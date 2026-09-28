// unified_attn_ext.cu
// PyTorch C++ extension wrapping the unified attention kernel.
// Compiled via torch.utils.cpp_extension.load() — no manual nvcc needed.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

// ============================================================================
// Kernel constants
// ============================================================================
constexpr int HEAD_DIM = 128;
constexpr int HMMA_K = 16;
constexpr int D_HEAD_TILES = HEAD_DIM / HMMA_K;  // 8

constexpr int BLOCK_M_PREFILL = 128;
constexpr int BLOCK_M_DECODE  = 16;
constexpr int BLOCK_N         = 128;  // shared by both paths — must be equal for bitwise consistency

// ============================================================================
// Online softmax state (FP32)
// ============================================================================
struct SoftmaxState {
    float row_max;
    float row_sum;
    float out[HEAD_DIM];
};

__device__ __forceinline__
void softmax_state_init(SoftmaxState& state) {
    state.row_max = -INFINITY;
    state.row_sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < HEAD_DIM; i++) {
        state.out[i] = 0.0f;
    }
}

__device__ __forceinline__
void softmax_rescale(SoftmaxState& state, float new_max) {
    float rescale = expf(state.row_max - new_max);
    state.row_sum *= rescale;
    #pragma unroll
    for (int i = 0; i < HEAD_DIM; i++) {
        state.out[i] *= rescale;
    }
    state.row_max = new_max;
}

// ============================================================================
// Reference QK^T dot product via FMA (matches HMMA accumulation order)
//
// Accumulation: d_head tiled into 8 chunks of 16, iterated sequentially.
// Within each chunk: sequential FMA over 16 elements.
// This is a REFERENCE implementation — correct but single-threaded.
// Production kernel would use PTX mma.sync.aligned.m16n8k16.
// ============================================================================
__device__ void compute_qk_dot(
    const __nv_bfloat16* __restrict__ q_smem,
    const __nv_bfloat16* __restrict__ k_smem,
    float* __restrict__ scores,
    int block_m,
    int block_n
) {
    for (int i = 0; i < block_m * block_n; i++) {
        scores[i] = 0.0f;
    }

    // d_head tiled into D_HEAD_TILES chunks of HMMA_K=16, sequential
    #pragma unroll
    for (int d_tile = 0; d_tile < D_HEAD_TILES; d_tile++) {
        int d_offset = d_tile * HMMA_K;
        for (int m = 0; m < block_m; m++) {
            for (int n = 0; n < block_n; n++) {
                float partial = 0.0f;
                #pragma unroll
                for (int k = 0; k < HMMA_K; k++) {
                    float q_val = __bfloat162float(q_smem[m * HEAD_DIM + d_offset + k]);
                    float k_val = __bfloat162float(k_smem[n * HEAD_DIM + d_offset + k]);
                    partial = fmaf(q_val, k_val, partial);
                }
                scores[m * block_n + n] += partial;
            }
        }
    }
}

// ============================================================================
// Attention output accumulation with online softmax
// ============================================================================
__device__ void compute_attn_output(
    float* __restrict__ scores,
    const __nv_bfloat16* __restrict__ v_smem,
    SoftmaxState* states,
    int block_m,
    int block_n,
    int valid_n
) {
    for (int m = 0; m < block_m; m++) {
        float block_max = -INFINITY;
        for (int n = 0; n < valid_n; n++) {
            block_max = fmaxf(block_max, scores[m * block_n + n]);
        }

        float new_max = fmaxf(states[m].row_max, block_max);
        softmax_rescale(states[m], new_max);

        float block_sum = 0.0f;
        for (int n = 0; n < valid_n; n++) {
            float p = expf(scores[m * block_n + n] - new_max);
            scores[m * block_n + n] = p;
            block_sum += p;
        }
        for (int n = valid_n; n < block_n; n++) {
            scores[m * block_n + n] = 0.0f;
        }
        states[m].row_sum += block_sum;

        for (int d = 0; d < HEAD_DIM; d++) {
            float v_acc = 0.0f;
            for (int n = 0; n < valid_n; n++) {
                v_acc += scores[m * block_n + n]
                         * __bfloat162float(v_smem[n * HEAD_DIM + d]);
            }
            states[m].out[d] += v_acc;
        }
    }
}

// ============================================================================
// Prefill kernel: processes BLOCK_M_PREFILL query tokens per CTA
// Grid: (num_q_blocks, batch * num_heads)
// ============================================================================
__global__ void unified_attn_prefill_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    int seq_len,
    int ctx_len,
    int num_heads,
    int num_kv_heads,
    float scale
) {
    int q_block_idx = blockIdx.x;
    int head_batch_idx = blockIdx.y;
    int batch_idx = head_batch_idx / num_heads;
    int head_idx = head_batch_idx % num_heads;
    int kv_head_idx = head_idx / (num_heads / num_kv_heads);

    int q_start = q_block_idx * BLOCK_M_PREFILL;
    int q_end = min(q_start + BLOCK_M_PREFILL, seq_len);
    int actual_m = q_end - q_start;
    if (actual_m <= 0) return;

    const __nv_bfloat16* q_ptr = Q + ((batch_idx * num_heads + head_idx) * seq_len) * HEAD_DIM;
    const __nv_bfloat16* k_ptr = K + ((batch_idx * num_kv_heads + kv_head_idx) * ctx_len) * HEAD_DIM;
    const __nv_bfloat16* v_ptr = V + ((batch_idx * num_kv_heads + kv_head_idx) * ctx_len) * HEAD_DIM;
    __nv_bfloat16* o_ptr = O + ((batch_idx * num_heads + head_idx) * seq_len) * HEAD_DIM;

    // Single-thread reference: all work on thread 0
    if (threadIdx.x != 0) return;

    // Process one query row at a time to avoid stack overflow.
    // BLOCK_M_PREFILL=128 SoftmaxState structs = 128*(2+128)*4 = ~67KB stack — too large.
    // Instead: outer loop over query rows, inner loop over KV blocks.
    // This matches the decode kernel's per-row processing exactly.
    __nv_bfloat16 k_local[BLOCK_N * HEAD_DIM];
    __nv_bfloat16 v_local[BLOCK_N * HEAD_DIM];
    float scores[BLOCK_N];

    int num_kv_blocks = (ctx_len + BLOCK_N - 1) / BLOCK_N;

    for (int m = 0; m < actual_m; m++) {
        int q_pos = q_start + m;

        // Load and scale this query row
        __nv_bfloat16 q_row[HEAD_DIM];
        for (int d = 0; d < HEAD_DIM; d++) {
            q_row[d] = __float2bfloat16(__bfloat162float(q_ptr[(q_start + m) * HEAD_DIM + d]) * scale);
        }

        SoftmaxState state;
        softmax_state_init(state);

        for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
            int kv_start = kv_block * BLOCK_N;
            int kv_end = min(kv_start + BLOCK_N, ctx_len);
            int valid_n = kv_end - kv_start;

            for (int i = 0; i < valid_n * HEAD_DIM; i++) {
                k_local[i] = k_ptr[kv_start * HEAD_DIM + i];
            }
            for (int i = 0; i < valid_n * HEAD_DIM; i++) {
                v_local[i] = v_ptr[kv_start * HEAD_DIM + i];
            }

            // QK^T for this single query row: same d_head tiling as decode
            for (int n = 0; n < valid_n; n++) {
                float dot = 0.0f;
                #pragma unroll
                for (int d_tile = 0; d_tile < D_HEAD_TILES; d_tile++) {
                    int d_off = d_tile * HMMA_K;
                    #pragma unroll
                    for (int k = 0; k < HMMA_K; k++) {
                        dot = fmaf(__bfloat162float(q_row[d_off + k]),
                                   __bfloat162float(k_local[n * HEAD_DIM + d_off + k]),
                                   dot);
                    }
                }
                // Causal mask: query at q_pos can only attend to kv_pos <= q_pos
                scores[n] = (kv_start + n <= q_pos) ? dot : -INFINITY;
            }
            for (int n = valid_n; n < BLOCK_N; n++) {
                scores[n] = -INFINITY;
            }

            // Online softmax update
            float block_max = -INFINITY;
            for (int n = 0; n < valid_n; n++) {
                block_max = fmaxf(block_max, scores[n]);
            }
            float new_max = fmaxf(state.row_max, block_max);
            softmax_rescale(state, new_max);

            float block_sum = 0.0f;
            for (int n = 0; n < valid_n; n++) {
                float p = expf(scores[n] - new_max);
                scores[n] = p;
                block_sum += p;
            }
            state.row_sum += block_sum;

            for (int d = 0; d < HEAD_DIM; d++) {
                float v_acc = 0.0f;
                for (int n = 0; n < valid_n; n++) {
                    v_acc += scores[n] * __bfloat162float(v_local[n * HEAD_DIM + d]);
                }
                state.out[d] += v_acc;
            }
        }

        float inv_sum = (state.row_sum > 0.0f) ? (1.0f / state.row_sum) : 0.0f;
        for (int d = 0; d < HEAD_DIM; d++) {
            o_ptr[(q_start + m) * HEAD_DIM + d] = __float2bfloat16(state.out[d] * inv_sum);
        }
    }
}

// ============================================================================
// Decode kernel: processes 1 query token per CTA (padded to BLOCK_M_DECODE=16)
// Grid: (1, batch * num_heads)
// ============================================================================
__global__ void unified_attn_decode_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    int ctx_len,
    int num_heads,
    int num_kv_heads,
    float scale
) {
    int head_batch_idx = blockIdx.y;
    int batch_idx = head_batch_idx / num_heads;
    int head_idx = head_batch_idx % num_heads;
    int kv_head_idx = head_idx / (num_heads / num_kv_heads);

    const __nv_bfloat16* q_ptr = Q + (batch_idx * num_heads + head_idx) * HEAD_DIM;
    const __nv_bfloat16* k_ptr = K + ((batch_idx * num_kv_heads + kv_head_idx) * ctx_len) * HEAD_DIM;
    const __nv_bfloat16* v_ptr = V + ((batch_idx * num_kv_heads + kv_head_idx) * ctx_len) * HEAD_DIM;
    __nv_bfloat16* o_ptr = O + (batch_idx * num_heads + head_idx) * HEAD_DIM;

    if (threadIdx.x != 0) return;

    // Load and scale the single query token
    __nv_bfloat16 q_row[HEAD_DIM];
    for (int d = 0; d < HEAD_DIM; d++) {
        q_row[d] = __float2bfloat16(__bfloat162float(q_ptr[d]) * scale);
    }

    __nv_bfloat16 k_local[BLOCK_N * HEAD_DIM];
    __nv_bfloat16 v_local[BLOCK_N * HEAD_DIM];
    float scores[BLOCK_N];

    SoftmaxState state;
    softmax_state_init(state);

    int num_kv_blocks = (ctx_len + BLOCK_N - 1) / BLOCK_N;
    for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        int kv_start = kv_block * BLOCK_N;
        int kv_end = min(kv_start + BLOCK_N, ctx_len);
        int valid_n = kv_end - kv_start;

        for (int i = 0; i < valid_n * HEAD_DIM; i++) {
            k_local[i] = k_ptr[kv_start * HEAD_DIM + i];
        }
        for (int i = 0; i < valid_n * HEAD_DIM; i++) {
            v_local[i] = v_ptr[kv_start * HEAD_DIM + i];
        }

        // QK^T: same d_head tiling as prefill (MUST match for consistency)
        for (int n = 0; n < valid_n; n++) {
            float dot = 0.0f;
            #pragma unroll
            for (int d_tile = 0; d_tile < D_HEAD_TILES; d_tile++) {
                int d_off = d_tile * HMMA_K;
                #pragma unroll
                for (int k = 0; k < HMMA_K; k++) {
                    dot = fmaf(__bfloat162float(q_row[d_off + k]),
                               __bfloat162float(k_local[n * HEAD_DIM + d_off + k]),
                               dot);
                }
            }
            scores[n] = dot;  // no causal mask: decode query attends to all ctx_len tokens
        }
        for (int n = valid_n; n < BLOCK_N; n++) {
            scores[n] = -INFINITY;
        }

        // Online softmax update (identical formula to prefill)
        float block_max = -INFINITY;
        for (int n = 0; n < valid_n; n++) {
            block_max = fmaxf(block_max, scores[n]);
        }
        float new_max = fmaxf(state.row_max, block_max);
        softmax_rescale(state, new_max);

        float block_sum = 0.0f;
        for (int n = 0; n < valid_n; n++) {
            float p = expf(scores[n] - new_max);
            scores[n] = p;
            block_sum += p;
        }
        state.row_sum += block_sum;

        for (int d = 0; d < HEAD_DIM; d++) {
            float v_acc = 0.0f;
            for (int n = 0; n < valid_n; n++) {
                v_acc += scores[n] * __bfloat162float(v_local[n * HEAD_DIM + d]);
            }
            state.out[d] += v_acc;
        }
    }

    float inv_sum = (state.row_sum > 0.0f) ? (1.0f / state.row_sum) : 0.0f;
    for (int d = 0; d < HEAD_DIM; d++) {
        o_ptr[d] = __float2bfloat16(state.out[d] * inv_sum);
    }
}

// ============================================================================
// PyTorch C++ bindings
// ============================================================================

// Q: [batch, num_heads, seq_len, head_dim]  (BF16)
// K: [batch, num_kv_heads, ctx_len, head_dim] (BF16)
// V: [batch, num_kv_heads, ctx_len, head_dim] (BF16)
// Returns O: [batch, num_heads, seq_len, head_dim] (BF16)
torch::Tensor unified_prefill(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    int num_heads, int num_kv_heads
) {
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous(), "Inputs must be contiguous");
    TORCH_CHECK(Q.dtype() == torch::kBFloat16, "Q must be BF16");

    int batch = Q.size(0);
    int seq_len = Q.size(2);
    int ctx_len = K.size(2);
    int head_dim = Q.size(3);
    TORCH_CHECK(head_dim == HEAD_DIM, "head_dim must be 128");

    auto O = torch::zeros_like(Q);
    float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    int num_q_blocks = (seq_len + BLOCK_M_PREFILL - 1) / BLOCK_M_PREFILL;
    dim3 grid(num_q_blocks, batch * num_heads);
    dim3 block(32);  // 1 warp, only thread 0 does work (reference impl)

    unified_attn_prefill_kernel<<<grid, block>>>(
        reinterpret_cast<const __nv_bfloat16*>(Q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(K.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(V.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(O.data_ptr()),
        seq_len, ctx_len, num_heads, num_kv_heads, scale
    );
    return O;
}

// Q: [batch, num_heads, 1, head_dim]  (BF16) — single decode token
// K: [batch, num_kv_heads, ctx_len, head_dim] (BF16)
// V: [batch, num_kv_heads, ctx_len, head_dim] (BF16)
// Returns O: [batch, num_heads, 1, head_dim] (BF16)
torch::Tensor unified_decode(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    int num_heads, int num_kv_heads
) {
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous(), "Inputs must be contiguous");
    TORCH_CHECK(Q.size(2) == 1, "Decode Q must have seq_len=1");

    int batch = Q.size(0);
    int ctx_len = K.size(2);
    int head_dim = Q.size(3);
    TORCH_CHECK(head_dim == HEAD_DIM, "head_dim must be 128");

    auto O = torch::zeros_like(Q);
    float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    dim3 grid(1, batch * num_heads);
    dim3 block(32);

    unified_attn_decode_kernel<<<grid, block>>>(
        reinterpret_cast<const __nv_bfloat16*>(Q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(K.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(V.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(O.data_ptr()),
        ctx_len, num_heads, num_kv_heads, scale
    );
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("unified_prefill", &unified_prefill, "Unified attention prefill (reference)");
    m.def("unified_decode", &unified_decode, "Unified attention decode (reference)");
}
