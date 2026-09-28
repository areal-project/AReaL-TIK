// unified_attn_ext.cu — kernel_02: multi-warp, shared memory, split-K over d_head
//
// Architecture:
//   4 warps per CTA (128 threads). Each warp owns 32 consecutive d_head elements.
//   K and V blocks are loaded cooperatively into shared memory by all 128 threads.
//   Dot product: each warp computes partial[warp_id][n] = sum_{d in warp_range} q[d]*k[n][d]
//   Warp 0 reduces the 4 partials sequentially (fixed order → bitwise consistent).
//   Warp 0 runs online softmax (sequential dependency, cheap).
//   All 4 warps accumulate V: warp w updates out[w*32..(w+1)*32-1].
//
// Bitwise consistency guarantee:
//   Both prefill and decode use the same 4-warp structure, same BLOCK_N=128,
//   same d_head partitioning, same sequential partial reduction order.
//   The floating-point path for any query position is identical in both paths.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>

// ============================================================================
// Constants
// ============================================================================
constexpr int HEAD_DIM       = 128;
constexpr int HMMA_K         = 16;
constexpr int D_HEAD_TILES   = HEAD_DIM / HMMA_K;   // 8
constexpr int NUM_WARPS      = 4;
constexpr int WARP_SIZE      = 32;
constexpr int BLOCK_THREADS  = NUM_WARPS * WARP_SIZE; // 128
constexpr int D_PER_WARP     = HEAD_DIM / NUM_WARPS;  // 32
constexpr int BLOCK_M_PREFILL = 128;
constexpr int BLOCK_N        = 128;  // must be equal in prefill and decode for bitwise consistency

// ============================================================================
// Shared memory layout (per CTA):
//
//   k_smem  [BLOCK_N * HEAD_DIM]  floats  = 128*128*4 = 64 KB
//   v_smem  [BLOCK_N * HEAD_DIM]  floats  = 128*128*4 = 64 KB
//   partial [NUM_WARPS * BLOCK_N] floats  = 4*128*4   = 2 KB
//   scores  [BLOCK_N]             floats  = 128*4     = 0.5 KB
//
//   Total: ~130.5 KB — fits in Hopper (228 KB per SM)
// ============================================================================
constexpr int SMEM_K_OFFSET       = 0;
constexpr int SMEM_V_OFFSET       = BLOCK_N * HEAD_DIM;
constexpr int SMEM_PARTIAL_OFFSET = 2 * BLOCK_N * HEAD_DIM;
constexpr int SMEM_SCORES_OFFSET  = SMEM_PARTIAL_OFFSET + NUM_WARPS * BLOCK_N;
constexpr int SMEM_FLOATS         = SMEM_SCORES_OFFSET + BLOCK_N + 1;  // +1 for rescale broadcast

// ============================================================================
// Online softmax state — held in registers by warp 0 only
// ============================================================================
struct SoftmaxState {
    float row_max;
    float row_sum;
    float out[HEAD_DIM];
};

__device__ __forceinline__
void softmax_state_init(SoftmaxState& s) {
    s.row_max = -INFINITY;
    s.row_sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < HEAD_DIM; i++) s.out[i] = 0.0f;
}

__device__ __forceinline__
void softmax_rescale(SoftmaxState& s, float new_max) {
    float r = expf(s.row_max - new_max);
    s.row_sum *= r;
    #pragma unroll
    for (int i = 0; i < HEAD_DIM; i++) s.out[i] *= r;
    s.row_max = new_max;
}

// ============================================================================
// Core per-query-row computation shared by prefill and decode.
//
// Called once per query row. All 128 threads participate.
// q_row[HEAD_DIM]: the scaled query row, held in registers by ALL threads
//   (each thread holds the full q_row — 128 floats = 512 bytes per thread,
//    acceptable for a reference impl; production would use register tiling).
// smem: pointer to the CTA's shared memory block.
// k_ptr, v_ptr: pointers to this head's K and V in global memory.
// ctx_len: number of KV tokens to attend to.
// q_pos: absolute position of this query token (for causal mask).
// state: softmax state, maintained by warp 0 only.
// ============================================================================
__device__ void process_query_row(
    const float* __restrict__ q_row,
    const __nv_bfloat16* __restrict__ k_ptr,
    const __nv_bfloat16* __restrict__ v_ptr,
    float* __restrict__ smem,
    int ctx_len,
    int q_pos,
    SoftmaxState& state,
    float* __restrict__ out_row  // output accumulator [HEAD_DIM], warp-partitioned
) {
    float* k_smem      = smem + SMEM_K_OFFSET;
    float* v_smem      = smem + SMEM_V_OFFSET;
    float* partial_smem = smem + SMEM_PARTIAL_OFFSET;
    float* scores_smem = smem + SMEM_SCORES_OFFSET;

    int tid      = threadIdx.x;
    int warp_id  = tid / WARP_SIZE;
    int lane_id  = tid % WARP_SIZE;

    // Each warp owns d_head elements [warp_id*32 .. (warp_id+1)*32 - 1]
    int d_start  = warp_id * D_PER_WARP;

    int num_kv_blocks = (ctx_len + BLOCK_N - 1) / BLOCK_N;

    for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        int kv_start = kv_block * BLOCK_N;
        int kv_end   = min(kv_start + BLOCK_N, ctx_len);
        int valid_n  = kv_end - kv_start;

        // ----------------------------------------------------------------
        // Step 1: Cooperative load of K block into shared memory.
        // 128 threads load BLOCK_N * HEAD_DIM = 16384 floats.
        // Each thread loads 16384 / 128 = 128 elements.
        // ----------------------------------------------------------------
        for (int i = tid; i < valid_n * HEAD_DIM; i += BLOCK_THREADS) {
            k_smem[i] = __bfloat162float(k_ptr[(kv_start) * HEAD_DIM + i]);
        }
        __syncthreads();

        // ----------------------------------------------------------------
        // Step 2: Each warp computes partial dot products.
        // Warp w: partial[w][n] = sum_{d=d_start}^{d_start+31} q[d] * k[n][d]
        //
        // Each thread in the warp handles one n value (lane_id = n index
        // within a group). Since BLOCK_N=128 and WARP_SIZE=32, each thread
        // handles 128/32 = 4 values of n.
        //
        // Accumulation order: d_tile 0..7, within each tile k=0..15.
        // This is the SAME order as kernel_01 — bitwise consistent.
        // ----------------------------------------------------------------
        for (int n_base = 0; n_base < BLOCK_N; n_base += WARP_SIZE) {
            int n = n_base + lane_id;
            float dot = 0.0f;
            if (n < valid_n) {
                #pragma unroll
                for (int d_tile = 0; d_tile < D_HEAD_TILES; d_tile++) {
                    int d_off = d_tile * HMMA_K;
                    // Only accumulate the d elements owned by this warp
                    if (d_off >= d_start && d_off < d_start + D_PER_WARP) {
                        #pragma unroll
                        for (int k = 0; k < HMMA_K; k++) {
                            dot = fmaf(q_row[d_off + k],
                                       k_smem[n * HEAD_DIM + d_off + k],
                                       dot);
                        }
                    }
                }
            }
            // Write partial to shared memory: partial[warp_id][n]
            if (n < BLOCK_N) {
                partial_smem[warp_id * BLOCK_N + n] = (n < valid_n) ? dot : 0.0f;
            }
        }
        __syncthreads();

        // ----------------------------------------------------------------
        // Step 3: Warp 0 reduces partials sequentially (fixed order).
        // scores[n] = partial[0][n] + partial[1][n] + partial[2][n] + partial[3][n]
        // Sequential addition — same order always — bitwise consistent.
        // Then apply causal mask and online softmax.
        // ----------------------------------------------------------------
        if (warp_id == 0) {
            for (int n = lane_id; n < BLOCK_N; n += WARP_SIZE) {
                float s = partial_smem[0 * BLOCK_N + n];
                s += partial_smem[1 * BLOCK_N + n];
                s += partial_smem[2 * BLOCK_N + n];
                s += partial_smem[3 * BLOCK_N + n];
                // Causal mask
                bool masked = (n >= valid_n) || (kv_start + n > q_pos);
                scores_smem[n] = masked ? -INFINITY : s;
            }
        }
        __syncthreads();

        // ----------------------------------------------------------------
        // Step 4: Warp 0 thread 0 runs online softmax.
        // Broadcasts rescale factor so all threads can update out_row.
        // ----------------------------------------------------------------
        // Borrow one float from smem for the rescale broadcast.
        // Safe: partial_smem and scores_smem are separate regions.
        float* rescale_smem = smem + SMEM_SCORES_OFFSET + BLOCK_N;  // 1 float after scores

        if (tid == 0) {
            float block_max = -INFINITY;
            for (int n = 0; n < valid_n; n++) {
                block_max = fmaxf(block_max, scores_smem[n]);
            }
            float new_max = fmaxf(state.row_max, block_max);
            // Compute rescale factor before updating state
            float rescale = expf(state.row_max - new_max);
            rescale_smem[0] = rescale;
            state.row_max = new_max;
            state.row_sum *= rescale;

            float block_sum = 0.0f;
            for (int n = 0; n < valid_n; n++) {
                float p = expf(scores_smem[n] - new_max);
                scores_smem[n] = p;
                block_sum += p;
            }
            for (int n = valid_n; n < BLOCK_N; n++) {
                scores_smem[n] = 0.0f;
            }
            state.row_sum += block_sum;
        }
        __syncthreads();

        // All threads apply the rescale to their out_row slice
        {
            float rescale = rescale_smem[0];
            for (int d_off = 0; d_off < D_PER_WARP; d_off++) {
                out_row[d_start + d_off] *= rescale;
            }
        }
        __syncthreads();

        // ----------------------------------------------------------------
        // Step 5: Cooperative load of V block into shared memory.
        // ----------------------------------------------------------------
        for (int i = tid; i < valid_n * HEAD_DIM; i += BLOCK_THREADS) {
            v_smem[i] = __bfloat162float(v_ptr[(kv_start) * HEAD_DIM + i]);
        }
        // Zero out padding
        for (int i = tid + valid_n * HEAD_DIM; i < BLOCK_N * HEAD_DIM; i += BLOCK_THREADS) {
            v_smem[i] = 0.0f;
        }
        __syncthreads();

        // ----------------------------------------------------------------
        // Step 6: All 4 warps accumulate V output in parallel.
        // Warp w updates out_row[d_start .. d_start+31].
        // Each thread in the warp handles one d value (lane_id = d offset
        // within the warp's range).
        // ----------------------------------------------------------------
        for (int d_off = 0; d_off < D_PER_WARP; d_off++) {
            int d = d_start + d_off;
            float v_acc = 0.0f;
            for (int n = 0; n < valid_n; n++) {
                v_acc += scores_smem[n] * v_smem[n * HEAD_DIM + d];
            }
            out_row[d] += v_acc;
        }
        __syncthreads();
    }
}

// ============================================================================
// Prefill kernel
// Grid: (num_q_blocks, batch * num_heads)
// Block: 128 threads (4 warps)
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
    extern __shared__ float smem[];

    int q_block_idx    = blockIdx.x;
    int head_batch_idx = blockIdx.y;
    int batch_idx      = head_batch_idx / num_heads;
    int head_idx       = head_batch_idx % num_heads;
    int kv_head_idx    = head_idx / (num_heads / num_kv_heads);

    int q_start  = q_block_idx * BLOCK_M_PREFILL;
    int q_end    = min(q_start + BLOCK_M_PREFILL, seq_len);
    int actual_m = q_end - q_start;
    if (actual_m <= 0) return;

    const __nv_bfloat16* q_ptr = Q + ((batch_idx * num_heads    + head_idx)    * seq_len) * HEAD_DIM;
    const __nv_bfloat16* k_ptr = K + ((batch_idx * num_kv_heads + kv_head_idx) * ctx_len) * HEAD_DIM;
    const __nv_bfloat16* v_ptr = V + ((batch_idx * num_kv_heads + kv_head_idx) * ctx_len) * HEAD_DIM;
    __nv_bfloat16*        o_ptr = O + ((batch_idx * num_heads    + head_idx)    * seq_len) * HEAD_DIM;

    int warp_id = threadIdx.x / WARP_SIZE;
    int d_start = warp_id * D_PER_WARP;

    // Process one query row at a time (outer loop over m)
    for (int m = 0; m < actual_m; m++) {
        int q_pos = q_start + m;

        // Load and scale Q row into registers (all threads hold full q_row)
        float q_row[HEAD_DIM];
        for (int d = threadIdx.x; d < HEAD_DIM; d += BLOCK_THREADS) {
            q_row[d] = __bfloat162float(q_ptr[(q_start + m) * HEAD_DIM + d]) * scale;
        }
        // Broadcast: all threads need full q_row. Use shared memory to share.
        // Reuse k_smem temporarily (safe: __syncthreads before k_smem is used).
        float* q_broadcast = smem;  // borrow first HEAD_DIM floats of smem
        for (int d = threadIdx.x; d < HEAD_DIM; d += BLOCK_THREADS) {
            q_broadcast[d] = q_row[d];
        }
        __syncthreads();
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d++) {
            q_row[d] = q_broadcast[d];
        }
        __syncthreads();

        // Per-warp output accumulator (each warp owns D_PER_WARP=32 dims)
        float out_row[HEAD_DIM];
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d++) out_row[d] = 0.0f;

        // Softmax state — only warp 0 thread 0 uses it, but declared for all
        SoftmaxState state;
        if (threadIdx.x == 0) softmax_state_init(state);

        process_query_row(q_row, k_ptr, v_ptr, smem,
                          ctx_len, q_pos, state, out_row);

        // Write output: warp w writes dims [d_start .. d_start+31]
        // Warp 0 thread 0 holds the final softmax state
        // Broadcast inv_sum via shared memory
        float* inv_sum_smem = smem;  // borrow 1 float
        if (threadIdx.x == 0) {
            inv_sum_smem[0] = (state.row_sum > 0.0f) ? (1.0f / state.row_sum) : 0.0f;
        }
        __syncthreads();
        float inv_sum = inv_sum_smem[0];
        __syncthreads();

        // Each warp writes its d_range to output
        // But out_row was accumulated independently per warp — each warp
        // only accumulated its own d_range in process_query_row step 6.
        for (int d_off = 0; d_off < D_PER_WARP; d_off++) {
            int d = d_start + d_off;
            o_ptr[(q_start + m) * HEAD_DIM + d] =
                __float2bfloat16(out_row[d] * inv_sum);
        }
        __syncthreads();
    }
}

// ============================================================================
// Decode kernel
// Grid: (1, batch * num_heads)
// Block: 128 threads (4 warps) — same structure as prefill
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
    extern __shared__ float smem[];

    int head_batch_idx = blockIdx.y;
    int batch_idx      = head_batch_idx / num_heads;
    int head_idx       = head_batch_idx % num_heads;
    int kv_head_idx    = head_idx / (num_heads / num_kv_heads);

    const __nv_bfloat16* q_ptr = Q + (batch_idx * num_heads    + head_idx)    * HEAD_DIM;
    const __nv_bfloat16* k_ptr = K + ((batch_idx * num_kv_heads + kv_head_idx) * ctx_len) * HEAD_DIM;
    const __nv_bfloat16* v_ptr = V + ((batch_idx * num_kv_heads + kv_head_idx) * ctx_len) * HEAD_DIM;
    __nv_bfloat16*        o_ptr = O + (batch_idx * num_heads    + head_idx)    * HEAD_DIM;

    int warp_id = threadIdx.x / WARP_SIZE;
    int d_start = warp_id * D_PER_WARP;

    // Load and scale Q row, broadcast to all threads
    float q_row[HEAD_DIM];
    float* q_broadcast = smem;
    for (int d = threadIdx.x; d < HEAD_DIM; d += BLOCK_THREADS) {
        q_broadcast[d] = __bfloat162float(q_ptr[d]) * scale;
    }
    __syncthreads();
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; d++) {
        q_row[d] = q_broadcast[d];
    }
    __syncthreads();

    float out_row[HEAD_DIM];
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; d++) out_row[d] = 0.0f;

    SoftmaxState state;
    if (threadIdx.x == 0) softmax_state_init(state);

    // q_pos = ctx_len - 1: decode query attends to all ctx_len tokens (no future)
    // Pass ctx_len as q_pos so causal mask (kv_start + n > q_pos) never triggers
    process_query_row(q_row, k_ptr, v_ptr, smem,
                      ctx_len, ctx_len - 1, state, out_row);

    float* inv_sum_smem = smem;
    if (threadIdx.x == 0) {
        inv_sum_smem[0] = (state.row_sum > 0.0f) ? (1.0f / state.row_sum) : 0.0f;
    }
    __syncthreads();
    float inv_sum = inv_sum_smem[0];
    __syncthreads();

    for (int d_off = 0; d_off < D_PER_WARP; d_off++) {
        int d = d_start + d_off;
        o_ptr[d] = __float2bfloat16(out_row[d] * inv_sum);
    }
}

// ============================================================================
// PyTorch bindings
// ============================================================================
torch::Tensor unified_prefill(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    int num_heads, int num_kv_heads
) {
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous(), "Inputs must be contiguous");
    TORCH_CHECK(Q.dtype() == torch::kBFloat16, "Q must be BF16");

    int batch   = Q.size(0);
    int seq_len = Q.size(2);
    int ctx_len = K.size(2);
    int head_dim = Q.size(3);
    TORCH_CHECK(head_dim == HEAD_DIM, "head_dim must be 128");

    auto O = torch::zeros_like(Q);
    float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    int num_q_blocks = (seq_len + BLOCK_M_PREFILL - 1) / BLOCK_M_PREFILL;
    dim3 grid(num_q_blocks, batch * num_heads);
    dim3 block(BLOCK_THREADS);
    size_t smem_bytes = SMEM_FLOATS * sizeof(float);

    cudaFuncSetAttribute(unified_attn_prefill_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem_bytes);

    unified_attn_prefill_kernel<<<grid, block, smem_bytes>>>(
        reinterpret_cast<const __nv_bfloat16*>(Q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(K.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(V.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(O.data_ptr()),
        seq_len, ctx_len, num_heads, num_kv_heads, scale
    );
    return O;
}

torch::Tensor unified_decode(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    int num_heads, int num_kv_heads
) {
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous(), "Inputs must be contiguous");
    TORCH_CHECK(Q.size(2) == 1, "Decode Q must have seq_len=1");

    int batch    = Q.size(0);
    int ctx_len  = K.size(2);
    int head_dim = Q.size(3);
    TORCH_CHECK(head_dim == HEAD_DIM, "head_dim must be 128");

    auto O = torch::zeros_like(Q);
    float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    dim3 grid(1, batch * num_heads);
    dim3 block(BLOCK_THREADS);
    size_t smem_bytes = SMEM_FLOATS * sizeof(float);

    cudaFuncSetAttribute(unified_attn_decode_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem_bytes);

    unified_attn_decode_kernel<<<grid, block, smem_bytes>>>(
        reinterpret_cast<const __nv_bfloat16*>(Q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(K.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(V.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(O.data_ptr()),
        ctx_len, num_heads, num_kv_heads, scale
    );
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("unified_prefill", &unified_prefill, "Unified attention prefill (multi-warp)");
    m.def("unified_decode",  &unified_decode,  "Unified attention decode (multi-warp)");
}
