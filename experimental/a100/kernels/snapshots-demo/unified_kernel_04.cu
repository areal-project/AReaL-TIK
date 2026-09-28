// unified_attn_ext.cu — kernel_04: HMMA via raw PTX mma.sync.aligned.m16n8k16
//
// Fragment layouts (verified against PTX ISA + blog reference):
//
// A fragment [16,16] bf16 — ldmatrix.x4:
//   Thread t provides ptr to: row = t%16, col = (t/16)*8
//   Threads 0-15 → left half (cols 0-7), threads 16-31 → right half (cols 8-15)
//
// B fragment [16,8] bf16 — ldmatrix.x2.trans:
//   .trans loads B in col-major format (K rows × N cols → N cols × K rows)
//   Thread t provides ptr to: row = t%16, col = 0
//   (threads 16-31 mirror threads 0-15 for x2)
//
// D/C fragment [16,8] f32 — 4 floats per thread:
//   groupID = lane_id >> 2,  tid_in_group = lane_id % 4
//   d[0] → (row=groupID,   col=tid_in_group*2)
//   d[1] → (row=groupID,   col=tid_in_group*2+1)
//   d[2] → (row=groupID+8, col=tid_in_group*2)
//   d[3] → (row=groupID+8, col=tid_in_group*2+1)
//
// Tile structure (4 warps, each handles N_TILES_PER_WARP=4 N-tiles):
//   Outer loop: M-tiles of 16 query rows (BLOCK_M_PREFILL/16 = 8 tiles)
//   Inner loop: KV blocks of BLOCK_N=128 tokens
//   Per KV block: scores[16,128] = 16 N-tiles × [16,8], 4 warps × 4 tiles each
//
// Bitwise consistency:
//   Both prefill and decode use identical PTX mma + ldmatrix calls.
//   Decode pads Q to 16 rows (15 zeros). HMMA accumulation is hardware-fixed.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>

constexpr int HEAD_DIM        = 128;
constexpr int NUM_WARPS       = 4;
constexpr int WARP_SIZE       = 32;
constexpr int BLOCK_THREADS   = NUM_WARPS * WARP_SIZE;  // 128
constexpr int BLOCK_M_PREFILL = 128;
constexpr int BLOCK_N         = 128;

constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 16;

constexpr int N_TILES          = BLOCK_N / MMA_N;          // 16
constexpr int K_TILES          = HEAD_DIM / MMA_K;         // 8
constexpr int N_TILES_PER_WARP = N_TILES / NUM_WARPS;      // 4

// Shared memory layout (bytes):
//   q_tile:   MMA_M * HEAD_DIM * 2 (bf16) = 4096 B
//   k_smem:   BLOCK_N * HEAD_DIM * 2 (bf16) = 32768 B
//   scores:   MMA_M * BLOCK_N * 4 (f32) = 8192 B
//   out_tile: MMA_M * HEAD_DIM * 4 (f32) = 8192 B
//   softmax:  MMA_M * 2 * 4 (row_max + row_sum) = 128 B
//   rescale:  MMA_M * 4 = 64 B
//   v_smem reuses k_smem (loaded after dot product)
//   Total: 4096 + 32768 + 8192 + 8192 + 128 + 64 = 53440 B ≈ 52 KB → needs opt-in

constexpr int Q_TILE_OFF   = 0;
constexpr int K_SMEM_OFF   = Q_TILE_OFF   + MMA_M * HEAD_DIM * 2;
constexpr int SCORES_OFF   = K_SMEM_OFF   + BLOCK_N * HEAD_DIM * 2;
constexpr int OUT_TILE_OFF = SCORES_OFF   + MMA_M * BLOCK_N * 4;
constexpr int SOFTMAX_OFF  = OUT_TILE_OFF + MMA_M * HEAD_DIM * 4;
constexpr int RESCALE_OFF  = SOFTMAX_OFF  + MMA_M * 2 * 4;
constexpr int SMEM_BYTES   = RESCALE_OFF  + MMA_M * 4;
constexpr int V_SMEM_OFF   = K_SMEM_OFF;  // reuse K smem for V

// ============================================================================
// PTX wrappers
// ============================================================================

// mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
__device__ __forceinline__
void mma_bf16(float& d0, float& d1, float& d2, float& d3,
              uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
              uint32_t b0, uint32_t b1,
              float c0, float c1, float c2, float c3) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3)
    );
}

// ldmatrix.x4 for A [16,16] bf16 (row-major in smem)
// Thread t provides ptr to row=(t%16), col=(t/16)*8
__device__ __forceinline__
void ldmatrix_a(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
                const __nv_bfloat16* smem_ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(addr)
    );
}

// Load B fragment manually from K smem [BLOCK_N, HEAD_DIM] row-major.
// B tile: K[n_col:n_col+8, kt*16:(kt+1)*16] — shape [N=8, K=16] row-major.
// mma.row.col expects B in col-major [K=16, N=8].
// Each thread holds 4 bf16 values packed as 2 uint32.
// PTX register layout for B in mma.m16n8k16:
//   Thread t holds B[k, n] where:
//     k = (t%4)*4 + {0,1,2,3} across the 2 registers (8 k-values per thread)
//     n = t/4  (0..7, one n-column per group of 4 threads)
//   Specifically: b0 = {B[k0,n], B[k1,n]}, b1 = {B[k8,n], B[k9,n]}
//   where k0=(t%4)*2, k1=k0+1, k8=k0+8, k9=k0+9, n=t/4
__device__ __forceinline__
void load_b_fragment(uint32_t& b0, uint32_t& b1,
                     const __nv_bfloat16* k_smem,
                     int n_col, int kt, int lane_id) {
    int n = n_col + (lane_id / 4);          // which N column (0..7 within tile)
    int k_base = (lane_id % 4) * 2;         // k offset within first half (0,2,4,6)
    int k0 = kt * MMA_K + k_base;           // first k index
    int k1 = k0 + 1;
    int k8 = k0 + 8;                        // second half
    int k9 = k8 + 1;

    __nv_bfloat16 v0 = k_smem[n * HEAD_DIM + k0];
    __nv_bfloat16 v1 = k_smem[n * HEAD_DIM + k1];
    __nv_bfloat16 v8 = k_smem[n * HEAD_DIM + k8];
    __nv_bfloat16 v9 = k_smem[n * HEAD_DIM + k9];

    b0 = (reinterpret_cast<uint16_t&>(v1) << 16) | reinterpret_cast<uint16_t&>(v0);
    b1 = (reinterpret_cast<uint16_t&>(v9) << 16) | reinterpret_cast<uint16_t&>(v8);
}

// ============================================================================
// Compute scores[MMA_M, BLOCK_N] for one M-tile using HMMA.
// ============================================================================
__device__ void compute_scores_hmma(
    const __nv_bfloat16* __restrict__ q_tile,  // [MMA_M, HEAD_DIM] smem, row-major
    const __nv_bfloat16* __restrict__ k_smem,  // [BLOCK_N, HEAD_DIM] smem, row-major
    float* __restrict__ scores,                 // [MMA_M, BLOCK_N] smem, row-major
    int valid_n
) {
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    int n_tile_start = warp_id * N_TILES_PER_WARP;

    // Accumulator: 4 N-tiles × 4 f32 per thread
    float d[N_TILES_PER_WARP][4] = {};

    for (int kt = 0; kt < K_TILES; kt++) {
        // Load A fragment: Q_tile[0:16, kt*16:(kt+1)*16]
        // ldmatrix.x4: thread t → row = t%16, col = (t/16)*8
        // In row-major smem with stride HEAD_DIM:
        //   ptr = q_tile + row * HEAD_DIM + kt*MMA_K + (t/16)*8
        uint32_t a0, a1, a2, a3;
        {
            int row = lane_id % MMA_M;
            int col = (lane_id / MMA_M) * 8 + kt * MMA_K;
            ldmatrix_a(a0, a1, a2, a3, q_tile + row * HEAD_DIM + col);
        }

        for (int nt = 0; nt < N_TILES_PER_WARP; nt++) {
            int n_col = (n_tile_start + nt) * MMA_N;

            uint32_t b0, b1;
            load_b_fragment(b0, b1, k_smem, n_col, kt, lane_id);

            mma_bf16(d[nt][0], d[nt][1], d[nt][2], d[nt][3],
                     a0, a1, a2, a3, b0, b1,
                     d[nt][0], d[nt][1], d[nt][2], d[nt][3]);
        }
    }

    // Scatter D to scores smem.
    // D layout: groupID = lane_id>>2, tid_in_group = lane_id%4
    //   d[0] → (row=groupID,   col=tid_in_group*2)
    //   d[1] → (row=groupID,   col=tid_in_group*2+1)
    //   d[2] → (row=groupID+8, col=tid_in_group*2)
    //   d[3] → (row=groupID+8, col=tid_in_group*2+1)
    {
        int group_id     = lane_id >> 2;
        int tid_in_group = lane_id % 4;
        int row0 = group_id;
        int row1 = group_id + 8;
        int col0 = tid_in_group * 2;
        int col1 = col0 + 1;

        for (int nt = 0; nt < N_TILES_PER_WARP; nt++) {
            int n_base = (n_tile_start + nt) * MMA_N;
            scores[row0 * BLOCK_N + n_base + col0] = d[nt][0];
            scores[row0 * BLOCK_N + n_base + col1] = d[nt][1];
            scores[row1 * BLOCK_N + n_base + col0] = d[nt][2];
            scores[row1 * BLOCK_N + n_base + col1] = d[nt][3];
        }
    }
}

// ============================================================================
// Process one M-tile (MMA_M=16 query rows) against all KV blocks.
// ============================================================================
__device__ void process_m_tile(
    char* __restrict__ smem,
    const __nv_bfloat16* __restrict__ k_ptr,
    const __nv_bfloat16* __restrict__ v_ptr,
    int ctx_len,
    int q_tile_start,
    int actual_rows
) {
    __nv_bfloat16* q_tile   = reinterpret_cast<__nv_bfloat16*>(smem + Q_TILE_OFF);
    __nv_bfloat16* k_smem   = reinterpret_cast<__nv_bfloat16*>(smem + K_SMEM_OFF);
    float*         scores   = reinterpret_cast<float*>(smem + SCORES_OFF);
    float*         out_tile = reinterpret_cast<float*>(smem + OUT_TILE_OFF);
    float*         softmax  = reinterpret_cast<float*>(smem + SOFTMAX_OFF);
    float*         rescale  = reinterpret_cast<float*>(smem + RESCALE_OFF);

    int tid = threadIdx.x;

    for (int i = tid; i < MMA_M * HEAD_DIM; i += BLOCK_THREADS) out_tile[i] = 0.0f;
    if (tid == 0) {
        for (int m = 0; m < MMA_M; m++) {
            softmax[m * 2]     = -INFINITY;
            softmax[m * 2 + 1] = 0.0f;
        }
    }
    __syncthreads();

    float inv_sqrt = 1.0f / sqrtf((float)HEAD_DIM);
    int num_kv_blocks = (ctx_len + BLOCK_N - 1) / BLOCK_N;

    for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        int kv_start = kv_block * BLOCK_N;
        int kv_end   = min(kv_start + BLOCK_N, ctx_len);
        int valid_n  = kv_end - kv_start;

        // Load K block (bf16, cooperative)
        for (int i = tid; i < valid_n * HEAD_DIM; i += BLOCK_THREADS)
            k_smem[i] = k_ptr[kv_start * HEAD_DIM + i];
        for (int i = tid + valid_n * HEAD_DIM; i < BLOCK_N * HEAD_DIM; i += BLOCK_THREADS)
            k_smem[i] = __float2bfloat16(0.0f);
        __syncthreads();

        // Compute scores via HMMA
        compute_scores_hmma(q_tile, k_smem, scores, valid_n);
        __syncthreads();

        // Apply scale + causal mask (thread 0)
        if (tid == 0) {
            for (int m = 0; m < MMA_M; m++) {
                int q_pos = q_tile_start + m;
                for (int n = 0; n < BLOCK_N; n++) {
                    float s = scores[m * BLOCK_N + n] * inv_sqrt;
                    bool masked = (n >= valid_n) || (kv_start + n > q_pos)
                                  || (m >= actual_rows);
                    scores[m * BLOCK_N + n] = masked ? -INFINITY : s;
                }
            }
        }
        __syncthreads();

        // Online softmax + broadcast rescale (thread 0)
        if (tid == 0) {
            for (int m = 0; m < MMA_M; m++) {
                float old_max = softmax[m * 2];
                float bmax = -INFINITY;
                for (int n = 0; n < valid_n; n++)
                    bmax = fmaxf(bmax, scores[m * BLOCK_N + n]);
                float new_max = fmaxf(old_max, bmax);
                // Guard: if new_max is -inf (all scores masked), rescale = 1 (no-op)
                float r = isinf(new_max) ? 1.0f : expf(old_max - new_max);
                rescale[m] = r;
                softmax[m * 2]     = new_max;
                softmax[m * 2 + 1] *= r;
                float bsum = 0.0f;
                for (int n = 0; n < valid_n; n++) {
                    float s = scores[m * BLOCK_N + n];
                    float p = isinf(s) ? 0.0f : expf(s - new_max);
                    scores[m * BLOCK_N + n] = p;
                    bsum += p;
                }
                for (int n = valid_n; n < BLOCK_N; n++) scores[m * BLOCK_N + n] = 0.0f;
                softmax[m * 2 + 1] += bsum;
            }
        }
        __syncthreads();

        // Rescale out_tile
        for (int i = tid; i < MMA_M * HEAD_DIM; i += BLOCK_THREADS)
            out_tile[i] *= rescale[i / HEAD_DIM];

        // Load V (reuse k_smem, bf16)
        __nv_bfloat16* v_smem = reinterpret_cast<__nv_bfloat16*>(smem + V_SMEM_OFF);
        for (int i = tid; i < valid_n * HEAD_DIM; i += BLOCK_THREADS)
            v_smem[i] = v_ptr[kv_start * HEAD_DIM + i];
        for (int i = tid + valid_n * HEAD_DIM; i < BLOCK_N * HEAD_DIM; i += BLOCK_THREADS)
            v_smem[i] = __float2bfloat16(0.0f);
        __syncthreads();

        // V accumulate
        for (int i = tid; i < MMA_M * HEAD_DIM; i += BLOCK_THREADS) {
            int m = i / HEAD_DIM, d = i % HEAD_DIM;
            float acc = 0.0f;
            for (int n = 0; n < valid_n; n++)
                acc += scores[m * BLOCK_N + n] * __bfloat162float(v_smem[n * HEAD_DIM + d]);
            out_tile[i] += acc;
        }
        __syncthreads();
    }

    // Finalize inv_sum
    if (tid == 0) {
        for (int m = 0; m < MMA_M; m++) {
            float s = softmax[m * 2 + 1];
            rescale[m] = (s > 0.0f) ? (1.0f / s) : 0.0f;
        }
    }
    __syncthreads();
}

// ============================================================================
// Prefill kernel
// ============================================================================
__global__ void unified_attn_prefill_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    int seq_len, int ctx_len, int num_heads, int num_kv_heads
) {
    extern __shared__ char smem[];

    int q_block_idx    = blockIdx.x;
    int head_batch_idx = blockIdx.y;
    int batch_idx      = head_batch_idx / num_heads;
    int head_idx       = head_batch_idx % num_heads;
    int kv_head_idx    = head_idx / (num_heads / num_kv_heads);

    int q_block_start = q_block_idx * BLOCK_M_PREFILL;
    if (q_block_start >= seq_len) return;
    int q_block_end = min(q_block_start + BLOCK_M_PREFILL, seq_len);

    const __nv_bfloat16* q_ptr = Q + ((batch_idx * num_heads    + head_idx)    * seq_len) * HEAD_DIM;
    const __nv_bfloat16* k_ptr = K + ((batch_idx * num_kv_heads + kv_head_idx) * ctx_len) * HEAD_DIM;
    const __nv_bfloat16* v_ptr = V + ((batch_idx * num_kv_heads + kv_head_idx) * ctx_len) * HEAD_DIM;
    __nv_bfloat16*        o_ptr = O + ((batch_idx * num_heads    + head_idx)    * seq_len) * HEAD_DIM;

    int tid = threadIdx.x;
    __nv_bfloat16* q_tile = reinterpret_cast<__nv_bfloat16*>(smem + Q_TILE_OFF);

    for (int mt = 0; mt < BLOCK_M_PREFILL / MMA_M; mt++) {
        int m_start    = q_block_start + mt * MMA_M;
        int m_end      = min(m_start + MMA_M, q_block_end);
        int actual_rows = m_end - m_start;
        if (actual_rows <= 0) break;

        // Load Q tile [MMA_M, HEAD_DIM] into smem (bf16, zero-pad)
        for (int i = tid; i < MMA_M * HEAD_DIM; i += BLOCK_THREADS) {
            int row = i / HEAD_DIM, col = i % HEAD_DIM;
            q_tile[i] = (row < actual_rows)
                        ? q_ptr[(m_start + row) * HEAD_DIM + col]
                        : __float2bfloat16(0.0f);
        }
        __syncthreads();

        process_m_tile(smem, k_ptr, v_ptr, ctx_len, m_start, actual_rows);

        float* out_tile = reinterpret_cast<float*>(smem + OUT_TILE_OFF);
        float* inv_sum  = reinterpret_cast<float*>(smem + RESCALE_OFF);
        for (int i = tid; i < actual_rows * HEAD_DIM; i += BLOCK_THREADS) {
            int row = i / HEAD_DIM, col = i % HEAD_DIM;
            o_ptr[(m_start + row) * HEAD_DIM + col] =
                __float2bfloat16(out_tile[i] * inv_sum[row]);
        }
        __syncthreads();
    }
}

// ============================================================================
// Decode kernel
// ============================================================================
__global__ void unified_attn_decode_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    int ctx_len, int num_heads, int num_kv_heads
) {
    extern __shared__ char smem[];

    int head_batch_idx = blockIdx.y;
    int batch_idx      = head_batch_idx / num_heads;
    int head_idx       = head_batch_idx % num_heads;
    int kv_head_idx    = head_idx / (num_heads / num_kv_heads);

    const __nv_bfloat16* q_ptr = Q + (batch_idx * num_heads    + head_idx)    * HEAD_DIM;
    const __nv_bfloat16* k_ptr = K + ((batch_idx * num_kv_heads + kv_head_idx) * ctx_len) * HEAD_DIM;
    const __nv_bfloat16* v_ptr = V + ((batch_idx * num_kv_heads + kv_head_idx) * ctx_len) * HEAD_DIM;
    __nv_bfloat16*        o_ptr = O + (batch_idx * num_heads    + head_idx)    * HEAD_DIM;

    int tid = threadIdx.x;
    __nv_bfloat16* q_tile = reinterpret_cast<__nv_bfloat16*>(smem + Q_TILE_OFF);

    for (int i = tid; i < MMA_M * HEAD_DIM; i += BLOCK_THREADS) {
        int row = i / HEAD_DIM, col = i % HEAD_DIM;
        q_tile[i] = (row == 0) ? q_ptr[col] : __float2bfloat16(0.0f);
    }
    __syncthreads();

    process_m_tile(smem, k_ptr, v_ptr, ctx_len, ctx_len - 1, 1);

    float* out_tile = reinterpret_cast<float*>(smem + OUT_TILE_OFF);
    float* inv_sum  = reinterpret_cast<float*>(smem + RESCALE_OFF);
    for (int col = tid; col < HEAD_DIM; col += BLOCK_THREADS)
        o_ptr[col] = __float2bfloat16(out_tile[col] * inv_sum[0]);
}

// ============================================================================
// PyTorch bindings
// ============================================================================
torch::Tensor unified_prefill(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                               int num_heads, int num_kv_heads) {
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda());
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous());
    TORCH_CHECK(Q.dtype() == torch::kBFloat16);
    TORCH_CHECK(Q.size(3) == HEAD_DIM);
    int batch = Q.size(0), seq_len = Q.size(2), ctx_len = K.size(2);
    auto O = torch::zeros_like(Q);
    int nq = (seq_len + BLOCK_M_PREFILL - 1) / BLOCK_M_PREFILL;
    cudaFuncSetAttribute(unified_attn_prefill_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
    unified_attn_prefill_kernel<<<dim3(nq, batch * num_heads), BLOCK_THREADS, SMEM_BYTES>>>(
        (const __nv_bfloat16*)Q.data_ptr(), (const __nv_bfloat16*)K.data_ptr(),
        (const __nv_bfloat16*)V.data_ptr(), (__nv_bfloat16*)O.data_ptr(),
        seq_len, ctx_len, num_heads, num_kv_heads);
    return O;
}

torch::Tensor unified_decode(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                              int num_heads, int num_kv_heads) {
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda());
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous());
    TORCH_CHECK(Q.size(2) == 1);
    TORCH_CHECK(Q.size(3) == HEAD_DIM);
    int batch = Q.size(0), ctx_len = K.size(2);
    auto O = torch::zeros_like(Q);
    cudaFuncSetAttribute(unified_attn_decode_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
    unified_attn_decode_kernel<<<dim3(1, batch * num_heads), BLOCK_THREADS, SMEM_BYTES>>>(
        (const __nv_bfloat16*)Q.data_ptr(), (const __nv_bfloat16*)K.data_ptr(),
        (const __nv_bfloat16*)V.data_ptr(), (__nv_bfloat16*)O.data_ptr(),
        ctx_len, num_heads, num_kv_heads);
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("unified_prefill", &unified_prefill, "Unified prefill (PTX HMMA m16n8k16 bf16)");
    m.def("unified_decode",  &unified_decode,  "Unified decode (PTX HMMA m16n8k16 bf16)");
}
