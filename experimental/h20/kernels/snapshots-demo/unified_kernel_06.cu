// unified_attn_ext.cu — kernel_06: vectorized K/V loads (LDG.128) + HMMA for QK^T and V
//
// Changes from kernel_04:
//   1. V accumulation via HMMA: O[16,128] += softmax_weights[16,128] * V[128,128]
//      Scores converted f32→bf16 after softmax, then used as A fragment for V GEMM.
//   2. Parallel softmax: 4 warps each handle 4 rows (MMA_M/NUM_WARPS=4 rows/warp).
//      No cross-warp communication needed (rows are independent).
//   3. Parallel causal mask: same 4-row-per-warp assignment.
//
// Fragment layouts (same as kernel_04 for QK^T):
//   A [16,16] bf16: ldmatrix.x4, thread t → row=t%16, col=(t/16)*8
//   B [N=8, K=16] bf16: manual load, thread t → n=n_col+(t/4), k=(t%4)*2+{0,1,8,9}
//   D/C [16,8] f32: groupID=lane>>2, col=tid_in_group*2+{0,1}, row=groupID+{0,8}
//
// Shared memory layout (bytes):
//   q_tile:       [16, 128] bf16 = 4096 B
//   k_smem:       [128, 128] bf16 = 32768 B  (reused for V)
//   scores_f32:   [16, 128] f32 = 8192 B     (softmax computation)
//   scores_bf16:  [16, 128] bf16 = 4096 B    (V HMMA input, after f32→bf16 conversion)
//   out_tile:     [16, 128] f32 = 8192 B     (output accumulator)
//   softmax:      [16, 2] f32 = 128 B        (row_max + row_sum)
//   rescale:      [16] f32 = 64 B
//   Total: 4096+32768+8192+4096+8192+128+64 = 57536 B ≈ 56 KB → needs opt-in
//
// Bitwise consistency:
//   Both prefill and decode use identical HMMA calls for QK^T and V accumulation.
//   f32→bf16 conversion of scores uses the same rounding in both paths.
//   BLOCK_N=128 unchanged.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>

constexpr int HEAD_DIM        = 128;
constexpr int NUM_WARPS       = 4;
constexpr int WARP_SIZE       = 32;
constexpr int BLOCK_THREADS   = NUM_WARPS * WARP_SIZE;
constexpr int BLOCK_M_PREFILL = 128;
constexpr int BLOCK_N         = 128;

constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 16;

constexpr int N_TILES          = BLOCK_N / MMA_N;      // 16
constexpr int K_TILES          = HEAD_DIM / MMA_K;     // 8
constexpr int N_TILES_PER_WARP = N_TILES / NUM_WARPS;  // 4
constexpr int ROWS_PER_WARP    = MMA_M / NUM_WARPS;    // 4 (for softmax parallelism)

// Shared memory byte offsets
constexpr int Q_TILE_OFF      = 0;
constexpr int K_SMEM_OFF      = Q_TILE_OFF      + MMA_M * HEAD_DIM * 2;
constexpr int SCORES_F32_OFF  = K_SMEM_OFF      + BLOCK_N * HEAD_DIM * 2;
constexpr int SCORES_BF16_OFF = SCORES_F32_OFF  + MMA_M * BLOCK_N * 4;
constexpr int OUT_TILE_OFF    = SCORES_BF16_OFF + MMA_M * BLOCK_N * 2;
constexpr int SOFTMAX_OFF     = OUT_TILE_OFF    + MMA_M * HEAD_DIM * 4;
constexpr int RESCALE_OFF     = SOFTMAX_OFF     + MMA_M * 2 * 4;
constexpr int SMEM_BYTES      = RESCALE_OFF     + MMA_M * 4;
constexpr int V_SMEM_OFF      = K_SMEM_OFF;  // reuse K smem for V

// ============================================================================
// PTX wrappers
// ============================================================================

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

// Load B fragment from a [BLOCK_N or HEAD_DIM, HEAD_DIM] bf16 smem array.
// B tile shape: [N=8, K=16] row-major in smem.
// mma.row.col expects B col-major [K=16, N=8].
// Thread t: n = n_col + (t/4), k0=(t%4)*2, k1=k0+1, k8=k0+8, k9=k0+9
__device__ __forceinline__
void load_b_fragment(uint32_t& b0, uint32_t& b1,
                     const __nv_bfloat16* smem,
                     int n_col, int kt, int lane_id, int row_stride) {
    int n      = n_col + (lane_id / 4);
    int k_base = (lane_id % 4) * 2;
    int k0 = kt * MMA_K + k_base;
    int k1 = k0 + 1;
    int k8 = k0 + 8;
    int k9 = k8 + 1;

    __nv_bfloat16 v0 = smem[n * row_stride + k0];
    __nv_bfloat16 v1 = smem[n * row_stride + k1];
    __nv_bfloat16 v8 = smem[n * row_stride + k8];
    __nv_bfloat16 v9 = smem[n * row_stride + k9];

    b0 = (reinterpret_cast<uint16_t&>(v1) << 16) | reinterpret_cast<uint16_t&>(v0);
    b1 = (reinterpret_cast<uint16_t&>(v9) << 16) | reinterpret_cast<uint16_t&>(v8);
}

// ============================================================================
// QK^T via HMMA: scores[MMA_M, BLOCK_N] = Q[MMA_M, HEAD_DIM] * K[BLOCK_N, HEAD_DIM]^T
// Identical to kernel_04.
// ============================================================================
__device__ void compute_qk_hmma(
    const __nv_bfloat16* __restrict__ q_tile,
    const __nv_bfloat16* __restrict__ k_smem,
    float* __restrict__ scores_f32
) {
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int n_tile_start = warp_id * N_TILES_PER_WARP;

    float d[N_TILES_PER_WARP][4] = {};

    for (int kt = 0; kt < K_TILES; kt++) {
        uint32_t a0, a1, a2, a3;
        {
            int row = lane_id % MMA_M;
            int col = (lane_id / MMA_M) * 8 + kt * MMA_K;
            ldmatrix_a(a0, a1, a2, a3, q_tile + row * HEAD_DIM + col);
        }
        for (int nt = 0; nt < N_TILES_PER_WARP; nt++) {
            uint32_t b0, b1;
            load_b_fragment(b0, b1, k_smem, (n_tile_start + nt) * MMA_N, kt, lane_id, HEAD_DIM);
            mma_bf16(d[nt][0], d[nt][1], d[nt][2], d[nt][3],
                     a0, a1, a2, a3, b0, b1,
                     d[nt][0], d[nt][1], d[nt][2], d[nt][3]);
        }
    }

    // Scatter to scores_f32 smem
    {
        int group_id     = lane_id >> 2;
        int tid_in_group = lane_id % 4;
        int row0 = group_id, row1 = group_id + 8;
        int col0 = tid_in_group * 2, col1 = col0 + 1;
        for (int nt = 0; nt < N_TILES_PER_WARP; nt++) {
            int n_base = (n_tile_start + nt) * MMA_N;
            scores_f32[row0 * BLOCK_N + n_base + col0] = d[nt][0];
            scores_f32[row0 * BLOCK_N + n_base + col1] = d[nt][1];
            scores_f32[row1 * BLOCK_N + n_base + col0] = d[nt][2];
            scores_f32[row1 * BLOCK_N + n_base + col1] = d[nt][3];
        }
    }
}

// ============================================================================
// V accumulation via HMMA: out_tile[MMA_M, HEAD_DIM] += W[MMA_M, BLOCK_N] * V[BLOCK_N, HEAD_DIM]
// W = softmax weights (bf16 in scores_bf16 smem).
// V = bf16 in v_smem (same layout as k_smem: [BLOCK_N, HEAD_DIM]).
// Tiled as: M=16, N=HEAD_DIM=128 (16 N-tiles of 8), K=BLOCK_N=128 (8 K-tiles of 16).
// 4 warps × 4 N-tiles each = 16 N-tiles covering all HEAD_DIM columns.
// ============================================================================
__device__ void compute_v_hmma(
    const __nv_bfloat16* __restrict__ w_smem,  // [MMA_M, BLOCK_N] bf16, row-major
    const __nv_bfloat16* __restrict__ v_smem,  // [BLOCK_N, HEAD_DIM] bf16, row-major
    float* __restrict__ out_tile               // [MMA_M, HEAD_DIM] f32, row-major
) {
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    // Each warp handles N_TILES_PER_WARP=4 N-tiles of HEAD_DIM
    int n_tile_start = warp_id * N_TILES_PER_WARP;

    // Accumulator: 4 N-tiles × 4 f32 per thread
    float d[N_TILES_PER_WARP][4] = {};

    // K-tiles over BLOCK_N=128 (the shared K dimension between W and V)
    for (int kt = 0; kt < K_TILES; kt++) {
        // A fragment: W[0:16, kt*16:(kt+1)*16] from scores_bf16 smem
        // W is [MMA_M=16, BLOCK_N=128] row-major with stride BLOCK_N
        uint32_t a0, a1, a2, a3;
        {
            int row = lane_id % MMA_M;
            int col = (lane_id / MMA_M) * 8 + kt * MMA_K;
            ldmatrix_a(a0, a1, a2, a3, w_smem + row * BLOCK_N + col);
        }

        for (int nt = 0; nt < N_TILES_PER_WARP; nt++) {
            int n_col = (n_tile_start + nt) * MMA_N;
            // B fragment: V[kt*16:(kt+1)*16, n_col:n_col+8]
            // V is [BLOCK_N, HEAD_DIM] row-major: V[row, col] = v_smem[row*HEAD_DIM + col]
            // mma.row.col B register layout: thread t holds B[k,n] where
            //   k = kt*MMA_K + (t%4)*2 + {0,1} for b0, + {8,9} for b1
            //   n = n_col + (t/4)
            {
                int k_base = (lane_id % 4) * 2;
                int k0 = kt * MMA_K + k_base;
                int k1 = k0 + 1;
                int k8 = k0 + 8;
                int k9 = k8 + 1;
                int n  = n_col + (lane_id / 4);
                __nv_bfloat16 e0 = v_smem[k0 * HEAD_DIM + n];
                __nv_bfloat16 e1 = v_smem[k1 * HEAD_DIM + n];
                __nv_bfloat16 e8 = v_smem[k8 * HEAD_DIM + n];
                __nv_bfloat16 e9 = v_smem[k9 * HEAD_DIM + n];
                uint32_t b0 = (reinterpret_cast<uint16_t&>(e1) << 16) | reinterpret_cast<uint16_t&>(e0);
                uint32_t b1 = (reinterpret_cast<uint16_t&>(e9) << 16) | reinterpret_cast<uint16_t&>(e8);
                mma_bf16(d[nt][0], d[nt][1], d[nt][2], d[nt][3],
                         a0, a1, a2, a3, b0, b1,
                         d[nt][0], d[nt][1], d[nt][2], d[nt][3]);
            }
        }
    }

    // Accumulate D into out_tile (atomic add since multiple KV blocks contribute)
    {
        int group_id     = lane_id >> 2;
        int tid_in_group = lane_id % 4;
        int row0 = group_id, row1 = group_id + 8;
        int col0 = tid_in_group * 2, col1 = col0 + 1;
        for (int nt = 0; nt < N_TILES_PER_WARP; nt++) {
            int n_base = (n_tile_start + nt) * MMA_N;
            out_tile[row0 * HEAD_DIM + n_base + col0] += d[nt][0];
            out_tile[row0 * HEAD_DIM + n_base + col1] += d[nt][1];
            out_tile[row1 * HEAD_DIM + n_base + col0] += d[nt][2];
            out_tile[row1 * HEAD_DIM + n_base + col1] += d[nt][3];
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
    __nv_bfloat16* q_tile      = reinterpret_cast<__nv_bfloat16*>(smem + Q_TILE_OFF);
    __nv_bfloat16* k_smem      = reinterpret_cast<__nv_bfloat16*>(smem + K_SMEM_OFF);
    float*         scores_f32  = reinterpret_cast<float*>(smem + SCORES_F32_OFF);
    __nv_bfloat16* scores_bf16 = reinterpret_cast<__nv_bfloat16*>(smem + SCORES_BF16_OFF);
    float*         out_tile    = reinterpret_cast<float*>(smem + OUT_TILE_OFF);
    float*         softmax     = reinterpret_cast<float*>(smem + SOFTMAX_OFF);
    float*         rescale     = reinterpret_cast<float*>(smem + RESCALE_OFF);

    int tid     = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    // Initialize out_tile and softmax state
    for (int i = tid; i < MMA_M * HEAD_DIM; i += BLOCK_THREADS) out_tile[i] = 0.0f;
    // Each warp initializes its own rows (ROWS_PER_WARP=4 rows per warp)
    int row_start = warp_id * ROWS_PER_WARP;
    if (lane_id < ROWS_PER_WARP) {
        int m = row_start + lane_id;
        softmax[m * 2]     = -INFINITY;
        softmax[m * 2 + 1] = 0.0f;
    }
    __syncthreads();

    float inv_sqrt = 1.0f / sqrtf((float)HEAD_DIM);
    int num_kv_blocks = (ctx_len + BLOCK_N - 1) / BLOCK_N;

    for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        int kv_start = kv_block * BLOCK_N;
        int kv_end   = min(kv_start + BLOCK_N, ctx_len);
        int valid_n  = kv_end - kv_start;

        // Load K block — vectorized LDG.128 (uint4 = 16 bytes = 8 bf16 per instruction)
        // valid_n * HEAD_DIM elements; BLOCK_N * HEAD_DIM total (zero-pad remainder)
        {
            const int vec_valid = valid_n * HEAD_DIM / 8;   // uint4 chunks with data
            const int vec_total = BLOCK_N * HEAD_DIM / 8;   // total uint4 chunks in smem
            const uint4* src = reinterpret_cast<const uint4*>(k_ptr + kv_start * HEAD_DIM);
            uint4*       dst = reinterpret_cast<uint4*>(k_smem);
            for (int i = tid; i < vec_valid; i += BLOCK_THREADS)
                dst[i] = src[i];
            for (int i = tid + vec_valid; i < vec_total; i += BLOCK_THREADS)
                dst[i] = make_uint4(0, 0, 0, 0);
        }
        __syncthreads();

        // QK^T via HMMA → scores_f32[MMA_M, BLOCK_N]
        compute_qk_hmma(q_tile, k_smem, scores_f32);
        __syncthreads();

        // Apply scale + causal mask — parallelized across warps (4 rows per warp)
        for (int r = lane_id; r < ROWS_PER_WARP; r += WARP_SIZE) {
            int m = row_start + r;
            if (m >= MMA_M) break;
            int q_pos = q_tile_start + m;
            for (int n = 0; n < BLOCK_N; n++) {
                float s = scores_f32[m * BLOCK_N + n] * inv_sqrt;
                bool masked = (n >= valid_n) || (kv_start + n > q_pos) || (m >= actual_rows);
                scores_f32[m * BLOCK_N + n] = masked ? -INFINITY : s;
            }
        }
        __syncthreads();

        // Online softmax — each warp handles its ROWS_PER_WARP rows independently
        // No cross-warp communication needed (rows are independent)
        for (int r = 0; r < ROWS_PER_WARP; r++) {
            int m = row_start + r;
            if (m >= MMA_M) break;
            // Only lane 0 of each warp does the scalar softmax for its rows
            // (avoids redundant work; other lanes will participate in V HMMA)
            if (lane_id == 0) {
                float old_max = softmax[m * 2];
                float bmax = -INFINITY;
                for (int n = 0; n < valid_n; n++)
                    bmax = fmaxf(bmax, scores_f32[m * BLOCK_N + n]);
                float new_max = fmaxf(old_max, bmax);
                float r_scale = isinf(new_max) ? 1.0f : expf(old_max - new_max);
                rescale[m] = r_scale;
                softmax[m * 2]     = new_max;
                softmax[m * 2 + 1] *= r_scale;
                float bsum = 0.0f;
                for (int n = 0; n < valid_n; n++) {
                    float s = scores_f32[m * BLOCK_N + n];
                    float p = isinf(s) ? 0.0f : expf(s - new_max);
                    scores_f32[m * BLOCK_N + n] = p;
                    bsum += p;
                }
                for (int n = valid_n; n < BLOCK_N; n++) scores_f32[m * BLOCK_N + n] = 0.0f;
                softmax[m * 2 + 1] += bsum;
            }
        }
        __syncthreads();

        // Rescale out_tile — each warp rescales its rows
        for (int i = tid; i < MMA_M * HEAD_DIM; i += BLOCK_THREADS)
            out_tile[i] *= rescale[i / HEAD_DIM];

        // Convert scores_f32 → scores_bf16 for V HMMA
        for (int i = tid; i < MMA_M * BLOCK_N; i += BLOCK_THREADS)
            scores_bf16[i] = __float2bfloat16(scores_f32[i]);

        // Load V — vectorized LDG.128 (reuse k_smem space)
        __nv_bfloat16* v_smem = reinterpret_cast<__nv_bfloat16*>(smem + V_SMEM_OFF);
        {
            const int vec_valid = valid_n * HEAD_DIM / 8;
            const int vec_total = BLOCK_N * HEAD_DIM / 8;
            const uint4* src = reinterpret_cast<const uint4*>(v_ptr + kv_start * HEAD_DIM);
            uint4*       dst = reinterpret_cast<uint4*>(v_smem);
            for (int i = tid; i < vec_valid; i += BLOCK_THREADS)
                dst[i] = src[i];
            for (int i = tid + vec_valid; i < vec_total; i += BLOCK_THREADS)
                dst[i] = make_uint4(0, 0, 0, 0);
        }
        __syncthreads();

        // V accumulation via HMMA: out_tile += scores_bf16 * V
        compute_v_hmma(scores_bf16, v_smem, out_tile);
        __syncthreads();
    }

    // Finalize inv_sum — each warp writes its rows
    if (lane_id == 0) {
        for (int r = 0; r < ROWS_PER_WARP; r++) {
            int m = row_start + r;
            if (m >= MMA_M) break;
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
    m.def("unified_prefill", &unified_prefill, "Unified prefill (HMMA QK^T + V)");
    m.def("unified_decode",  &unified_decode,  "Unified decode (HMMA QK^T + V)");
}
