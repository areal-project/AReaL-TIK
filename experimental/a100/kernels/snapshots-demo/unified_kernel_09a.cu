// unified_kernel_09a.cu — kernel_09a: register-resident output accumulator only
//
// Base: kernel_08 (split-Q, chunked HMMA, smem softmax/scores)
// Single change: out_tile moves from smem → register array acc_o[OUT_N_TILES][4]
//
// Everything else is IDENTICAL to kernel_08:
//   - scores_f32, scores_b16 still in per-warp smem
//   - softmax state (row_max, row_sum) still in per-warp smem
//   - serial lane-0 softmax loop unchanged
//   - expf() unchanged (not exp2f)
//   - __syncwarp() pattern unchanged
//   - causal mask loop unchanged
//   - N_CHUNK=4 chunked QK^T unchanged
//
// What changes:
//   - out_tile[MMA_M*HEAD_DIM] f32 smem removed from per-warp region
//   - acc_o[OUT_N_TILES][4] f32 registers added per thread
//   - compute_v_hmma_reg: MMA accumulates into acc_o registers (C=acc_o, D=acc_o)
//     instead of out_tile[smem] += d[nt][i]
//   - Rescale step: acc_o[nt][0,1] *= rescale[row0], acc_o[nt][2,3] *= rescale[row1]
//     (register multiply, no smem read-modify-write)
//   - Output write: reads acc_o registers directly, multiplies by inv_sum
//
// Smem layout (121 KB, was 129 KB — saved 8 KB by removing out_tile):
//   k_smem:         [128,128] bf16 = 32 KB
//   warp_region[4]: each 16576 B = 16.2 KB
//     q_tile:       [16,128] bf16 =  4 KB
//     scores_f32:   [16,128] f32  =  8 KB
//     scores_b16:   [16,128] bf16 =  4 KB
//     softmax:      [16,2]   f32  =  128 B
//     rescale:      [16]     f32  =   64 B
//   Total: 32768 + 4×16576 = 99072 B ≈ 97 KB

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>

constexpr int HEAD_DIM        = 128;
constexpr int NUM_WARPS       = 4;
constexpr int WARP_SIZE       = 32;
constexpr int BLOCK_THREADS   = NUM_WARPS * WARP_SIZE;
constexpr int BLOCK_M_PREFILL = NUM_WARPS * 16;  // 64
constexpr int BLOCK_N         = 128;

constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 16;

constexpr int N_TILES    = BLOCK_N  / MMA_N;   // 16
constexpr int K_TILES    = HEAD_DIM / MMA_K;   // 8
constexpr int KV_K_TILES = BLOCK_N  / MMA_K;   // 8
constexpr int OUT_N_TILES= HEAD_DIM / MMA_N;   // 16
constexpr int N_CHUNK    = 4;  // chunked QK^T (same as kernel_08)

// Shared memory layout — out_tile removed vs kernel_08
constexpr int K_SMEM_BYTES     = BLOCK_N * HEAD_DIM * 2;   // 32768 B
constexpr int Q_TILE_BYTES     = MMA_M * HEAD_DIM * 2;     //  4096 B
constexpr int SCORES_F32_BYTES = MMA_M * BLOCK_N * 4;      //  8192 B
constexpr int SCORES_B16_BYTES = MMA_M * BLOCK_N * 2;      //  4096 B
constexpr int SOFTMAX_BYTES    = MMA_M * 2 * 4;            //   128 B
constexpr int RESCALE_BYTES    = MMA_M * 4;                //    64 B
constexpr int PER_WARP_BYTES   = Q_TILE_BYTES + SCORES_F32_BYTES
                                + SCORES_B16_BYTES + SOFTMAX_BYTES + RESCALE_BYTES;
// = 16576 B per warp

constexpr int K_SMEM_OFF       = 0;
constexpr int WARP_BASE        = K_SMEM_BYTES;  // 32768
constexpr int SMEM_BYTES       = WARP_BASE + NUM_WARPS * PER_WARP_BYTES;  // 99072 B

// Per-warp offsets (relative to warp's private region base) — no W_OUT_TILE_OFF
constexpr int W_Q_TILE_OFF      = 0;
constexpr int W_SCORES_F32_OFF  = W_Q_TILE_OFF      + Q_TILE_BYTES;
constexpr int W_SCORES_B16_OFF  = W_SCORES_F32_OFF  + SCORES_F32_BYTES;
constexpr int W_SOFTMAX_OFF     = W_SCORES_B16_OFF  + SCORES_B16_BYTES;
constexpr int W_RESCALE_OFF     = W_SOFTMAX_OFF     + SOFTMAX_BYTES;

// ============================================================================
// PTX wrappers — identical to kernel_08
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

// ============================================================================
// QK^T — identical to kernel_08 (chunked, scatters to scores_f32 smem)
// ============================================================================
__device__ __forceinline__ void compute_qk_hmma(
    const __nv_bfloat16* __restrict__ q_tile,
    const __nv_bfloat16* __restrict__ k_smem,
    float* __restrict__ scores_f32
) {
    int lane_id      = threadIdx.x % WARP_SIZE;
    int group_id     = lane_id >> 2;
    int tid_in_group = lane_id & 3;
    int row0 = group_id, row1 = group_id + 8;
    int col0 = tid_in_group * 2, col1 = col0 + 1;

    for (int nc = 0; nc < N_TILES; nc += N_CHUNK) {
        float d[N_CHUNK][4] = {};

        for (int kt = 0; kt < K_TILES; kt++) {
            uint32_t a0, a1, a2, a3;
            {
                int row = lane_id % MMA_M;
                int col = (lane_id / MMA_M) * 8 + kt * MMA_K;
                ldmatrix_a(a0, a1, a2, a3, q_tile + row * HEAD_DIM + col);
            }
            for (int nt = 0; nt < N_CHUNK; nt++) {
                int n_col  = (nc + nt) * MMA_N;
                int n      = n_col + (lane_id / 4);
                int k_base = (lane_id % 4) * 2;
                int k0 = kt * MMA_K + k_base;
                int k1 = k0+1, k8 = k0+8, k9 = k8+1;
                __nv_bfloat16 v0 = k_smem[n * HEAD_DIM + k0];
                __nv_bfloat16 v1 = k_smem[n * HEAD_DIM + k1];
                __nv_bfloat16 v8 = k_smem[n * HEAD_DIM + k8];
                __nv_bfloat16 v9 = k_smem[n * HEAD_DIM + k9];
                uint32_t b0 = (reinterpret_cast<uint16_t&>(v1) << 16) | reinterpret_cast<uint16_t&>(v0);
                uint32_t b1 = (reinterpret_cast<uint16_t&>(v9) << 16) | reinterpret_cast<uint16_t&>(v8);
                mma_bf16(d[nt][0], d[nt][1], d[nt][2], d[nt][3],
                         a0, a1, a2, a3, b0, b1,
                         d[nt][0], d[nt][1], d[nt][2], d[nt][3]);
            }
        }

        // Scatter to scores_f32 smem — identical to kernel_08
        for (int nt = 0; nt < N_CHUNK; nt++) {
            int n_base = (nc + nt) * MMA_N;
            scores_f32[row0 * BLOCK_N + n_base + col0] = d[nt][0];
            scores_f32[row0 * BLOCK_N + n_base + col1] = d[nt][1];
            scores_f32[row1 * BLOCK_N + n_base + col0] = d[nt][2];
            scores_f32[row1 * BLOCK_N + n_base + col1] = d[nt][3];
        }
    }
}

// ============================================================================
// PV — register-resident accumulator (THE ONLY CHANGE vs kernel_08)
//
// kernel_08: out_tile[smem] += d[nt][i]   (smem read-modify-write)
// kernel_09a: acc_o[nt][i] is passed in as C and written as D in the MMA
//             (pure register accumulation, no smem traffic)
//
// acc_o[OUT_N_TILES][4] persists across KV blocks in the caller's registers.
// Each call to this function adds the current KV block's contribution.
// ============================================================================
__device__ __forceinline__ void compute_v_hmma_reg(
    const __nv_bfloat16* __restrict__ w_smem,  // warp-private [16,128] bf16 (scores_b16)
    const __nv_bfloat16* __restrict__ v_smem,  // shared [128,128] bf16
    float acc_o[OUT_N_TILES][4]                // in/out: register accumulator
) {
    int lane_id      = threadIdx.x % WARP_SIZE;
    int group_id     = lane_id >> 2;
    int tid_in_group = lane_id & 3;
    int row0 = group_id, row1 = group_id + 8;
    int col0 = tid_in_group * 2, col1 = col0 + 1;

    for (int nc = 0; nc < OUT_N_TILES; nc += N_CHUNK) {
        // Load current acc_o values for this chunk into local d[][]
        // so the MMA can use them as C (accumulate into existing output)
        float d[N_CHUNK][4];
        for (int nt = 0; nt < N_CHUNK; nt++) {
            d[nt][0] = acc_o[nc + nt][0];
            d[nt][1] = acc_o[nc + nt][1];
            d[nt][2] = acc_o[nc + nt][2];
            d[nt][3] = acc_o[nc + nt][3];
        }

        for (int kt = 0; kt < KV_K_TILES; kt++) {
            uint32_t a0, a1, a2, a3;
            {
                int row = lane_id % MMA_M;
                int col = (lane_id / MMA_M) * 8 + kt * MMA_K;
                ldmatrix_a(a0, a1, a2, a3, w_smem + row * BLOCK_N + col);
            }
            for (int nt = 0; nt < N_CHUNK; nt++) {
                int n_col  = (nc + nt) * MMA_N;
                int k_base = (lane_id % 4) * 2;
                int k0 = kt * MMA_K + k_base;
                int k1 = k0+1, k8 = k0+8, k9 = k8+1;
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

        // Write back to acc_o registers
        for (int nt = 0; nt < N_CHUNK; nt++) {
            acc_o[nc + nt][0] = d[nt][0];
            acc_o[nc + nt][1] = d[nt][1];
            acc_o[nc + nt][2] = d[nt][2];
            acc_o[nc + nt][3] = d[nt][3];
        }
    }
}

// ============================================================================
// Prefill kernel — split-Q, register acc_o, smem softmax/scores (kernel_08 base)
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

    const __nv_bfloat16* q_ptr = Q + ((batch_idx * num_heads    + head_idx)    * seq_len) * HEAD_DIM;
    const __nv_bfloat16* k_ptr = K + ((batch_idx * num_kv_heads + kv_head_idx) * ctx_len) * HEAD_DIM;
    const __nv_bfloat16* v_ptr = V + ((batch_idx * num_kv_heads + kv_head_idx) * ctx_len) * HEAD_DIM;
    __nv_bfloat16*        o_ptr = O + ((batch_idx * num_heads    + head_idx)    * seq_len) * HEAD_DIM;

    int tid     = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    // Per-warp smem region (no out_tile)
    char* warp_smem    = smem + WARP_BASE + warp_id * PER_WARP_BYTES;
    __nv_bfloat16* q_tile    = reinterpret_cast<__nv_bfloat16*>(warp_smem + W_Q_TILE_OFF);
    float*         scores_f32= reinterpret_cast<float*>(warp_smem + W_SCORES_F32_OFF);
    __nv_bfloat16* scores_b16= reinterpret_cast<__nv_bfloat16*>(warp_smem + W_SCORES_B16_OFF);
    float*         softmax   = reinterpret_cast<float*>(warp_smem + W_SOFTMAX_OFF);
    float*         rescale   = reinterpret_cast<float*>(warp_smem + W_RESCALE_OFF);
    __nv_bfloat16* k_smem_p  = reinterpret_cast<__nv_bfloat16*>(smem + K_SMEM_OFF);

    int warp_q_start = q_block_start + warp_id * MMA_M;
    int warp_q_end   = min(warp_q_start + MMA_M, seq_len);
    int actual_rows  = max(0, warp_q_end - warp_q_start);

    // Load Q tile (warp-local, no sync needed)
    for (int i = lane_id; i < MMA_M * HEAD_DIM; i += WARP_SIZE) {
        int row = i / HEAD_DIM, col = i % HEAD_DIM;
        q_tile[i] = (row < actual_rows)
                    ? q_ptr[(warp_q_start + row) * HEAD_DIM + col]
                    : __float2bfloat16(0.0f);
    }

    // Initialize softmax state in smem (identical to kernel_08)
    if (lane_id < MMA_M) {
        softmax[lane_id * 2]     = -INFINITY;
        softmax[lane_id * 2 + 1] = 0.0f;
    }

    // Register output accumulator — replaces smem out_tile
    float acc_o[OUT_N_TILES][4];
    #pragma unroll
    for (int nt = 0; nt < OUT_N_TILES; nt++)
        acc_o[nt][0] = acc_o[nt][1] = acc_o[nt][2] = acc_o[nt][3] = 0.0f;

    float inv_sqrt = 1.0f / sqrtf((float)HEAD_DIM);
    int num_kv_blocks = (ctx_len + BLOCK_N - 1) / BLOCK_N;

    for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        int kv_start = kv_block * BLOCK_N;
        int kv_end   = min(kv_start + BLOCK_N, ctx_len);
        int valid_n  = kv_end - kv_start;

        // Cooperative K load — identical to kernel_08
        {
            const int vec_valid = valid_n * HEAD_DIM / 8;
            const int vec_total = BLOCK_N * HEAD_DIM / 8;
            const uint4* src = reinterpret_cast<const uint4*>(k_ptr + kv_start * HEAD_DIM);
            uint4*       dst = reinterpret_cast<uint4*>(k_smem_p);
            for (int i = tid; i < vec_valid; i += BLOCK_THREADS) dst[i] = src[i];
            for (int i = tid + vec_valid; i < vec_total; i += BLOCK_THREADS)
                dst[i] = make_uint4(0, 0, 0, 0);
        }
        __syncthreads();  // sync #1

        if (actual_rows > 0) {
            // QK^T → scores_f32 smem (identical to kernel_08)
            compute_qk_hmma(q_tile, k_smem_p, scores_f32);

            // Scale + causal mask (identical to kernel_08)
            for (int m = lane_id; m < MMA_M; m += WARP_SIZE) {
                int q_pos = warp_q_start + m;
                for (int n = 0; n < BLOCK_N; n++) {
                    float s = scores_f32[m * BLOCK_N + n] * inv_sqrt;
                    bool masked = (n >= valid_n) || (kv_start + n > q_pos) || (m >= actual_rows);
                    scores_f32[m * BLOCK_N + n] = masked ? -INFINITY : s;
                }
            }
            __syncwarp();

            // Online softmax — lane 0 serial (identical to kernel_08)
            if (lane_id == 0) {
                for (int m = 0; m < actual_rows; m++) {
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
            __syncwarp();

            // Rescale acc_o registers (replaces: out_tile[smem] *= rescale[row])
            {
                int group_id     = lane_id >> 2;
                int row0 = group_id, row1 = group_id + 8;
                float r0 = rescale[row0];
                float r1 = (row1 < MMA_M) ? rescale[row1] : 1.0f;
                #pragma unroll
                for (int nt = 0; nt < OUT_N_TILES; nt++) {
                    acc_o[nt][0] *= r0;
                    acc_o[nt][1] *= r0;
                    acc_o[nt][2] *= r1;
                    acc_o[nt][3] *= r1;
                }
            }

            // Convert scores f32→bf16 (identical to kernel_08)
            for (int i = lane_id; i < MMA_M * BLOCK_N; i += WARP_SIZE)
                scores_b16[i] = __float2bfloat16(scores_f32[i]);
        }
        __syncthreads();  // sync #2

        // Cooperative V load — identical to kernel_08
        {
            const int vec_valid = valid_n * HEAD_DIM / 8;
            const int vec_total = BLOCK_N * HEAD_DIM / 8;
            const uint4* src = reinterpret_cast<const uint4*>(v_ptr + kv_start * HEAD_DIM);
            uint4*       dst = reinterpret_cast<uint4*>(k_smem_p);
            for (int i = tid; i < vec_valid; i += BLOCK_THREADS) dst[i] = src[i];
            for (int i = tid + vec_valid; i < vec_total; i += BLOCK_THREADS)
                dst[i] = make_uint4(0, 0, 0, 0);
        }
        __syncthreads();  // sync #3

        // PV — accumulate into register acc_o (THE CHANGE)
        if (actual_rows > 0) {
            compute_v_hmma_reg(scores_b16, k_smem_p, acc_o);
        }
        __syncthreads();  // sync #4
    }

    // Write output from registers
    if (actual_rows > 0) {
        // Finalize inv_sum (read from smem softmax, identical to kernel_08)
        if (lane_id < MMA_M) {
            float s = softmax[lane_id * 2 + 1];
            rescale[lane_id] = (s > 0.0f) ? (1.0f / s) : 0.0f;
        }
        __syncwarp();

        // Write acc_o registers → global memory
        int group_id     = lane_id >> 2;
        int tid_in_group = lane_id & 3;
        int row0 = group_id, row1 = group_id + 8;
        int col0 = tid_in_group * 2, col1 = col0 + 1;
        float inv0 = rescale[row0];
        float inv1 = (row1 < actual_rows) ? rescale[row1] : 0.0f;

        #pragma unroll
        for (int nt = 0; nt < OUT_N_TILES; nt++) {
            int n_base = nt * MMA_N;
            if (row0 < actual_rows) {
                o_ptr[(warp_q_start + row0) * HEAD_DIM + n_base + col0] =
                    __float2bfloat16(acc_o[nt][0] * inv0);
                o_ptr[(warp_q_start + row0) * HEAD_DIM + n_base + col1] =
                    __float2bfloat16(acc_o[nt][1] * inv0);
            }
            if (row1 < actual_rows) {
                o_ptr[(warp_q_start + row1) * HEAD_DIM + n_base + col0] =
                    __float2bfloat16(acc_o[nt][2] * inv1);
                o_ptr[(warp_q_start + row1) * HEAD_DIM + n_base + col1] =
                    __float2bfloat16(acc_o[nt][3] * inv1);
            }
        }
    }
}

// ============================================================================
// Decode kernel — register acc_o, smem softmax/scores (kernel_08 base)
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

    int tid     = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    // Use warp 0's private region (no out_tile)
    char* w0_smem    = smem + WARP_BASE;
    __nv_bfloat16* q_tile    = reinterpret_cast<__nv_bfloat16*>(w0_smem + W_Q_TILE_OFF);
    float*         scores_f32= reinterpret_cast<float*>(w0_smem + W_SCORES_F32_OFF);
    __nv_bfloat16* scores_b16= reinterpret_cast<__nv_bfloat16*>(w0_smem + W_SCORES_B16_OFF);
    float*         softmax   = reinterpret_cast<float*>(w0_smem + W_SOFTMAX_OFF);
    float*         rescale   = reinterpret_cast<float*>(w0_smem + W_RESCALE_OFF);
    __nv_bfloat16* k_smem_p  = reinterpret_cast<__nv_bfloat16*>(smem + K_SMEM_OFF);

    // Warp 0 init (identical to kernel_08, minus out_tile init)
    if (warp_id == 0) {
        for (int i = lane_id; i < MMA_M * HEAD_DIM; i += WARP_SIZE) {
            int row = i / HEAD_DIM, col = i % HEAD_DIM;
            q_tile[i] = (row == 0) ? q_ptr[col] : __float2bfloat16(0.0f);
        }
        if (lane_id < MMA_M) {
            softmax[lane_id * 2]     = -INFINITY;
            softmax[lane_id * 2 + 1] = 0.0f;
        }
    }
    __syncthreads();

    // Register output accumulator for warp 0
    float acc_o[OUT_N_TILES][4];
    if (warp_id == 0) {
        #pragma unroll
        for (int nt = 0; nt < OUT_N_TILES; nt++)
            acc_o[nt][0] = acc_o[nt][1] = acc_o[nt][2] = acc_o[nt][3] = 0.0f;
    }

    float inv_sqrt = 1.0f / sqrtf((float)HEAD_DIM);
    int num_kv_blocks = (ctx_len + BLOCK_N - 1) / BLOCK_N;

    for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        int kv_start = kv_block * BLOCK_N;
        int kv_end   = min(kv_start + BLOCK_N, ctx_len);
        int valid_n  = kv_end - kv_start;

        // Cooperative K load — identical to kernel_08
        {
            const int vec_valid = valid_n * HEAD_DIM / 8;
            const int vec_total = BLOCK_N * HEAD_DIM / 8;
            const uint4* src = reinterpret_cast<const uint4*>(k_ptr + kv_start * HEAD_DIM);
            uint4*       dst = reinterpret_cast<uint4*>(k_smem_p);
            for (int i = tid; i < vec_valid; i += BLOCK_THREADS) dst[i] = src[i];
            for (int i = tid + vec_valid; i < vec_total; i += BLOCK_THREADS)
                dst[i] = make_uint4(0, 0, 0, 0);
        }
        __syncthreads();  // sync #1

        if (warp_id == 0) {
            // QK^T → scores_f32 smem (identical to kernel_08)
            compute_qk_hmma(q_tile, k_smem_p, scores_f32);

            // Causal mask (identical to kernel_08)
            for (int m = lane_id; m < MMA_M; m += WARP_SIZE) {
                int q_pos = ctx_len - 1;
                for (int n = 0; n < BLOCK_N; n++) {
                    float s = scores_f32[m * BLOCK_N + n] * inv_sqrt;
                    bool masked = (n >= valid_n) || (kv_start + n > q_pos) || (m >= 1);
                    scores_f32[m * BLOCK_N + n] = masked ? -INFINITY : s;
                }
            }
            __syncwarp();

            // Online softmax lane 0 (identical to kernel_08)
            if (lane_id == 0) {
                float old_max = softmax[0];
                float bmax = -INFINITY;
                for (int n = 0; n < valid_n; n++)
                    bmax = fmaxf(bmax, scores_f32[n]);
                float new_max = fmaxf(old_max, bmax);
                float r_scale = isinf(new_max) ? 1.0f : expf(old_max - new_max);
                rescale[0] = r_scale;
                softmax[0] = new_max;
                softmax[1] *= r_scale;
                float bsum = 0.0f;
                for (int n = 0; n < valid_n; n++) {
                    float s = scores_f32[n];
                    float p = isinf(s) ? 0.0f : expf(s - new_max);
                    scores_f32[n] = p;
                    bsum += p;
                }
                for (int n = valid_n; n < BLOCK_N; n++) scores_f32[n] = 0.0f;
                softmax[1] += bsum;
            }
            __syncwarp();

            // Rescale acc_o registers (replaces: out_tile[0..HEAD_DIM] *= rescale[0])
            // Only row0=0 (group_id=0) is valid for decode; rescale all uniformly
            // since row1 elements are never written to output.
            {
                float r0 = rescale[0];
                #pragma unroll
                for (int nt = 0; nt < OUT_N_TILES; nt++) {
                    acc_o[nt][0] *= r0;
                    acc_o[nt][1] *= r0;
                    // acc_o[nt][2,3] are row1 (padding rows), rescale anyway for consistency
                    acc_o[nt][2] *= r0;
                    acc_o[nt][3] *= r0;
                }
            }

            // Convert scores f32→bf16 (identical to kernel_08)
            for (int i = lane_id; i < MMA_M * BLOCK_N; i += WARP_SIZE)
                scores_b16[i] = __float2bfloat16(scores_f32[i]);
        }
        __syncthreads();  // sync #2

        // Cooperative V load — identical to kernel_08
        {
            const int vec_valid = valid_n * HEAD_DIM / 8;
            const int vec_total = BLOCK_N * HEAD_DIM / 8;
            const uint4* src = reinterpret_cast<const uint4*>(v_ptr + kv_start * HEAD_DIM);
            uint4*       dst = reinterpret_cast<uint4*>(k_smem_p);
            for (int i = tid; i < vec_valid; i += BLOCK_THREADS) dst[i] = src[i];
            for (int i = tid + vec_valid; i < vec_total; i += BLOCK_THREADS)
                dst[i] = make_uint4(0, 0, 0, 0);
        }
        __syncthreads();  // sync #3

        // PV — register acc_o (THE CHANGE)
        if (warp_id == 0) {
            compute_v_hmma_reg(scores_b16, k_smem_p, acc_o);
        }
        __syncthreads();  // sync #4
    }

    // Warp 0 writes output from registers
    if (warp_id == 0) {
        if (lane_id == 0) {
            float s = softmax[1];
            rescale[0] = (s > 0.0f) ? (1.0f / s) : 0.0f;
        }
        __syncwarp();

        // Only group_id==0 (row0=0) writes valid decode output
        int group_id     = lane_id >> 2;
        int tid_in_group = lane_id & 3;
        int col0 = tid_in_group * 2, col1 = col0 + 1;
        float inv0 = rescale[0];

        if (group_id == 0) {
            #pragma unroll
            for (int nt = 0; nt < OUT_N_TILES; nt++) {
                int n_base = nt * MMA_N;
                o_ptr[n_base + col0] = __float2bfloat16(acc_o[nt][0] * inv0);
                o_ptr[n_base + col1] = __float2bfloat16(acc_o[nt][1] * inv0);
            }
        }
    }
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
    m.def("unified_prefill", &unified_prefill, "Unified prefill (09a: reg acc_o)");
    m.def("unified_decode",  &unified_decode,  "Unified decode  (09a: reg acc_o)");
}
