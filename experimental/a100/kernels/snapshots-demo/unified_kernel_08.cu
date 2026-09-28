// unified_kernel_08.cu — kernel_08: split-Q warp partitioning (FA2-style)
//
// Changes from kernel_06:
//   1. Split-Q warp partitioning: each of 4 warps independently owns MMA_M=16
//      query rows and processes the full K/V sequence with no cross-warp
//      communication on scores, softmax, or output accumulators.
//   2. BLOCK_M_PREFILL = 4 × MMA_M = 64.
//   3. Per-warp private smem: q_tile, out_tile, scores_f32, scores_bf16,
//      softmax, rescale — all non-overlapping between warps.
//   4. Cooperative K/V load (all 4 warps together), then each warp independently
//      runs QK^T → softmax → PV without __syncthreads on accumulators.
//   5. Syncs per KV block: 4 (K-load, K-done, V-load, V-done) vs 6 in kernel_06.
//
// Bug fixes vs initial draft:
//   - process_warp_rows no longer calls compute_v_hmma (was passing v_smem=nullptr)
//   - __syncwarp() added after mask loop (before lane 0 reads scores_f32 for softmax)
//   - N_TILES=16 accumulator array d[16][4] replaced with serialized scatter to
//     avoid 64 f32/thread register spill: compute 4 N-tiles at a time (N_CHUNK=4),
//     scatter immediately, repeat — keeps register pressure at d[4][4]=64 bytes.
//   - Decode Q load restricted to warp 0 only (prevent warps 1-3 corrupting warp0 smem)
//
// Shared memory layout (129 KB total, 1 CTA/SM on 228 KB limit):
//   k_smem:          [128, 128] bf16 = 32 KB  (cooperative load, reused for V)
//   warp_region[4]:  each 24768 B = 24.2 KB
//     q_tile:        [16, 128] bf16 =  4 KB
//     out_tile:      [16, 128] f32  =  8 KB
//     scores_f32:    [16, 128] f32  =  8 KB
//     scores_bf16:   [16, 128] bf16 =  4 KB
//     softmax:       [16, 2]   f32  =  128 B
//     rescale:       [16]      f32  =   64 B
//   Total: 32768 + 4×24768 = 131840 B ≈ 129 KB

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

constexpr int N_TILES   = BLOCK_N  / MMA_N;   // 16 — QK^T output N-tiles per warp
constexpr int K_TILES   = HEAD_DIM / MMA_K;   // 8  — QK^T K-tiles
constexpr int KV_K_TILES = BLOCK_N / MMA_K;   // 8  — PV K-tiles (over BLOCK_N)
constexpr int OUT_N_TILES = HEAD_DIM / MMA_N;  // 16 — PV output N-tiles per warp

// To avoid register spill from d[N_TILES=16][4] = 64 f32 per thread,
// process QK^T in chunks of N_CHUNK N-tiles, scatter immediately.
constexpr int N_CHUNK = 4;  // 4 N-tiles at a time → d[4][4] = 16 f32 per thread

// Shared memory layout
constexpr int K_SMEM_BYTES     = BLOCK_N * HEAD_DIM * 2;   // 32768 B — shared K/V
constexpr int Q_TILE_BYTES     = MMA_M * HEAD_DIM * 2;     //  4096 B
constexpr int OUT_TILE_BYTES   = MMA_M * HEAD_DIM * 4;     //  8192 B
constexpr int SCORES_F32_BYTES = MMA_M * BLOCK_N * 4;      //  8192 B
constexpr int SCORES_B16_BYTES = MMA_M * BLOCK_N * 2;      //  4096 B
constexpr int SOFTMAX_BYTES    = MMA_M * 2 * 4;            //   128 B
constexpr int RESCALE_BYTES    = MMA_M * 4;                //    64 B
constexpr int PER_WARP_BYTES   = Q_TILE_BYTES + OUT_TILE_BYTES + SCORES_F32_BYTES
                                + SCORES_B16_BYTES + SOFTMAX_BYTES + RESCALE_BYTES;
// = 24768 B per warp

constexpr int K_SMEM_OFF  = 0;
constexpr int WARP_BASE   = K_SMEM_BYTES;  // 32768
constexpr int SMEM_BYTES  = WARP_BASE + NUM_WARPS * PER_WARP_BYTES;  // 131840 B ≈ 129 KB

// Per-warp offsets (relative to warp's private region base)
constexpr int W_Q_TILE_OFF      = 0;
constexpr int W_OUT_TILE_OFF    = W_Q_TILE_OFF      + Q_TILE_BYTES;
constexpr int W_SCORES_F32_OFF  = W_OUT_TILE_OFF    + OUT_TILE_BYTES;
constexpr int W_SCORES_B16_OFF  = W_SCORES_F32_OFF  + SCORES_F32_BYTES;
constexpr int W_SOFTMAX_OFF     = W_SCORES_B16_OFF  + SCORES_B16_BYTES;
constexpr int W_RESCALE_OFF     = W_SOFTMAX_OFF     + SOFTMAX_BYTES;

// ============================================================================
// PTX wrappers (identical to kernel_06)
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
// Per-warp QK^T: scores_f32[MMA_M=16, BLOCK_N=128] = Q[16,128] * K[128,128]^T
//
// Register pressure fix: process N_CHUNK=4 N-tiles at a time, scatter to smem
// immediately, then repeat. Keeps accumulator at d[4][4]=16 f32 per thread
// instead of d[16][4]=64 f32 (which would spill to local memory).
// ============================================================================
__device__ __forceinline__ void compute_qk_hmma(
    const __nv_bfloat16* __restrict__ q_tile,   // warp-private [16,128] bf16
    const __nv_bfloat16* __restrict__ k_smem,   // shared [128,128] bf16
    float* __restrict__ scores_f32              // warp-private [16,128] f32
) {
    int lane_id = threadIdx.x % WARP_SIZE;
    int group_id     = lane_id >> 2;
    int tid_in_group = lane_id % 4;
    int row0 = group_id, row1 = group_id + 8;
    int col0 = tid_in_group * 2, col1 = col0 + 1;

    // Process N_TILES=16 in chunks of N_CHUNK=4
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

        // Scatter this chunk to warp-private scores_f32
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
// Per-warp PV: out_tile[MMA_M=16, HEAD_DIM=128] += W[16,128] * V[128,128]
// Same chunked approach: OUT_N_TILES=16 in chunks of N_CHUNK=4.
// ============================================================================
__device__ __forceinline__ void compute_v_hmma(
    const __nv_bfloat16* __restrict__ w_smem,  // warp-private [16, 128] bf16
    const __nv_bfloat16* __restrict__ v_smem,  // shared [128, 128] bf16
    float* __restrict__ out_tile               // warp-private [16, 128] f32
) {
    int lane_id = threadIdx.x % WARP_SIZE;
    int group_id     = lane_id >> 2;
    int tid_in_group = lane_id % 4;
    int row0 = group_id, row1 = group_id + 8;
    int col0 = tid_in_group * 2, col1 = col0 + 1;

    for (int nc = 0; nc < OUT_N_TILES; nc += N_CHUNK) {
        float d[N_CHUNK][4] = {};

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

        // Accumulate chunk into warp-private out_tile
        for (int nt = 0; nt < N_CHUNK; nt++) {
            int n_base = (nc + nt) * MMA_N;
            out_tile[row0 * HEAD_DIM + n_base + col0] += d[nt][0];
            out_tile[row0 * HEAD_DIM + n_base + col1] += d[nt][1];
            out_tile[row1 * HEAD_DIM + n_base + col0] += d[nt][2];
            out_tile[row1 * HEAD_DIM + n_base + col1] += d[nt][3];
        }
    }
}

// ============================================================================
// Prefill kernel — split-Q
// 4 warps × 16 rows = 64 rows per CTA. Each warp is fully independent after
// the cooperative K/V loads. Syncs: 4 per KV block.
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

    // Each warp's private smem region
    char* warp_smem   = smem + WARP_BASE + warp_id * PER_WARP_BYTES;
    __nv_bfloat16* q_tile    = reinterpret_cast<__nv_bfloat16*>(warp_smem + W_Q_TILE_OFF);
    float*         out_tile  = reinterpret_cast<float*>(warp_smem + W_OUT_TILE_OFF);
    float*         scores_f32= reinterpret_cast<float*>(warp_smem + W_SCORES_F32_OFF);
    __nv_bfloat16* scores_b16= reinterpret_cast<__nv_bfloat16*>(warp_smem + W_SCORES_B16_OFF);
    float*         softmax   = reinterpret_cast<float*>(warp_smem + W_SOFTMAX_OFF);
    float*         rescale   = reinterpret_cast<float*>(warp_smem + W_RESCALE_OFF);
    __nv_bfloat16* k_smem_p  = reinterpret_cast<__nv_bfloat16*>(smem + K_SMEM_OFF);

    // This warp's Q row range
    int warp_q_start = q_block_start + warp_id * MMA_M;
    int warp_q_end   = min(warp_q_start + MMA_M, seq_len);
    int actual_rows  = max(0, warp_q_end - warp_q_start);

    // Load this warp's Q tile into its private smem (warp-local, no sync needed)
    for (int i = lane_id; i < MMA_M * HEAD_DIM; i += WARP_SIZE) {
        int row = i / HEAD_DIM, col = i % HEAD_DIM;
        q_tile[i] = (row < actual_rows)
                    ? q_ptr[(warp_q_start + row) * HEAD_DIM + col]
                    : __float2bfloat16(0.0f);
    }

    // Initialize per-warp out_tile and softmax state (warp-local, no sync needed)
    for (int i = lane_id; i < MMA_M * HEAD_DIM; i += WARP_SIZE) out_tile[i] = 0.0f;
    if (lane_id < MMA_M) {
        softmax[lane_id * 2]     = -INFINITY;
        softmax[lane_id * 2 + 1] = 0.0f;
    }

    float inv_sqrt = 1.0f / sqrtf((float)HEAD_DIM);
    int num_kv_blocks = (ctx_len + BLOCK_N - 1) / BLOCK_N;

    for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        int kv_start = kv_block * BLOCK_N;
        int kv_end   = min(kv_start + BLOCK_N, ctx_len);
        int valid_n  = kv_end - kv_start;

        // ── Cooperative K load ─────────────────────────────────────────────
        {
            const int vec_valid = valid_n * HEAD_DIM / 8;
            const int vec_total = BLOCK_N * HEAD_DIM / 8;
            const uint4* src = reinterpret_cast<const uint4*>(k_ptr + kv_start * HEAD_DIM);
            uint4*       dst = reinterpret_cast<uint4*>(k_smem_p);
            for (int i = tid; i < vec_valid; i += BLOCK_THREADS) dst[i] = src[i];
            for (int i = tid + vec_valid; i < vec_total; i += BLOCK_THREADS)
                dst[i] = make_uint4(0, 0, 0, 0);
        }
        __syncthreads();  // sync #1: K visible to all warps

        // ── Per-warp QK^T ──────────────────────────────────────────────────
        if (actual_rows > 0) {
            compute_qk_hmma(q_tile, k_smem_p, scores_f32);

            // Scale + causal mask: lanes 0..15 handle rows 0..15
            // (MMA_M=16 < WARP_SIZE=32; lanes 16..31 are idle but that's OK)
            for (int m = lane_id; m < MMA_M; m += WARP_SIZE) {
                int q_pos = warp_q_start + m;
                for (int n = 0; n < BLOCK_N; n++) {
                    float s = scores_f32[m * BLOCK_N + n] * inv_sqrt;
                    bool masked = (n >= valid_n) || (kv_start + n > q_pos) || (m >= actual_rows);
                    scores_f32[m * BLOCK_N + n] = masked ? -INFINITY : s;
                }
            }
            __syncwarp();  // lanes 0..15 wrote scores_f32; lane 0 reads next

            // Online softmax — lane 0 does all MMA_M rows serially
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
            __syncwarp();  // broadcast rescale[] and scores_f32[] within warp

            // Rescale out_tile + convert scores f32→bf16
            for (int i = lane_id; i < MMA_M * HEAD_DIM; i += WARP_SIZE)
                out_tile[i] *= rescale[i / HEAD_DIM];
            for (int i = lane_id; i < MMA_M * BLOCK_N; i += WARP_SIZE)
                scores_b16[i] = __float2bfloat16(scores_f32[i]);
        }
        __syncthreads();  // sync #2: all warps done with K smem, safe to load V

        // ── Cooperative V load (reuse k_smem) ─────────────────────────────
        {
            const int vec_valid = valid_n * HEAD_DIM / 8;
            const int vec_total = BLOCK_N * HEAD_DIM / 8;
            const uint4* src = reinterpret_cast<const uint4*>(v_ptr + kv_start * HEAD_DIM);
            uint4*       dst = reinterpret_cast<uint4*>(k_smem_p);
            for (int i = tid; i < vec_valid; i += BLOCK_THREADS) dst[i] = src[i];
            for (int i = tid + vec_valid; i < vec_total; i += BLOCK_THREADS)
                dst[i] = make_uint4(0, 0, 0, 0);
        }
        __syncthreads();  // sync #3: V visible to all warps

        // ── Per-warp PV ────────────────────────────────────────────────────
        if (actual_rows > 0) {
            compute_v_hmma(scores_b16, k_smem_p, out_tile);
        }
        __syncthreads();  // sync #4: all warps done with V smem
    }

    // Write output
    if (actual_rows > 0) {
        // Finalize inv_sum
        if (lane_id < MMA_M) {
            float s = softmax[lane_id * 2 + 1];
            rescale[lane_id] = (s > 0.0f) ? (1.0f / s) : 0.0f;
        }
        __syncwarp();

        for (int i = lane_id; i < actual_rows * HEAD_DIM; i += WARP_SIZE) {
            int row = i / HEAD_DIM, col = i % HEAD_DIM;
            o_ptr[(warp_q_start + row) * HEAD_DIM + col] =
                __float2bfloat16(out_tile[i] * rescale[row]);
        }
    }
}

// ============================================================================
// Decode kernel — split-Q is a no-op (single query row).
// Only warp 0 does meaningful work; other warps help with cooperative K/V loads.
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

    // Use warp 0's private region for all compute
    char* w0_smem    = smem + WARP_BASE;  // warp 0's region
    __nv_bfloat16* q_tile    = reinterpret_cast<__nv_bfloat16*>(w0_smem + W_Q_TILE_OFF);
    float*         out_tile  = reinterpret_cast<float*>(w0_smem + W_OUT_TILE_OFF);
    float*         scores_f32= reinterpret_cast<float*>(w0_smem + W_SCORES_F32_OFF);
    __nv_bfloat16* scores_b16= reinterpret_cast<__nv_bfloat16*>(w0_smem + W_SCORES_B16_OFF);
    float*         softmax   = reinterpret_cast<float*>(w0_smem + W_SOFTMAX_OFF);
    float*         rescale   = reinterpret_cast<float*>(w0_smem + W_RESCALE_OFF);
    __nv_bfloat16* k_smem_p  = reinterpret_cast<__nv_bfloat16*>(smem + K_SMEM_OFF);

    // Load Q into warp 0's q_tile — ONLY warp 0 writes (avoids corrupting its smem)
    if (warp_id == 0) {
        for (int i = lane_id; i < MMA_M * HEAD_DIM; i += WARP_SIZE) {
            int row = i / HEAD_DIM, col = i % HEAD_DIM;
            q_tile[i] = (row == 0) ? q_ptr[col] : __float2bfloat16(0.0f);
        }
        for (int i = lane_id; i < MMA_M * HEAD_DIM; i += WARP_SIZE) out_tile[i] = 0.0f;
        if (lane_id < MMA_M) {
            softmax[lane_id * 2]     = -INFINITY;
            softmax[lane_id * 2 + 1] = 0.0f;
        }
    }
    __syncthreads();  // ensure warp 0's init is visible before KV loop

    float inv_sqrt = 1.0f / sqrtf((float)HEAD_DIM);
    int num_kv_blocks = (ctx_len + BLOCK_N - 1) / BLOCK_N;

    for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        int kv_start = kv_block * BLOCK_N;
        int kv_end   = min(kv_start + BLOCK_N, ctx_len);
        int valid_n  = kv_end - kv_start;

        // Cooperative K load (all warps)
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

        // Warp 0: QK^T + softmax
        if (warp_id == 0) {
            compute_qk_hmma(q_tile, k_smem_p, scores_f32);

            // Causal mask (q_pos = ctx_len - 1, actual_rows = 1)
            for (int m = lane_id; m < MMA_M; m += WARP_SIZE) {
                int q_pos = ctx_len - 1;
                for (int n = 0; n < BLOCK_N; n++) {
                    float s = scores_f32[m * BLOCK_N + n] * inv_sqrt;
                    bool masked = (n >= valid_n) || (kv_start + n > q_pos) || (m >= 1);
                    scores_f32[m * BLOCK_N + n] = masked ? -INFINITY : s;
                }
            }
            __syncwarp();

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

            for (int i = lane_id; i < HEAD_DIM; i += WARP_SIZE)
                out_tile[i] *= rescale[0];
            for (int i = lane_id; i < MMA_M * BLOCK_N; i += WARP_SIZE)
                scores_b16[i] = __float2bfloat16(scores_f32[i]);
        }
        __syncthreads();  // sync #2: warp 0 done with K

        // Cooperative V load
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

        // Warp 0: PV
        if (warp_id == 0) {
            compute_v_hmma(scores_b16, k_smem_p, out_tile);
        }
        __syncthreads();  // sync #4
    }

    // Warp 0 writes output
    if (warp_id == 0) {
        if (lane_id == 0) {
            float s = softmax[1];
            rescale[0] = (s > 0.0f) ? (1.0f / s) : 0.0f;
        }
        __syncwarp();
        for (int col = lane_id; col < HEAD_DIM; col += WARP_SIZE)
            o_ptr[col] = __float2bfloat16(out_tile[col] * rescale[0]);
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
    m.def("unified_prefill", &unified_prefill, "Unified prefill (split-Q, chunked HMMA)");
    m.def("unified_decode",  &unified_decode,  "Unified decode (split-Q, chunked HMMA)");
}
