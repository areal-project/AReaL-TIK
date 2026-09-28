// unified_kernel_09d.cu — kernel_09d: minimal [16×16] smem buffer per PV K-tile
//
// Base: kernel_09c (reg acc_s, smem scores_b16[16×128])
// Change: Replace scores_b16[16×128] with pv_buf[16×16] — write only 2 N-tiles
//         (16 columns) per PV K-tile iteration, not all 16 N-tiles upfront.
//
// Key insight: We don't need the full [16×128] scores_b16 buffer. The PV loop
// processes K-tiles sequentially (kt=0..7), each needing only 2 N-tiles worth
// of weights (columns kt*16..kt*16+15). Write those 16 columns to a small
// [16×16] temp buffer, sync, ldmatrix, then proceed to next K-tile.
//
// What is NEW vs 09c:
//   - scores_b16[16×128] (4 KB per warp) REPLACED with pv_buf[16×16] (512 B per warp)
//   - write_ktile_to_smem: writes only acc_s[kt*2] and acc_s[kt*2+1] (2 N-tiles)
//   - PV loop: write→sync→ldmatrix→MMA per K-tile (8 iterations)
//
// What is IDENTICAL to 09c:
//   - acc_s[N_TILES][4] in registers (QK^T output, softmax input/output)
//   - acc_o[OUT_N_TILES][4] in registers
//   - row_max[2], row_sum[2] in registers
//   - quad_max/quad_sum, expf(), all 4 __syncthreads() per KV block
//
// Smem layout (50 KB, was 64 KB in 09c, 129 KB in kernel_08):
//   k_smem:         [128,128] bf16 = 32 KB
//   warp_region[4]: each 4608 B = 4.5 KB
//     q_tile:       [16,128] bf16 =  4096 B
//     pv_buf:       [16,16]  bf16 =   512 B
//   Total: 32768 + 4*4608 = 51200 B ≈ 50 KB

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

// Smem layout — small pv_buf instead of large scores_b16
constexpr int K_SMEM_BYTES   = BLOCK_N * HEAD_DIM * 2;   // 32768 B
constexpr int Q_TILE_BYTES   = MMA_M * HEAD_DIM * 2;     //  4096 B
constexpr int PV_BUF_BYTES   = MMA_M * MMA_K * 2;        //   512 B (16×16 bf16)
constexpr int PER_WARP_BYTES = Q_TILE_BYTES + PV_BUF_BYTES;  // 4608 B

constexpr int K_SMEM_OFF     = 0;
constexpr int WARP_BASE      = K_SMEM_BYTES;  // 32768
constexpr int SMEM_BYTES     = WARP_BASE + NUM_WARPS * PER_WARP_BYTES;  // 51200 B

constexpr int W_Q_TILE_OFF   = 0;
constexpr int W_PV_BUF_OFF   = W_Q_TILE_OFF + Q_TILE_BYTES;

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

// ============================================================================
// Quad reductions
// ============================================================================
__device__ __forceinline__ float quad_max(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 1));
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 2));
    return v;
}
__device__ __forceinline__ float quad_sum(float v) {
    v += __shfl_xor_sync(0xffffffff, v, 1);
    v += __shfl_xor_sync(0xffffffff, v, 2);
    return v;
}

// ============================================================================
// QK^T — register-resident (from 09c, unchanged)
// ============================================================================
__device__ __forceinline__ void compute_qk_reg(
    const __nv_bfloat16* __restrict__ q_tile,
    const __nv_bfloat16* __restrict__ k_smem,
    float acc_s[N_TILES][4]
) {
    int lane_id = threadIdx.x % WARP_SIZE;

    #pragma unroll
    for (int nt = 0; nt < N_TILES; nt++)
        acc_s[nt][0] = acc_s[nt][1] = acc_s[nt][2] = acc_s[nt][3] = 0.0f;

    #pragma unroll
    for (int kt = 0; kt < K_TILES; kt++) {
        uint32_t a0, a1, a2, a3;
        {
            int row = lane_id % MMA_M;
            int col = (lane_id / MMA_M) * 8 + kt * MMA_K;
            ldmatrix_a(a0, a1, a2, a3, q_tile + row * HEAD_DIM + col);
        }
        #pragma unroll
        for (int nt = 0; nt < N_TILES; nt++) {
            int n_col  = nt * MMA_N;
            int n      = n_col + (lane_id >> 2);
            int k_base = (lane_id & 3) * 2;
            int k0 = kt * MMA_K + k_base;
            int k1 = k0+1, k8 = k0+8, k9 = k8+1;
            __nv_bfloat16 v0 = k_smem[n * HEAD_DIM + k0];
            __nv_bfloat16 v1 = k_smem[n * HEAD_DIM + k1];
            __nv_bfloat16 v8 = k_smem[n * HEAD_DIM + k8];
            __nv_bfloat16 v9 = k_smem[n * HEAD_DIM + k9];
            uint32_t b0 = (reinterpret_cast<uint16_t&>(v1) << 16) | reinterpret_cast<uint16_t&>(v0);
            uint32_t b1 = (reinterpret_cast<uint16_t&>(v9) << 16) | reinterpret_cast<uint16_t&>(v8);
            mma_bf16(acc_s[nt][0], acc_s[nt][1], acc_s[nt][2], acc_s[nt][3],
                     a0, a1, a2, a3, b0, b1,
                     acc_s[nt][0], acc_s[nt][1], acc_s[nt][2], acc_s[nt][3]);
        }
    }
}

// ============================================================================
// Softmax — register-resident (from 09c, unchanged)
// ============================================================================
__device__ __forceinline__ void softmax_update_reg(
    float acc_s[N_TILES][4],
    float acc_o[OUT_N_TILES][4],
    float row_max[2],
    float row_sum[2],
    float inv_sqrt,
    int valid_n,
    int kv_start,
    int warp_q_start,
    int actual_rows
) {
    int lane_id  = threadIdx.x % WARP_SIZE;
    int group_id = lane_id >> 2;
    int col_pair = lane_id & 3;
    int row0 = group_id, row1 = group_id + 8;
    int col0 = col_pair * 2, col1 = col0 + 1;
    int q_pos0 = warp_q_start + row0;
    int q_pos1 = warp_q_start + row1;

    float bmax0 = -INFINITY, bmax1 = -INFINITY;
    #pragma unroll
    for (int nt = 0; nt < N_TILES; nt++) {
        int n_base = nt * MMA_N;
        int n0 = n_base + col0, n1 = n_base + col1;
        float s0 = acc_s[nt][0] * inv_sqrt;
        float s1 = acc_s[nt][1] * inv_sqrt;
        float s2 = acc_s[nt][2] * inv_sqrt;
        float s3 = acc_s[nt][3] * inv_sqrt;
        bool m00 = (n0 >= valid_n) || (kv_start + n0 > q_pos0) || (row0 >= actual_rows);
        bool m01 = (n1 >= valid_n) || (kv_start + n1 > q_pos0) || (row0 >= actual_rows);
        bool m10 = (n0 >= valid_n) || (kv_start + n0 > q_pos1) || (row1 >= actual_rows);
        bool m11 = (n1 >= valid_n) || (kv_start + n1 > q_pos1) || (row1 >= actual_rows);
        acc_s[nt][0] = m00 ? -INFINITY : s0;
        acc_s[nt][1] = m01 ? -INFINITY : s1;
        acc_s[nt][2] = m10 ? -INFINITY : s2;
        acc_s[nt][3] = m11 ? -INFINITY : s3;
        bmax0 = fmaxf(bmax0, fmaxf(acc_s[nt][0], acc_s[nt][1]));
        bmax1 = fmaxf(bmax1, fmaxf(acc_s[nt][2], acc_s[nt][3]));
    }
    bmax0 = quad_max(bmax0);
    bmax1 = quad_max(bmax1);

    float new_max0 = fmaxf(row_max[0], bmax0);
    float new_max1 = fmaxf(row_max[1], bmax1);
    float rescale0 = isinf(new_max0) ? 1.0f : expf(row_max[0] - new_max0);
    float rescale1 = isinf(new_max1) ? 1.0f : expf(row_max[1] - new_max1);
    row_max[0] = new_max0;
    row_max[1] = new_max1;

    #pragma unroll
    for (int nt = 0; nt < OUT_N_TILES; nt++) {
        acc_o[nt][0] *= rescale0;  acc_o[nt][1] *= rescale0;
        acc_o[nt][2] *= rescale1;  acc_o[nt][3] *= rescale1;
    }
    row_sum[0] *= rescale0;
    row_sum[1] *= rescale1;

    float bsum0 = 0.0f, bsum1 = 0.0f;
    #pragma unroll
    for (int nt = 0; nt < N_TILES; nt++) {
        float p0 = isinf(acc_s[nt][0]) ? 0.0f : expf(acc_s[nt][0] - new_max0);
        float p1 = isinf(acc_s[nt][1]) ? 0.0f : expf(acc_s[nt][1] - new_max0);
        float p2 = isinf(acc_s[nt][2]) ? 0.0f : expf(acc_s[nt][2] - new_max1);
        float p3 = isinf(acc_s[nt][3]) ? 0.0f : expf(acc_s[nt][3] - new_max1);
        acc_s[nt][0] = p0;  acc_s[nt][1] = p1;
        acc_s[nt][2] = p2;  acc_s[nt][3] = p3;
        bsum0 += p0 + p1;
        bsum1 += p2 + p3;
    }
    bsum0 = quad_sum(bsum0);
    bsum1 = quad_sum(bsum1);
    row_sum[0] += bsum0;
    row_sum[1] += bsum1;
}

// ============================================================================
// Write K-tile to small pv_buf — THE KEY CHANGE vs 09c
//
// For PV K-tile kt (covers weight columns kt*16..kt*16+15):
//   N-tile kt*2   = columns kt*16..kt*16+7
//   N-tile kt*2+1 = columns kt*16+8..kt*16+15
//
// Thread t writes:
//   acc_s[kt*2][0,1]   → pv_buf[row0, 0..1, 8..9]  (N-tile kt*2 at cols 0-7)
//   acc_s[kt*2+1][0,1] → pv_buf[row0, 8..9, ...]   (N-tile kt*2+1 at cols 8-15)
//   acc_s[kt*2][2,3]   → pv_buf[row1, ...]
//   acc_s[kt*2+1][2,3] → pv_buf[row1, ...]
//
// pv_buf layout: [16 rows, 16 cols] bf16, row-major
// Thread t (group_id, col_pair) writes to:
//   pv_buf[row0 * 16 + col_offset]
//   where col_offset = (nt - kt*2)*8 + col_pair*2
// ============================================================================
__device__ __forceinline__ void write_ktile_to_smem(
    float acc_s[N_TILES][4],
    int kt,  // PV K-tile index (0..7)
    __nv_bfloat16* __restrict__ pv_buf
) {
    int lane_id      = threadIdx.x % WARP_SIZE;
    int group_id     = lane_id >> 2;
    int tid_in_group = lane_id & 3;
    int row0 = group_id, row1 = group_id + 8;
    int col0 = tid_in_group * 2, col1 = col0 + 1;

    // N-tile kt*2 (columns 0..7 of pv_buf)
    int nt0 = kt * 2;
    pv_buf[row0 * 16 + col0] = __float2bfloat16(acc_s[nt0][0]);
    pv_buf[row0 * 16 + col1] = __float2bfloat16(acc_s[nt0][1]);
    pv_buf[row1 * 16 + col0] = __float2bfloat16(acc_s[nt0][2]);
    pv_buf[row1 * 16 + col1] = __float2bfloat16(acc_s[nt0][3]);

    // N-tile kt*2+1 (columns 8..15 of pv_buf)
    int nt1 = kt * 2 + 1;
    pv_buf[row0 * 16 + 8 + col0] = __float2bfloat16(acc_s[nt1][0]);
    pv_buf[row0 * 16 + 8 + col1] = __float2bfloat16(acc_s[nt1][1]);
    pv_buf[row1 * 16 + 8 + col0] = __float2bfloat16(acc_s[nt1][2]);
    pv_buf[row1 * 16 + 8 + col1] = __float2bfloat16(acc_s[nt1][3]);
}

// ============================================================================
// PV with per-K-tile smem write — THE KEY CHANGE vs 09c
//
// For each of 8 K-tiles:
//   1. Write acc_s[kt*2, kt*2+1] to pv_buf (16 columns)
//   2. __syncwarp() — ensure all lanes wrote to pv_buf
//   3. ldmatrix from pv_buf (16-wide, not BLOCK_N=128-wide)
//   4. MMA with V
//
// This reduces smem writes from 16 N-tiles (128 cols) to 2 N-tiles (16 cols)
// per K-tile — 8× reduction in scatter traffic.
// ============================================================================
__device__ __forceinline__ void compute_pv_reg_per_ktile(
    float acc_s[N_TILES][4],
    __nv_bfloat16* __restrict__ pv_buf,
    const __nv_bfloat16* __restrict__ v_smem,
    float acc_o[OUT_N_TILES][4]
) {
    int lane_id = threadIdx.x % WARP_SIZE;

    for (int kt = 0; kt < KV_K_TILES; kt++) {
        // Write this K-tile's weights to pv_buf
        write_ktile_to_smem(acc_s, kt, pv_buf);
        __syncwarp();  // Ensure all lanes wrote to pv_buf

        // ldmatrix from pv_buf[16×16]
        uint32_t a0, a1, a2, a3;
        {
            int row = lane_id % MMA_M;
            int col = (lane_id / MMA_M) * 8;  // col in [0, 8] for pv_buf[16]
            ldmatrix_a(a0, a1, a2, a3, pv_buf + row * 16 + col);
        }

        // MMA for all output N-tiles using this K-tile's weights
        #pragma unroll
        for (int nt = 0; nt < OUT_N_TILES; nt++) {
            int n_col  = nt * MMA_N;
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
            mma_bf16(acc_o[nt][0], acc_o[nt][1], acc_o[nt][2], acc_o[nt][3],
                     a0, a1, a2, a3, b0, b1,
                     acc_o[nt][0], acc_o[nt][1], acc_o[nt][2], acc_o[nt][3]);
        }
    }
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

    const __nv_bfloat16* q_ptr = Q + ((batch_idx * num_heads    + head_idx)    * seq_len) * HEAD_DIM;
    const __nv_bfloat16* k_ptr = K + ((batch_idx * num_kv_heads + kv_head_idx) * ctx_len) * HEAD_DIM;
    const __nv_bfloat16* v_ptr = V + ((batch_idx * num_kv_heads + kv_head_idx) * ctx_len) * HEAD_DIM;
    __nv_bfloat16*        o_ptr = O + ((batch_idx * num_heads    + head_idx)    * seq_len) * HEAD_DIM;

    int tid     = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    char* warp_smem    = smem + WARP_BASE + warp_id * PER_WARP_BYTES;
    __nv_bfloat16* q_tile = reinterpret_cast<__nv_bfloat16*>(warp_smem + W_Q_TILE_OFF);
    __nv_bfloat16* pv_buf = reinterpret_cast<__nv_bfloat16*>(warp_smem + W_PV_BUF_OFF);
    __nv_bfloat16* k_smem_p = reinterpret_cast<__nv_bfloat16*>(smem + K_SMEM_OFF);

    int warp_q_start = q_block_start + warp_id * MMA_M;
    int warp_q_end   = min(warp_q_start + MMA_M, seq_len);
    int actual_rows  = max(0, warp_q_end - warp_q_start);

    for (int i = lane_id; i < MMA_M * HEAD_DIM; i += WARP_SIZE) {
        int row = i / HEAD_DIM, col = i % HEAD_DIM;
        q_tile[i] = (row < actual_rows)
                    ? q_ptr[(warp_q_start + row) * HEAD_DIM + col]
                    : __float2bfloat16(0.0f);
    }

    float acc_o[OUT_N_TILES][4];
    float row_max[2] = {-INFINITY, -INFINITY};
    float row_sum[2] = {0.0f, 0.0f};
    #pragma unroll
    for (int nt = 0; nt < OUT_N_TILES; nt++)
        acc_o[nt][0] = acc_o[nt][1] = acc_o[nt][2] = acc_o[nt][3] = 0.0f;

    float inv_sqrt = 1.0f / sqrtf((float)HEAD_DIM);
    int num_kv_blocks = (ctx_len + BLOCK_N - 1) / BLOCK_N;

    for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        int kv_start = kv_block * BLOCK_N;
        int kv_end   = min(kv_start + BLOCK_N, ctx_len);
        int valid_n  = kv_end - kv_start;

        {
            const int vec_valid = valid_n * HEAD_DIM / 8;
            const int vec_total = BLOCK_N * HEAD_DIM / 8;
            const uint4* src = reinterpret_cast<const uint4*>(k_ptr + kv_start * HEAD_DIM);
            uint4*       dst = reinterpret_cast<uint4*>(k_smem_p);
            for (int i = tid; i < vec_valid; i += BLOCK_THREADS) dst[i] = src[i];
            for (int i = tid + vec_valid; i < vec_total; i += BLOCK_THREADS)
                dst[i] = make_uint4(0, 0, 0, 0);
        }
        __syncthreads();  // sync #1: K ready

        float acc_s[N_TILES][4];
        if (actual_rows > 0) {
            compute_qk_reg(q_tile, k_smem_p, acc_s);
            softmax_update_reg(acc_s, acc_o, row_max, row_sum,
                               inv_sqrt, valid_n, kv_start, warp_q_start, actual_rows);
        }
        __syncthreads();  // sync #2: all warps done with K smem

        {
            const int vec_valid = valid_n * HEAD_DIM / 8;
            const int vec_total = BLOCK_N * HEAD_DIM / 8;
            const uint4* src = reinterpret_cast<const uint4*>(v_ptr + kv_start * HEAD_DIM);
            uint4*       dst = reinterpret_cast<uint4*>(k_smem_p);
            for (int i = tid; i < vec_valid; i += BLOCK_THREADS) dst[i] = src[i];
            for (int i = tid + vec_valid; i < vec_total; i += BLOCK_THREADS)
                dst[i] = make_uint4(0, 0, 0, 0);
        }
        __syncthreads();  // sync #3: V ready

        if (actual_rows > 0) {
            compute_pv_reg_per_ktile(acc_s, pv_buf, k_smem_p, acc_o);
        }
        __syncthreads();  // sync #4: all warps done with V smem
    }

    if (actual_rows > 0) {
        int group_id     = lane_id >> 2;
        int tid_in_group = lane_id & 3;
        int row0 = group_id, row1 = group_id + 8;
        int col0 = tid_in_group * 2, col1 = col0 + 1;
        float inv0 = (row_sum[0] > 0.0f) ? (1.0f / row_sum[0]) : 0.0f;
        float inv1 = (row_sum[1] > 0.0f) ? (1.0f / row_sum[1]) : 0.0f;

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

    int tid     = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    char* w0_smem    = smem + WARP_BASE;
    __nv_bfloat16* q_tile = reinterpret_cast<__nv_bfloat16*>(w0_smem + W_Q_TILE_OFF);
    __nv_bfloat16* pv_buf = reinterpret_cast<__nv_bfloat16*>(w0_smem + W_PV_BUF_OFF);
    __nv_bfloat16* k_smem_p = reinterpret_cast<__nv_bfloat16*>(smem + K_SMEM_OFF);

    if (warp_id == 0) {
        for (int i = lane_id; i < MMA_M * HEAD_DIM; i += WARP_SIZE) {
            int row = i / HEAD_DIM, col = i % HEAD_DIM;
            q_tile[i] = (row == 0) ? q_ptr[col] : __float2bfloat16(0.0f);
        }
    }
    __syncthreads();

    float acc_o[OUT_N_TILES][4];
    float row_max[2] = {-INFINITY, -INFINITY};
    float row_sum[2] = {0.0f, 0.0f};
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

        float acc_s[N_TILES][4];
        if (warp_id == 0) {
            compute_qk_reg(q_tile, k_smem_p, acc_s);
            softmax_update_reg(acc_s, acc_o, row_max, row_sum,
                               inv_sqrt, valid_n, kv_start, ctx_len - 1, 1);
        }
        __syncthreads();  // sync #2

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

        if (warp_id == 0) {
            compute_pv_reg_per_ktile(acc_s, pv_buf, k_smem_p, acc_o);
        }
        __syncthreads();  // sync #4
    }

    if (warp_id == 0) {
        int group_id     = lane_id >> 2;
        int tid_in_group = lane_id & 3;
        int col0 = tid_in_group * 2, col1 = col0 + 1;
        float inv0 = (row_sum[0] > 0.0f) ? (1.0f / row_sum[0]) : 0.0f;

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
    m.def("unified_prefill", &unified_prefill, "Unified prefill (09d: minimal pv_buf per K-tile)");
    m.def("unified_decode",  &unified_decode,  "Unified decode  (09d: minimal pv_buf per K-tile)");
}
