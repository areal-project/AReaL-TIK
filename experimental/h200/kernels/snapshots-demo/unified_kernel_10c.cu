// unified_kernel_10c.cu — kernel_10c: unrolled K_TILES loop to eliminate branch + register spill
//
// Base: kernel_10b (TMA + SWIZZLE_128B split tiles, 4.0x, correct)
// Change: SWIZZLE_128B via split tiles — split HEAD_DIM=128 into two 64-element halves
//
// Problem: SWIZZLE_128B requires inner dim ≤ 128 bytes = 64 bf16 elements
//          Our rows are 128 bf16 = 256 bytes → too large for SWIZZLE_128B
//
// Solution: Split each row into lo [0..63] and hi [64..127] halves
//   Each half = 64 bf16 = 128 bytes ≤ 128 byte limit ✓
//   Two TMA calls per K/V load (lo half + hi half)
//   One TMA descriptor per tensor (same desc, different coord_col: 0 or 64)
//
// Smem layout (same total size as 10a):
//   k_smem_lo: [BLOCK_N=128, 64] bf16 = 16 KB  (cols 0..63)
//   k_smem_hi: [BLOCK_N=128, 64] bf16 = 16 KB  (cols 64..127)
//   v_smem_lo: [BLOCK_N=128, 64] bf16 = 16 KB
//   v_smem_hi: [BLOCK_N=128, 64] bf16 = 16 KB
//   Total K/V: 64 KB (was 32 KB in 10a — doubled due to split)
//
// Wait — this doubles smem! Let me reconsider.
// Actually we can reuse: k_lo and k_hi share the same 32KB buffer
// by loading lo first, computing partial QK, then loading hi, accumulating.
// But that changes the algorithm...
//
// Simpler: keep single 32KB K buffer, split into lo/hi halves within it:
//   k_smem[0..16KB-1] = lo half [BLOCK_N, 64]
//   k_smem[16KB..32KB-1] = hi half [BLOCK_N, 64]
//   Total: 32 KB (same as 10a)
//
// Swizzle formula for 64-element box with SWIZZLE_128B:
//   physical_col = logical_col XOR ((row * 8) & 63)
//   (8 = 16B_chunk / sizeof(bf16), period = 64 elements)
//
// TMA replaces the scalar load loops with a single hardware instruction.
// Everything else is IDENTICAL to 09i:
//   - Same smem layout (54 KB, K_SMEM_STRIDE=136, V_SMEM_STRIDE=136)
//   - Same 4 __syncthreads() per KV block
//   - Same compute functions (QK, softmax, PV)
//   - Same bitwise correctness
//
// TMA descriptor encodes the padded smem layout:
//   K: global [BLOCK_N, HEAD_DIM] → smem [BLOCK_N, K_SMEM_STRIDE] (padded)
//   V: global [BLOCK_N, HEAD_DIM] → smem [BLOCK_N, V_SMEM_STRIDE] (padded)
//
// Synchronization: mbarrier replaces the implicit sync in scalar loads
//   - Thread 0 issues TMA + signals mbarrier
//   - All threads wait on mbarrier before compute
//   - __syncthreads() still used after compute (same as 09i)
//
// Host: cuTensorMapEncodeTiled creates TMA descriptors with padded smem stride
//
// NCU profiling showed 5.6-way bank conflicts on shared loads (68% speedup potential).
// Root cause: k_smem[n * HEAD_DIM + k] where HEAD_DIM=128 bf16 = 256 bytes
//   → row stride = 256 bytes = 8 × 32-byte bank lines
//   → all rows start at same bank offset → threads in different rows hit same bank
//
// Fix: pad each row by K_SMEM_PAD=8 bf16 (16 bytes)
//   → row stride = (128+8) × 2 = 272 bytes
//   → 272 % 64 = 16 → rows start at different bank offsets → no conflicts
//
// Analysis (bank = addr % 32 for bf16 elements):
//   Original: n=0→bank 0, n=1→bank 0, n=2→bank 0, n=3→bank 0  (4-way conflict!)
//   Padded:   n=0→bank 0, n=1→bank 8, n=2→bank 16, n=3→bank 24 (zero conflicts!)
//
// What is NEW vs 09e:
//   - K_SMEM_PAD = 8 (bf16 padding per row)
//   - K_SMEM_STRIDE = HEAD_DIM + K_SMEM_PAD = 136
//   - K smem size: BLOCK_N * K_SMEM_STRIDE * 2 = 34816 B (was 32768 B)
//   - K load: same coalesced uint4 loads, same layout — no change needed
//   - compute_qk_reg: k_smem[n * K_SMEM_STRIDE + k] instead of k_smem[n * HEAD_DIM + k]
//   - compute_pv_reg_per_ktile: v_smem[k * HEAD_DIM + n] unchanged (V not padded yet)
//
// What is IDENTICAL to 09e:
//   - All register arrays, softmax, exp2f — unchanged
//   - K/V load pattern (coalesced uint4) — unchanged
//   - pv_buf[16×16] — unchanged
//   - All 4 __syncthreads() per KV block — unchanged
//
// Smem layout (52 KB, was 50 KB in 09e):
//   k_smem:         [128, 136] bf16 = 34816 B  (padded, reused for V unpadded)
//   warp_region[4]: each 4608 B = 4.5 KB
//     q_tile:       [16,128] bf16 =  4096 B
//     pv_buf:       [16,16]  bf16 =   512 B
//   Total: 34816 + 4*4608 = 53248 B ≈ 52 KB

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cuda.h>  // cuTensorMapEncodeTiled (driver API)

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

// Mathematical constant for exp2f trick: exp(x) = exp2(x * LOG2E)
constexpr float LOG2E = 1.442695040888963407359924681001892137426646f;

// Smem layout — padded K/V smem + padded pv_buf to eliminate bank conflicts
// K/V share same padded buffer (both use stride=136)
// pv_buf padded to 24 cols (ldmatrix-aligned: 24*2=48 bytes, divisible by 16)
constexpr int KV_SMEM_PAD    = 8;                                    // bf16 padding per row
constexpr int K_SMEM_STRIDE  = HEAD_DIM + KV_SMEM_PAD;              // 136 bf16 per row (K)
constexpr int V_SMEM_STRIDE  = BLOCK_N  + KV_SMEM_PAD;              // 136 bf16 per row (V)
// Split tile: HEAD_DIM=128 split into lo [0..63] and hi [64..127]
// Each half = 64 bf16 = 128 bytes ≤ SWIZZLE_128B limit ✓
constexpr int HALF_DIM       = HEAD_DIM / 2;                             // 64
constexpr int KV_HALF_BYTES  = BLOCK_N * HALF_DIM * 2;                  // 16384 B per half
constexpr int KV_LO_OFF      = 0;                                        // lo half offset in bf16 elements
constexpr int KV_HI_OFF      = BLOCK_N * HALF_DIM;                      // hi half offset in bf16 elements = 8192
constexpr int KV_SMEM_BYTES  = KV_HALF_BYTES * 2;                       // 32768 B total
constexpr int Q_TILE_BYTES   = MMA_M * HEAD_DIM * 2;                    //  4096 B
constexpr int PV_BUF_COLS    = 24;                                       // padded: 24*2=48B, ldmatrix-aligned
constexpr int PV_BUF_BYTES   = MMA_M * PV_BUF_COLS * 2;                 //   768 B (16×24 bf16)
constexpr int PER_WARP_BYTES = Q_TILE_BYTES + PV_BUF_BYTES;             //  4864 B

// mbarrier storage: 2 barriers (K_ready, V_ready), each 8 bytes, 8-byte aligned
constexpr int MBAR_BYTES     = 2 * 8;                                   //    16 B
constexpr int K_SMEM_OFF     = 0;
constexpr int MBAR_OFF       = KV_SMEM_BYTES;                           // 32768
constexpr int WARP_BASE      = MBAR_OFF + MBAR_BYTES;                   // 32784
constexpr int SMEM_BYTES     = WARP_BASE + NUM_WARPS * PER_WARP_BYTES;  // 52256 B (unchanged)

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
// mbarrier helpers (Hopper hardware barriers for TMA synchronization)
// ============================================================================

// Initialize mbarrier: expects `count` arrivals before releasing waiters
__device__ __forceinline__ void mbar_init(uint64_t* b, uint32_t count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"((uint32_t)__cvta_generic_to_shared(b)), "r"(count));
}

// Arrive and declare N bytes of pending TMA transaction
// TMA hardware will arrive automatically when copy completes
__device__ __forceinline__ void mbar_arrive_tx(uint64_t* b, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"((uint32_t)__cvta_generic_to_shared(b)), "r"(tx_bytes));
}

// Wait for mbarrier to complete (phase alternates 0/1 each cycle)
__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t phase) {
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n"
        "@!P bra WAIT;\n"
        "}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(b)), "r"(phase)
    );
}

// ============================================================================
// Swizzle address helper for SWIZZLE_128B
// TMA with SWIZZLE_128B writes element (row, col) at:
//   physical_col = col XOR ((row * 8) % 64)
//   physical_addr = row * row_stride + physical_col
// where 8 = 16B_chunk_size / sizeof(bf16) = 8 elements
// ============================================================================
// Correct SWIZZLE_128B formula for bf16:
// 128B span, 16B chunks = 8 bf16 per chunk
// physical_col = ((col/8) ^ (row%8)) * 8 + (col%8)
// For 64-element half: col in [0..63], row in [0..127]
// row%8 gives the XOR value (repeats every 8 rows)
__device__ __forceinline__ int swizzle_col(int row, int col) {
    return ((col / 8) ^ (row % 8)) * 8 + (col % 8);
}

// ============================================================================
// TMA load: one thread issues async 2D tile copy global → smem
// The TMA descriptor encodes the global tensor layout and smem box size.
// coord_col: column offset in global tensor (always 0 for us)
// coord_row: row offset in global tensor (= kv_row_base + kv_start)
// ============================================================================
__device__ __forceinline__ void tma_load_2d(
    const CUtensorMap* __restrict__ desc,
    __nv_bfloat16* __restrict__ smem_dst,
    uint64_t* __restrict__ mbar,
    int coord_col,   // column offset (0)
    int coord_row    // row offset (absolute row in global tensor)
) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem_dst)),
           "l"(desc),
           "r"(coord_col), "r"(coord_row),
           "r"((uint32_t)__cvta_generic_to_shared(mbar))
        : "memory"
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
    // Unrolled K_TILES loop: kt=0..3 uses lo half, kt=4..7 uses hi half
    // Eliminates runtime branch (k < HALF_DIM) → no register spill
    int lane_id = threadIdx.x % WARP_SIZE;
    const __nv_bfloat16* k_lo = k_smem + KV_LO_OFF;
    const __nv_bfloat16* k_hi = k_smem + KV_HI_OFF;

    #pragma unroll
    for (int nt = 0; nt < N_TILES; nt++)
        acc_s[nt][0] = acc_s[nt][1] = acc_s[nt][2] = acc_s[nt][3] = 0.0f;

    // Lo half: kt=0..3, k0 in [0..54], k8 in [8..62] → all < HALF_DIM=64
    #pragma unroll
    for (int kt = 0; kt < K_TILES / 2; kt++) {
        uint32_t a0, a1, a2, a3;
        ldmatrix_a(a0, a1, a2, a3,
                   q_tile + (lane_id % MMA_M) * HEAD_DIM + (lane_id / MMA_M) * 8 + kt * MMA_K);
        #pragma unroll
        for (int nt = 0; nt < N_TILES; nt++) {
            int n      = nt * MMA_N + (lane_id >> 2);
            int k_base = (lane_id & 3) * 2;
            int k0 = kt * MMA_K + k_base;
            int k1 = k0+1, k8 = k0+8, k9 = k8+1;
            __nv_bfloat16 v0 = k_lo[n * HALF_DIM + swizzle_col(n, k0)];
            __nv_bfloat16 v1 = k_lo[n * HALF_DIM + swizzle_col(n, k1)];
            __nv_bfloat16 v8 = k_lo[n * HALF_DIM + swizzle_col(n, k8)];
            __nv_bfloat16 v9 = k_lo[n * HALF_DIM + swizzle_col(n, k9)];
            uint32_t b0 = (reinterpret_cast<uint16_t&>(v1) << 16) | reinterpret_cast<uint16_t&>(v0);
            uint32_t b1 = (reinterpret_cast<uint16_t&>(v9) << 16) | reinterpret_cast<uint16_t&>(v8);
            mma_bf16(acc_s[nt][0], acc_s[nt][1], acc_s[nt][2], acc_s[nt][3],
                     a0, a1, a2, a3, b0, b1,
                     acc_s[nt][0], acc_s[nt][1], acc_s[nt][2], acc_s[nt][3]);
        }
    }

    // Hi half: kt=4..7, k0 in [64..118], k8 in [72..126] → all >= HALF_DIM=64
    #pragma unroll
    for (int kt = K_TILES / 2; kt < K_TILES; kt++) {
        uint32_t a0, a1, a2, a3;
        ldmatrix_a(a0, a1, a2, a3,
                   q_tile + (lane_id % MMA_M) * HEAD_DIM + (lane_id / MMA_M) * 8 + kt * MMA_K);
        #pragma unroll
        for (int nt = 0; nt < N_TILES; nt++) {
            int n      = nt * MMA_N + (lane_id >> 2);
            int k_base = (lane_id & 3) * 2;
            int k0 = kt * MMA_K + k_base;
            int k1 = k0+1, k8 = k0+8, k9 = k8+1;
            int lk0 = k0 - HALF_DIM, lk1 = k1 - HALF_DIM;
            int lk8 = k8 - HALF_DIM, lk9 = k9 - HALF_DIM;
            __nv_bfloat16 v0 = k_hi[n * HALF_DIM + swizzle_col(n, lk0)];
            __nv_bfloat16 v1 = k_hi[n * HALF_DIM + swizzle_col(n, lk1)];
            __nv_bfloat16 v8 = k_hi[n * HALF_DIM + swizzle_col(n, lk8)];
            __nv_bfloat16 v9 = k_hi[n * HALF_DIM + swizzle_col(n, lk9)];
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
    float rescale0 = isinf(new_max0) ? 1.0f : exp2f((row_max[0] - new_max0) * LOG2E);
    float rescale1 = isinf(new_max1) ? 1.0f : exp2f((row_max[1] - new_max1) * LOG2E);
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
        float p0 = isinf(acc_s[nt][0]) ? 0.0f : exp2f((acc_s[nt][0] - new_max0) * LOG2E);
        float p1 = isinf(acc_s[nt][1]) ? 0.0f : exp2f((acc_s[nt][1] - new_max0) * LOG2E);
        float p2 = isinf(acc_s[nt][2]) ? 0.0f : exp2f((acc_s[nt][2] - new_max1) * LOG2E);
        float p3 = isinf(acc_s[nt][3]) ? 0.0f : exp2f((acc_s[nt][3] - new_max1) * LOG2E);
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

    // N-tile kt*2 (columns 0..7 of pv_buf, padded to PV_BUF_COLS=24)
    int nt0 = kt * 2;
    pv_buf[row0 * PV_BUF_COLS + col0] = __float2bfloat16(acc_s[nt0][0]);
    pv_buf[row0 * PV_BUF_COLS + col1] = __float2bfloat16(acc_s[nt0][1]);
    pv_buf[row1 * PV_BUF_COLS + col0] = __float2bfloat16(acc_s[nt0][2]);
    pv_buf[row1 * PV_BUF_COLS + col1] = __float2bfloat16(acc_s[nt0][3]);

    // N-tile kt*2+1 (columns 8..15 of pv_buf)
    int nt1 = kt * 2 + 1;
    pv_buf[row0 * PV_BUF_COLS + 8 + col0] = __float2bfloat16(acc_s[nt1][0]);
    pv_buf[row0 * PV_BUF_COLS + 8 + col1] = __float2bfloat16(acc_s[nt1][1]);
    pv_buf[row1 * PV_BUF_COLS + 8 + col0] = __float2bfloat16(acc_s[nt1][2]);
    pv_buf[row1 * PV_BUF_COLS + 8 + col1] = __float2bfloat16(acc_s[nt1][3]);
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
    // Unrolled: split OUT_N_TILES into lo (nt=0..7, n<64) and hi (nt=8..15, n>=64)
    // Eliminates runtime branch (n < HALF_DIM) → no register spill
    int lane_id = threadIdx.x % WARP_SIZE;
    const __nv_bfloat16* v_lo = v_smem + KV_LO_OFF;
    const __nv_bfloat16* v_hi = v_smem + KV_HI_OFF;

    for (int kt = 0; kt < KV_K_TILES; kt++) {
        write_ktile_to_smem(acc_s, kt, pv_buf);
        __syncwarp();

        uint32_t a0, a1, a2, a3;
        ldmatrix_a(a0, a1, a2, a3,
                   pv_buf + (lane_id % MMA_M) * PV_BUF_COLS + (lane_id / MMA_M) * 8);

        int k_base = (lane_id % 4) * 2;
        int k0 = kt * MMA_K + k_base;
        int k1 = k0+1, k8 = k0+8, k9 = k8+1;

        // Lo half: nt=0..7, n = nt*8 + (lane/4) in [0..63] → all < HALF_DIM
        #pragma unroll
        for (int nt = 0; nt < OUT_N_TILES / 2; nt++) {
            int n = nt * MMA_N + (lane_id / 4);  // in [0..63]
            __nv_bfloat16 e0 = v_lo[k0 * HALF_DIM + swizzle_col(k0, n)];
            __nv_bfloat16 e1 = v_lo[k1 * HALF_DIM + swizzle_col(k1, n)];
            __nv_bfloat16 e8 = v_lo[k8 * HALF_DIM + swizzle_col(k8, n)];
            __nv_bfloat16 e9 = v_lo[k9 * HALF_DIM + swizzle_col(k9, n)];
            uint32_t b0 = (reinterpret_cast<uint16_t&>(e1) << 16) | reinterpret_cast<uint16_t&>(e0);
            uint32_t b1 = (reinterpret_cast<uint16_t&>(e9) << 16) | reinterpret_cast<uint16_t&>(e8);
            mma_bf16(acc_o[nt][0], acc_o[nt][1], acc_o[nt][2], acc_o[nt][3],
                     a0, a1, a2, a3, b0, b1,
                     acc_o[nt][0], acc_o[nt][1], acc_o[nt][2], acc_o[nt][3]);
        }

        // Hi half: nt=8..15, n = nt*8 + (lane/4) in [64..127] → all >= HALF_DIM
        #pragma unroll
        for (int nt = OUT_N_TILES / 2; nt < OUT_N_TILES; nt++) {
            int n = nt * MMA_N + (lane_id / 4);  // in [64..127]
            int ln = n - HALF_DIM;                // local index in [0..63]
            __nv_bfloat16 e0 = v_hi[k0 * HALF_DIM + swizzle_col(k0, ln)];
            __nv_bfloat16 e1 = v_hi[k1 * HALF_DIM + swizzle_col(k1, ln)];
            __nv_bfloat16 e8 = v_hi[k8 * HALF_DIM + swizzle_col(k8, ln)];
            __nv_bfloat16 e9 = v_hi[k9 * HALF_DIM + swizzle_col(k9, ln)];
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
    const CUtensorMap* __restrict__ tma_k,  // TMA descriptor for K
    const CUtensorMap* __restrict__ tma_v,  // TMA descriptor for V
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

    const __nv_bfloat16* q_ptr = Q + ((batch_idx * num_heads + head_idx) * seq_len) * HEAD_DIM;
    // k_ptr and v_ptr removed: TMA uses descriptor + kv_row_base coordinate
    __nv_bfloat16* o_ptr = O + ((batch_idx * num_heads + head_idx) * seq_len) * HEAD_DIM;

    int tid     = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    char* warp_smem    = smem + WARP_BASE + warp_id * PER_WARP_BYTES;
    __nv_bfloat16* q_tile = reinterpret_cast<__nv_bfloat16*>(warp_smem + W_Q_TILE_OFF);
    __nv_bfloat16* pv_buf = reinterpret_cast<__nv_bfloat16*>(warp_smem + W_PV_BUF_OFF);
    __nv_bfloat16* k_smem_p = reinterpret_cast<__nv_bfloat16*>(smem + K_SMEM_OFF);
    // mbarrier: [0]=K_ready, [1]=V_ready
    uint64_t* mbar = reinterpret_cast<uint64_t*>(smem + MBAR_OFF);
    // kv_row_base: absolute row in global K/V tensor for this (batch, kv_head)
    const int kv_row_base = (batch_idx * num_kv_heads + kv_head_idx) * ctx_len;
    const int kv_bytes    = BLOCK_N * HEAD_DIM * 2;  // bytes per K or V tile

    int warp_q_start = q_block_start + warp_id * MMA_M;
    int warp_q_end   = min(warp_q_start + MMA_M, seq_len);
    int actual_rows  = max(0, warp_q_end - warp_q_start);

    // Initialize mbarriers (thread 0 only, before any TMA)
    // Each mbarrier expects exactly 1 arrival (from TMA hardware completion)
    if (tid == 0) {
        mbar_init(&mbar[0], 1);  // K_ready
        mbar_init(&mbar[1], 1);  // V_ready
    }
    __syncthreads();  // ensure mbarrier init visible to all threads

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

        // Load K via TMA: 2 calls for lo [0..63] and hi [64..127] halves
        // Each half = 64 bf16 = 128 bytes → fits SWIZZLE_128B limit
        // Both halves issued together; mbar[0] expects 2x kv_half_bytes
        {
            const int kphase = kv_block & 1;
            const int kv_half_bytes = BLOCK_N * HALF_DIM * 2;
            if (tid == 0) {
                mbar_arrive_tx(&mbar[0], kv_half_bytes * 2);  // expect both halves
                tma_load_2d(tma_k, k_smem_p + KV_LO_OFF, &mbar[0], 0,  kv_row_base + kv_start);
                tma_load_2d(tma_k, k_smem_p + KV_HI_OFF, &mbar[0], HALF_DIM, kv_row_base + kv_start);
            }
            mbar_wait(&mbar[0], kphase);
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        }
        __syncthreads();  // sync #1: K ready

        float acc_s[N_TILES][4];
        if (actual_rows > 0) {
            compute_qk_reg(q_tile, k_smem_p, acc_s);
            softmax_update_reg(acc_s, acc_o, row_max, row_sum,
                               inv_sqrt, valid_n, kv_start, warp_q_start, actual_rows);
        }
        __syncthreads();  // sync #2: all warps done with K smem

        // Load V via TMA: 2 calls for lo [0..63] and hi [64..127] halves
        {
            const int vphase = kv_block & 1;
            const int kv_half_bytes = BLOCK_N * HALF_DIM * 2;
            if (tid == 0) {
                mbar_arrive_tx(&mbar[1], kv_half_bytes * 2);
                tma_load_2d(tma_v, k_smem_p + KV_LO_OFF, &mbar[1], 0,  kv_row_base + kv_start);
                tma_load_2d(tma_v, k_smem_p + KV_HI_OFF, &mbar[1], HALF_DIM, kv_row_base + kv_start);
            }
            mbar_wait(&mbar[1], vphase);
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
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

        // Load K with split+swizzle: vectorized global reads, swizzled smem writes
        // Each uint4 = 8 bf16 = one 16B chunk → maps to exactly one swizzle chunk
        // For chunk c in row n: physical position = swizzle_col(n, c*8) = ((c)^(n%8))*8
        {
            __nv_bfloat16* k_lo = k_smem_p + KV_LO_OFF;
            __nv_bfloat16* k_hi = k_smem_p + KV_HI_OFF;
            // HEAD_DIM/8 = 16 uint4 per row, BLOCK_N rows = 128*16 = 2048 uint4 total
            const int vecs_per_row = HEAD_DIM / 8;  // 16
            const int vec_total = BLOCK_N * vecs_per_row;  // 2048
            const uint4* src = reinterpret_cast<const uint4*>(k_ptr + kv_start * HEAD_DIM);
            for (int vi = tid; vi < vec_total; vi += BLOCK_THREADS) {
                int n = vi / vecs_per_row;
                int chunk = vi % vecs_per_row;  // which 8-element chunk (0..15)
                uint4 data = (n < valid_n) ? src[vi] : make_uint4(0, 0, 0, 0);
                // chunk < 8: lo half; chunk >= 8: hi half
                if (chunk < 8) {
                    // lo half: chunk in [0..7], physical = swizzle_col(n, chunk*8)/8 = chunk^(n%8)
                    int phys_chunk = chunk ^ (n % 8);
                    reinterpret_cast<uint4*>(k_lo + n * HALF_DIM)[phys_chunk] = data;
                } else {
                    // hi half: chunk in [8..15], local_chunk = chunk-8 in [0..7]
                    int local_chunk = chunk - 8;
                    int phys_chunk = local_chunk ^ (n % 8);
                    reinterpret_cast<uint4*>(k_hi + n * HALF_DIM)[phys_chunk] = data;
                }
            }
        }
        __syncthreads();  // sync #1

        float acc_s[N_TILES][4];
        if (warp_id == 0) {
            compute_qk_reg(q_tile, k_smem_p, acc_s);
            softmax_update_reg(acc_s, acc_o, row_max, row_sum,
                               inv_sqrt, valid_n, kv_start, ctx_len - 1, 1);
        }
        __syncthreads();  // sync #2

        // Load V into padded smem (decode)
        {
            const int vecs_per_row = HEAD_DIM / 8;
            const int vec_valid = valid_n * vecs_per_row;
            const int vec_total = BLOCK_N * vecs_per_row;
            const uint4* src_vec = reinterpret_cast<const uint4*>(v_ptr + kv_start * HEAD_DIM);
            for (int vi = tid; vi < vec_total; vi += BLOCK_THREADS) {
                const int n      = vi / vecs_per_row;
                const int vi_row = vi % vecs_per_row;
                uint4 data = (vi < vec_valid) ? src_vec[vi] : make_uint4(0, 0, 0, 0);
                // V with split+swizzle: each uint4 = one 16B chunk
                // V[k, n] stored as: lo[k*HALF_DIM + swizzle] for n<64, hi[...] for n>=64
                {
                    const int k_row   = vi / vecs_per_row;
                    const int chunk   = vi % vecs_per_row;  // 0..15
                    __nv_bfloat16* v_lo = k_smem_p + KV_LO_OFF;
                    __nv_bfloat16* v_hi = k_smem_p + KV_HI_OFF;
                    if (chunk < 8) {
                        int phys_chunk = chunk ^ (k_row % 8);
                        reinterpret_cast<uint4*>(v_lo + k_row * HALF_DIM)[phys_chunk] = data;
                    } else {
                        int local_chunk = chunk - 8;
                        int phys_chunk = local_chunk ^ (k_row % 8);
                        reinterpret_cast<uint4*>(v_hi + k_row * HALF_DIM)[phys_chunk] = data;
                    }
                }
            }
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
// ============================================================================
// TMA descriptor creation helper
// Creates a 2D TMA descriptor for a tensor stored as [num_rows, HEAD_DIM] bf16.
// The smem box is [BLOCK_N, HEAD_DIM] with padded smem stride.
// ============================================================================
static CUtensorMap make_tma_desc(
    const void* global_ptr,
    int num_rows,           // total rows in global tensor
    int smem_rows,          // BLOCK_N = 128
    int smem_col_stride     // padded smem stride in elements (K_SMEM_STRIDE or V_SMEM_STRIDE)
) {
    CUtensorMap desc;
     // Global tensor: [num_rows, HEAD_DIM] bf16, row-major
     // Split tile: load HALF_DIM=64 columns at a time
     // globalDim: fastest dim first → {HEAD_DIM, num_rows}
     // smem_box: {HALF_DIM=64, BLOCK_N=128} — 64 cols × 128 rows per tile
     // SWIZZLE_128B: 64 bf16 = 128 bytes ≤ 128B limit ✓
     uint64_t global_dim[2]    = {(uint64_t)HEAD_DIM, (uint64_t)num_rows};
     uint64_t global_stride[1] = {(uint64_t)HEAD_DIM * sizeof(__nv_bfloat16)};
     uint32_t smem_box[2]      = {(uint32_t)HALF_DIM, (uint32_t)smem_rows};  // 64 cols
     uint32_t smem_stride[2]   = {1, 1};

     CUresult res = cuTensorMapEncodeTiled(
         &desc,
         CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
         2,
         const_cast<void*>(global_ptr),
         global_dim,
         global_stride,
         smem_box,
         smem_stride,
         CU_TENSOR_MAP_INTERLEAVE_NONE,
         CU_TENSOR_MAP_SWIZZLE_128B,  // 64 bf16 = 128 bytes ≤ 128B limit ✓
         CU_TENSOR_MAP_L2_PROMOTION_NONE,
         CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    TORCH_CHECK(res == CUDA_SUCCESS,
                "cuTensorMapEncodeTiled failed with error: ", (int)res);
    return desc;
}

torch::Tensor unified_prefill(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                               int num_heads, int num_kv_heads) {
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda());
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous());
    TORCH_CHECK(Q.dtype() == torch::kBFloat16);
    TORCH_CHECK(Q.size(3) == HEAD_DIM);

    int batch = Q.size(0), seq_len = Q.size(2), ctx_len = K.size(2);
    int num_kv_heads_actual = K.size(1);
    auto O = torch::zeros_like(Q);

    // Total rows in K/V global tensor: batch * num_kv_heads * ctx_len
    int total_kv_rows = batch * num_kv_heads_actual * ctx_len;

    // Create TMA descriptors on host
    // K: global [total_kv_rows, HEAD_DIM] → smem [BLOCK_N, HEAD_DIM] with K_SMEM_STRIDE
    // V: global [total_kv_rows, HEAD_DIM] → smem [BLOCK_N, HEAD_DIM] with V_SMEM_STRIDE
    // Note: TMA descriptor encodes the global layout; smem stride is handled by
    // the kernel writing to k_smem_p[n * K_SMEM_STRIDE + col] (for K)
    // and k_smem_p[n * V_SMEM_STRIDE + col] (for V, same buffer reused)
    // The TMA descriptor uses HEAD_DIM as the smem box width (no padding in descriptor)
    // Padding is applied by the kernel's access pattern, not the TMA descriptor.
    CUtensorMap tma_k_desc = make_tma_desc(K.data_ptr(), total_kv_rows, BLOCK_N, K_SMEM_STRIDE);
    CUtensorMap tma_v_desc = make_tma_desc(V.data_ptr(), total_kv_rows, BLOCK_N, V_SMEM_STRIDE);

    // Copy descriptors to device constant memory
    CUtensorMap *d_tma_k, *d_tma_v;
    cudaMalloc(&d_tma_k, sizeof(CUtensorMap));
    cudaMalloc(&d_tma_v, sizeof(CUtensorMap));
    cudaMemcpy(d_tma_k, &tma_k_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tma_v, &tma_v_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice);

    int nq = (seq_len + BLOCK_M_PREFILL - 1) / BLOCK_M_PREFILL;
    cudaFuncSetAttribute(unified_attn_prefill_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
    unified_attn_prefill_kernel<<<dim3(nq, batch * num_heads), BLOCK_THREADS, SMEM_BYTES>>>(
        (const __nv_bfloat16*)Q.data_ptr(),
        d_tma_k, d_tma_v,
        (__nv_bfloat16*)O.data_ptr(),
        seq_len, ctx_len, num_heads, num_kv_heads);

    cudaFree(d_tma_k);
    cudaFree(d_tma_v);
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
    m.def("unified_prefill", &unified_prefill, "Unified prefill (10c: unrolled K/V loops, no branch spill)");
    m.def("unified_decode",  &unified_decode,  "Unified decode  (10c: unrolled K/V loops)");
}
