// kernel_iter2 — Remove unused per-warp smem (iteration 2)
//
// Base: kernel_11c (WGMMA PV + V^T TMA, 67 KB smem → 3 CTAs/SM)
// Change: Remove smem allocation for per-warp q_tile (4096B/warp × 4) and pv_buf
//         (768B/warp × 4). These were used by the legacy mma.sync PV path which is
//         no longer called (both prefill and decode use WGMMA PV with register-based P).
//         Smem drops from 68640B (~67KB) to 49168B (~48KB) → 4 CTAs/SM (up from 3).
//
// More CTA/SM means better HBM latency hiding via warp scheduling. The SM switches
// between CTAs while one waits for memory → higher effective throughput.
//
// Legacy functions (write_ktile_to_smem, compute_pv_reg_per_ktile) and their
// dependent constants preserved for reference but not allocated in smem.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cuda.h>

#include <cute/tensor.hpp>
#include <cute/util/debug.hpp>
#include <cutlass/device_kernel.h>
#include <cutlass/pipeline/sm90_pipeline.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/mma_sm90.h>
#include <cutlass/numeric_conversion.h>

// ── Core constants first so Int<HEAD_DIM> etc. work in using-declarations ──
constexpr int HEAD_DIM        = 128;
constexpr int NUM_WARPS       = 4;
constexpr int WARP_SIZE       = 32;
constexpr int BLOCK_THREADS   = NUM_WARPS * WARP_SIZE;
constexpr int BLOCK_M_PREFILL = NUM_WARPS * 16;  // 64
constexpr int BLOCK_N         = 128;

constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 16;

using namespace cute;
using BF16 = cutlass::bfloat16_t;
using SmemLayoutAtom = decltype(GMMA::Layout_K_SW128_Atom<BF16>{});
using SmemLayoutQ    = decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<BLOCK_M_PREFILL>, Int<HEAD_DIM>>{}));
using SmemLayoutKW   = decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<BLOCK_N>,         Int<HEAD_DIM>>{}));
using TiledMmaQK     = decltype(make_tiled_mma(
    SM90_64x128x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>{}));

using TmaK = decltype(make_tma_atom(
    SM90_TMA_LOAD{},
    make_tensor(static_cast<BF16 const*>(nullptr),
                make_shape(1, Int<HEAD_DIM>{}),
                make_stride(Int<HEAD_DIM>{}, Int<1>{})),
    SmemLayoutKW{},
    Shape<Int<BLOCK_N>, Int<HEAD_DIM>>{}));

// V^T smem layout: MN-major SW128 [HEAD_DIM=128, BLOCK_N=128]
// V^T is stored transposed relative to V: rows=HEAD_DIM, cols=BLOCK_N
// MN-major means BLOCK_N (cols) is the fast dimension for WGMMA PV
using SmemLayoutAtomVt = decltype(GMMA::Layout_MN_SW128_Atom<BF16>{});
using SmemLayoutVt     = decltype(tile_to_shape(SmemLayoutAtomVt{},
                                                Shape<Int<HEAD_DIM>, Int<BLOCK_N>>{}));

// TMA for V^T: global tensor viewed as [HEAD_DIM, total_kv_rows] with stride (1, HEAD_DIM)
// dim0 (HEAD_DIM) has stride 1 -> satisfies TMA gmem_prob_stride[0]==1 for MN-major smem
using TmaVt = decltype(make_tma_atom(
    SM90_TMA_LOAD{},
    make_tensor(static_cast<BF16 const*>(nullptr),
                make_shape(Int<HEAD_DIM>{}, 1),
                make_stride(Int<1>{}, Int<HEAD_DIM>{})),
    SmemLayoutVt{},
    Shape<Int<HEAD_DIM>, Int<BLOCK_N>>{}));

// WGMMA PV: P[64,128] (K-major, in registers RS) x Vt[128,128] (MN-major, in smem SS)
// -> O[64,128] accumulator
// RS = P stays in registers (no smem write needed)
// MN = Vt is MN-major (BLOCK_N is fast dimension)
using TiledMmaPV = decltype(make_tiled_mma(
    SM90_64x128x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::MN>{}));

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
// Per-warp smem (Q_TILE + PV_BUF) is unused by WGMMA paths — not allocated
// Legacy constants (WARP_BASE, Q_TILE_BYTES, PV_BUF_BYTES, PER_WARP_BYTES)
// preserved below for reference-only legacy functions.
constexpr int SMEM_BYTES     = MBAR_OFF + MBAR_BYTES;                   // 32784 B (no warp smem)

constexpr int W_Q_TILE_OFF   = 0;
constexpr int W_PV_BUF_OFF   = W_Q_TILE_OFF + Q_TILE_BYTES;

// Extra smem for WGMMA: Q in K-major SW128 layout
constexpr int Q_WGMMA_BYTES  = 64  * 128 * 2;   // 16384 B
constexpr int Q_WGMMA_OFF    = SMEM_BYTES;                        // 32784
constexpr int SMEM_BYTES_10H = Q_WGMMA_OFF + Q_WGMMA_BYTES;      // 49168 B (~48 KB)

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
// QK^T — cute::gemm WGMMA SS (replaces per-warp mma.sync compute_qk_reg)
// q_smem: [BLOCK_M=64, HEAD_DIM=128] K-major SW128
// k_wgmma: [BLOCK_N=128, HEAD_DIM=128] K-major SW128
// All 128 threads must call unconditionally (.aligned requirement)
// Output acc_s[N_TILES][4] in same per-thread layout as mma.sync (warp 0 rows 0-15)
// ============================================================================
__device__ __forceinline__ void compute_qk_cute(
    const BF16* __restrict__ q_smem,
    const BF16* __restrict__ k_wgmma,
    float acc_s[N_TILES][4]
) {
    TiledMmaQK tiled_mma;
    ThrMMA thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
    Tensor sQ = make_tensor(make_smem_ptr(q_smem),  SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(k_wgmma), SmemLayoutKW{});
    Tensor tCsQ = thr_mma.partition_A(sQ);
    Tensor tCsK = thr_mma.partition_B(sK);
    auto tCrQ = thr_mma.make_fragment_A(tCsQ);
    auto tCrK = thr_mma.make_fragment_B(tCsK);
    auto tCrC = partition_fragment_C(tiled_mma, Shape<Int<64>, Int<128>>{});
    clear(tCrC);
    warpgroup_arrive();
    gemm(tiled_mma, tCrQ, tCrK, tCrC);
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    #pragma unroll
    for (int nt = 0; nt < N_TILES; nt++) {
        acc_s[nt][0] = tCrC(nt*4+0, 0, 0);
        acc_s[nt][1] = tCrC(nt*4+1, 0, 0);
        acc_s[nt][2] = tCrC(nt*4+2, 0, 0);
        acc_s[nt][3] = tCrC(nt*4+3, 0, 0);
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
// write_ktile_to_smem + compute_pv_reg_per_ktile — legacy mma.sync PV helpers
// (kept for reference; decode now uses compute_pv_cute like prefill)
// ============================================================================
__device__ __forceinline__ void write_ktile_to_smem(
    float acc_s[N_TILES][4],
    int kt,
    __nv_bfloat16* __restrict__ pv_buf
) {
    int lane_id = threadIdx.x % WARP_SIZE;
    int group_id = lane_id >> 2, tid_in_group = lane_id & 3;
    int row0 = group_id, row1 = group_id + 8;
    int col0 = tid_in_group * 2, col1 = col0 + 1;
    int nt0 = kt * 2, nt1 = kt * 2 + 1;
    pv_buf[row0 * PV_BUF_COLS + col0]     = __float2bfloat16(acc_s[nt0][0]);
    pv_buf[row0 * PV_BUF_COLS + col1]     = __float2bfloat16(acc_s[nt0][1]);
    pv_buf[row1 * PV_BUF_COLS + col0]     = __float2bfloat16(acc_s[nt0][2]);
    pv_buf[row1 * PV_BUF_COLS + col1]     = __float2bfloat16(acc_s[nt0][3]);
    pv_buf[row0 * PV_BUF_COLS + 8 + col0] = __float2bfloat16(acc_s[nt1][0]);
    pv_buf[row0 * PV_BUF_COLS + 8 + col1] = __float2bfloat16(acc_s[nt1][1]);
    pv_buf[row1 * PV_BUF_COLS + 8 + col0] = __float2bfloat16(acc_s[nt1][2]);
    pv_buf[row1 * PV_BUF_COLS + 8 + col1] = __float2bfloat16(acc_s[nt1][3]);
}

__device__ __forceinline__ void compute_pv_reg_per_ktile(
    float acc_s[N_TILES][4],
    __nv_bfloat16* __restrict__ pv_buf,
    const __nv_bfloat16* __restrict__ v_smem,
    float acc_o[OUT_N_TILES][4]
) {
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
        int k0 = kt * MMA_K + k_base, k1 = k0+1, k8 = k0+8, k9 = k8+1;
        #pragma unroll
        for (int nt = 0; nt < OUT_N_TILES / 2; nt++) {
            int n = nt * MMA_N + (lane_id / 4);
            __nv_bfloat16 e0 = v_lo[k0*HALF_DIM+swizzle_col(k0,n)], e1 = v_lo[k1*HALF_DIM+swizzle_col(k1,n)];
            __nv_bfloat16 e8 = v_lo[k8*HALF_DIM+swizzle_col(k8,n)], e9 = v_lo[k9*HALF_DIM+swizzle_col(k9,n)];
            uint32_t b0 = (reinterpret_cast<uint16_t&>(e1)<<16)|reinterpret_cast<uint16_t&>(e0);
            uint32_t b1 = (reinterpret_cast<uint16_t&>(e9)<<16)|reinterpret_cast<uint16_t&>(e8);
            mma_bf16(acc_o[nt][0],acc_o[nt][1],acc_o[nt][2],acc_o[nt][3],a0,a1,a2,a3,b0,b1,
                     acc_o[nt][0],acc_o[nt][1],acc_o[nt][2],acc_o[nt][3]);
        }
        #pragma unroll
        for (int nt = OUT_N_TILES/2; nt < OUT_N_TILES; nt++) {
            int n = nt * MMA_N + (lane_id / 4), ln = n - HALF_DIM;
            __nv_bfloat16 e0 = v_hi[k0*HALF_DIM+swizzle_col(k0,ln)], e1 = v_hi[k1*HALF_DIM+swizzle_col(k1,ln)];
            __nv_bfloat16 e8 = v_hi[k8*HALF_DIM+swizzle_col(k8,ln)], e9 = v_hi[k9*HALF_DIM+swizzle_col(k9,ln)];
            uint32_t b0 = (reinterpret_cast<uint16_t&>(e1)<<16)|reinterpret_cast<uint16_t&>(e0);
            uint32_t b1 = (reinterpret_cast<uint16_t&>(e9)<<16)|reinterpret_cast<uint16_t&>(e8);
            mma_bf16(acc_o[nt][0],acc_o[nt][1],acc_o[nt][2],acc_o[nt][3],a0,a1,a2,a3,b0,b1,
                     acc_o[nt][0],acc_o[nt][1],acc_o[nt][2],acc_o[nt][3]);
        }
    }
}

// ============================================================================
// convert_layout_acc_Aregs — copied from FA3 utils.h
//
// Transforms the QK C-accumulator layout into the RS A-register layout for PV.
//
// Why needed: cute::gemm for RS WGMMA requires A to have shape (vals, MMA_M, K_tiles)
// where K_tiles = HEAD_DIM/16 = 8. The QK C accumulator has shape ((2,2,C<16>), 1, 1)
// which has size<2>=1. The RS A layout needs size<2>=8 (K_tiles).
//
// What it does for bf16 SM90:
//   Input:  ((2, 2, C<16>), 1, 1)  with strides ((1, 2, 4), 0, 0)  [64 float32]
//   Output: ((2, 2, 2), 2, C<8>)   with strides ((1, 2, 4), 8, 16) [64 bf16]
//
// The transformation: split mode <0,2> (size C<16>) into (2, C<8>) via logical_divide.
// The "2" goes into mode <0,2> and "C<8>" becomes the new mode 2 (K_tiles).
// This matches the RS WGMMA A register layout exactly.
// ============================================================================
template<typename MMA_Traits, typename Layout0>
__device__ __forceinline__ auto convert_layout_acc_Aregs(Layout0 acc_layout) {
    // SM90 path: rank<0>(acc_layout) == 3, size<0,0>==2, size<0,1>==2
    // bf16 path: sizeof(ValTypeA) == 2
    auto l = logical_divide(get<0, 2>(acc_layout), Tile<_2>{});  // splits C<16> → (2, C<8>)
    // Reassemble: mode0 = (size<0,0>, size<0,1>, first_half_of_split)
    //             mode1 = size<1> (MMA_M)
    //             mode2 = coalesced second_half_of_split + size<2> (K_tiles)
    return make_layout(
        make_layout(get<0, 0>(acc_layout), get<0, 1>(acc_layout), get<0, 0>(l)),
        get<1>(acc_layout),
        coalesce(make_layout(get<0, 1>(l), get<2>(acc_layout)))
    );
}

// ============================================================================
// WGMMA PV — FA3-style RS WGMMA with P in registers and V^T in MN-major SW128 smem
//
// Exactly mirrors FA3's mma() function (mainloop_fwd_sm90_tma_gmma_ws.hpp lines 1157-1182):
//
//   Tensor tOrP_acc = make_tensor(tSrS.data(), convert_layout_acc_Aregs<TiledMmaPV>(tSrS.layout()));
//   Tensor tOrP = make_tensor_like<Element>(tOrP_acc);
//   convert_type_out(tOrP_acc, tOrP);   // float32 → bf16, same register layout
//   flash::gemm<false, 0>(tiled_mma_pv, tOrP, tOrV(...), tOrO);
//
// All 128 threads must call unconditionally (.aligned requirement for WGMMA).
//
// acc_s[N_TILES][4]: QK output (float32), used as P after softmax
// vt_smem: V^T in MN-major SW128 [HEAD_DIM=128, BLOCK_N=128]
// acc_o[OUT_N_TILES][4]: O accumulator (loaded before gemm — online softmax, don't clear)
// ============================================================================
__device__ __forceinline__ void compute_pv_cute(
    float acc_s[N_TILES][4],
    const BF16* __restrict__ vt_smem,
    float acc_o[OUT_N_TILES][4]
) {
    TiledMmaPV tiled_mma;
    ThrMMA thr_mma = tiled_mma.get_thread_slice(threadIdx.x);

    // Step 1: Build V^T smem tensor and its B fragment (smem descriptor)
    // partition_B + make_fragment_B gives the GMMA descriptor iterator for V^T
    Tensor sVt  = make_tensor(make_smem_ptr(vt_smem), SmemLayoutVt{});
    Tensor tCsVt = thr_mma.partition_B(sVt);
    auto   tCrVt = thr_mma.make_fragment_B(tCsVt);

    // Step 2: Build O accumulator (C fragment), load existing acc_o
    // partition_fragment_C gives shape ((2,2,C<16>), 1, 1) = 64 float32 per thread
    auto tCrC = partition_fragment_C(tiled_mma, Shape<Int<BLOCK_M_PREFILL>, Int<HEAD_DIM>>{});
    #pragma unroll
    for (int nt = 0; nt < OUT_N_TILES; nt++) {
        tCrC(nt*4+0, 0, 0) = acc_o[nt][0];
        tCrC(nt*4+1, 0, 0) = acc_o[nt][1];
        tCrC(nt*4+2, 0, 0) = acc_o[nt][2];
        tCrC(nt*4+3, 0, 0) = acc_o[nt][3];
    }

    // Step 3: Build P register tensor (A fragment for RS WGMMA)
    // FA3: tOrP_acc = make_tensor(tSrS.data(), convert_layout_acc_Aregs<TiledMmaPV>(tSrS.layout()))
    //      tOrP     = make_tensor_like<BF16>(tOrP_acc)
    //      convert_type_out(tOrP_acc, tOrP)
    //
    // convert_layout_acc_Aregs transforms ((2,2,C<16>),1,1) → ((2,2,2),2,C<8>)
    // This gives tOrP shape ((2,2,2), 2, C<8>) = 64 bf16 with K_tiles=8 in mode 2
    // which is exactly what RS WGMMA expects for A.
    //
    // We build tSrS (float32) from acc_s using the same layout as tCrC,
    // then apply convert_layout_acc_Aregs to get the RS A layout.
    auto tSrS_layout = tCrC.layout();  // ((2,2,C<16>), 1, 1)
    auto tOrP_layout = convert_layout_acc_Aregs<TiledMmaPV>(tSrS_layout);  // ((2,2,2), 2, C<8>)

    // Fill float32 source with acc_s values (same flat indexing as tCrC)
    auto tSrS = make_tensor<float>(tSrS_layout);
    #pragma unroll
    for (int nt = 0; nt < N_TILES; nt++) {
        tSrS(nt*4+0, 0, 0) = acc_s[nt][0];
        tSrS(nt*4+1, 0, 0) = acc_s[nt][1];
        tSrS(nt*4+2, 0, 0) = acc_s[nt][2];
        tSrS(nt*4+3, 0, 0) = acc_s[nt][3];
    }

    // Reinterpret as RS A layout and convert float32 → bf16
    // make_tensor(data_ptr, new_layout) reuses the same register storage with new shape
    auto tOrP_acc = make_tensor(tSrS.data(), tOrP_layout);  // float32, shape ((2,2,2),2,C<8>)
    auto tOrP     = make_tensor_like<BF16>(tOrP_acc);        // bf16,    shape ((2,2,2),2,C<8>)
    // Convert float32 → bf16 using vectorized NumericArrayConverter (FA3's convert_type_out)
    {
        using From_t = float;
        using To_t   = BF16;
        constexpr int N = decltype(size(tOrP_acc))::value;  // 64
        cutlass::NumericArrayConverter<To_t, From_t, N> cvt;
        auto src = recast<cutlass::Array<From_t, N> const>(tOrP_acc);
        auto dst = recast<cutlass::Array<To_t,   N>      >(tOrP);
        dst[0] = cvt(src[0]);
    }

    // Step 4: WGMMA PV — FA3's flash::gemm<zero_init=false, wg_wait=0>
    // warpgroup_fence_operand on tOrP (RS) and tCrC before/after
    warpgroup_fence_operand(tOrP);
    warpgroup_fence_operand(tCrC);
    warpgroup_arrive();
    // Iterate over K_tiles (size<2>(tOrP) = C<8> = 8 tiles of 16)
    // tiled_mma.accumulate_ starts as One (we loaded acc_o into tCrC above)
    #pragma unroll
    for (int k = 0; k < size<2>(tOrP); ++k) {
        cute::gemm(tiled_mma, tOrP(_,_,k), tCrVt(_,_,k), tCrC);
        tiled_mma.accumulate_ = GMMA::ScaleOut::One;
    }
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    warpgroup_fence_operand(tCrC);
    warpgroup_fence_operand(tOrP);

    // Step 5: Extract results back to acc_o
    #pragma unroll
    for (int nt = 0; nt < OUT_N_TILES; nt++) {
        acc_o[nt][0] = tCrC(nt*4+0, 0, 0);
        acc_o[nt][1] = tCrC(nt*4+1, 0, 0);
        acc_o[nt][2] = tCrC(nt*4+2, 0, 0);
        acc_o[nt][3] = tCrC(nt*4+3, 0, 0);
    }
}


// ============================================================================
// Prefill kernel
// ============================================================================
__global__
__launch_bounds__(BLOCK_THREADS)
void unified_attn_prefill_kernel(
    const __nv_bfloat16* __restrict__ Q,
    CUTLASS_GRID_CONSTANT TmaK  const tma_k,   // cute TMA for K → K-major SW128
    CUTLASS_GRID_CONSTANT TmaVt const tma_vt,  // cute TMA for V^T → MN-major SW128
    __nv_bfloat16* __restrict__ O,
    int seq_len, int ctx_len, int num_heads, int num_kv_heads,
    int total_kv_rows
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
    __nv_bfloat16* o_ptr = O + ((batch_idx * num_heads + head_idx) * seq_len) * HEAD_DIM;

    int tid     = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    BF16*          k_wgmma = reinterpret_cast<BF16*>(smem + K_SMEM_OFF);  // K-major SW128; reused as Vt after QK
    BF16*          vt_smem = reinterpret_cast<BF16*>(smem + K_SMEM_OFF);  // MN-major SW128 Vt (same slot as K)
    BF16*          q_smem  = reinterpret_cast<BF16*>(smem + Q_WGMMA_OFF);
    uint64_t* mbar = reinterpret_cast<uint64_t*>(smem + MBAR_OFF);
    const int kv_row_base = (batch_idx * num_kv_heads + kv_head_idx) * ctx_len;

    int warp_q_start = q_block_start + warp_id * MMA_M;
    int warp_q_end   = min(warp_q_start + MMA_M, seq_len);
    int actual_rows  = max(0, warp_q_end - warp_q_start);

    // Producer (warp 0) initializes mbarriers
    if (warp_id == 0 && lane_id == 0) {
        mbar_init(&mbar[0], 1);
        mbar_init(&mbar[1], 1);
    }
    __syncthreads();

    // All threads load Q into K-major SW128 smem (cooperative, before loop)
    {
        Tensor sQ = make_tensor(make_smem_ptr(q_smem), SmemLayoutQ{});
        const int total = BLOCK_M_PREFILL * HEAD_DIM;
        for (int i = tid; i < total; i += BLOCK_THREADS) {
            int m = i / HEAD_DIM, k = i % HEAD_DIM;
            int global_row = q_block_start + m;
            sQ(m, k) = (global_row < seq_len) ? BF16(q_ptr[global_row * HEAD_DIM + k]) : BF16(0.0f);
        }
    }
    __syncthreads();

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

        // ── PRODUCER role (warp 0 lane 0): issue TMA K load ──────────────────
        // All other threads are idle here — in step 2 they will overlap with compute
        if (warp_id == 0 && lane_id == 0) {
            const int k_bytes = BLOCK_N * HEAD_DIM * sizeof(BF16);
            mbar_arrive_tx(&mbar[0], k_bytes);
            Tensor mK = tma_k.get_tma_tensor(make_shape(total_kv_rows, Int<HEAD_DIM>{}));
            Tensor sK = make_tensor(make_smem_ptr(k_wgmma), SmemLayoutKW{});
            Tensor mK_off = domain_offset(make_coord(kv_row_base + kv_start, 0), mK);
            Tensor gK = local_tile(mK_off, Shape<Int<BLOCK_N>, Int<HEAD_DIM>>{},
                                   make_coord(0, 0));
            auto [tKgK, tKsK] = tma_partition(tma_k, Int<0>{}, Layout<_1>{},
                                               group_modes<0,2>(sK),
                                               group_modes<0,2>(gK));
            copy(tma_k.with(mbar[0]), tKgK, tKsK);
        }
        // All warps wait for K
        {
            const int kphase = kv_block & 1;
            mbar_wait(&mbar[0], kphase);
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        }
        __syncthreads();

        // ── ALL WARPS: WGMMA QK (.aligned requires all 128 threads) ──────────
        float acc_s[N_TILES][4];
        compute_qk_cute(q_smem, k_wgmma, acc_s);
        __syncthreads();  // barrier: K smem consumed, safe for Vt TMA to overwrite

        // ── Issue V^T TMA early so it overlaps with softmax ──────────────────
        // V^T reuses K smem slot (K is done after QK + sync). TMA runs async.
        if (warp_id == 0 && lane_id == 0) {
            const int vt_bytes = HEAD_DIM * BLOCK_N * sizeof(BF16);
            mbar_arrive_tx(&mbar[1], vt_bytes);
            Tensor mVt = tma_vt.get_tma_tensor(make_shape(Int<HEAD_DIM>{}, total_kv_rows));
            Tensor sVt = make_tensor(make_smem_ptr(vt_smem), SmemLayoutVt{});
            Tensor mVt_off = domain_offset(make_coord(0, kv_row_base + kv_start), mVt);
            Tensor gVt = local_tile(mVt_off, Shape<Int<HEAD_DIM>, Int<BLOCK_N>>{}, make_coord(0, 0));
            auto [tVgVt, tVsVt] = tma_partition(tma_vt, Int<0>{}, Layout<_1>{},
                                                 group_modes<0,2>(sVt),
                                                 group_modes<0,2>(gVt));
            copy(tma_vt.with(mbar[1]), tVgVt, tVsVt);
        }
        // ALL warps run softmax for their own rows (overlapped with V^T TMA)
        if (actual_rows > 0) {
            softmax_update_reg(acc_s, acc_o, row_max, row_sum,
                               inv_sqrt, valid_n, kv_start, warp_q_start, actual_rows);
        }
        // All warps wait for V^T (may still be in flight after softmax)
        {
            const int vphase = kv_block & 1;
            mbar_wait(&mbar[1], vphase);
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        }
        __syncthreads();

        // ── ALL WARPS: WGMMA PV (.aligned requires all 128 threads) ──────────
        if (actual_rows > 0)
            compute_pv_cute(acc_s, vt_smem, acc_o);
        __syncthreads();
    }

    // ── ALL WARPS: write output for their own rows ────────────────────────────
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
    CUTLASS_GRID_CONSTANT TmaK const tma_k,
    CUTLASS_GRID_CONSTANT TmaVt const tma_vt,
    __nv_bfloat16* __restrict__ O,
    int ctx_len, int num_heads, int num_kv_heads,
    int total_kv_rows
) {
    extern __shared__ char smem[];

    int head_batch_idx = blockIdx.y;
    int batch_idx      = head_batch_idx / num_heads;
    int head_idx       = head_batch_idx % num_heads;
    int kv_head_idx    = head_idx / (num_heads / num_kv_heads);

    const __nv_bfloat16* q_ptr = Q + (batch_idx * num_heads    + head_idx)    * HEAD_DIM;
    __nv_bfloat16*        o_ptr = O + (batch_idx * num_heads    + head_idx)    * HEAD_DIM;

    int tid     = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    BF16*          q_smem   = reinterpret_cast<BF16*>(smem + Q_WGMMA_OFF);
    BF16*          k_wgmma  = reinterpret_cast<BF16*>(smem + K_SMEM_OFF);
    BF16*          vt_smem  = reinterpret_cast<BF16*>(smem + K_SMEM_OFF);
    uint64_t*      mbar     = reinterpret_cast<uint64_t*>(smem + MBAR_OFF);
    const int kv_row_base   = (batch_idx * num_kv_heads + kv_head_idx) * ctx_len;

    if (tid == 0) {
        mbar_init(&mbar[0], 1);
        mbar_init(&mbar[1], 1);
    }

    // Load Q into K-major SW128 smem for WGMMA (all 128 threads)
    {
        Tensor sQ = make_tensor(make_smem_ptr(q_smem), SmemLayoutQ{});
        const int total = BLOCK_M_PREFILL * HEAD_DIM;
        for (int i = tid; i < total; i += BLOCK_THREADS) {
            int m = i / HEAD_DIM, k = i % HEAD_DIM;
            sQ(m, k) = (m == 0) ? BF16(q_ptr[k]) : BF16(0.0f);
        }
    }
    __syncthreads();

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
        const int vphase = kv_block & 1;

        // Load K via cute TMA directly into K-major SW128 smem (same as prefill)
        {
            const int kphase = kv_block & 1;
            const int k_bytes = BLOCK_N * HEAD_DIM * sizeof(BF16);
            if (tid == 0) {
                mbar_arrive_tx(&mbar[0], k_bytes);
                Tensor mK = tma_k.get_tma_tensor(make_shape(total_kv_rows, Int<HEAD_DIM>{}));
                Tensor sK = make_tensor(make_smem_ptr(k_wgmma), SmemLayoutKW{});
                Tensor mK_off = domain_offset(make_coord(kv_row_base + kv_start, 0), mK);
                Tensor gK = local_tile(mK_off, Shape<Int<BLOCK_N>, Int<HEAD_DIM>>{},
                                       make_coord(0, 0));
                auto [tKgK, tKsK] = tma_partition(tma_k, Int<0>{}, Layout<_1>{},
                                                   group_modes<0,2>(sK),
                                                   group_modes<0,2>(gK));
                copy(tma_k.with(mbar[0]), tKgK, tKsK);
            }
            mbar_wait(&mbar[0], kphase);
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        }
        __syncthreads();  // sync #1

        // WGMMA QK — all 128 threads unconditionally
        float acc_s[N_TILES][4];
        compute_qk_cute(q_smem, k_wgmma, acc_s);
        __syncthreads();  // sync #2

        // Issue V^T TMA after QK finishes consuming K smem; overlap the load with softmax.
        {
            const int vt_bytes = HEAD_DIM * BLOCK_N * sizeof(BF16);
            if (tid == 0) {
                mbar_arrive_tx(&mbar[1], vt_bytes);
                Tensor mVt = tma_vt.get_tma_tensor(make_shape(Int<HEAD_DIM>{}, total_kv_rows));
                Tensor sVt = make_tensor(make_smem_ptr(vt_smem), SmemLayoutVt{});
                Tensor mVt_off = domain_offset(make_coord(0, kv_row_base + kv_start), mVt);
                Tensor gVt = local_tile(mVt_off, Shape<Int<HEAD_DIM>, Int<BLOCK_N>>{},
                                        make_coord(0, 0));
                auto [tVgVt, tVsVt] = tma_partition(tma_vt, Int<0>{}, Layout<_1>{},
                                                    group_modes<0,2>(sVt),
                                                    group_modes<0,2>(gVt));
                copy(tma_vt.with(mbar[1]), tVgVt, tVsVt);
            }
        }
        if (warp_id == 0) {
            softmax_update_reg(acc_s, acc_o, row_max, row_sum,
                               inv_sqrt, valid_n, kv_start, ctx_len - 1, 1);
        }
        {
            mbar_wait(&mbar[1], vphase);
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        }
        __syncthreads();  // sync #3

        compute_pv_cute(acc_s, vt_smem, acc_o);
        __syncthreads();  // sync #4
    }

    // Broadcast row_sum[0] from warp 0 to all warps via smem so all warps can
    // normalize their portion of the WGMMA accumulator for row 0.
    // We reuse the mbar slot (already done, safe to overwrite) as a float scratch.
    {
        float* scratch = reinterpret_cast<float*>(smem + MBAR_OFF);
        if (warp_id == 0 && lane_id == 0)
            scratch[0] = row_sum[0];
        __syncthreads();
        float shared_row_sum = scratch[0];
        __syncthreads();

        // All 4 warps write their portion of row 0 of the WGMMA output.
        // In SM90_64x128x16 RS, the 64-row tile is split across 4 warps:
        //   warp w owns rows [w*16 .. w*16+15] of the tile.
        // Decode has only 1 query row (row 0 of the tile = warp 0, rows 0-15).
        // Within warp 0: group_id = lane_id>>2, row0 = group_id, row1 = group_id+8.
        // Row 0 is owned by threads with group_id == 0 (lane_id 0-3) in warp 0.
        // Warps 1-3 own rows 16-63 — none of which are valid for decode.
        int warp_row_base = warp_id * MMA_M;  // 0, 16, 32, 48
        int group_id      = lane_id >> 2;
        int tid_in_group  = lane_id & 3;
        int row0 = warp_row_base + group_id;        // absolute row in 64-row tile
        int row1 = warp_row_base + group_id + 8;
        int col0 = tid_in_group * 2, col1 = col0 + 1;
        float inv0 = (shared_row_sum > 0.0f) ? (1.0f / shared_row_sum) : 0.0f;

        #pragma unroll
        for (int nt = 0; nt < OUT_N_TILES; nt++) {
            int n_base = nt * MMA_N;
            // Only write row 0 (the single decode query row)
            if (row0 == 0) {
                o_ptr[n_base + col0] = __float2bfloat16(acc_o[nt][0] * inv0);
                o_ptr[n_base + col1] = __float2bfloat16(acc_o[nt][1] * inv0);
            }
            // row1 is always >= 8, never 0 — skip
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
    int total_kv_rows = batch * num_kv_heads_actual * ctx_len;

    // cute TMA for K: [BLOCK_N=128, HEAD_DIM=128] → K-major SW128 smem
    auto dK = make_stride(Int<HEAD_DIM>{}, Int<1>{});
    Tensor mK = make_tensor(reinterpret_cast<BF16 const*>(K.data_ptr()),
                            make_shape(total_kv_rows, Int<HEAD_DIM>{}), dK);
    auto tma_k = make_tma_atom(SM90_TMA_LOAD{}, mK, SmemLayoutKW{},
                               Shape<Int<BLOCK_N>, Int<HEAD_DIM>>{});

    // cute TMA for V^T: global [HEAD_DIM, total_kv_rows] stride (1, HEAD_DIM)
    // V^T has dim0=HEAD_DIM with stride 1 → satisfies TMA gmem_prob_stride[0]==1
    auto dVt = make_stride(Int<1>{}, Int<HEAD_DIM>{});
    Tensor mVt = make_tensor(reinterpret_cast<BF16 const*>(V.data_ptr()),
                             make_shape(Int<HEAD_DIM>{}, total_kv_rows), dVt);
    auto tma_vt = make_tma_atom(SM90_TMA_LOAD{}, mVt, SmemLayoutVt{},
                                Shape<Int<HEAD_DIM>, Int<BLOCK_N>>{});

    int nq = (seq_len + BLOCK_M_PREFILL - 1) / BLOCK_M_PREFILL;
    cudaFuncSetAttribute(unified_attn_prefill_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES_10H);
    unified_attn_prefill_kernel<<<dim3(nq, batch * num_heads), BLOCK_THREADS, SMEM_BYTES_10H>>>(
        (const __nv_bfloat16*)Q.data_ptr(),
        tma_k, tma_vt,
        (__nv_bfloat16*)O.data_ptr(),
        seq_len, ctx_len, num_heads, num_kv_heads, total_kv_rows);

    return O;
}

torch::Tensor unified_decode(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                              int num_heads, int num_kv_heads) {
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda());
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous());
    TORCH_CHECK(Q.size(2) == 1);
    TORCH_CHECK(Q.size(3) == HEAD_DIM);
    int batch = Q.size(0), ctx_len = K.size(2);
    int num_kv_heads_actual = K.size(1);
    auto O = torch::zeros_like(Q);
    int total_kv_rows = batch * num_kv_heads_actual * ctx_len;

    auto dK = make_stride(Int<HEAD_DIM>{}, Int<1>{});
    Tensor mK = make_tensor(reinterpret_cast<BF16 const*>(K.data_ptr()),
                            make_shape(total_kv_rows, Int<HEAD_DIM>{}), dK);
    auto tma_k = make_tma_atom(SM90_TMA_LOAD{}, mK, SmemLayoutKW{},
                               Shape<Int<BLOCK_N>, Int<HEAD_DIM>>{});

    // cute TMA for V^T: global [HEAD_DIM, total_kv_rows] stride (1, HEAD_DIM)
    auto dVt = make_stride(Int<1>{}, Int<HEAD_DIM>{});
    Tensor mVt = make_tensor(reinterpret_cast<BF16 const*>(V.data_ptr()),
                             make_shape(Int<HEAD_DIM>{}, total_kv_rows), dVt);
    auto tma_vt = make_tma_atom(SM90_TMA_LOAD{}, mVt, SmemLayoutVt{},
                                Shape<Int<HEAD_DIM>, Int<BLOCK_N>>{});

    cudaFuncSetAttribute(unified_attn_decode_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES_10H);
    unified_attn_decode_kernel<<<dim3(1, batch * num_heads), BLOCK_THREADS, SMEM_BYTES_10H>>>(
        (const __nv_bfloat16*)Q.data_ptr(),
        tma_k,
        tma_vt,
        (__nv_bfloat16*)O.data_ptr(),
        ctx_len, num_heads, num_kv_heads, total_kv_rows);
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("unified_prefill", &unified_prefill, "Unified prefill (10h: cute TMA K → K-major SW128, WGMMA QK)");
    m.def("unified_decode",  &unified_decode,  "Unified decode  (10h: WGMMA QK, vectorized K/V)");
}
