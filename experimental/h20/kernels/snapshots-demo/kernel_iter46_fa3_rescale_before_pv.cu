// unified_kernel_11c.cu — kernel_11c: WGMMA PV + V^T TMA for decode
//
// Base: unified_kernel_11b.cu (V TMA overlapped with softmax, 0.7× FlashInfer at ctx=1024)
// Change: decode V load switched from split-half row-major to V^T MN-major SW128 (cute TMA),
//         decode PV switched from mma.sync (warp 0 only) to WGMMA RS (all 128 threads).
//         Reuses compute_pv_cute and tma_vt already present for prefill.
//
// What changes vs 11b:
//   - unified_attn_decode_kernel: tma_v (CUtensorMap) → tma_vt (TmaVt, cute TMA)
//   - V smem layout: split-half KV_LO/HI → V^T MN-major SW128 (same 32KB slot as K)
//   - PV compute: compute_pv_reg_per_ktile (mma.sync, warp 0) → compute_pv_cute (WGMMA, all threads)
//   - pv_buf smem: removed from decode (no longer needed)
//   - Output write: adapted for WGMMA accumulator layout (single row, m_tile=0)
//   - unified_decode host: creates tma_vt instead of tma_v
//
// What is IDENTICAL to 11b:
//   - unified_attn_prefill_kernel: zero changes
//   - compute_pv_cute function: unchanged
//   - SmemLayoutVt, TiledMmaPV: unchanged
//   - Smem total size: unchanged (K and V^T share same 32KB slot)
//   - Bitwise consistency: maintained (both paths use compute_pv_cute with same V^T layout)
//
// Expected: ~1.5-2.0× FlashInfer at ctx=1024 (vs 0.7× in 11b)
// WGMMA PV uses all 128 threads vs warp-0-only mma.sync → 4× more parallelism for PV.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <optional>
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
// FUNCTION: SmemLayoutVt
// FA3 tiles MN-major V with its column mode first.  Without this Step order,
// CuTe decomposes one logical 32-KiB V^T tile into 32 one-KiB TMA copies.
using SmemLayoutVt     = decltype(tile_to_shape(SmemLayoutAtomVt{},
                                                Shape<Int<HEAD_DIM>, Int<BLOCK_N>>{},
                                                Step<_2, _1>{}));

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
constexpr int WARP_BASE      = MBAR_OFF + MBAR_BYTES;                   // 32784
constexpr int SMEM_BYTES     = WARP_BASE + NUM_WARPS * PER_WARP_BYTES;  // 52256 B (unchanged)

constexpr int W_Q_TILE_OFF   = 0;
constexpr int W_PV_BUF_OFF   = W_Q_TILE_OFF + Q_TILE_BYTES;

// Extra smem for WGMMA: Q in K-major SW128 layout
// K smem is now K-major SW128 [BLOCK_N=128, HEAD_DIM=128] = 32KB (same size as 10c split-half)
// placed at K_SMEM_OFF=0, reusing the same slot — no extra K buffer needed
constexpr int Q_WGMMA_BYTES  = 64  * 128 * 2;   // 16384 B  [BLOCK_M=64,  HEAD_DIM=128]
constexpr int Q_WGMMA_OFF    = SMEM_BYTES;                        // 52256
constexpr int SMEM_BYTES_10H = Q_WGMMA_OFF + Q_WGMMA_BYTES;      // 68640 B (~67 KB)

// Decode-only two-stage K/V^T pipeline.  K and V use disjoint ping-pong
// buffers so block i+1 can be loaded while block i executes unchanged math.
constexpr int DECODE_K_STAGE0_OFF = 0;
constexpr int DECODE_K_STAGE1_OFF = KV_SMEM_BYTES;
constexpr int DECODE_V_STAGE0_OFF = 2 * KV_SMEM_BYTES;
constexpr int DECODE_V_STAGE1_OFF = 3 * KV_SMEM_BYTES;
constexpr int DECODE_MBAR_OFF      = 4 * KV_SMEM_BYTES;
constexpr int DECODE_MBAR_BYTES    = 4 * sizeof(uint64_t);
constexpr int DECODE_Q_OFF         = DECODE_MBAR_OFF + DECODE_MBAR_BYTES;
constexpr int SMEM_BYTES_DECODE_PIPE = DECODE_Q_OFF + Q_WGMMA_BYTES;

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
// FUNCTION: mbar_wait
__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t phase) {
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    auto const* barrier = reinterpret_cast<Barrier::ValueType const*>(b);
    // Match FA3/FlashInfer's pipeline wait: keep the ready fast path, then use
    // CUTLASS's timeout-qualified wait so a delayed TMA does not busy-poll.
    if (!Barrier::try_wait(barrier, phase)) {
        Barrier::wait(barrier, phase);
    }
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

// FUNCTION: fast_exp2_ftz
__device__ __forceinline__ float fast_exp2_ftz(float x) {
    float result;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(x));
    return result;
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
// FUNCTION: softmax_update_reg
template <bool ApplyCausalMask, bool CheckInf = true, bool IsFirst = false,
          bool CheckTail = true>
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
        bool m00, m01, m10, m11;
        if constexpr (CheckTail) {
            m00 = (n0 >= valid_n) || (row0 >= actual_rows);
            m01 = (n1 >= valid_n) || (row0 >= actual_rows);
            m10 = (n0 >= valid_n) || (row1 >= actual_rows);
            m11 = (n1 >= valid_n) || (row1 >= actual_rows);
        } else {
            m00 = (row0 >= actual_rows);
            m01 = (row0 >= actual_rows);
            m10 = (row1 >= actual_rows);
            m11 = (row1 >= actual_rows);
        }
        if constexpr (ApplyCausalMask) {
            m00 = m00 || (kv_start + n0 > q_pos0);
            m01 = m01 || (kv_start + n1 > q_pos0);
            m10 = m10 || (kv_start + n0 > q_pos1);
            m11 = m11 || (kv_start + n1 > q_pos1);
        }
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
    float rescale0, rescale1;
    if constexpr (!IsFirst) {
        if constexpr (CheckInf) {
            rescale0 = isinf(new_max0) ? 1.0f : fast_exp2_ftz((row_max[0] - new_max0) * LOG2E);
            rescale1 = isinf(new_max1) ? 1.0f : fast_exp2_ftz((row_max[1] - new_max1) * LOG2E);
        } else {
            rescale0 = fast_exp2_ftz((row_max[0] - new_max0) * LOG2E);
            rescale1 = fast_exp2_ftz((row_max[1] - new_max1) * LOG2E);
        }
    }
    row_max[0] = new_max0;
    row_max[1] = new_max1;

    if constexpr (!IsFirst) {
        #pragma unroll
        for (int nt = 0; nt < OUT_N_TILES; nt++) {
            acc_o[nt][0] *= rescale0;  acc_o[nt][1] *= rescale0;
            acc_o[nt][2] *= rescale1;  acc_o[nt][3] *= rescale1;
        }
        row_sum[0] *= rescale0;
        row_sum[1] *= rescale1;
    }

    float bsum0 = 0.0f, bsum1 = 0.0f;
    #pragma unroll
    for (int nt = 0; nt < N_TILES; nt++) {
        float p0, p1, p2, p3;
        if constexpr (CheckInf) {
            p0 = isinf(acc_s[nt][0]) ? 0.0f : fast_exp2_ftz((acc_s[nt][0] - new_max0) * LOG2E);
            p1 = isinf(acc_s[nt][1]) ? 0.0f : fast_exp2_ftz((acc_s[nt][1] - new_max0) * LOG2E);
            p2 = isinf(acc_s[nt][2]) ? 0.0f : fast_exp2_ftz((acc_s[nt][2] - new_max1) * LOG2E);
            p3 = isinf(acc_s[nt][3]) ? 0.0f : fast_exp2_ftz((acc_s[nt][3] - new_max1) * LOG2E);
        } else {
            p0 = fast_exp2_ftz((acc_s[nt][0] - new_max0) * LOG2E);
            p1 = fast_exp2_ftz((acc_s[nt][1] - new_max0) * LOG2E);
            p2 = fast_exp2_ftz((acc_s[nt][2] - new_max1) * LOG2E);
            p3 = fast_exp2_ftz((acc_s[nt][3] - new_max1) * LOG2E);
        }
        acc_s[nt][0] = p0;  acc_s[nt][1] = p1;
        acc_s[nt][2] = p2;  acc_s[nt][3] = p3;
        bsum0 += p0 + p1;
        bsum1 += p2 + p3;
    }
    // Keep lane-local partial sums across KV blocks. FA3/FlashInfer defer the
    // four-lane reduction until final normalization, avoiding four shuffles here.
    row_sum[0] += bsum0;
    row_sum[1] += bsum1;
}

// FUNCTION: softmax_update_reg_deferred_o_rescale
// FA3 IntraWGOverlap computes the next score tile while the previous PV WGMMA
// group is still outstanding.  This is the same online-softmax operation order
// as softmax_update_reg, except that the O rescale is returned to the caller and
// carried into the next PV step, matching FA3's RescaleOBeforeGemm schedule.
template <bool ApplyCausalMask, bool CheckInf = true, bool IsFirst = false,
          bool CheckTail = true>
__device__ __forceinline__ void softmax_update_reg_deferred_o_rescale(
    float acc_s[N_TILES][4],
    float row_max[2],
    float row_sum[2],
    float output_rescale[2],
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
    for (int nt = 0; nt < N_TILES; ++nt) {
        int n_base = nt * MMA_N;
        int n0 = n_base + col0, n1 = n0 + 1;
        float s0 = acc_s[nt][0] * inv_sqrt;
        float s1 = acc_s[nt][1] * inv_sqrt;
        float s2 = acc_s[nt][2] * inv_sqrt;
        float s3 = acc_s[nt][3] * inv_sqrt;
        bool m00, m01, m10, m11;
        if constexpr (CheckTail) {
            m00 = (n0 >= valid_n) || (row0 >= actual_rows);
            m01 = (n1 >= valid_n) || (row0 >= actual_rows);
            m10 = (n0 >= valid_n) || (row1 >= actual_rows);
            m11 = (n1 >= valid_n) || (row1 >= actual_rows);
        } else {
            m00 = (row0 >= actual_rows);
            m01 = (row0 >= actual_rows);
            m10 = (row1 >= actual_rows);
            m11 = (row1 >= actual_rows);
        }
        if constexpr (ApplyCausalMask) {
            m00 = m00 || (kv_start + n0 > q_pos0);
            m01 = m01 || (kv_start + n1 > q_pos0);
            m10 = m10 || (kv_start + n0 > q_pos1);
            m11 = m11 || (kv_start + n1 > q_pos1);
        }
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
    float rescale0 = 1.0f, rescale1 = 1.0f;
    if constexpr (!IsFirst) {
        if constexpr (CheckInf) {
            rescale0 = isinf(new_max0) ? 1.0f
                                       : fast_exp2_ftz((row_max[0] - new_max0) * LOG2E);
            rescale1 = isinf(new_max1) ? 1.0f
                                       : fast_exp2_ftz((row_max[1] - new_max1) * LOG2E);
        } else {
            rescale0 = fast_exp2_ftz((row_max[0] - new_max0) * LOG2E);
            rescale1 = fast_exp2_ftz((row_max[1] - new_max1) * LOG2E);
        }
    }
    row_max[0] = new_max0;
    row_max[1] = new_max1;
    output_rescale[0] = rescale0;
    output_rescale[1] = rescale1;

    if constexpr (!IsFirst) {
        row_sum[0] *= rescale0;
        row_sum[1] *= rescale1;
    }

    float bsum0 = 0.0f, bsum1 = 0.0f;
    #pragma unroll
    for (int nt = 0; nt < N_TILES; ++nt) {
        float p0, p1, p2, p3;
        if constexpr (CheckInf) {
            p0 = isinf(acc_s[nt][0]) ? 0.0f
                                      : fast_exp2_ftz((acc_s[nt][0] - new_max0) * LOG2E);
            p1 = isinf(acc_s[nt][1]) ? 0.0f
                                      : fast_exp2_ftz((acc_s[nt][1] - new_max0) * LOG2E);
            p2 = isinf(acc_s[nt][2]) ? 0.0f
                                      : fast_exp2_ftz((acc_s[nt][2] - new_max1) * LOG2E);
            p3 = isinf(acc_s[nt][3]) ? 0.0f
                                      : fast_exp2_ftz((acc_s[nt][3] - new_max1) * LOG2E);
        } else {
            p0 = fast_exp2_ftz((acc_s[nt][0] - new_max0) * LOG2E);
            p1 = fast_exp2_ftz((acc_s[nt][1] - new_max0) * LOG2E);
            p2 = fast_exp2_ftz((acc_s[nt][2] - new_max1) * LOG2E);
            p3 = fast_exp2_ftz((acc_s[nt][3] - new_max1) * LOG2E);
        }
        acc_s[nt][0] = p0;
        acc_s[nt][1] = p1;
        acc_s[nt][2] = p2;
        acc_s[nt][3] = p3;
        bsum0 += p0 + p1;
        bsum1 += p2 + p3;
    }
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
// acc_o[OUT_N_TILES][4]: O accumulator (loaded on later blocks; first PV uses ScaleOut::Zero)
// ============================================================================
// FUNCTION: compute_pv_cute
template <bool ZeroInit = false>
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

    // Step 2: Build O accumulator (C fragment). The first PV update uses
    // WGMMA ScaleOut::Zero, so its known-zero acc_o input need not be loaded.
    // partition_fragment_C gives shape ((2,2,C<16>), 1, 1) = 64 float32 per thread
    auto tCrC = partition_fragment_C(tiled_mma, Shape<Int<BLOCK_M_PREFILL>, Int<HEAD_DIM>>{});
    if constexpr (!ZeroInit) {
        #pragma unroll
        for (int nt = 0; nt < OUT_N_TILES; nt++) {
            tCrC(nt*4+0, 0, 0) = acc_o[nt][0];
            tCrC(nt*4+1, 0, 0) = acc_o[nt][1];
            tCrC(nt*4+2, 0, 0) = acc_o[nt][2];
            tCrC(nt*4+3, 0, 0) = acc_o[nt][3];
        }
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

    // Step 4: WGMMA PV — FA3/FlashInfer zero-initialize the first PV GEMM.
    // warpgroup_fence_operand on tOrP (RS) and tCrC before/after
    warpgroup_fence_operand(tOrP);
    warpgroup_fence_operand(tCrC);
    warpgroup_arrive();
    // Iterate over K_tiles (size<2>(tOrP) = C<8> = 8 tiles of 16)
    if constexpr (ZeroInit) {
        tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
    } else {
        tiled_mma.accumulate_ = GMMA::ScaleOut::One;
    }
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
// FUNCTION: unified_attn_prefill_kernel
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

    // FA3-style causal LPT exposure: adjacent x blocks retain GQA K/V locality,
    // while reversed y exposes the longest causal query tiles first.
    int head_batch_idx = blockIdx.x;
    int q_block_idx    = int(gridDim.y) - 1 - int(blockIdx.y);
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

    char* warp_smem = smem + WARP_BASE + warp_id * PER_WARP_BYTES;
    __nv_bfloat16* pv_buf  = reinterpret_cast<__nv_bfloat16*>(warp_smem + W_PV_BUF_OFF);
    BF16*          k_wgmma = reinterpret_cast<BF16*>(smem + K_SMEM_OFF);  // K-major SW128; reused as Vt after QK
    BF16*          vt_smem = reinterpret_cast<BF16*>(smem + K_SMEM_OFF);  // MN-major SW128 Vt (same slot as K)
    BF16*          q_smem  = reinterpret_cast<BF16*>(smem + Q_WGMMA_OFF);
    uint64_t* mbar = reinterpret_cast<uint64_t*>(smem + MBAR_OFF);
    const int kv_row_base = (batch_idx * num_kv_heads + kv_head_idx) * ctx_len;

    int warp_q_start = q_block_start + warp_id * MMA_M;
    int warp_q_end   = min(warp_q_start + MMA_M, seq_len);
    int actual_rows  = max(0, warp_q_end - warp_q_start);

    // Producer (warp 0) prefetches TMA descriptors and initializes mbarriers.
    // FA3/FlashInfer do this from one elected thread before the first TMA use.
    if (warp_id == 0 && lane_id == 0) {
        cute::prefetch_tma_descriptor(tma_k.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_vt.get_tma_descriptor());
        mbar_init(&mbar[0], 1);
        mbar_init(&mbar[1], 1);
    }
    __syncthreads();

    // FA3-style 128-bit Q staging: one uint4 is eight contiguous BF16 values
    // and exactly one aligned 16-byte segment of the SW128 logical row.
    {
        Tensor sQ = make_tensor(make_smem_ptr(q_smem), SmemLayoutQ{});
        constexpr int ELEMS_PER_VEC = sizeof(uint4) / sizeof(BF16);
        constexpr int VECS_PER_ROW  = HEAD_DIM / ELEMS_PER_VEC;
        constexpr int VEC_TOTAL     = BLOCK_M_PREFILL * VECS_PER_ROW;
        for (int vi = tid; vi < VEC_TOTAL; vi += BLOCK_THREADS) {
            int m = vi / VECS_PER_ROW;
            int vec_in_row = vi % VECS_PER_ROW;
            int k = vec_in_row * ELEMS_PER_VEC;
            int global_row = q_block_start + m;
            uint4 q_vec = make_uint4(0, 0, 0, 0);
            if (global_row < seq_len) {
                const uint4* src = reinterpret_cast<const uint4*>(
                    q_ptr + global_row * HEAD_DIM);
                q_vec = src[vec_in_row];
            }
            uint4* dst = reinterpret_cast<uint4*>(&sQ(m, k));
            *dst = q_vec;
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

    // FA3 BlockMN-style causal upper bound: this Q tile cannot attend to any
    // complete KV block strictly to its right.  Keep the final partially
    // visible block; softmax_update_reg still applies the per-element mask.
    const int q_block_end = min(q_block_start + BLOCK_M_PREFILL, seq_len);
    const int causal_kv_end = max(0, min(ctx_len, q_block_end + ctx_len - seq_len));
    const int num_kv_blocks = (causal_kv_end + BLOCK_N - 1) / BLOCK_N;

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
        }
        __syncthreads();

        // ── ALL WARPS: WGMMA QK (.aligned requires all 128 threads) ──────────
        float acc_s[N_TILES][4];
        compute_qk_cute(q_smem, k_wgmma, acc_s);
        // ALL warps run softmax for their own rows
        if (actual_rows > 0) {
            const bool needs_causal_mask = kv_start + valid_n - 1 > warp_q_start;
            if (kv_block == 0 && needs_causal_mask) {
                if (valid_n == BLOCK_N) {
                    softmax_update_reg<true, true, true, false>(
                        acc_s, acc_o, row_max, row_sum, inv_sqrt, valid_n,
                        kv_start, warp_q_start, actual_rows);
                } else {
                    softmax_update_reg<true, true, true, true>(
                        acc_s, acc_o, row_max, row_sum, inv_sqrt, valid_n,
                        kv_start, warp_q_start, actual_rows);
                }
            } else if (kv_block == 0 && actual_rows == MMA_M) {
                if (valid_n == BLOCK_N) {
                    softmax_update_reg<false, false, true, false>(
                        acc_s, acc_o, row_max, row_sum, inv_sqrt, valid_n,
                        kv_start, warp_q_start, actual_rows);
                } else {
                    softmax_update_reg<false, false, true, true>(
                        acc_s, acc_o, row_max, row_sum, inv_sqrt, valid_n,
                        kv_start, warp_q_start, actual_rows);
                }
            } else if (kv_block == 0) {
                if (valid_n == BLOCK_N) {
                    softmax_update_reg<false, true, true, false>(
                        acc_s, acc_o, row_max, row_sum, inv_sqrt, valid_n,
                        kv_start, warp_q_start, actual_rows);
                } else {
                    softmax_update_reg<false, true, true, true>(
                        acc_s, acc_o, row_max, row_sum, inv_sqrt, valid_n,
                        kv_start, warp_q_start, actual_rows);
                }
            } else if (needs_causal_mask) {
                if (valid_n == BLOCK_N) {
                    softmax_update_reg<true, true, false, false>(
                        acc_s, acc_o, row_max, row_sum, inv_sqrt, valid_n,
                        kv_start, warp_q_start, actual_rows);
                } else {
                    softmax_update_reg<true, true, false, true>(
                        acc_s, acc_o, row_max, row_sum, inv_sqrt, valid_n,
                        kv_start, warp_q_start, actual_rows);
                }
            } else if (actual_rows == MMA_M) {
                // FA3's full-tile steady state compiles out infinity guards once
                // every represented row is valid and has a visible finite key.
                if (valid_n == BLOCK_N) {
                    softmax_update_reg<false, false, false, false>(
                        acc_s, acc_o, row_max, row_sum, inv_sqrt, valid_n,
                        kv_start, warp_q_start, actual_rows);
                } else {
                    softmax_update_reg<false, false, false, true>(
                        acc_s, acc_o, row_max, row_sum, inv_sqrt, valid_n,
                        kv_start, warp_q_start, actual_rows);
                }
            } else {
                if (valid_n == BLOCK_N) {
                    softmax_update_reg<false, true, false, false>(
                        acc_s, acc_o, row_max, row_sum, inv_sqrt, valid_n,
                        kv_start, warp_q_start, actual_rows);
                } else {
                    softmax_update_reg<false, true, false, true>(
                        acc_s, acc_o, row_max, row_sum, inv_sqrt, valid_n,
                        kv_start, warp_q_start, actual_rows);
                }
            }
        }
        __syncthreads();

        // ── PRODUCER role (warp 0 lane 0): issue TMA V^T load ────────────────
        // V^T reuses K smem slot (K is done after QK). cute TMA → MN-major SW128.
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
        // All warps wait for V^T
        {
            const int vphase = kv_block & 1;
            mbar_wait(&mbar[1], vphase);
        }
        __syncthreads();

        // ── ALL WARPS: WGMMA PV (.aligned requires all 128 threads) ──────────
        if (actual_rows > 0) {
            if (kv_block == 0) {
                compute_pv_cute<true>(acc_s, vt_smem, acc_o);
            } else {
                compute_pv_cute<false>(acc_s, vt_smem, acc_o);
            }
        }
        __syncthreads();
    }

    // ── ALL WARPS: write output for their own rows ────────────────────────────
    if (actual_rows > 0) {
        int group_id     = lane_id >> 2;
        int tid_in_group = lane_id & 3;
        int row0 = group_id, row1 = group_id + 8;
        int col0 = tid_in_group * 2, col1 = col0 + 1;
        float final_sum0 = quad_sum(row_sum[0]);
        float final_sum1 = quad_sum(row_sum[1]);
        float inv0 = (final_sum0 > 0.0f) ? (1.0f / final_sum0) : 0.0f;
        float inv1 = (final_sum1 > 0.0f) ? (1.0f / final_sum1) : 0.0f;

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
// FUNCTION: unified_attn_decode_kernel
__global__ void unified_attn_decode_kernel(
    const __nv_bfloat16* __restrict__ Q,
    CUTLASS_GRID_CONSTANT TmaK const tma_k,
    CUTLASS_GRID_CONSTANT TmaVt const tma_vt,
    __nv_bfloat16* __restrict__ O,
    int ctx_len, int num_heads, int num_kv_heads,
    int total_kv_rows
) {
    extern __shared__ char smem[];

    // Pack two adjacent Q heads from the same GQA group into logical rows 0-1.
    // The pair shares one K/V stream and one CTA's QK/PV work.
    constexpr int DECODE_HEADS_PER_CTA = 2;
    // Encode the GQA hierarchy directly in the 3-D grid. This preserves the
    // original x-fastest packed-head order without runtime div/mod expansion.
    int head_pair_in_kv = int(blockIdx.x);
    int kv_head_idx     = int(blockIdx.y);
    int batch_idx       = int(blockIdx.z);
    int head_pair_idx   = kv_head_idx * int(gridDim.x) + head_pair_in_kv;
    int head_idx       = head_pair_idx * DECODE_HEADS_PER_CTA;

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
        // Prefetch both streaming TMA descriptors before their first use.
        cute::prefetch_tma_descriptor(tma_k.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_vt.get_tma_descriptor());
        mbar_init(&mbar[0], 1);
        mbar_init(&mbar[1], 1);
    }

    // Like FA3 PackGQA, flatten heads sharing K/V into WGMMA's row dimension.
    // Q is [batch, head, 1, dim], so the two adjacent heads are contiguous.
    // Leave the other 62 rows untouched because their outputs are discarded.
    {
        Tensor sQ = make_tensor(make_smem_ptr(q_smem), SmemLayoutQ{});
        constexpr int PACKED_Q_ELEMS = DECODE_HEADS_PER_CTA * HEAD_DIM;
        for (int idx = tid; idx < PACKED_Q_ELEMS; idx += BLOCK_THREADS) {
            int row = idx / HEAD_DIM;
            int k   = idx % HEAD_DIM;
            sQ(row, k) = BF16(q_ptr[row * HEAD_DIM + k]);
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
            if (kv_block == 0) {
                if (valid_n == BLOCK_N) {
                    softmax_update_reg<false, true, true, false>(
                        acc_s, acc_o, row_max, row_sum,
                        inv_sqrt, valid_n, kv_start, ctx_len - 1,
                        DECODE_HEADS_PER_CTA);
                } else {
                    softmax_update_reg<false, true, true, true>(
                        acc_s, acc_o, row_max, row_sum,
                        inv_sqrt, valid_n, kv_start, ctx_len - 1,
                        DECODE_HEADS_PER_CTA);
                }
            } else {
                if (valid_n == BLOCK_N) {
                    softmax_update_reg<false, true, false, false>(
                        acc_s, acc_o, row_max, row_sum,
                        inv_sqrt, valid_n, kv_start, ctx_len - 1,
                        DECODE_HEADS_PER_CTA);
                } else {
                    softmax_update_reg<false, true, false, true>(
                        acc_s, acc_o, row_max, row_sum,
                        inv_sqrt, valid_n, kv_start, ctx_len - 1,
                        DECODE_HEADS_PER_CTA);
                }
            }
        }
        {
            mbar_wait(&mbar[1], vphase);
        }
        __syncthreads();  // sync #3

        if (kv_block == 0) {
            compute_pv_cute<true>(acc_s, vt_smem, acc_o);
        } else {
            compute_pv_cute<false>(acc_s, vt_smem, acc_o);
        }
        __syncthreads();  // sync #4
    }

    // Finalize the lane-local partial row sums once, after all KV blocks.
    if (warp_id == 0) {
        row_sum[0] = quad_sum(row_sum[0]);
    }

    // Each packed head belongs to one independent four-lane group in warp 0.
    // Normalize and store rows 0-1 directly to their adjacent head outputs.
    int packed_row = lane_id >> 2;
    if (warp_id == 0 && packed_row < DECODE_HEADS_PER_CTA) {
        float row_sum0 = row_sum[0];
        float inv0 = (row_sum0 > 0.0f) ? (1.0f / row_sum0) : 0.0f;
        int tid_in_group = lane_id & 3;
        int col0 = tid_in_group * 2, col1 = col0 + 1;
        __nv_bfloat16* packed_o_ptr = o_ptr + packed_row * HEAD_DIM;

        #pragma unroll
        for (int nt = 0; nt < OUT_N_TILES; nt++) {
            int n_base = nt * MMA_N;
            packed_o_ptr[n_base + col0] = __float2bfloat16(acc_o[nt][0] * inv0);
            packed_o_ptr[n_base + col1] = __float2bfloat16(acc_o[nt][1] * inv0);
        }
    }
}

// FUNCTION: unified_attn_decode_pipelined_kernel
__global__ void unified_attn_decode_pipelined_kernel(
    const __nv_bfloat16* __restrict__ Q,
    CUTLASS_GRID_CONSTANT TmaK const tma_k,
    CUTLASS_GRID_CONSTANT TmaVt const tma_vt,
    __nv_bfloat16* __restrict__ O,
    int ctx_len, int num_heads, int num_kv_heads,
    int total_kv_rows
) {
    extern __shared__ char smem[];

    constexpr int DECODE_HEADS_PER_CTA = 2;
    int head_pair_in_kv = int(blockIdx.x);
    int kv_head_idx     = int(blockIdx.y);
    int batch_idx       = int(blockIdx.z);
    int head_pair_idx   = kv_head_idx * int(gridDim.x) + head_pair_in_kv;
    int head_idx        = head_pair_idx * DECODE_HEADS_PER_CTA;

    const __nv_bfloat16* q_ptr = Q +
        (batch_idx * num_heads + head_idx) * HEAD_DIM;
    __nv_bfloat16* o_ptr = O +
        (batch_idx * num_heads + head_idx) * HEAD_DIM;

    int tid     = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    BF16* q_smem = reinterpret_cast<BF16*>(smem + DECODE_Q_OFF);
    uint64_t* mbar = reinterpret_cast<uint64_t*>(smem + DECODE_MBAR_OFF);
    const int kv_row_base =
        (batch_idx * num_kv_heads + kv_head_idx) * ctx_len;

    if (tid == 0) {
        cute::prefetch_tma_descriptor(tma_k.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_vt.get_tma_descriptor());
        mbar_init(&mbar[0], 1);
        mbar_init(&mbar[1], 1);
        mbar_init(&mbar[2], 1);
        mbar_init(&mbar[3], 1);
    }

    {
        Tensor sQ = make_tensor(make_smem_ptr(q_smem), SmemLayoutQ{});
        constexpr int PACKED_Q_ELEMS = DECODE_HEADS_PER_CTA * HEAD_DIM;
        for (int idx = tid; idx < PACKED_Q_ELEMS; idx += BLOCK_THREADS) {
            int row = idx / HEAD_DIM;
            int k   = idx % HEAD_DIM;
            sQ(row, k) = BF16(q_ptr[row * HEAD_DIM + k]);
        }
    }
    __syncthreads();

    float row_max[2] = {-INFINITY, -INFINITY};
    float row_sum[2] = {0.0f, 0.0f};
    float output_rescale[2] = {1.0f, 1.0f};
    float acc_s[N_TILES][4];

    float inv_sqrt = 1.0f / sqrtf((float)HEAD_DIM);
    int num_kv_blocks = (ctx_len + BLOCK_N - 1) / BLOCK_N;

    // Persistent register fragments are required for FA3's IntraWGOverlap:
    // QK(i) and PV(i-1) remain as separate committed WGMMA groups, while the
    // previous P and the running O stay live until their group is retired.
    TiledMmaQK tiled_mma_qk;
    ThrMMA thr_mma_qk = tiled_mma_qk.get_thread_slice(threadIdx.x);
    Tensor sQ = make_tensor(make_smem_ptr(q_smem), SmemLayoutQ{});
    Tensor tCsQ = thr_mma_qk.partition_A(sQ);
    auto tCrQ = thr_mma_qk.make_fragment_A(tCsQ);
    auto tSrS = partition_fragment_C(
        tiled_mma_qk, Shape<Int<BLOCK_M_PREFILL>, Int<BLOCK_N>>{});

    TiledMmaPV tiled_mma_pv;
    ThrMMA thr_mma_pv = tiled_mma_pv.get_thread_slice(threadIdx.x);
    auto tOrO = partition_fragment_C(
        tiled_mma_pv, Shape<Int<BLOCK_M_PREFILL>, Int<HEAD_DIM>>{});
    clear(tOrO);
    auto tOrP_layout = convert_layout_acc_Aregs<TiledMmaPV>(tSrS.layout());
    auto tOrP_acc = make_tensor(tSrS.data(), tOrP_layout);
    auto tOrP = make_tensor_like<BF16>(tOrP_acc);
    // The first PV's first K-slice overwrites the known-zero O accumulator.
    // The assignment inside each K loop leaves subsequent blocks at ScaleOut::One.
    tiled_mma_pv.accumulate_ = GMMA::ScaleOut::Zero;

    auto convert_scores_to_p = [&]() {
        using From_t = float;
        using To_t = BF16;
        constexpr int N = decltype(size(tOrP_acc))::value;
        cutlass::NumericArrayConverter<To_t, From_t, N> cvt;
        auto src = recast<cutlass::Array<From_t, N> const>(tOrP_acc);
        auto dst = recast<cutlass::Array<To_t, N>>(tOrP);
        dst[0] = cvt(src[0]);
    };

    auto issue_k = [&](int kv_block) {
        int stage = kv_block & 1;
        int kv_start = kv_block * BLOCK_N;
        BF16* k_stage = reinterpret_cast<BF16*>(
            smem + (stage == 0 ? DECODE_K_STAGE0_OFF : DECODE_K_STAGE1_OFF));
        const int k_bytes = BLOCK_N * HEAD_DIM * sizeof(BF16);
        mbar_arrive_tx(&mbar[stage], k_bytes);
        Tensor mK = tma_k.get_tma_tensor(
            make_shape(total_kv_rows, Int<HEAD_DIM>{}));
        Tensor sK = make_tensor(make_smem_ptr(k_stage), SmemLayoutKW{});
        Tensor mK_off = domain_offset(
            make_coord(kv_row_base + kv_start, 0), mK);
        Tensor gK = local_tile(
            mK_off, Shape<Int<BLOCK_N>, Int<HEAD_DIM>>{}, make_coord(0, 0));
        auto [tKgK, tKsK] = tma_partition(
            tma_k, Int<0>{}, Layout<_1>{},
            group_modes<0,2>(sK), group_modes<0,2>(gK));
        copy(tma_k.with(mbar[stage]), tKgK, tKsK);
    };

    auto issue_v = [&](int kv_block) {
        int stage = kv_block & 1;
        int kv_start = kv_block * BLOCK_N;
        BF16* v_stage = reinterpret_cast<BF16*>(
            smem + (stage == 0 ? DECODE_V_STAGE0_OFF : DECODE_V_STAGE1_OFF));
        const int vt_bytes = HEAD_DIM * BLOCK_N * sizeof(BF16);
        mbar_arrive_tx(&mbar[2 + stage], vt_bytes);
        Tensor mVt = tma_vt.get_tma_tensor(
            make_shape(Int<HEAD_DIM>{}, total_kv_rows));
        Tensor sVt = make_tensor(make_smem_ptr(v_stage), SmemLayoutVt{});
        Tensor mVt_off = domain_offset(
            make_coord(0, kv_row_base + kv_start), mVt);
        Tensor gVt = local_tile(
            mVt_off, Shape<Int<HEAD_DIM>, Int<BLOCK_N>>{}, make_coord(0, 0));
        auto [tVgVt, tVsVt] = tma_partition(
            tma_vt, Int<0>{}, Layout<_1>{},
            group_modes<0,2>(sVt), group_modes<0,2>(gVt));
        copy(tma_vt.with(mbar[2 + stage]), tVgVt, tVsVt);
    };

    // Prologue.  Only K is advanced to block 1 here.  V remains one block
    // behind K, exactly as FA3's IntraWGOverlap producer loop requires, so V0
    // cannot be overwritten before the overlapped PV0 group consumes it.
    if (num_kv_blocks > 0 && tid == 0) {
        issue_k(0);
        issue_v(0);
    }
    if (num_kv_blocks > 0) {
        mbar_wait(&mbar[0], 0);
    }
    if (num_kv_blocks > 1 && tid == 0) {
        issue_k(1);
    }
    __syncthreads();

    // QK0 is the only non-overlapped QK.  It creates P0, which becomes the A
    // operand of the first overlapped PV group in the steady-state loop.
    if (num_kv_blocks > 0) {
        BF16* k0 = reinterpret_cast<BF16*>(smem + DECODE_K_STAGE0_OFF);
        Tensor sK0 = make_tensor(make_smem_ptr(k0), SmemLayoutKW{});
        Tensor tCsK0 = thr_mma_qk.partition_B(sK0);
        auto tCrK0 = thr_mma_qk.make_fragment_B(tCsK0);
        clear(tSrS);
        warpgroup_arrive();
        gemm(tiled_mma_qk, tCrQ, tCrK0, tSrS);
        warpgroup_commit_batch();
        warpgroup_wait<0>();

        #pragma unroll
        for (int nt = 0; nt < N_TILES; ++nt) {
            acc_s[nt][0] = tSrS(nt * 4 + 0, 0, 0);
            acc_s[nt][1] = tSrS(nt * 4 + 1, 0, 0);
            acc_s[nt][2] = tSrS(nt * 4 + 2, 0, 0);
            acc_s[nt][3] = tSrS(nt * 4 + 3, 0, 0);
        }
        if (warp_id == 0) {
            int valid_n = min(BLOCK_N, ctx_len);
            if (valid_n == BLOCK_N) {
                softmax_update_reg_deferred_o_rescale<false, true, true, false>(
                    acc_s, row_max, row_sum, output_rescale,
                    inv_sqrt, valid_n, 0, ctx_len - 1,
                    DECODE_HEADS_PER_CTA);
            } else {
                softmax_update_reg_deferred_o_rescale<false, true, true, true>(
                    acc_s, row_max, row_sum, output_rescale,
                    inv_sqrt, valid_n, 0, ctx_len - 1,
                    DECODE_HEADS_PER_CTA);
            }
        }
        #pragma unroll
        for (int nt = 0; nt < N_TILES; ++nt) {
            tSrS(nt * 4 + 0, 0, 0) = acc_s[nt][0];
            tSrS(nt * 4 + 1, 0, 0) = acc_s[nt][1];
            tSrS(nt * 4 + 2, 0, 0) = acc_s[nt][2];
            tSrS(nt * 4 + 3, 0, 0) = acc_s[nt][3];
        }
        convert_scores_to_p();
    }
    // K0 can now be reused for K2; V0 remains live for PV0.
    __syncthreads();

    // FA3 mainloop_fwd_sm90_tma_gmma_ws.hpp IntraWGOverlap: each step commits
    // QK(i) first and PV(i-1) second.  wait<1> exposes QK scores while PV stays
    // outstanding, so scalar softmax overlaps PV.  wait<0> precedes both the O
    // rescale and overwriting P, preserving the original online-attention order.
    for (int score_block = 1; score_block < num_kv_blocks; ++score_block) {
        int k_stage = score_block & 1;
        int k_phase = (score_block >> 1) & 1;
        int pv_block = score_block - 1;
        int v_stage = pv_block & 1;
        int v_phase = (pv_block >> 1) & 1;

        mbar_wait(&mbar[k_stage], k_phase);
        mbar_wait(&mbar[2 + v_stage], v_phase);

        // The prior step's final CTA barrier released K(i-1) and V(i-2).
        // K stays one block ahead; V(i) is loaded while PV(i-1) executes.
        if (tid == 0) {
            if (score_block + 1 < num_kv_blocks) {
                issue_k(score_block + 1);
            }
            issue_v(score_block);
        }
        __syncthreads();

        BF16* k_cur = reinterpret_cast<BF16*>(
            smem + (k_stage == 0 ? DECODE_K_STAGE0_OFF : DECODE_K_STAGE1_OFF));
        BF16* v_prev = reinterpret_cast<BF16*>(
            smem + (v_stage == 0 ? DECODE_V_STAGE0_OFF : DECODE_V_STAGE1_OFF));

        Tensor sK = make_tensor(make_smem_ptr(k_cur), SmemLayoutKW{});
        Tensor tCsK = thr_mma_qk.partition_B(sK);
        auto tCrK = thr_mma_qk.make_fragment_B(tCsK);
        clear(tSrS);
        warpgroup_arrive();
        gemm(tiled_mma_qk, tCrQ, tCrK, tSrS);
        warpgroup_commit_batch();

        // FA3 RescaleOBeforeGemm: output_rescale belongs to the score tile
        // processed in the previous step.  The prior PV has already retired at
        // that step's wait<0>, so scale O now while this step's QK is
        // outstanding and before adding the corresponding PV contribution.
        if (warp_id == 0) {
            #pragma unroll
            for (int nt = 0; nt < OUT_N_TILES; ++nt) {
                tOrO(nt * 4 + 0, 0, 0) *= output_rescale[0];
                tOrO(nt * 4 + 1, 0, 0) *= output_rescale[0];
                tOrO(nt * 4 + 2, 0, 0) *= output_rescale[1];
                tOrO(nt * 4 + 3, 0, 0) *= output_rescale[1];
            }
        }

        Tensor sVt = make_tensor(make_smem_ptr(v_prev), SmemLayoutVt{});
        Tensor tCsVt = thr_mma_pv.partition_B(sVt);
        auto tCrVt = thr_mma_pv.make_fragment_B(tCsVt);
        warpgroup_fence_operand(tOrP);
        warpgroup_fence_operand(tOrO);
        warpgroup_arrive();
        #pragma unroll
        for (int k = 0; k < size<2>(tOrP); ++k) {
            cute::gemm(tiled_mma_pv, tOrP(_,_,k), tCrVt(_,_,k), tOrO);
            tiled_mma_pv.accumulate_ = GMMA::ScaleOut::One;
        }
        warpgroup_commit_batch();

        // QK is the older group, so wait<1> makes tSrS available but deliberately
        // leaves the PV group outstanding.
        warpgroup_wait<1>();
        #pragma unroll
        for (int nt = 0; nt < N_TILES; ++nt) {
            acc_s[nt][0] = tSrS(nt * 4 + 0, 0, 0);
            acc_s[nt][1] = tSrS(nt * 4 + 1, 0, 0);
            acc_s[nt][2] = tSrS(nt * 4 + 2, 0, 0);
            acc_s[nt][3] = tSrS(nt * 4 + 3, 0, 0);
        }

        int kv_start = score_block * BLOCK_N;
        int valid_n = min(BLOCK_N, ctx_len - kv_start);
        if (warp_id == 0) {
            if (valid_n == BLOCK_N) {
                softmax_update_reg_deferred_o_rescale<false, true, false, false>(
                    acc_s, row_max, row_sum, output_rescale,
                    inv_sqrt, valid_n, kv_start, ctx_len - 1,
                    DECODE_HEADS_PER_CTA);
            } else {
                softmax_update_reg_deferred_o_rescale<false, true, false, true>(
                    acc_s, row_max, row_sum, output_rescale,
                    inv_sqrt, valid_n, kv_start, ctx_len - 1,
                    DECODE_HEADS_PER_CTA);
            }
        }
        #pragma unroll
        for (int nt = 0; nt < N_TILES; ++nt) {
            tSrS(nt * 4 + 0, 0, 0) = acc_s[nt][0];
            tSrS(nt * 4 + 1, 0, 0) = acc_s[nt][1];
            tSrS(nt * 4 + 2, 0, 0) = acc_s[nt][2];
            tSrS(nt * 4 + 3, 0, 0) = acc_s[nt][3];
        }

        // PV(i-1) must retire before P's registers are overwritten with the
        // probabilities for block i.  Its newly computed O rescale is carried
        // into the next step instead of extending this post-wait critical path.
        warpgroup_wait<0>();
        warpgroup_fence_operand(tOrO);
        warpgroup_fence_operand(tOrP);
        convert_scores_to_p();

        // Releases K(i) and V(i-1) before their stages are targeted again.
        __syncthreads();
    }

    // Epilogue PV for the last score block.  V(last) was issued in the final
    // steady-state step and has overlapped all of that step's math.
    if (num_kv_blocks > 0) {
        int last_block = num_kv_blocks - 1;
        int v_stage = last_block & 1;
        int v_phase = (last_block >> 1) & 1;
        mbar_wait(&mbar[2 + v_stage], v_phase);
        __syncthreads();

        BF16* v_last = reinterpret_cast<BF16*>(
            smem + (v_stage == 0 ? DECODE_V_STAGE0_OFF : DECODE_V_STAGE1_OFF));
        Tensor sVt = make_tensor(make_smem_ptr(v_last), SmemLayoutVt{});
        Tensor tCsVt = thr_mma_pv.partition_B(sVt);
        auto tCrVt = thr_mma_pv.make_fragment_B(tCsVt);

        // The final score tile has no following steady-state step in which to
        // consume its deferred scale.  Apply it immediately before the final
        // PV so the arithmetic order remains scale(O_previous) + P_last * V.
        if (warp_id == 0) {
            #pragma unroll
            for (int nt = 0; nt < OUT_N_TILES; ++nt) {
                tOrO(nt * 4 + 0, 0, 0) *= output_rescale[0];
                tOrO(nt * 4 + 1, 0, 0) *= output_rescale[0];
                tOrO(nt * 4 + 2, 0, 0) *= output_rescale[1];
                tOrO(nt * 4 + 3, 0, 0) *= output_rescale[1];
            }
        }
        warpgroup_fence_operand(tOrP);
        warpgroup_fence_operand(tOrO);
        warpgroup_arrive();
        #pragma unroll
        for (int k = 0; k < size<2>(tOrP); ++k) {
            cute::gemm(tiled_mma_pv, tOrP(_,_,k), tCrVt(_,_,k), tOrO);
            tiled_mma_pv.accumulate_ = GMMA::ScaleOut::One;
        }
        warpgroup_commit_batch();
        warpgroup_wait<0>();
        warpgroup_fence_operand(tOrO);
        warpgroup_fence_operand(tOrP);
        __syncthreads();
    }

    if (warp_id == 0) {
        row_sum[0] = quad_sum(row_sum[0]);
    }

    int packed_row = lane_id >> 2;
    if (warp_id == 0 && packed_row < DECODE_HEADS_PER_CTA) {
        float row_sum0 = row_sum[0];
        float inv0 = (row_sum0 > 0.0f) ? (1.0f / row_sum0) : 0.0f;
        int tid_in_group = lane_id & 3;
        int col0 = tid_in_group * 2;
        int col1 = col0 + 1;
        __nv_bfloat16* packed_o_ptr = o_ptr + packed_row * HEAD_DIM;

        #pragma unroll
        for (int nt = 0; nt < OUT_N_TILES; ++nt) {
            int n_base = nt * MMA_N;
            packed_o_ptr[n_base + col0] =
                __float2bfloat16(tOrO(nt * 4 + 0, 0, 0) * inv0);
            packed_o_ptr[n_base + col1] =
                __float2bfloat16(tOrO(nt * 4 + 1, 0, 0) * inv0);
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

// FUNCTION: unified_prefill
torch::Tensor unified_prefill(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                               int num_heads, int num_kv_heads) {
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda());
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous());
    TORCH_CHECK(Q.dtype() == torch::kBFloat16);
    TORCH_CHECK(Q.size(3) == HEAD_DIM);

    int batch = Q.size(0), seq_len = Q.size(2), ctx_len = K.size(2);
    int num_kv_heads_actual = K.size(1);
    // Every logical output element is written by the epilogue, including the
    // zero-context path, so avoid a redundant device-wide initialization.
    auto O = torch::empty_like(Q);
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
    // Dynamic-smem capacity is invariant for this loaded kernel. Match CUTLASS's
    // initialize/run split and configure it once before repeated launches.
    static const cudaError_t attr_status =
        cudaFuncSetAttribute(unified_attn_prefill_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES_10H);
    TORCH_CHECK(attr_status == cudaSuccess,
                "prefill dynamic-smem configuration failed: ",
                cudaGetErrorString(attr_status));
    unified_attn_prefill_kernel<<<dim3(batch * num_heads, nq), BLOCK_THREADS, SMEM_BYTES_10H>>>(
        (const __nv_bfloat16*)Q.data_ptr(),
        tma_k, tma_vt,
        (__nv_bfloat16*)O.data_ptr(),
        seq_len, ctx_len, num_heads, num_kv_heads, total_kv_rows);

    return O;
}

// FUNCTION: unified_decode
torch::Tensor unified_decode(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                              int num_heads, int num_kv_heads) {
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda());
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous());
    TORCH_CHECK(Q.size(2) == 1);
    TORCH_CHECK(Q.size(3) == HEAD_DIM);
    constexpr int DECODE_HEADS_PER_CTA = 2;
    TORCH_CHECK(num_heads % num_kv_heads == 0);
    int q_heads_per_kv = num_heads / num_kv_heads;
    TORCH_CHECK(q_heads_per_kv % DECODE_HEADS_PER_CTA == 0,
                "two-head decode packing requires an even Q-heads-per-KV-head ratio");
    int batch = Q.size(0), ctx_len = K.size(2);
    int num_kv_heads_actual = K.size(1);
    // Each packed row's four participating lanes cover all 128 output columns.
    // The zero-context path also stores explicit zeros from initialized state.
    auto O = torch::empty_like(Q);
    int total_kv_rows = batch * num_kv_heads_actual * ctx_len;

    // FlashInfer's plan/run split and CUTLASS's initialize/run split retain
    // immutable launch metadata. Do the same for the two tensor maps: repeated
    // benchmark calls use the same K/V storage and encoded flattened shape.
    struct DecodeTmaCacheEntry {
        int device;
        const void* k_ptr;
        const void* v_ptr;
        int total_rows;
        TmaK tma_k;
        TmaVt tma_vt;
    };
    static thread_local std::optional<DecodeTmaCacheEntry> tma_cache;

    const int device = Q.get_device();
    const void* k_ptr = K.data_ptr();
    const void* v_ptr = V.data_ptr();
    const bool cache_hit = tma_cache.has_value() &&
        tma_cache->device == device &&
        tma_cache->k_ptr == k_ptr &&
        tma_cache->v_ptr == v_ptr &&
        tma_cache->total_rows == total_kv_rows;
    if (!cache_hit) {
        auto dK = make_stride(Int<HEAD_DIM>{}, Int<1>{});
        Tensor mK = make_tensor(reinterpret_cast<BF16 const*>(k_ptr),
                                make_shape(total_kv_rows, Int<HEAD_DIM>{}), dK);
        auto new_tma_k = make_tma_atom(SM90_TMA_LOAD{}, mK, SmemLayoutKW{},
                                       Shape<Int<BLOCK_N>, Int<HEAD_DIM>>{});

        auto dVt = make_stride(Int<1>{}, Int<HEAD_DIM>{});
        Tensor mVt = make_tensor(reinterpret_cast<BF16 const*>(v_ptr),
                                 make_shape(Int<HEAD_DIM>{}, total_kv_rows), dVt);
        auto new_tma_vt = make_tma_atom(SM90_TMA_LOAD{}, mVt, SmemLayoutVt{},
                                        Shape<Int<HEAD_DIM>, Int<BLOCK_N>>{});
        tma_cache = DecodeTmaCacheEntry{
            device, k_ptr, v_ptr, total_kv_rows, new_tma_k, new_tma_vt
        };
    }
    TmaK const& tma_k = tma_cache->tma_k;
    TmaVt const& tma_vt = tma_cache->tma_vt;

    int head_pairs_per_kv = q_heads_per_kv / DECODE_HEADS_PER_CTA;
    dim3 decode_grid(head_pairs_per_kv, num_kv_heads, batch);
    if (ctx_len > 1024) {
        // The deeper pipeline repays its larger shared-memory footprint only
        // when enough KV blocks repeat.  Keep the iteration-40 short path so
        // launch-sensitive decode-128/1024 and the consistency denominator do
        // not pay the long-context staging cost.
        static const cudaError_t pipeline_attr_status =
            cudaFuncSetAttribute(unified_attn_decode_pipelined_kernel,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 SMEM_BYTES_DECODE_PIPE);
        TORCH_CHECK(pipeline_attr_status == cudaSuccess,
                    "pipelined decode dynamic-smem configuration failed: ",
                    cudaGetErrorString(pipeline_attr_status));
        unified_attn_decode_pipelined_kernel<<<
            decode_grid, BLOCK_THREADS, SMEM_BYTES_DECODE_PIPE>>>(
                (const __nv_bfloat16*)Q.data_ptr(),
                tma_k,
                tma_vt,
                (__nv_bfloat16*)O.data_ptr(),
                ctx_len, num_heads, num_kv_heads, total_kv_rows);
    } else {
        static const cudaError_t short_attr_status =
            cudaFuncSetAttribute(unified_attn_decode_kernel,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 SMEM_BYTES_10H);
        TORCH_CHECK(short_attr_status == cudaSuccess,
                    "decode dynamic-smem configuration failed: ",
                    cudaGetErrorString(short_attr_status));
        unified_attn_decode_kernel<<<decode_grid, BLOCK_THREADS, SMEM_BYTES_10H>>>(
            (const __nv_bfloat16*)Q.data_ptr(),
            tma_k,
            tma_vt,
            (__nv_bfloat16*)O.data_ptr(),
            ctx_len, num_heads, num_kv_heads, total_kv_rows);
    }
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("unified_prefill", &unified_prefill, "Unified prefill (10h: cute TMA K → K-major SW128, WGMMA QK)");
    m.def("unified_decode",  &unified_decode,  "Unified decode  (10h: WGMMA QK, vectorized K/V)");
}
