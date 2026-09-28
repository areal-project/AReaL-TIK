// kernel_iter1 kBlockM=128 with AtomLayout<2,1,1> for QK, dual-PV (single-atom PV ×2)
//
// Base: kernel_11c (WGMMA PV + V^T TMA, 2.0x FA3 at seq=1024)
// Change: BLOCK_M_PREFILL 64→128, TiledMmaQK gets AtomLayout<_2,_1,_1>
//         Both atoms properly handled: QK extraction → 8-wide arrays,
//         4-row softmax, PV called twice (once per atom with single-atom TiledMmaPV).
//         Halves tile count (nq = seq_len/128), reducing launch overhead.
//
// Why dual-PV not single AtomLayout PV: convert_layout_acc_Aregs does not handle
// the atom dimension correctly for the NumericArrayConverter. Calling single-atom
// PV twice avoids this compilation issue at the cost of 2× PV WGMMA calls.
//
// Smem: Q_WGMMA doubles to 32KB (128 rows), total ~83KB → 2 CTAs/SM

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

// ── Core constants ──
constexpr int HEAD_DIM        = 128;
constexpr int NUM_WARPS       = 4;
constexpr int WARP_SIZE       = 32;
constexpr int BLOCK_THREADS   = NUM_WARPS * WARP_SIZE;
constexpr int BLOCK_M_PREFILL = 128;
constexpr int BLOCK_N         = 128;

constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 16;

using namespace cute;
using BF16 = cutlass::bfloat16_t;
using SmemLayoutAtom = decltype(GMMA::Layout_K_SW128_Atom<BF16>{});
using SmemLayoutQ    = decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<BLOCK_M_PREFILL>, Int<HEAD_DIM>>{}));
using SmemLayoutKW   = decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<BLOCK_N>,         Int<HEAD_DIM>>{}));

// TiledMmaQK with AtomLayout<2,1,1> for kBlockM=128 — handles all 128 query rows
using TiledMmaQK = decltype(make_tiled_mma(
    SM90_64x128x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>{},
    Layout<Shape<_2, _1, _1>>{}));

using TmaK = decltype(make_tma_atom(
    SM90_TMA_LOAD{},
    make_tensor(static_cast<BF16 const*>(nullptr),
                make_shape(1, Int<HEAD_DIM>{}),
                make_stride(Int<HEAD_DIM>{}, Int<1>{})),
    SmemLayoutKW{},
    Shape<Int<BLOCK_N>, Int<HEAD_DIM>>{}));

using SmemLayoutAtomVt = decltype(GMMA::Layout_MN_SW128_Atom<BF16>{});
using SmemLayoutVt     = decltype(tile_to_shape(SmemLayoutAtomVt{},
                                                Shape<Int<HEAD_DIM>, Int<BLOCK_N>>{}));

using TmaVt = decltype(make_tma_atom(
    SM90_TMA_LOAD{},
    make_tensor(static_cast<BF16 const*>(nullptr),
                make_shape(Int<HEAD_DIM>{}, 1),
                make_stride(Int<1>{}, Int<HEAD_DIM>{})),
    SmemLayoutVt{},
    Shape<Int<HEAD_DIM>, Int<BLOCK_N>>{}));

// TiledMmaPV with AtomLayout<2,1,1> for kBlockM=128 — handles both atoms in one WGMMA
using TiledMmaPV = decltype(make_tiled_mma(
    SM90_64x128x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::MN>{},
    Layout<Shape<_2, _1, _1>>{}));

constexpr int N_TILES    = BLOCK_N  / MMA_N;   // 16
constexpr int KV_K_TILES = BLOCK_N  / MMA_K;   // 8
constexpr int OUT_N_TILES= HEAD_DIM / MMA_N;   // 16

constexpr float LOG2E = 1.442695040888963407359924681001892137426646f;

constexpr int KV_SMEM_PAD    = 8;
constexpr int K_SMEM_STRIDE  = HEAD_DIM + KV_SMEM_PAD;
constexpr int V_SMEM_STRIDE  = BLOCK_N  + KV_SMEM_PAD;
constexpr int HALF_DIM       = HEAD_DIM / 2;
constexpr int KV_HALF_BYTES  = BLOCK_N * HALF_DIM * 2;
constexpr int KV_LO_OFF      = 0;
constexpr int KV_HI_OFF      = BLOCK_N * HALF_DIM;
constexpr int KV_SMEM_BYTES  = KV_HALF_BYTES * 2;
constexpr int Q_TILE_BYTES   = MMA_M * HEAD_DIM * 2;
constexpr int PV_BUF_COLS    = 24;
constexpr int PV_BUF_BYTES   = MMA_M * PV_BUF_COLS * 2;
constexpr int PER_WARP_BYTES = Q_TILE_BYTES + PV_BUF_BYTES;

constexpr int MBAR_BYTES     = 2 * 8;
constexpr int K_SMEM_OFF     = 0;
constexpr int MBAR_OFF       = KV_SMEM_BYTES;
constexpr int WARP_BASE      = MBAR_OFF + MBAR_BYTES;
constexpr int SMEM_BYTES     = WARP_BASE + NUM_WARPS * PER_WARP_BYTES;

constexpr int W_Q_TILE_OFF   = 0;
constexpr int W_PV_BUF_OFF   = W_Q_TILE_OFF + Q_TILE_BYTES;

constexpr int Q_WGMMA_BYTES  = 128 * 128 * 2;
constexpr int Q_WGMMA_OFF    = SMEM_BYTES;
constexpr int SMEM_BYTES_10H = Q_WGMMA_OFF + Q_WGMMA_BYTES;

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

__device__ __forceinline__ void mbar_init(uint64_t* b, uint32_t count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"((uint32_t)__cvta_generic_to_shared(b)), "r"(count));
}

__device__ __forceinline__ void mbar_arrive_tx(uint64_t* b, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"((uint32_t)__cvta_generic_to_shared(b)), "r"(tx_bytes));
}

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

__device__ __forceinline__ int swizzle_col(int row, int col) {
    return ((col / 8) ^ (row % 8)) * 8 + (col % 8);
}

__device__ __forceinline__ void tma_load_2d(
    const CUtensorMap* __restrict__ desc,
    __nv_bfloat16* __restrict__ smem_dst,
    uint64_t* __restrict__ mbar,
    int coord_col, int coord_row
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
// FUNCTION: compute_qk_cute
// WGMMA QK with AtomLayout<2,1,1>. Extracts both atoms into acc_s[N_TILES][8].
// ============================================================================
__device__ __forceinline__ void compute_qk_cute(
    const BF16* __restrict__ q_smem,
    const BF16* __restrict__ k_wgmma,
    float acc_s[N_TILES][8]
) {
    TiledMmaQK tiled_mma;
    ThrMMA thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
    Tensor sQ = make_tensor(make_smem_ptr(q_smem),  SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(k_wgmma), SmemLayoutKW{});
    Tensor tCsQ = thr_mma.partition_A(sQ);
    Tensor tCsK = thr_mma.partition_B(sK);
    auto tCrQ = thr_mma.make_fragment_A(tCsQ);
    auto tCrK = thr_mma.make_fragment_B(tCsK);
    auto tCrC = partition_fragment_C(tiled_mma, Shape<Int<128>, Int<128>>{});
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
        acc_s[nt][4] = tCrC(nt*4+0, 1, 0);
        acc_s[nt][5] = tCrC(nt*4+1, 1, 0);
        acc_s[nt][6] = tCrC(nt*4+2, 1, 0);
        acc_s[nt][7] = tCrC(nt*4+3, 1, 0);
    }
}

// ============================================================================
// FUNCTION: softmax_update_reg
// 4 rows per thread: atom0-row0 [0,1], atom0-row1 [2,3], atom1-row0 [4,5], atom1-row1 [6,7]
// ============================================================================
__device__ __forceinline__ void softmax_update_reg(
    float acc_s[N_TILES][8],
    float acc_o[OUT_N_TILES][8],
    float row_max[4],
    float row_sum[4],
    float inv_sqrt,
    int valid_n,
    int kv_start,
    int warp_q_start_a0,
    int warp_q_start_a1,
    int actual_rows_a0,
    int actual_rows_a1
) {
    int lane_id  = threadIdx.x % WARP_SIZE;
    int group_id = lane_id >> 2;
    int col_pair = lane_id & 3;
    int col0 = col_pair * 2, col1 = col0 + 1;
    int r0 = group_id, r1 = group_id + 8;
    int q_pos0 = warp_q_start_a0 + r0;
    int q_pos1 = warp_q_start_a0 + r1;
    int q_pos2 = warp_q_start_a1 + r0;
    int q_pos3 = warp_q_start_a1 + r1;

    float bmax0 = -INFINITY, bmax1 = -INFINITY;
    float bmax2 = -INFINITY, bmax3 = -INFINITY;
    #pragma unroll
    for (int nt = 0; nt < N_TILES; nt++) {
        int n_base = nt * MMA_N;
        int n0 = n_base + col0, n1 = n_base + col1;
        float s0 = acc_s[nt][0] * inv_sqrt;
        float s1 = acc_s[nt][1] * inv_sqrt;
        float s2 = acc_s[nt][2] * inv_sqrt;
        float s3 = acc_s[nt][3] * inv_sqrt;
        float s4 = acc_s[nt][4] * inv_sqrt;
        float s5 = acc_s[nt][5] * inv_sqrt;
        float s6 = acc_s[nt][6] * inv_sqrt;
        float s7 = acc_s[nt][7] * inv_sqrt;
        bool m00 = (n0 >= valid_n) || (kv_start + n0 > q_pos0) || (r0 >= actual_rows_a0);
        bool m01 = (n1 >= valid_n) || (kv_start + n1 > q_pos0) || (r0 >= actual_rows_a0);
        bool m10 = (n0 >= valid_n) || (kv_start + n0 > q_pos1) || (r1 >= actual_rows_a0);
        bool m11 = (n1 >= valid_n) || (kv_start + n1 > q_pos1) || (r1 >= actual_rows_a0);
        bool m20 = (n0 >= valid_n) || (kv_start + n0 > q_pos2) || (r0 >= actual_rows_a1);
        bool m21 = (n1 >= valid_n) || (kv_start + n1 > q_pos2) || (r0 >= actual_rows_a1);
        bool m30 = (n0 >= valid_n) || (kv_start + n0 > q_pos3) || (r1 >= actual_rows_a1);
        bool m31 = (n1 >= valid_n) || (kv_start + n1 > q_pos3) || (r1 >= actual_rows_a1);
        acc_s[nt][0] = m00 ? -INFINITY : s0;  acc_s[nt][1] = m01 ? -INFINITY : s1;
        acc_s[nt][2] = m10 ? -INFINITY : s2;  acc_s[nt][3] = m11 ? -INFINITY : s3;
        acc_s[nt][4] = m20 ? -INFINITY : s4;  acc_s[nt][5] = m21 ? -INFINITY : s5;
        acc_s[nt][6] = m30 ? -INFINITY : s6;  acc_s[nt][7] = m31 ? -INFINITY : s7;
        bmax0 = fmaxf(bmax0, fmaxf(acc_s[nt][0], acc_s[nt][1]));
        bmax1 = fmaxf(bmax1, fmaxf(acc_s[nt][2], acc_s[nt][3]));
        bmax2 = fmaxf(bmax2, fmaxf(acc_s[nt][4], acc_s[nt][5]));
        bmax3 = fmaxf(bmax3, fmaxf(acc_s[nt][6], acc_s[nt][7]));
    }
    bmax0 = quad_max(bmax0);  bmax1 = quad_max(bmax1);
    bmax2 = quad_max(bmax2);  bmax3 = quad_max(bmax3);

    float new_max0 = fmaxf(row_max[0], bmax0);
    float new_max1 = fmaxf(row_max[1], bmax1);
    float new_max2 = fmaxf(row_max[2], bmax2);
    float new_max3 = fmaxf(row_max[3], bmax3);
    float rescale0 = isinf(new_max0) ? 1.0f : exp2f((row_max[0] - new_max0) * LOG2E);
    float rescale1 = isinf(new_max1) ? 1.0f : exp2f((row_max[1] - new_max1) * LOG2E);
    float rescale2 = isinf(new_max2) ? 1.0f : exp2f((row_max[2] - new_max2) * LOG2E);
    float rescale3 = isinf(new_max3) ? 1.0f : exp2f((row_max[3] - new_max3) * LOG2E);
    row_max[0] = new_max0;  row_max[1] = new_max1;
    row_max[2] = new_max2;  row_max[3] = new_max3;

    #pragma unroll
    for (int nt = 0; nt < OUT_N_TILES; nt++) {
        acc_o[nt][0] *= rescale0;  acc_o[nt][1] *= rescale0;
        acc_o[nt][2] *= rescale1;  acc_o[nt][3] *= rescale1;
        acc_o[nt][4] *= rescale2;  acc_o[nt][5] *= rescale2;
        acc_o[nt][6] *= rescale3;  acc_o[nt][7] *= rescale3;
    }
    row_sum[0] *= rescale0;  row_sum[1] *= rescale1;
    row_sum[2] *= rescale2;  row_sum[3] *= rescale3;

    float bsum0 = 0.0f, bsum1 = 0.0f, bsum2 = 0.0f, bsum3 = 0.0f;
    #pragma unroll
    for (int nt = 0; nt < N_TILES; nt++) {
        float p0 = isinf(acc_s[nt][0]) ? 0.0f : exp2f((acc_s[nt][0] - new_max0) * LOG2E);
        float p1 = isinf(acc_s[nt][1]) ? 0.0f : exp2f((acc_s[nt][1] - new_max0) * LOG2E);
        float p2 = isinf(acc_s[nt][2]) ? 0.0f : exp2f((acc_s[nt][2] - new_max1) * LOG2E);
        float p3 = isinf(acc_s[nt][3]) ? 0.0f : exp2f((acc_s[nt][3] - new_max1) * LOG2E);
        float p4 = isinf(acc_s[nt][4]) ? 0.0f : exp2f((acc_s[nt][4] - new_max2) * LOG2E);
        float p5 = isinf(acc_s[nt][5]) ? 0.0f : exp2f((acc_s[nt][5] - new_max2) * LOG2E);
        float p6 = isinf(acc_s[nt][6]) ? 0.0f : exp2f((acc_s[nt][6] - new_max3) * LOG2E);
        float p7 = isinf(acc_s[nt][7]) ? 0.0f : exp2f((acc_s[nt][7] - new_max3) * LOG2E);
        acc_s[nt][0] = p0;  acc_s[nt][1] = p1;
        acc_s[nt][2] = p2;  acc_s[nt][3] = p3;
        acc_s[nt][4] = p4;  acc_s[nt][5] = p5;
        acc_s[nt][6] = p6;  acc_s[nt][7] = p7;
        bsum0 += p0 + p1;  bsum1 += p2 + p3;
        bsum2 += p4 + p5;  bsum3 += p6 + p7;
    }
    bsum0 = quad_sum(bsum0);  bsum1 = quad_sum(bsum1);
    bsum2 = quad_sum(bsum2);  bsum3 = quad_sum(bsum3);
    row_sum[0] += bsum0;  row_sum[1] += bsum1;
    row_sum[2] += bsum2;  row_sum[3] += bsum3;
}

// ============================================================================
// Legacy helpers (unused)
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
// FUNCTION: convert_layout_acc_Aregs
// Same as original — for single-atom layout ((2,2,C<16>), 1, 1) → ((2,2,2), 1, C<8>)
// ============================================================================
template<typename MMA_Traits, typename Layout0>
__device__ __forceinline__ auto convert_layout_acc_Aregs(Layout0 acc_layout) {
    auto l = logical_divide(get<0, 2>(acc_layout), Tile<_2>{});
    return make_layout(
        make_layout(get<0, 0>(acc_layout), get<0, 1>(acc_layout), get<0, 0>(l)),
        get<1>(acc_layout),
        coalesce(make_layout(get<0, 1>(l), get<2>(acc_layout)))
    );
}

// ============================================================================
// FUNCTION: compute_pv_cute_single
// Single-atom WGMMA PV — same as original compute_pv_cute with [4] arrays.
// Processes one 64-row atom of post-softmax probabilities.
// ============================================================================
__device__ __forceinline__ void compute_pv_cute_single(
    float acc_s_atom[N_TILES][4],
    const BF16* __restrict__ vt_smem,
    float acc_o_atom[OUT_N_TILES][4]
) {
    TiledMmaPV tiled_mma;
    ThrMMA thr_mma = tiled_mma.get_thread_slice(threadIdx.x);

    Tensor sVt  = make_tensor(make_smem_ptr(vt_smem), SmemLayoutVt{});
    Tensor tCsVt = thr_mma.partition_B(sVt);
    auto   tCrVt = thr_mma.make_fragment_B(tCsVt);

    auto tCrC = partition_fragment_C(tiled_mma, Shape<Int<64>, Int<HEAD_DIM>>{});
    #pragma unroll
    for (int nt = 0; nt < OUT_N_TILES; nt++) {
        tCrC(nt*4+0, 0, 0) = acc_o_atom[nt][0];
        tCrC(nt*4+1, 0, 0) = acc_o_atom[nt][1];
        tCrC(nt*4+2, 0, 0) = acc_o_atom[nt][2];
        tCrC(nt*4+3, 0, 0) = acc_o_atom[nt][3];
    }

    auto tSrS_layout = tCrC.layout();
    auto tOrP_layout = convert_layout_acc_Aregs<TiledMmaPV>(tSrS_layout);

    auto tSrS = make_tensor<float>(tSrS_layout);
    #pragma unroll
    for (int nt = 0; nt < N_TILES; nt++) {
        tSrS(nt*4+0, 0, 0) = acc_s_atom[nt][0];
        tSrS(nt*4+1, 0, 0) = acc_s_atom[nt][1];
        tSrS(nt*4+2, 0, 0) = acc_s_atom[nt][2];
        tSrS(nt*4+3, 0, 0) = acc_s_atom[nt][3];
    }

    auto tOrP_acc = make_tensor(tSrS.data(), tOrP_layout);
    auto tOrP     = make_tensor_like<BF16>(tOrP_acc);
    {
        using From_t = float;
        using To_t   = BF16;
        constexpr int N = decltype(size(tOrP_acc))::value;
        cutlass::NumericArrayConverter<To_t, From_t, N> cvt;
        auto src = recast<cutlass::Array<From_t, N> const>(tOrP_acc);
        auto dst = recast<cutlass::Array<To_t,   N>      >(tOrP);
        dst[0] = cvt(src[0]);
    }

    warpgroup_fence_operand(tOrP);
    warpgroup_fence_operand(tCrC);
    warpgroup_arrive();
    #pragma unroll
    for (int k = 0; k < size<2>(tOrP); ++k) {
        cute::gemm(tiled_mma, tOrP(_,_,k), tCrVt(_,_,k), tCrC);
        tiled_mma.accumulate_ = GMMA::ScaleOut::One;
    }
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    warpgroup_fence_operand(tCrC);
    warpgroup_fence_operand(tOrP);

    #pragma unroll
    for (int nt = 0; nt < OUT_N_TILES; nt++) {
        acc_o_atom[nt][0] = tCrC(nt*4+0, 0, 0);
        acc_o_atom[nt][1] = tCrC(nt*4+1, 0, 0);
        acc_o_atom[nt][2] = tCrC(nt*4+2, 0, 0);
        acc_o_atom[nt][3] = tCrC(nt*4+3, 0, 0);
    }
}

// ============================================================================
// FUNCTION: compute_pv_cute
// Wrapper: extracts atom 0 from 8-wide arrays, runs PV, stores back.
//          Then extracts atom 1, runs PV, stores back.
// ============================================================================
__device__ __forceinline__ void compute_pv_cute(
    float acc_s[N_TILES][8],
    const BF16* __restrict__ vt_smem,
    float acc_o[OUT_N_TILES][8]
) {
    float s_atom[N_TILES][4], o_atom[OUT_N_TILES][4];

    // Atom 0: acc_s[0..3], acc_o[0..3]
    #pragma unroll
    for (int nt = 0; nt < N_TILES; nt++) {
        s_atom[nt][0] = acc_s[nt][0];  s_atom[nt][1] = acc_s[nt][1];
        s_atom[nt][2] = acc_s[nt][2];  s_atom[nt][3] = acc_s[nt][3];
    }
    #pragma unroll
    for (int nt = 0; nt < OUT_N_TILES; nt++) {
        o_atom[nt][0] = acc_o[nt][0];  o_atom[nt][1] = acc_o[nt][1];
        o_atom[nt][2] = acc_o[nt][2];  o_atom[nt][3] = acc_o[nt][3];
    }
    compute_pv_cute_single(s_atom, vt_smem, o_atom);
    #pragma unroll
    for (int nt = 0; nt < OUT_N_TILES; nt++) {
        acc_o[nt][0] = o_atom[nt][0];  acc_o[nt][1] = o_atom[nt][1];
        acc_o[nt][2] = o_atom[nt][2];  acc_o[nt][3] = o_atom[nt][3];
    }

    // Atom 1: acc_s[4..7], acc_o[4..7]
    #pragma unroll
    for (int nt = 0; nt < N_TILES; nt++) {
        s_atom[nt][0] = acc_s[nt][4];  s_atom[nt][1] = acc_s[nt][5];
        s_atom[nt][2] = acc_s[nt][6];  s_atom[nt][3] = acc_s[nt][7];
    }
    #pragma unroll
    for (int nt = 0; nt < OUT_N_TILES; nt++) {
        o_atom[nt][0] = acc_o[nt][4];  o_atom[nt][1] = acc_o[nt][5];
        o_atom[nt][2] = acc_o[nt][6];  o_atom[nt][3] = acc_o[nt][7];
    }
    compute_pv_cute_single(s_atom, vt_smem, o_atom);
    #pragma unroll
    for (int nt = 0; nt < OUT_N_TILES; nt++) {
        acc_o[nt][4] = o_atom[nt][0];  acc_o[nt][5] = o_atom[nt][1];
        acc_o[nt][6] = o_atom[nt][2];  acc_o[nt][7] = o_atom[nt][3];
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
    CUTLASS_GRID_CONSTANT TmaK  const tma_k,
    CUTLASS_GRID_CONSTANT TmaVt const tma_vt,
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

    BF16*          k_wgmma = reinterpret_cast<BF16*>(smem + K_SMEM_OFF);
    BF16*          vt_smem = reinterpret_cast<BF16*>(smem + K_SMEM_OFF);
    BF16*          q_smem  = reinterpret_cast<BF16*>(smem + Q_WGMMA_OFF);
    uint64_t* mbar = reinterpret_cast<uint64_t*>(smem + MBAR_OFF);
    const int kv_row_base = (batch_idx * num_kv_heads + kv_head_idx) * ctx_len;

    int warp_q_start = q_block_start + warp_id * MMA_M;
    int atom0_start = warp_q_start;
    int atom0_end   = min(atom0_start + MMA_M, seq_len);
    int atom0_rows  = max(0, atom0_end - atom0_start);

    int atom1_start = q_block_start + 64 + warp_id * MMA_M;
    int atom1_end   = min(atom1_start + MMA_M, seq_len);
    int atom1_rows  = max(0, atom1_end - atom1_start);

    int any_rows = (atom0_rows > 0 || atom1_rows > 0) ? 1 : 0;

    if (warp_id == 0 && lane_id == 0) {
        mbar_init(&mbar[0], 1);
        mbar_init(&mbar[1], 1);
    }
    __syncthreads();

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

    float acc_o[OUT_N_TILES][8];
    float row_max[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
    float row_sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    #pragma unroll
    for (int nt = 0; nt < OUT_N_TILES; nt++) {
        acc_o[nt][0] = acc_o[nt][1] = acc_o[nt][2] = acc_o[nt][3] = 0.0f;
        acc_o[nt][4] = acc_o[nt][5] = acc_o[nt][6] = acc_o[nt][7] = 0.0f;
    }

    float inv_sqrt = 1.0f / sqrtf((float)HEAD_DIM);
    int num_kv_blocks = (ctx_len + BLOCK_N - 1) / BLOCK_N;

    for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        int kv_start = kv_block * BLOCK_N;
        int kv_end   = min(kv_start + BLOCK_N, ctx_len);
        int valid_n  = kv_end - kv_start;

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
        {
            const int kphase = kv_block & 1;
            mbar_wait(&mbar[0], kphase);
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        }
        __syncthreads();

        float acc_s[N_TILES][8];
        compute_qk_cute(q_smem, k_wgmma, acc_s);
        if (any_rows) {
            softmax_update_reg(acc_s, acc_o, row_max, row_sum,
                               inv_sqrt, valid_n, kv_start,
                               atom0_start, atom1_start,
                               atom0_rows, atom1_rows);
        }
        __syncthreads();

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
        {
            const int vphase = kv_block & 1;
            mbar_wait(&mbar[1], vphase);
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        }
        __syncthreads();

        if (any_rows)
            compute_pv_cute(acc_s, vt_smem, acc_o);
        __syncthreads();
    }

    // ── Epilogue: write both atoms' rows ──────────────────────────
    if (any_rows) {
        int group_id     = lane_id >> 2;
        int tid_in_group = lane_id & 3;
        int r0 = group_id, r1 = group_id + 8;
        int col0 = tid_in_group * 2, col1 = col0 + 1;
        float inv0 = (row_sum[0] > 0.0f) ? (1.0f / row_sum[0]) : 0.0f;
        float inv1 = (row_sum[1] > 0.0f) ? (1.0f / row_sum[1]) : 0.0f;
        float inv2 = (row_sum[2] > 0.0f) ? (1.0f / row_sum[2]) : 0.0f;
        float inv3 = (row_sum[3] > 0.0f) ? (1.0f / row_sum[3]) : 0.0f;

        #pragma unroll
        for (int nt = 0; nt < OUT_N_TILES; nt++) {
            int n_base = nt * MMA_N;
            int out_n0 = n_base + col0, out_n1 = n_base + col1;

            if (r0 < atom0_rows) {
                o_ptr[(atom0_start + r0) * HEAD_DIM + out_n0] =
                    __float2bfloat16(acc_o[nt][0] * inv0);
                o_ptr[(atom0_start + r0) * HEAD_DIM + out_n1] =
                    __float2bfloat16(acc_o[nt][1] * inv0);
            }
            if (r1 < atom0_rows) {
                o_ptr[(atom0_start + r1) * HEAD_DIM + out_n0] =
                    __float2bfloat16(acc_o[nt][2] * inv1);
                o_ptr[(atom0_start + r1) * HEAD_DIM + out_n1] =
                    __float2bfloat16(acc_o[nt][3] * inv1);
            }
            if (r0 < atom1_rows) {
                o_ptr[(atom1_start + r0) * HEAD_DIM + out_n0] =
                    __float2bfloat16(acc_o[nt][4] * inv2);
                o_ptr[(atom1_start + r0) * HEAD_DIM + out_n1] =
                    __float2bfloat16(acc_o[nt][5] * inv2);
            }
            if (r1 < atom1_rows) {
                o_ptr[(atom1_start + r1) * HEAD_DIM + out_n0] =
                    __float2bfloat16(acc_o[nt][6] * inv3);
                o_ptr[(atom1_start + r1) * HEAD_DIM + out_n1] =
                    __float2bfloat16(acc_o[nt][7] * inv3);
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

    {
        Tensor sQ = make_tensor(make_smem_ptr(q_smem), SmemLayoutQ{});
        const int total = BLOCK_M_PREFILL * HEAD_DIM;
        for (int i = tid; i < total; i += BLOCK_THREADS) {
            int m = i / HEAD_DIM, k = i % HEAD_DIM;
            sQ(m, k) = (m == 0) ? BF16(q_ptr[k]) : BF16(0.0f);
        }
    }
    __syncthreads();

    float acc_o[OUT_N_TILES][8];
    float row_max[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
    float row_sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    #pragma unroll
    for (int nt = 0; nt < OUT_N_TILES; nt++) {
        acc_o[nt][0] = acc_o[nt][1] = acc_o[nt][2] = acc_o[nt][3] = 0.0f;
        acc_o[nt][4] = acc_o[nt][5] = acc_o[nt][6] = acc_o[nt][7] = 0.0f;
    }

    float inv_sqrt = 1.0f / sqrtf((float)HEAD_DIM);
    int num_kv_blocks = (ctx_len + BLOCK_N - 1) / BLOCK_N;

    for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        int kv_start = kv_block * BLOCK_N;
        int kv_end   = min(kv_start + BLOCK_N, ctx_len);
        int valid_n  = kv_end - kv_start;
        const int vphase = kv_block & 1;

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
        __syncthreads();

        float acc_s[N_TILES][8];
        compute_qk_cute(q_smem, k_wgmma, acc_s);
        __syncthreads();

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
                               inv_sqrt, valid_n, kv_start,
                               ctx_len - 1, ctx_len - 1, 1, 0);
        }
        {
            mbar_wait(&mbar[1], vphase);
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        }
        __syncthreads();

        compute_pv_cute(acc_s, vt_smem, acc_o);
        __syncthreads();
    }

    {
        float* scratch = reinterpret_cast<float*>(smem + MBAR_OFF);
        if (warp_id == 0 && lane_id == 0)
            scratch[0] = row_sum[0];
        __syncthreads();
        float shared_row_sum = scratch[0];
        __syncthreads();

        int group_id     = lane_id >> 2;
        int tid_in_group = lane_id & 3;
        float inv0 = (shared_row_sum > 0.0f) ? (1.0f / shared_row_sum) : 0.0f;
        int col0 = tid_in_group * 2, col1 = col0 + 1;

        if (warp_id == 0 && group_id == 0) {
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
// Host functions
// ============================================================================

static CUtensorMap make_tma_desc(
    const void* global_ptr,
    int num_rows,
    int smem_rows,
    int smem_col_stride
) {
    CUtensorMap desc;
    uint64_t global_dim[2]    = {(uint64_t)HEAD_DIM, (uint64_t)num_rows};
    uint64_t global_stride[1] = {(uint64_t)HEAD_DIM * sizeof(__nv_bfloat16)};
    uint32_t smem_box[2]      = {(uint32_t)HALF_DIM, (uint32_t)smem_rows};
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
        CU_TENSOR_MAP_SWIZZLE_128B,
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

    auto dK = make_stride(Int<HEAD_DIM>{}, Int<1>{});
    Tensor mK = make_tensor(reinterpret_cast<BF16 const*>(K.data_ptr()),
                            make_shape(total_kv_rows, Int<HEAD_DIM>{}), dK);
    auto tma_k = make_tma_atom(SM90_TMA_LOAD{}, mK, SmemLayoutKW{},
                               Shape<Int<BLOCK_N>, Int<HEAD_DIM>>{});

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
    m.def("unified_prefill", &unified_prefill, "Unified prefill (kblockm128)");
    m.def("unified_decode",  &unified_decode,  "Unified decode  (kblockm128)");
}
