// unified_kernel_12b_v5.cu — kernel_12b step 5: replace Vt raw mbarrier with PipelineTmaAsync
//
// Base: unified_kernel_12b_v4.cu (PipelineK for K, raw mbarrier for Vt, sync A + sync B)
// Change: Vt barrier also replaced with PipelineTmaAsync<2>.
//         Both K and Vt now use PipelineTmaAsync — ready for sync B removal in v6.
//         Sync A and sync B: still kept (same as v4).
//
// Expected: same correctness as v4, similar performance (still slower than v3 due to
//           empty barrier overhead with sync B present — payoff comes in v6).
//
// What changes vs 12a:
//   - BLOCK_THREADS_WS = 256 (was 128)
//   - warpgroup_reg_dealloc<24>() for WG0 (threads 0-127)
//   - warpgroup_reg_alloc<240>() for WG1 (threads 128-255)
//   - WG0 (warp_id 0-3): producer — only warp 0 issues TMA, rest idle
//   - WG1 (warp_id 4-7): consumer — runs WGMMA using threadIdx.x-128
//   - compute_qk_cute_ws / compute_pv_cute_ws: take explicit thread_idx
//   - Softmax and output write use consumer_warp_id = warp_id - 4
//   - ALL __syncthreads() kept — WG0 and WG1 alternate in lockstep
//   - Raw mbarriers (same as 11c) — no PipelineTmaAsync
//   - Smem layout: IDENTICAL to 11c (68KB, no double buffering)
//
// What is IDENTICAL to 12a/11c:
//   - unified_attn_decode_kernel: zero changes
//   - Persistent grid dim3(num_sm) with LPT ordering
//   - Raw mbarrier pattern (mbar_init, mbar_arrive_tx, mbar_wait)
//   - All __syncthreads() sync points
//   - Smem layout (SMEM_BYTES_10H = 68640 B)
//   - Bitwise consistency: maintained
//
// Why this is safe:
//   - All __syncthreads() ensure WG0 and WG1 never race
//   - mbar_wait is called by all 256 threads (spin-loop, safe)
//   - warpgroup_arrive() inside WGMMA functions is called only by WG1 (128 threads)
//   - No PipelineTmaAsync deadlock possible
//
// Expected: same correctness as 11c, possibly slightly faster due to more registers
//           for consumer WGMMA accumulators (240 vs ~64 in 11c)

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
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/numeric_conversion.h>

constexpr int HEAD_DIM        = 128;
constexpr int NUM_WARPS       = 4;
constexpr int WARP_SIZE       = 32;
constexpr int BLOCK_THREADS   = NUM_WARPS * WARP_SIZE;
constexpr int BLOCK_M_PREFILL = NUM_WARPS * 16;  // 64
constexpr int BLOCK_N         = 128;

constexpr int BLOCK_THREADS_WS   = 256;
constexpr int NUM_WARPS_WS       = BLOCK_THREADS_WS / WARP_SIZE;
constexpr int PRODUCER_WG_IDX    = 0;
constexpr int CONSUMER_WG_IDX    = 1;
constexpr int MMA_THREAD_OFFSET  = 128;
constexpr int CONSUMER_WARP_BASE = 4;

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

using TiledMmaPV = decltype(make_tiled_mma(
    SM90_64x128x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::MN>{}));

constexpr int N_TILES     = BLOCK_N  / MMA_N;   // 16
constexpr int K_TILES     = HEAD_DIM / MMA_K;   // 8
constexpr int OUT_N_TILES = HEAD_DIM / MMA_N;   // 16

constexpr float LOG2E = 1.442695040888963407359924681001892137426646f;

// Same smem layout as 11c / 12a.
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

constexpr int Q_WGMMA_BYTES  = 64 * 128 * 2;
constexpr int Q_WGMMA_OFF    = SMEM_BYTES;
constexpr int SMEM_BYTES_10H = Q_WGMMA_OFF + Q_WGMMA_BYTES;

// 12b-v3: double-buffered K and Vt (2 slots each, ping-pong)
constexpr int K_STAGE_BYTES      = BLOCK_N * HEAD_DIM * 2;           // 32768 B per slot
constexpr int VT_STAGE_BYTES     = HEAD_DIM * BLOCK_N * 2;           // 32768 B per slot
constexpr int K_BUF_0_OFF        = 0;                                 // K slot 0
constexpr int K_BUF_1_OFF        = K_STAGE_BYTES;                    // K slot 1 = 32768
constexpr int VT_BUF_0_OFF       = K_STAGE_BYTES * 2;                // Vt slot 0 = 65536
constexpr int VT_BUF_1_OFF       = K_STAGE_BYTES * 2 + VT_STAGE_BYTES; // Vt slot 1 = 98304

// PipelineK and PipelineVt storage: each 2 stages × (FullBarrier + EmptyBarrier)
using PipelineK  = cutlass::PipelineTmaAsync<2>;
using PipelineVt = cutlass::PipelineTmaAsync<2>;
using PipelineStateWS = cutlass::PipelineState<2>;
constexpr int PIPELINE_K_OFF     = K_STAGE_BYTES * 2 + VT_STAGE_BYTES * 2;  // 131072
constexpr int PIPELINE_K_BYTES   = sizeof(PipelineK::SharedStorage);
constexpr int PIPELINE_VT_OFF    = PIPELINE_K_OFF + PIPELINE_K_BYTES;
constexpr int PIPELINE_VT_BYTES  = sizeof(PipelineVt::SharedStorage);

constexpr int WARP_BASE_V5       = PIPELINE_VT_OFF + PIPELINE_VT_BYTES;
constexpr int SMEM_BYTES_V5_BASE = WARP_BASE_V5 + 4 * PER_WARP_BYTES;
constexpr int Q_WGMMA_OFF_V5     = SMEM_BYTES_V5_BASE;
constexpr int SMEM_BYTES_12B_V5  = Q_WGMMA_OFF_V5 + Q_WGMMA_BYTES;

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

__device__ __forceinline__ void compute_qk_cute(
    const BF16* __restrict__ q_smem,
    const BF16* __restrict__ k_wgmma,
    float acc_s[N_TILES][4]
) {
    TiledMmaQK tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
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
        acc_s[nt][0] = tCrC(nt * 4 + 0, 0, 0);
        acc_s[nt][1] = tCrC(nt * 4 + 1, 0, 0);
        acc_s[nt][2] = tCrC(nt * 4 + 2, 0, 0);
        acc_s[nt][3] = tCrC(nt * 4 + 3, 0, 0);
    }
}

__device__ __forceinline__ void compute_qk_cute_ws(
    const BF16* __restrict__ q_smem,
    const BF16* __restrict__ k_wgmma,
    float acc_s[N_TILES][4],
    int thread_idx
) {
    TiledMmaQK tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
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
        acc_s[nt][0] = tCrC(nt * 4 + 0, 0, 0);
        acc_s[nt][1] = tCrC(nt * 4 + 1, 0, 0);
        acc_s[nt][2] = tCrC(nt * 4 + 2, 0, 0);
        acc_s[nt][3] = tCrC(nt * 4 + 3, 0, 0);
    }
}

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

template<typename MMA_Traits, typename Layout0>
__device__ __forceinline__ auto convert_layout_acc_Aregs(Layout0 acc_layout) {
    auto l = logical_divide(get<0, 2>(acc_layout), Tile<_2>{});
    return make_layout(
        make_layout(get<0, 0>(acc_layout), get<0, 1>(acc_layout), get<0, 0>(l)),
        get<1>(acc_layout),
        coalesce(make_layout(get<0, 1>(l), get<2>(acc_layout)))
    );
}

__device__ __forceinline__ void compute_pv_cute(
    float acc_s[N_TILES][4],
    const BF16* __restrict__ vt_smem,
    float acc_o[OUT_N_TILES][4]
) {
    TiledMmaPV tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);

    Tensor sVt = make_tensor(make_smem_ptr(vt_smem), SmemLayoutVt{});
    Tensor tCsVt = thr_mma.partition_B(sVt);
    auto tCrVt = thr_mma.make_fragment_B(tCsVt);

    auto tCrC = partition_fragment_C(tiled_mma, Shape<Int<BLOCK_M_PREFILL>, Int<HEAD_DIM>>{});
    #pragma unroll
    for (int nt = 0; nt < OUT_N_TILES; nt++) {
        tCrC(nt * 4 + 0, 0, 0) = acc_o[nt][0];
        tCrC(nt * 4 + 1, 0, 0) = acc_o[nt][1];
        tCrC(nt * 4 + 2, 0, 0) = acc_o[nt][2];
        tCrC(nt * 4 + 3, 0, 0) = acc_o[nt][3];
    }

    auto tSrS_layout = tCrC.layout();
    auto tOrP_layout = convert_layout_acc_Aregs<TiledMmaPV>(tSrS_layout);
    auto tSrS = make_tensor<float>(tSrS_layout);
    #pragma unroll
    for (int nt = 0; nt < N_TILES; nt++) {
        tSrS(nt * 4 + 0, 0, 0) = acc_s[nt][0];
        tSrS(nt * 4 + 1, 0, 0) = acc_s[nt][1];
        tSrS(nt * 4 + 2, 0, 0) = acc_s[nt][2];
        tSrS(nt * 4 + 3, 0, 0) = acc_s[nt][3];
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
        cute::gemm(tiled_mma, tOrP(_, _, k), tCrVt(_, _, k), tCrC);
        tiled_mma.accumulate_ = GMMA::ScaleOut::One;
    }
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    warpgroup_fence_operand(tCrC);
    warpgroup_fence_operand(tOrP);

    #pragma unroll
    for (int nt = 0; nt < OUT_N_TILES; nt++) {
        acc_o[nt][0] = tCrC(nt * 4 + 0, 0, 0);
        acc_o[nt][1] = tCrC(nt * 4 + 1, 0, 0);
        acc_o[nt][2] = tCrC(nt * 4 + 2, 0, 0);
        acc_o[nt][3] = tCrC(nt * 4 + 3, 0, 0);
    }
}

__device__ __forceinline__ void compute_pv_cute_ws(
    float acc_s[N_TILES][4],
    const BF16* __restrict__ vt_smem,
    float acc_o[OUT_N_TILES][4],
    int thread_idx
) {
    TiledMmaPV tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);

    Tensor sVt = make_tensor(make_smem_ptr(vt_smem), SmemLayoutVt{});
    Tensor tCsVt = thr_mma.partition_B(sVt);
    auto tCrVt = thr_mma.make_fragment_B(tCsVt);

    auto tCrC = partition_fragment_C(tiled_mma, Shape<Int<BLOCK_M_PREFILL>, Int<HEAD_DIM>>{});
    #pragma unroll
    for (int nt = 0; nt < OUT_N_TILES; nt++) {
        tCrC(nt * 4 + 0, 0, 0) = acc_o[nt][0];
        tCrC(nt * 4 + 1, 0, 0) = acc_o[nt][1];
        tCrC(nt * 4 + 2, 0, 0) = acc_o[nt][2];
        tCrC(nt * 4 + 3, 0, 0) = acc_o[nt][3];
    }

    auto tSrS_layout = tCrC.layout();
    auto tOrP_layout = convert_layout_acc_Aregs<TiledMmaPV>(tSrS_layout);
    auto tSrS = make_tensor<float>(tSrS_layout);
    #pragma unroll
    for (int nt = 0; nt < N_TILES; nt++) {
        tSrS(nt * 4 + 0, 0, 0) = acc_s[nt][0];
        tSrS(nt * 4 + 1, 0, 0) = acc_s[nt][1];
        tSrS(nt * 4 + 2, 0, 0) = acc_s[nt][2];
        tSrS(nt * 4 + 3, 0, 0) = acc_s[nt][3];
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
        cute::gemm(tiled_mma, tOrP(_, _, k), tCrVt(_, _, k), tCrC);
        tiled_mma.accumulate_ = GMMA::ScaleOut::One;
    }
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    warpgroup_fence_operand(tCrC);
    warpgroup_fence_operand(tOrP);

    #pragma unroll
    for (int nt = 0; nt < OUT_N_TILES; nt++) {
        acc_o[nt][0] = tCrC(nt * 4 + 0, 0, 0);
        acc_o[nt][1] = tCrC(nt * 4 + 1, 0, 0);
        acc_o[nt][2] = tCrC(nt * 4 + 2, 0, 0);
        acc_o[nt][3] = tCrC(nt * 4 + 3, 0, 0);
    }
}

__global__
__launch_bounds__(BLOCK_THREADS_WS)
void unified_attn_prefill_kernel(
    const __nv_bfloat16* __restrict__ Q,
    CUTLASS_GRID_CONSTANT TmaK  const tma_k,
    CUTLASS_GRID_CONSTANT TmaVt const tma_vt,
    __nv_bfloat16* __restrict__ O,
    int seq_len, int ctx_len, int num_heads, int num_kv_heads,
    int total_kv_rows,
    int nq,
    int total_tiles
) {
    extern __shared__ char smem[];

    int tid     = threadIdx.x;
    int wg_idx  = tid / 128;
    int wg_tid  = tid % 128;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    if (wg_idx == PRODUCER_WG_IDX) {
        cutlass::arch::warpgroup_reg_dealloc<24>();
    } else {
        cutlass::arch::warpgroup_reg_alloc<240>();
    }

    // K and Vt double-buffered: 2 slots each, indexed by kv_block & 1
    BF16* k_bufs[2]  = { reinterpret_cast<BF16*>(smem + K_BUF_0_OFF),
                         reinterpret_cast<BF16*>(smem + K_BUF_1_OFF) };
    BF16* vt_bufs[2] = { reinterpret_cast<BF16*>(smem + VT_BUF_0_OFF),
                         reinterpret_cast<BF16*>(smem + VT_BUF_1_OFF) };
    BF16*     q_smem  = reinterpret_cast<BF16*>(smem + Q_WGMMA_OFF_V5);
    auto& pipeline_k_storage  = *reinterpret_cast<PipelineK::SharedStorage*>(smem + PIPELINE_K_OFF);
    auto& pipeline_vt_storage = *reinterpret_cast<PipelineVt::SharedStorage*>(smem + PIPELINE_VT_OFF);

    float inv_sqrt = 1.0f / sqrtf((float)HEAD_DIM);
    int num_kv_blocks = (ctx_len + BLOCK_N - 1) / BLOCK_N;

    for (int tile_idx = blockIdx.x; tile_idx < total_tiles; tile_idx += gridDim.x) {
        int head_batch_idx = tile_idx / nq;
        int m_block        = tile_idx % nq;
        int q_block_idx    = (nq - 1) - m_block;

        int batch_idx   = head_batch_idx / num_heads;
        int head_idx    = head_batch_idx % num_heads;
        int kv_head_idx = head_idx / (num_heads / num_kv_heads);

        int q_block_start = q_block_idx * BLOCK_M_PREFILL;
        const __nv_bfloat16* q_ptr = Q + ((batch_idx * num_heads + head_idx) * seq_len) * HEAD_DIM;
        __nv_bfloat16* o_ptr = O + ((batch_idx * num_heads + head_idx) * seq_len) * HEAD_DIM;
        const int kv_row_base = (batch_idx * num_kv_heads + kv_head_idx) * ctx_len;

        int consumer_thread_idx = 0;
        int consumer_warp_id = 0;
        int warp_q_start = 0;
        int warp_q_end = 0;
        int actual_rows = 0;
        if (wg_idx == CONSUMER_WG_IDX) {
            consumer_thread_idx = tid - MMA_THREAD_OFFSET;
            consumer_warp_id = warp_id - CONSUMER_WARP_BASE;
            warp_q_start = q_block_start + consumer_warp_id * MMA_M;
            warp_q_end = min(warp_q_start + MMA_M, seq_len);
            actual_rows = max(0, warp_q_end - warp_q_start);
        }

        // Construct PipelineK and PipelineVt — constructors initialize full+empty barriers
        typename PipelineK::Params pk_params;
        pk_params.role = (wg_idx == PRODUCER_WG_IDX)
            ? PipelineK::ThreadCategory::Producer
            : PipelineK::ThreadCategory::Consumer;
        pk_params.is_leader = (wg_idx == PRODUCER_WG_IDX && warp_id == 0 && lane_id == 0);
        pk_params.transaction_bytes = (uint32_t)K_STAGE_BYTES;
        pk_params.num_consumers = cutlass::NumThreadsPerWarpGroup;
        PipelineK pipeline_k(pipeline_k_storage, pk_params, Shape<_1,_1,_1>{});

        typename PipelineVt::Params pvt_params;
        pvt_params.role = (wg_idx == PRODUCER_WG_IDX)
            ? PipelineVt::ThreadCategory::Producer
            : PipelineVt::ThreadCategory::Consumer;
        pvt_params.is_leader = (wg_idx == PRODUCER_WG_IDX && warp_id == 0 && lane_id == 0);
        pvt_params.transaction_bytes = (uint32_t)VT_STAGE_BYTES;
        pvt_params.num_consumers = cutlass::NumThreadsPerWarpGroup;
        PipelineVt pipeline_vt(pipeline_vt_storage, pvt_params, Shape<_1,_1,_1>{});
        __syncthreads();  // ensure all barriers visible before use

        {
            Tensor sQ = make_tensor(make_smem_ptr(q_smem), SmemLayoutQ{});
            const int total = BLOCK_M_PREFILL * HEAD_DIM;
            for (int i = tid; i < total; i += BLOCK_THREADS_WS) {
                int m = i / HEAD_DIM;
                int k = i % HEAD_DIM;
                int global_row = q_block_start + m;
                sQ(m, k) = (global_row < seq_len) ? BF16(q_ptr[global_row * HEAD_DIM + k]) : BF16(0.0f);
            }
        }
        __syncthreads();

        float acc_o[OUT_N_TILES][4];
        float row_max[2] = {-INFINITY, -INFINITY};
        float row_sum[2] = {0.0f, 0.0f};
        if (wg_idx == CONSUMER_WG_IDX) {
            #pragma unroll
            for (int nt = 0; nt < OUT_N_TILES; nt++) {
                acc_o[nt][0] = 0.0f;
                acc_o[nt][1] = 0.0f;
                acc_o[nt][2] = 0.0f;
                acc_o[nt][3] = 0.0f;
            }
        }

        // Per-slot use counter for Vt raw mbarrier phase
        int vt_use[2] = {0, 0};
        PipelineStateWS write_state = cutlass::make_producer_start_state<PipelineK>();
        PipelineStateWS read_state  = {};
        PipelineStateWS write_state_vt = cutlass::make_producer_start_state<PipelineVt>();
        PipelineStateWS read_state_vt  = {};

        for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
            int kv_start = kv_block * BLOCK_N;
            int kv_end   = min(kv_start + BLOCK_N, ctx_len);
            int valid_n  = kv_end - kv_start;
            int buf      = kv_block & 1;  // ping-pong slot (same index for K and Vt)

            // WG0: acquire K slot (wait for consumer to release), issue K TMA, issue Vt TMA
            if (wg_idx == PRODUCER_WG_IDX) {
                // PipelineK: wait for empty slot, then announce TMA bytes
                pipeline_k.producer_acquire(write_state);
                if (warp_id == 0 && lane_id == 0) {
                    // TMA K → k_bufs[write_state.index()] using pipeline barrier
                    Tensor mK = tma_k.get_tma_tensor(make_shape(total_kv_rows, Int<HEAD_DIM>{}));
                    Tensor sK = make_tensor(make_smem_ptr(k_bufs[write_state.index()]), SmemLayoutKW{});
                    Tensor mK_off = domain_offset(make_coord(kv_row_base + kv_start, 0), mK);
                    Tensor gK = local_tile(mK_off, Shape<Int<BLOCK_N>, Int<HEAD_DIM>>{}, make_coord(0, 0));
                    auto [tKgK, tKsK] = tma_partition(tma_k, Int<0>{}, Layout<_1>{},
                                                      group_modes<0,2>(sK), group_modes<0,2>(gK));
                    // TMA signals pipeline_k's full_barrier automatically on completion
                    copy(tma_k.with(*pipeline_k.producer_get_barrier(write_state)), tKgK, tKsK);
                }
                pipeline_k.producer_commit(write_state, (uint32_t)K_STAGE_BYTES);
                ++write_state;

                // PipelineVt: acquire Vt slot, issue Vt TMA
                pipeline_vt.producer_acquire(write_state_vt);
                if (warp_id == 0 && lane_id == 0) {
                    Tensor mVt = tma_vt.get_tma_tensor(make_shape(Int<HEAD_DIM>{}, total_kv_rows));
                    Tensor sVt = make_tensor(make_smem_ptr(vt_bufs[write_state_vt.index()]), SmemLayoutVt{});
                    Tensor mVt_off = domain_offset(make_coord(0, kv_row_base + kv_start), mVt);
                    Tensor gVt = local_tile(mVt_off, Shape<Int<HEAD_DIM>, Int<BLOCK_N>>{}, make_coord(0, 0));
                    auto [tVgVt, tVsVt] = tma_partition(tma_vt, Int<0>{}, Layout<_1>{},
                                                        group_modes<0,2>(sVt), group_modes<0,2>(gVt));
                    copy(tma_vt.with(*pipeline_vt.producer_get_barrier(write_state_vt)), tVgVt, tVsVt);
                }
                pipeline_vt.producer_commit(write_state_vt, (uint32_t)VT_STAGE_BYTES);
                ++write_state_vt;
            }
            __syncthreads();  // sync A: both TMA issued, WG1 can start waiting

            // WG1: wait K (pipeline) → QK+softmax → release K → wait Vt (raw) → PV
            float acc_s[N_TILES][4];
            if (wg_idx == CONSUMER_WG_IDX) {
                // PipelineK: wait for full_barrier (TMA completed)
                pipeline_k.consumer_wait(read_state);
                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");

                compute_qk_cute_ws(q_smem, k_bufs[read_state.index()], acc_s, consumer_thread_idx);
                if (actual_rows > 0) {
                    softmax_update_reg(acc_s, acc_o, row_max, row_sum,
                                       inv_sqrt, valid_n, kv_start, warp_q_start, actual_rows);
                }

                // PipelineK: signal empty_barrier (slot free for producer to reuse)
                pipeline_k.consumer_release(read_state);

                // PipelineVt: wait for Vt data, run PV, release Vt slot
                pipeline_vt.consumer_wait(read_state_vt);
                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");

                if (actual_rows > 0) {
                    compute_pv_cute_ws(acc_s, vt_bufs[read_state_vt.index()], acc_o, consumer_thread_idx);
                }
                pipeline_vt.consumer_release(read_state_vt);

                ++read_state;
                ++read_state_vt;
            }
            __syncthreads();  // sync B: PV done
        }

        // Producer tail: drain both pipelines
        if (wg_idx == PRODUCER_WG_IDX) {
            pipeline_k.producer_tail(write_state);
            pipeline_vt.producer_tail(write_state_vt);
        }

        if (wg_idx == CONSUMER_WG_IDX && actual_rows > 0) {
            int group_id     = lane_id >> 2;
            int tid_in_group = lane_id & 3;
            int row0 = group_id;
            int row1 = group_id + 8;
            int col0 = tid_in_group * 2;
            int col1 = col0 + 1;
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

        __syncthreads();
    }
}

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
    int total_tiles = nq * batch * num_heads;
    int num_sm = 0;
    cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0);

    cudaFuncSetAttribute(unified_attn_prefill_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES_12B_V5);
    unified_attn_prefill_kernel<<<dim3(num_sm), BLOCK_THREADS_WS, SMEM_BYTES_12B_V5>>>(
        (const __nv_bfloat16*)Q.data_ptr(),
        tma_k, tma_vt,
        (__nv_bfloat16*)O.data_ptr(),
        seq_len, ctx_len, num_heads, num_kv_heads, total_kv_rows,
        nq, total_tiles);

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
    m.def("unified_prefill", &unified_prefill, "Unified prefill (12b-v1: 256-thread warp-spec, raw mbarriers)");
    m.def("unified_decode",  &unified_decode,  "Unified decode  (10h: WGMMA QK, vectorized K/V)");
}
