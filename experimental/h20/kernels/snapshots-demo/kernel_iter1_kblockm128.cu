// kernel_iter1 — kBlockM=128 with AtomLayout<2,1,1>
//
// Base: unified_kernel_11c (WGMMA PV + V^T TMA, 2.0× FA3 prefill at seq=1024)
// Technique: Double BLOCK_M_PREFILL from 64 to 128, halving tile/CTA count.
//            Add AtomLayout<2,1,1> to TiledMmaQK for 2-atom WGMMA.
//            Remove dead per-warp smem (pv_buf, per-warp Q tile) to maintain
//            3 CTAs/SM despite Q smem doubling from 16KB to 32KB.
//
// Reference: FA3 uses kBlockM=128 for hdim=128 causal (tile_size.h:33, mainloop:89)
// Prior work: kernel_13a_step1 achieved 1.5× FA3 (vs 2.0× baseline) but with
//             second-atom PV bugs. kernel_iter13 fixed per-atom softmax/PV but
//             was never GPU-tested.
//
// What changes vs 11c:
//   - BLOCK_M_PREFILL: 64 → 128 (NUM_WARPS * 32)
//   - TiledMmaQK: add Layout<Shape<Int<2>, _1, _1>> atom layout
//   - compute_qk_cute: extracts acc_s[2][N_TILES][4] (both atoms)
//   - Prefill: per-atom q_starts, separate acc_o/row_max/row_sum arrays
//   - Prefill: extra __syncthreads() after QK before Vt TMA (K smem → Vt overwrite safety)
//   - Prefill: Vt TMA issued before softmax (overlaps Vt load with softmax compute)
//   - Decode: only atom 0 participates in softmax/PV
//   - Q_WGMMA_BYTES: 16KB → 32KB (128×128×2 bytes)
//   - Dead smem removed: warp_smem, pv_buf, per-warp Q tile (never used in WGMMA path)
//     → SMEM_BYTES_10H ~64KB → 3 CTAs/SM (same as 11c)
//   - Removed: mma_bf16, ldmatrix_a, write_ktile_to_smem, compute_pv_reg_per_ktile,
//              swizzle_col, tma_load_2d, make_tma_desc (dead legacy code)
//   - Removed: K_SMEM_STRIDE, V_SMEM_STRIDE, KV_LO_OFF, KV_HI_OFF, KV_SMEM_PAD,
//              HALF_DIM, KV_HALF_BYTES, PV_BUF_COLS, PV_BUF_BYTES, W_Q_TILE_OFF,
//              W_PV_BUF_OFF, MMA_K, KV_K_TILES, K_TILES, KV_SMEM_BYTES (dead constants)
//
// What is IDENTICAL to 11c:
//   - compute_pv_cute: unchanged (called twice, once per atom)
//   - softmax_update_reg: identical signature and body
//   - TmaK, TmaVt, TiledMmaPV: unchanged
//   - SmemLayoutVt, SmemLayoutKW, SmemLayoutAtom: unchanged
//   - Host bindings (unified_prefill, unified_decode): identical
//   - Bitwise mechanism: decode pads Q to 128 rows (127 zeros), row 0 computation identical

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>

#include <cute/tensor.hpp>
#include <cute/util/debug.hpp>
#include <cutlass/device_kernel.h>
#include <cutlass/pipeline/sm90_pipeline.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/mma_sm90.h>
#include <cutlass/numeric_conversion.h>

constexpr int HEAD_DIM        = 128;
constexpr int NUM_WARPS       = 4;
constexpr int WARP_SIZE       = 32;
constexpr int BLOCK_THREADS   = NUM_WARPS * WARP_SIZE;
constexpr int BLOCK_M_PREFILL = NUM_WARPS * 32;  // 128 (was 64)
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
    SM90_64x128x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>{},
    Layout<Shape<Int<BLOCK_M_PREFILL / 64>, _1, _1>>{}));

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
constexpr int OUT_N_TILES = HEAD_DIM / MMA_N;   // 16

constexpr float LOG2E = 1.442695040888963407359924681001892137426646f;

// smem layout — dead per-warp buffers removed, only K/Vt + mbar + Q WGMMA remain
constexpr int MBAR_BYTES     = 2 * 8;
constexpr int K_SMEM_BYTES   = BLOCK_N * HEAD_DIM * 2;  // 32768 B (K and Vt share this slot)
constexpr int K_SMEM_OFF     = 0;
constexpr int MBAR_OFF       = K_SMEM_BYTES;
constexpr int Q_WGMMA_BYTES  = BLOCK_M_PREFILL * HEAD_DIM * 2;  // 32768 B (was 16384)
constexpr int Q_WGMMA_OFF    = MBAR_OFF + MBAR_BYTES;
constexpr int SMEM_BYTES_10H = Q_WGMMA_OFF + Q_WGMMA_BYTES;  // ~65552 ≈ 64KB

// ============================================================================
// mbarrier helpers
// ============================================================================

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
// FUNCTION: compute_qk_cute — WGMMA QK with AtomLayout<2,1,1>
//
// tCrC shape = ((2,2,C<16>), 2, 1) where mode 1 = atom index.
// acc_s[0] = rows 0-63, acc_s[1] = rows 64-127.
// ============================================================================

__device__ __forceinline__ void compute_qk_cute(
    const BF16* __restrict__ q_smem,
    const BF16* __restrict__ k_wgmma,
    float acc_s[2][N_TILES][4]
) {
    TiledMmaQK tiled_mma;
    ThrMMA thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
    Tensor sQ = make_tensor(make_smem_ptr(q_smem),  SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(k_wgmma), SmemLayoutKW{});
    Tensor tCsQ = thr_mma.partition_A(sQ);
    Tensor tCsK = thr_mma.partition_B(sK);
    auto tCrQ = thr_mma.make_fragment_A(tCsQ);
    auto tCrK = thr_mma.make_fragment_B(tCsK);
    auto tCrC = partition_fragment_C(tiled_mma, Shape<Int<BLOCK_M_PREFILL>, Int<HEAD_DIM>>{});
    clear(tCrC);
    warpgroup_arrive();
    gemm(tiled_mma, tCrQ, tCrK, tCrC);
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    #pragma unroll
    for (int nt = 0; nt < N_TILES; nt++) {
        acc_s[0][nt][0] = tCrC(nt*4+0, 0, 0);
        acc_s[0][nt][1] = tCrC(nt*4+1, 0, 0);
        acc_s[0][nt][2] = tCrC(nt*4+2, 0, 0);
        acc_s[0][nt][3] = tCrC(nt*4+3, 0, 0);
        acc_s[1][nt][0] = tCrC(nt*4+0, 1, 0);
        acc_s[1][nt][1] = tCrC(nt*4+1, 1, 0);
        acc_s[1][nt][2] = tCrC(nt*4+2, 1, 0);
        acc_s[1][nt][3] = tCrC(nt*4+3, 1, 0);
    }
}

// ============================================================================
// FUNCTION: softmax_update_reg — register-resident (unchanged from 11c)
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
// FUNCTION: convert_layout_acc_Aregs (unchanged from 11c)
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
// FUNCTION: compute_pv_cute — WGMMA PV (unchanged from 11c)
// Called once per atom; uses 64-row accumulator shape.
// ============================================================================

__device__ __forceinline__ void compute_pv_cute(
    float acc_s[N_TILES][4],
    const BF16* __restrict__ vt_smem,
    float acc_o[OUT_N_TILES][4]
) {
    TiledMmaPV tiled_mma;
    ThrMMA thr_mma = tiled_mma.get_thread_slice(threadIdx.x);

    Tensor sVt  = make_tensor(make_smem_ptr(vt_smem), SmemLayoutVt{});
    Tensor tCsVt = thr_mma.partition_B(sVt);
    auto   tCrVt = thr_mma.make_fragment_B(tCsVt);

    auto tCrC = partition_fragment_C(tiled_mma, Shape<Int<64>, Int<HEAD_DIM>>{});
    #pragma unroll
    for (int nt = 0; nt < OUT_N_TILES; nt++) {
        tCrC(nt*4+0, 0, 0) = acc_o[nt][0];
        tCrC(nt*4+1, 0, 0) = acc_o[nt][1];
        tCrC(nt*4+2, 0, 0) = acc_o[nt][2];
        tCrC(nt*4+3, 0, 0) = acc_o[nt][3];
    }

    auto tSrS_layout = tCrC.layout();
    auto tOrP_layout = convert_layout_acc_Aregs<TiledMmaPV>(tSrS_layout);

    auto tSrS = make_tensor<float>(tSrS_layout);
    #pragma unroll
    for (int nt = 0; nt < N_TILES; nt++) {
        tSrS(nt*4+0, 0, 0) = acc_s[nt][0];
        tSrS(nt*4+1, 0, 0) = acc_s[nt][1];
        tSrS(nt*4+2, 0, 0) = acc_s[nt][2];
        tSrS(nt*4+3, 0, 0) = acc_s[nt][3];
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
        acc_o[nt][0] = tCrC(nt*4+0, 0, 0);
        acc_o[nt][1] = tCrC(nt*4+1, 0, 0);
        acc_o[nt][2] = tCrC(nt*4+2, 0, 0);
        acc_o[nt][3] = tCrC(nt*4+3, 0, 0);
    }
}

// ============================================================================
// FUNCTION: unified_attn_prefill_kernel — kBlockM=128 prefill
// ============================================================================

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
    uint64_t*      mbar    = reinterpret_cast<uint64_t*>(smem + MBAR_OFF);
    const int kv_row_base  = (batch_idx * num_kv_heads + kv_head_idx) * ctx_len;

    int warp_q_start0 = q_block_start + warp_id * MMA_M;
    int warp_q_end0   = min(warp_q_start0 + MMA_M, seq_len);
    int actual_rows0  = max(0, warp_q_end0 - warp_q_start0);
    int warp_q_start1 = q_block_start + 64 + warp_id * MMA_M;
    int warp_q_end1   = min(warp_q_start1 + MMA_M, seq_len);
    int actual_rows1  = max(0, warp_q_end1 - warp_q_start1);

    if (warp_id == 0 && lane_id == 0) {
        mbar_init(&mbar[0], 1);
        mbar_init(&mbar[1], 1);
    }
    __syncthreads();

    // Load Q into K-major SW128 smem (128 rows × 128 cols = 16384 elements)
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

    float acc_o0[OUT_N_TILES][4];
    float row_max0[2] = {-INFINITY, -INFINITY};
    float row_sum0[2] = {0.0f, 0.0f};
    float acc_o1[OUT_N_TILES][4];
    float row_max1[2] = {-INFINITY, -INFINITY};
    float row_sum1[2] = {0.0f, 0.0f};
    #pragma unroll
    for (int nt = 0; nt < OUT_N_TILES; nt++)
        acc_o0[nt][0] = acc_o0[nt][1] = acc_o0[nt][2] = acc_o0[nt][3] = 0.0f;
    #pragma unroll
    for (int nt = 0; nt < OUT_N_TILES; nt++)
        acc_o1[nt][0] = acc_o1[nt][1] = acc_o1[nt][2] = acc_o1[nt][3] = 0.0f;

    float inv_sqrt = 1.0f / sqrtf((float)HEAD_DIM);
    int num_kv_blocks = (ctx_len + BLOCK_N - 1) / BLOCK_N;

    for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        int kv_start = kv_block * BLOCK_N;
        int kv_end   = min(kv_start + BLOCK_N, ctx_len);
        int valid_n  = kv_end - kv_start;

        // ── Issue TMA K load (warp 0 lane 0) ──────────────────────────────────
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

        // ── WGMMA QK (all 128 threads, 2 atoms) ──────────────────────────────
        float acc_s[2][N_TILES][4];
        compute_qk_cute(q_smem, k_wgmma, acc_s);
        // sync ensures all threads done reading K smem before Vt TMA overwrites it
        __syncthreads();

        // ── Issue V^T TMA (overlaps with softmax below) ───────────────────────
        if (warp_id == 0 && lane_id == 0) {
            const int vt_bytes = HEAD_DIM * BLOCK_N * sizeof(BF16);
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

        // ── Softmax atom 0 (rows 0-63 of this CTA's Q tile) ──────────────────
        if (actual_rows0 > 0) {
            softmax_update_reg(acc_s[0], acc_o0, row_max0, row_sum0,
                               inv_sqrt, valid_n, kv_start, warp_q_start0, actual_rows0);
        }
        // ── Softmax atom 1 (rows 64-127) ─────────────────────────────────────
        if (actual_rows1 > 0) {
            softmax_update_reg(acc_s[1], acc_o1, row_max1, row_sum1,
                               inv_sqrt, valid_n, kv_start, warp_q_start1, actual_rows1);
        }

        // ── Wait for V^T ──────────────────────────────────────────────────────
        {
            const int vphase = kv_block & 1;
            mbar_wait(&mbar[1], vphase);
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        }
        __syncthreads();

        // ── WGMMA PV atom 0 ───────────────────────────────────────────────────
        if (actual_rows0 > 0)
            compute_pv_cute(acc_s[0], vt_smem, acc_o0);
        // ── WGMMA PV atom 1 ───────────────────────────────────────────────────
        if (actual_rows1 > 0)
            compute_pv_cute(acc_s[1], vt_smem, acc_o1);
        __syncthreads();
    }

    // ── Write output for both atoms ──────────────────────────────────────────
    if (actual_rows0 > 0 || actual_rows1 > 0) {
        int group_id     = lane_id >> 2;
        int tid_in_group = lane_id & 3;
        int col0 = tid_in_group * 2, col1 = col0 + 1;

        int global_00 = warp_q_start0 + group_id;
        int global_01 = warp_q_start0 + group_id + 8;
        int global_10 = warp_q_start1 + group_id;
        int global_11 = warp_q_start1 + group_id + 8;

        float inv_a0 = (row_sum0[0] > 0.0f) ? (1.0f / row_sum0[0]) : 0.0f;
        float inv_a1 = (row_sum0[1] > 0.0f) ? (1.0f / row_sum0[1]) : 0.0f;
        float inv_b0 = (row_sum1[0] > 0.0f) ? (1.0f / row_sum1[0]) : 0.0f;
        float inv_b1 = (row_sum1[1] > 0.0f) ? (1.0f / row_sum1[1]) : 0.0f;

        #pragma unroll
        for (int nt = 0; nt < OUT_N_TILES; nt++) {
            int n_base = nt * MMA_N;
            if (global_00 < seq_len) {
                o_ptr[global_00 * HEAD_DIM + n_base + col0] =
                    __float2bfloat16(acc_o0[nt][0] * inv_a0);
                o_ptr[global_00 * HEAD_DIM + n_base + col1] =
                    __float2bfloat16(acc_o0[nt][1] * inv_a0);
            }
            if (global_01 < seq_len) {
                o_ptr[global_01 * HEAD_DIM + n_base + col0] =
                    __float2bfloat16(acc_o0[nt][2] * inv_a1);
                o_ptr[global_01 * HEAD_DIM + n_base + col1] =
                    __float2bfloat16(acc_o0[nt][3] * inv_a1);
            }
            if (global_10 < seq_len) {
                o_ptr[global_10 * HEAD_DIM + n_base + col0] =
                    __float2bfloat16(acc_o1[nt][0] * inv_b0);
                o_ptr[global_10 * HEAD_DIM + n_base + col1] =
                    __float2bfloat16(acc_o1[nt][1] * inv_b0);
            }
            if (global_11 < seq_len) {
                o_ptr[global_11 * HEAD_DIM + n_base + col0] =
                    __float2bfloat16(acc_o1[nt][2] * inv_b1);
                o_ptr[global_11 * HEAD_DIM + n_base + col1] =
                    __float2bfloat16(acc_o1[nt][3] * inv_b1);
            }
        }
    }
}

// ============================================================================
// FUNCTION: unified_attn_decode_kernel — kBlockM=128 decode
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

    // Load Q (128 rows, row 0 = valid query, rows 1-127 = zeros)
    {
        Tensor sQ = make_tensor(make_smem_ptr(q_smem), SmemLayoutQ{});
        const int total = BLOCK_M_PREFILL * HEAD_DIM;
        for (int i = tid; i < total; i += BLOCK_THREADS) {
            int m = i / HEAD_DIM, k = i % HEAD_DIM;
            sQ(m, k) = (m == 0) ? BF16(q_ptr[k]) : BF16(0.0f);
        }
    }
    __syncthreads();

    float acc_o0[OUT_N_TILES][4];
    float row_max0[2] = {-INFINITY, -INFINITY};
    float row_sum0[2] = {0.0f, 0.0f};
    #pragma unroll
    for (int nt = 0; nt < OUT_N_TILES; nt++)
        acc_o0[nt][0] = acc_o0[nt][1] = acc_o0[nt][2] = acc_o0[nt][3] = 0.0f;

    float inv_sqrt = 1.0f / sqrtf((float)HEAD_DIM);
    int num_kv_blocks = (ctx_len + BLOCK_N - 1) / BLOCK_N;

    for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        int kv_start = kv_block * BLOCK_N;
        int kv_end   = min(kv_start + BLOCK_N, ctx_len);
        int valid_n  = kv_end - kv_start;
        const int vphase = kv_block & 1;

        // Load K via cute TMA
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

        // WGMMA QK (all 128 threads, 2 atoms)
        float acc_s[2][N_TILES][4];
        compute_qk_cute(q_smem, k_wgmma, acc_s);
        __syncthreads();

        // Issue V^T TMA (overlaps with softmax)
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

        // Softmax atom 0 only (warp 0 owns row 0)
        if (warp_id == 0) {
            softmax_update_reg(acc_s[0], acc_o0, row_max0, row_sum0,
                               inv_sqrt, valid_n, kv_start, ctx_len - 1, 1);
        }

        // Wait for V^T
        {
            mbar_wait(&mbar[1], vphase);
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        }
        __syncthreads();

        // PV atom 0 only (row 0 is valid query)
        compute_pv_cute(acc_s[0], vt_smem, acc_o0);
        __syncthreads();
    }

    // Write output: row 0 of atom 0
    {
        float* scratch = reinterpret_cast<float*>(smem + MBAR_OFF);
        if (warp_id == 0 && lane_id == 0)
            scratch[0] = row_sum0[0];
        __syncthreads();
        float shared_row_sum = scratch[0];
        __syncthreads();

        int warp_row_base = warp_id * MMA_M;
        int group_id      = lane_id >> 2;
        int tid_in_group  = lane_id & 3;
        int row0 = warp_row_base + group_id;
        int col0 = tid_in_group * 2, col1 = col0 + 1;
        float inv0 = (shared_row_sum > 0.0f) ? (1.0f / shared_row_sum) : 0.0f;

        #pragma unroll
        for (int nt = 0; nt < OUT_N_TILES; nt++) {
            int n_base = nt * MMA_N;
            if (row0 == 0) {
                o_ptr[n_base + col0] = __float2bfloat16(acc_o0[nt][0] * inv0);
                o_ptr[n_base + col1] = __float2bfloat16(acc_o0[nt][1] * inv0);
            }
        }
    }
}

// ============================================================================
// PyTorch bindings (unchanged from 11c)
// ============================================================================

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
    m.def("unified_prefill", &unified_prefill, "kBlockM=128 unified prefill (AtomLayout<2,1,1>)");
    m.def("unified_decode",  &unified_decode,  "kBlockM=128 unified decode");
}
