// kernel_iter1 — kBlockM=128 with AtomLayout<2,1,1> for QK WGMMA
//
// Base: kernel_11c (BLOCK_M_PREFILL=64, WGMMA PV + V^T TMA)
// Change: BLOCK_M_PREFILL=128 → halves tile count
//         AtomLayout<2,1,1> for QK WGMMA (FA3 pattern for hdim=128)
//         Explicit 2-atom handling: acc_s atom0/atom1, per-atom softmax+PV
//
// Why: 13a_step1 showed 1.5x FA3 at seq=1024 (vs 2.0x for 11c)
//      but had correctness bug: second atom (rows 64-127) was zeroed.
//      This implementation properly processes both atoms.
//
// Changes vs 11c:
//   1. BLOCK_M_PREFILL=128 (was 64)
//   2. TiledMmaQK: AtomLayout<2,1,1>
//   3. compute_qk_cute_128 outputs acc_s0 (rows 0-63) and acc_s1 (rows 64-127)
//   4. Per-atom softmax_update_reg + compute_pv_cute calls
//   5. Epilogue writes both atoms' rows
//   6. Q_WGMMA_BYTES=32768, SMEM_BYTES_10H~85KB (2 CTAs/SM)
//   7. Decode: Q padded to 128 rows, only atom 0 processed

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
constexpr int BLOCK_M_PREFILL = 128;  // was 64
constexpr int BLOCK_N         = 128;

constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 16;

using namespace cute;
using BF16 = cutlass::bfloat16_t;
using SmemLayoutAtom = decltype(GMMA::Layout_K_SW128_Atom<BF16>{});
using SmemLayoutQ    = decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<BLOCK_M_PREFILL>, Int<HEAD_DIM>>{}));
using SmemLayoutKW   = decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<BLOCK_N>,         Int<HEAD_DIM>>{}));

// TiledMmaQK with AtomLayout<2,1,1> — FA3 pattern for hdim=128 causal
// AtomLayout splits the 128-row output into 2 atoms of 64 rows each
using TiledMmaQK = decltype(make_tiled_mma(
    SM90_64x128x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>{},
    Layout<Shape<Int<2>, _1, _1>>{}));

using TmaK = decltype(make_tma_atom(
    SM90_TMA_LOAD{},
    make_tensor(static_cast<BF16 const*>(nullptr),
                make_shape(1, Int<HEAD_DIM>{}),
                make_stride(Int<HEAD_DIM>{}, Int<1>{})),
    SmemLayoutKW{},
    Shape<Int<BLOCK_N>, Int<HEAD_DIM>>{}));

// V^T smem layout: MN-major SW128 [HEAD_DIM=128, BLOCK_N=128]
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

// WGMMA PV: P[64,128] (K-major, in registers RS) x Vt[128,128] (MN-major, in smem SS)
// Each atom processes 64 rows independently — no AtomLayout needed for PV
using TiledMmaPV = decltype(make_tiled_mma(
    SM90_64x128x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::MN>{}));

constexpr int N_TILES     = BLOCK_N  / MMA_N;   // 16
constexpr int OUT_N_TILES = HEAD_DIM / MMA_N;   // 16

constexpr float LOG2E = 1.442695040888963407359924681001892137426646f;

// Smem layout
constexpr int KV_SMEM_BYTES  = BLOCK_N * HEAD_DIM * 2;  // 32768 B
constexpr int PER_WARP_BYTES = 4864;                     // kept for WARP_BASE calc (backward compat)

constexpr int MBAR_BYTES     = 2 * 8;                   // 16 B
constexpr int K_SMEM_OFF     = 0;
constexpr int MBAR_OFF       = KV_SMEM_BYTES;           // 32768
constexpr int WARP_BASE      = MBAR_OFF + MBAR_BYTES;   // 32784
constexpr int SMEM_BYTES     = WARP_BASE + NUM_WARPS * PER_WARP_BYTES;  // 52256 B

// Q in K-major SW128 layout — now 128 rows × 128 cols = 32 KB
constexpr int Q_WGMMA_BYTES  = BLOCK_M_PREFILL * HEAD_DIM * 2;  // 32768 B
constexpr int Q_WGMMA_OFF    = SMEM_BYTES;                      // 52256
constexpr int SMEM_BYTES_10H = Q_WGMMA_OFF + Q_WGMMA_BYTES;     // 85024 B (~83 KB)

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
// QK^T — WGMMA SS with AtomLayout<2,1,1> for 128-row output
// All 128 threads must call unconditionally.
// Output: acc_s0[N_TILES][4] = rows 0-63, acc_s1[N_TILES][4] = rows 64-127
// ============================================================================
__device__ __forceinline__ void compute_qk_cute_128(
    const BF16* __restrict__ q_smem,
    const BF16* __restrict__ k_wgmma,
    float acc_s0[N_TILES][4],
    float acc_s1[N_TILES][4]
) {
    TiledMmaQK tiled_mma;
    ThrMMA thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
    Tensor sQ = make_tensor(make_smem_ptr(q_smem),  SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(k_wgmma), SmemLayoutKW{});
    Tensor tCsQ = thr_mma.partition_A(sQ);
    Tensor tCsK = thr_mma.partition_B(sK);
    auto tCrQ = thr_mma.make_fragment_A(tCsQ);
    auto tCrK = thr_mma.make_fragment_B(tCsK);
    // tCrC shape: ((2,2,C<16>), 2, 1) = 128 elements per thread
    // mode 0: 64 within-atom elements, mode 1: atom index (0 or 1)
    auto tCrC = partition_fragment_C(tiled_mma, Shape<Int<128>, Int<128>>{});
    clear(tCrC);
    warpgroup_arrive();
    gemm(tiled_mma, tCrQ, tCrK, tCrC);
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    #pragma unroll
    for (int nt = 0; nt < N_TILES; nt++) {
        // Atom 0 (rows 0-63): tCrC(idx, 0, 0)
        acc_s0[nt][0] = tCrC(nt*4+0, 0, 0);
        acc_s0[nt][1] = tCrC(nt*4+1, 0, 0);
        acc_s0[nt][2] = tCrC(nt*4+2, 0, 0);
        acc_s0[nt][3] = tCrC(nt*4+3, 0, 0);
        // Atom 1 (rows 64-127): tCrC(idx, 1, 0)
        acc_s1[nt][0] = tCrC(nt*4+0, 1, 0);
        acc_s1[nt][1] = tCrC(nt*4+1, 1, 0);
        acc_s1[nt][2] = tCrC(nt*4+2, 1, 0);
        acc_s1[nt][3] = tCrC(nt*4+3, 1, 0);
    }
}

// ============================================================================
// Softmax — register-resident (unchanged, operates on single atom's [N_TILES][4])
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
// convert_layout_acc_Aregs — copied from FA3 utils.h
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
// WGMMA PV — FA3-style RS WGMMA (single atom: 64 rows × 128 cols)
// Uses TiledMmaPV (64x128 RS) with no AtomLayout.
// acc_s[N_TILES][4]: softmax probabilities (one atom)
// vt_smem: V^T in MN-major SW128 [HEAD_DIM=128, BLOCK_N=128]
// acc_o[OUT_N_TILES][4]: O accumulator (loaded before gemm)
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

    // Single atom: 64 rows (not BLOCK_M_PREFILL=128)
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
// Prefill kernel — kBlockM=128 with 2-atom handling
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

    BF16* k_wgmma = reinterpret_cast<BF16*>(smem + K_SMEM_OFF);
    BF16* vt_smem = reinterpret_cast<BF16*>(smem + K_SMEM_OFF);
    BF16* q_smem  = reinterpret_cast<BF16*>(smem + Q_WGMMA_OFF);
    uint64_t* mbar = reinterpret_cast<uint64_t*>(smem + MBAR_OFF);
    const int kv_row_base = (batch_idx * num_kv_heads + kv_head_idx) * ctx_len;

    // Atom 0 rows: [q_block_start + warp_id*16, q_block_start + warp_id*16 + 15]
    int warp_q_start = q_block_start + warp_id * MMA_M;
    int warp_q_end   = min(warp_q_start + MMA_M, seq_len);
    int actual_rows  = max(0, warp_q_end - warp_q_start);

    // Atom 1 rows: [q_block_start + 64 + warp_id*16, q_block_start + 64 + warp_id*16 + 15]
    int warp_q_start1 = q_block_start + 64 + warp_id * MMA_M;
    int warp_q_end1   = min(warp_q_start1 + MMA_M, seq_len);
    int actual_rows1  = max(0, warp_q_end1 - warp_q_start1);

    // Producer (warp 0) initializes mbarriers
    if (warp_id == 0 && lane_id == 0) {
        mbar_init(&mbar[0], 1);
        mbar_init(&mbar[1], 1);
    }
    __syncthreads();

    // All threads load Q into K-major SW128 smem (cooperative, 128 rows)
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

    // Per-atom state
    float acc_o0[OUT_N_TILES][4];
    float acc_o1[OUT_N_TILES][4];
    float row_max0[2] = {-INFINITY, -INFINITY};
    float row_max1[2] = {-INFINITY, -INFINITY};
    float row_sum0[2] = {0.0f, 0.0f};
    float row_sum1[2] = {0.0f, 0.0f};
    #pragma unroll
    for (int nt = 0; nt < OUT_N_TILES; nt++) {
        acc_o0[nt][0] = acc_o0[nt][1] = acc_o0[nt][2] = acc_o0[nt][3] = 0.0f;
        acc_o1[nt][0] = acc_o1[nt][1] = acc_o1[nt][2] = acc_o1[nt][3] = 0.0f;
    }

    float inv_sqrt = 1.0f / sqrtf((float)HEAD_DIM);
    int num_kv_blocks = (ctx_len + BLOCK_N - 1) / BLOCK_N;

    for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        int kv_start = kv_block * BLOCK_N;
        int kv_end   = min(kv_start + BLOCK_N, ctx_len);
        int valid_n  = kv_end - kv_start;

        // ── PRODUCER role (warp 0 lane 0): issue TMA K load ──
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

        // ── ALL WARPS: WGMMA QK with AtomLayout<2,1,1> (128-row output) ──
        float acc_s0[N_TILES][4], acc_s1[N_TILES][4];
        compute_qk_cute_128(q_smem, k_wgmma, acc_s0, acc_s1);

        // Atom 0 softmax (rows 0-63)
        if (actual_rows > 0) {
            softmax_update_reg(acc_s0, acc_o0, row_max0, row_sum0,
                               inv_sqrt, valid_n, kv_start, warp_q_start, actual_rows);
        }
        // Atom 1 softmax (rows 64-127)
        if (actual_rows1 > 0) {
            softmax_update_reg(acc_s1, acc_o1, row_max1, row_sum1,
                               inv_sqrt, valid_n, kv_start, warp_q_start1, actual_rows1);
        }
        __syncthreads();

        // ── PRODUCER role (warp 0 lane 0): issue TMA V^T load ──
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

        // ── ALL WARPS: WGMMA PV for each atom ──
        if (actual_rows > 0)
            compute_pv_cute(acc_s0, vt_smem, acc_o0);
        if (actual_rows1 > 0)
            compute_pv_cute(acc_s1, vt_smem, acc_o1);
        __syncthreads();
    }

    // ── Output epilogue: write both atoms ──
    int group_id     = lane_id >> 2;
    int tid_in_group = lane_id & 3;
    int row0 = group_id, row1 = group_id + 8;
    int col0 = tid_in_group * 2, col1 = col0 + 1;

    // Atom 0 (rows 0-63)
    if (actual_rows > 0) {
        float inv0 = (row_sum0[0] > 0.0f) ? (1.0f / row_sum0[0]) : 0.0f;
        float inv1 = (row_sum0[1] > 0.0f) ? (1.0f / row_sum0[1]) : 0.0f;
        #pragma unroll
        for (int nt = 0; nt < OUT_N_TILES; nt++) {
            int n_base = nt * MMA_N;
            if (row0 < actual_rows) {
                o_ptr[(warp_q_start + row0) * HEAD_DIM + n_base + col0] =
                    __float2bfloat16(acc_o0[nt][0] * inv0);
                o_ptr[(warp_q_start + row0) * HEAD_DIM + n_base + col1] =
                    __float2bfloat16(acc_o0[nt][1] * inv0);
            }
            if (row1 < actual_rows) {
                o_ptr[(warp_q_start + row1) * HEAD_DIM + n_base + col0] =
                    __float2bfloat16(acc_o0[nt][2] * inv1);
                o_ptr[(warp_q_start + row1) * HEAD_DIM + n_base + col1] =
                    __float2bfloat16(acc_o0[nt][3] * inv1);
            }
        }
    }

    // Atom 1 (rows 64-127)
    if (actual_rows1 > 0) {
        float inv0_1 = (row_sum1[0] > 0.0f) ? (1.0f / row_sum1[0]) : 0.0f;
        float inv1_1 = (row_sum1[1] > 0.0f) ? (1.0f / row_sum1[1]) : 0.0f;
        #pragma unroll
        for (int nt = 0; nt < OUT_N_TILES; nt++) {
            int n_base = nt * MMA_N;
            if (row0 < actual_rows1) {
                o_ptr[(warp_q_start1 + row0) * HEAD_DIM + n_base + col0] =
                    __float2bfloat16(acc_o1[nt][0] * inv0_1);
                o_ptr[(warp_q_start1 + row0) * HEAD_DIM + n_base + col1] =
                    __float2bfloat16(acc_o1[nt][1] * inv0_1);
            }
            if (row1 < actual_rows1) {
                o_ptr[(warp_q_start1 + row1) * HEAD_DIM + n_base + col0] =
                    __float2bfloat16(acc_o1[nt][2] * inv1_1);
                o_ptr[(warp_q_start1 + row1) * HEAD_DIM + n_base + col1] =
                    __float2bfloat16(acc_o1[nt][3] * inv1_1);
            }
        }
    }
}

// ============================================================================
// Decode kernel — kBlockM=128
// Q loaded as 128 rows (row 0 = query, rest = 0)
// Only atom 0 processed (row 0 is in atom 0)
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

    // Load Q into K-major SW128 smem (128 rows, row 0 = query, rest = 0)
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

        // WGMMA QK — all 128 threads, produces 2 atoms
        float acc_s0[N_TILES][4], acc_s1[N_TILES][4];
        compute_qk_cute_128(q_smem, k_wgmma, acc_s0, acc_s1);
        __syncthreads();

        // Issue V^T TMA after QK finishes consuming K smem
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

        // Only process atom 0 (row 0 is in atom 0, rows 0-63 of 128-row tile)
        if (warp_id == 0) {
            softmax_update_reg(acc_s0, acc_o0, row_max0, row_sum0,
                               inv_sqrt, valid_n, kv_start, ctx_len - 1, 1);
        }
        {
            mbar_wait(&mbar[1], vphase);
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        }
        __syncthreads();

        compute_pv_cute(acc_s0, vt_smem, acc_o0);
        __syncthreads();
    }

    // Broadcast row_sum0[0] from warp 0 to all warps
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
        int row0_w = warp_row_base + group_id;
        int col0 = tid_in_group * 2, col1 = col0 + 1;
        float inv0 = (shared_row_sum > 0.0f) ? (1.0f / shared_row_sum) : 0.0f;

        #pragma unroll
        for (int nt = 0; nt < OUT_N_TILES; nt++) {
            int n_base = nt * MMA_N;
            if (row0_w == 0) {
                o_ptr[n_base + col0] = __float2bfloat16(acc_o0[nt][0] * inv0);
                o_ptr[n_base + col1] = __float2bfloat16(acc_o0[nt][1] * inv0);
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
    m.def("unified_prefill", &unified_prefill, "Unified prefill (kBlockM=128, 2-atom WGMMA QK)");
    m.def("unified_decode",  &unified_decode,  "Unified decode (kBlockM=128, padded Q)");
}
