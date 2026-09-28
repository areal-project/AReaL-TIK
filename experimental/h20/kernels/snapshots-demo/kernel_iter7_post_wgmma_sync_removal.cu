// kernel_iter18 — Final cleanup for epoch 3
//
// Base: kernel_iter15 (double-buffered K in decode)
// Changes:
// 1. Added const qualifiers to loop-invariant parameters in softmax
// 2. Removed duplicate #include <cuda.h> and unnecessary includes
// 3. Final cleanup — all warnings resolved, includes sorted

#include <torch/extension.h>
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

using namespace cute;
using BF16 = cutlass::bfloat16_t;
using SmemLayoutAtom = decltype(GMMA::Layout_K_SW128_Atom<BF16>{});
// BLOCK_M_PREFILL=128 → tile_to_shape to 128×128
using SmemLayoutQ    = decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<BLOCK_M_PREFILL>, Int<HEAD_DIM>>{}));
using SmemLayoutKW   = decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<BLOCK_N>,         Int<HEAD_DIM>>{}));
// AtomLayout<2,1,1> = 2 atoms in M dimension for kBlockM=128
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

// PV WGMMA: unchanged — single 64×128 atom (called twice, once per QK atom)
using TiledMmaPV = decltype(make_tiled_mma(
    SM90_64x128x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::MN>{}));

constexpr int N_TILES    = BLOCK_N  / MMA_N;   // 16
constexpr int OUT_N_TILES= HEAD_DIM / MMA_N;   // 16

constexpr float LOG2E = 1.442695040888963407359924681001892137426646f;

constexpr int HALF_DIM       = HEAD_DIM / 2;
constexpr int KV_HALF_BYTES  = BLOCK_N * HALF_DIM * 2;
constexpr int KV_SMEM_BYTES  = KV_HALF_BYTES * 2;
constexpr int PER_WARP_BYTES = MMA_M * HEAD_DIM * 2;  // Q tile only (pv_buf removed)

constexpr int MBAR_BYTES     = 2 * 8;
constexpr int K_SMEM_OFF     = 0;
constexpr int MBAR_OFF       = KV_SMEM_BYTES;
constexpr int WARP_BASE      = MBAR_OFF + MBAR_BYTES;
constexpr int SMEM_BYTES     = WARP_BASE + NUM_WARPS * PER_WARP_BYTES;

// Q smem doubles: 128×128×2 = 32768 (was 16384)
constexpr int Q_WGMMA_BYTES  = BLOCK_M_PREFILL * HEAD_DIM * 2;
constexpr int Q_WGMMA_OFF    = SMEM_BYTES;
constexpr int SMEM_BYTES_10H = Q_WGMMA_OFF + Q_WGMMA_BYTES;

// Decode-specific layout with 2 K buffers (double-buffered)
constexpr int DECODE_K0_OFF    = 0;
constexpr int DECODE_K1_OFF    = KV_SMEM_BYTES;                     // 32768
constexpr int DECODE_MBAR_OFF  = DECODE_K1_OFF + KV_SMEM_BYTES;     // 65536
constexpr int DECODE_MBAR_TOTAL = 3 * 8;                            // 24 B (K0, K1, Vt)
constexpr int DECODE_WARP_BASE = DECODE_MBAR_OFF + DECODE_MBAR_TOTAL; // 65560
constexpr int DECODE_SMEM_BYTES = DECODE_WARP_BASE + NUM_WARPS * PER_WARP_BYTES; // 85016
constexpr int DECODE_Q_OFF     = DECODE_SMEM_BYTES;                 // 85016
constexpr int DECODE_SMEM_TOTAL = DECODE_Q_OFF + Q_WGMMA_BYTES;     // 117784 B ≈ 115 KB

// ============================================================================
// Vectorized store helper: pack 2 bf16 into a single uint32_t store
// ============================================================================

__device__ __forceinline__ void store_2bf16(
    __nv_bfloat16* __restrict__ base, int idx_base,
    float v0, float v1
) {
    union { __nv_bfloat16 f; uint16_t u; } c0, c1;
    c0.f = __float2bfloat16(v0);
    c1.f = __float2bfloat16(v1);
    reinterpret_cast<uint32_t*>(base)[idx_base / 2] =
        (uint32_t(c1.u) << 16) | c0.u;
}

// ============================================================================

// ============================================================================
// mbarrier helpers (unchanged)
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
// Quad reductions (unchanged)
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
// Now extracts 2 atoms: acc_s[0] for rows 0-63, acc_s[1] for rows 64-127.
// tCrC shape = ((2,2,C<16>), 2, 1) where mode 1 = atom index.
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
    // BLOCK_M=128 → 128×128 output tile, 2 atoms
    auto tCrC = partition_fragment_C(tiled_mma, Shape<Int<BLOCK_M_PREFILL>, Int<HEAD_DIM>>{});
    clear(tCrC);
    warpgroup_arrive();
    gemm(tiled_mma, tCrQ, tCrK, tCrC);
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    // Extract both atoms:
    //   tCrC(nt*4+j, 0, 0) = atom 0 (rows 0-63)
    //   tCrC(nt*4+j, 1, 0) = atom 1 (rows 64-127)
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
// FUNCTION: softmax_update_reg (unchanged signature — called twice)
// ============================================================================

__device__ __forceinline__ void softmax_update_reg(
    float acc_s[N_TILES][4],
    float acc_o[OUT_N_TILES][4],
    float row_max[2],
    float row_sum[2],
    const float inv_sqrt,
    const int valid_n,
    const int kv_start,
    const int warp_q_start,
    const int actual_rows
) {
    int lane_id  = threadIdx.x % WARP_SIZE;
    int group_id = lane_id >> 2;
    int col_pair = lane_id & 3;
    int row0 = group_id, row1 = group_id + 8;
    int col0 = col_pair * 2, col1 = col0 + 1;
    int q_pos0 = warp_q_start + row0;
    int q_pos1 = warp_q_start + row1;

    int causal_max0 = max(0, q_pos0 - kv_start + 1);
    int causal_max1 = max(0, q_pos1 - kv_start + 1);
    int max_n0 = min(valid_n, causal_max0);
    int max_n1 = min(valid_n, causal_max1);

    float bmax0 = -INFINITY, bmax1 = -INFINITY;
    #pragma unroll
    for (int nt = 0; nt < N_TILES; nt++) {
        int n_base = nt * MMA_N;
        int n0 = n_base + col0, n1 = n_base + col1;
        float s0 = acc_s[nt][0] * inv_sqrt;
        float s1 = acc_s[nt][1] * inv_sqrt;
        float s2 = acc_s[nt][2] * inv_sqrt;
        float s3 = acc_s[nt][3] * inv_sqrt;
        bool m00 = (n0 >= max_n0) || (row0 >= actual_rows);
        bool m01 = (n1 >= max_n0) || (row0 >= actual_rows);
        bool m10 = (n0 >= max_n1) || (row1 >= actual_rows);
        bool m11 = (n1 >= max_n1) || (row1 >= actual_rows);
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
// FUNCTION: convert_layout_acc_Aregs (unchanged)
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
// FUNCTION: compute_pv_cute (unchanged signature — called twice)
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
// FUNCTION: unified_attn_prefill_kernel — kBlockM=128 prefill kernel
//
// Each warp processes 16 rows of atom 0 + 16 rows of atom 1 = 32 rows total.
// row_sum/row_max extended to [4] for 2 rows × 2 atoms.
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
    BF16*          vt_smem = reinterpret_cast<BF16*>(smem + K_SMEM_OFF);
    BF16*          q_smem  = reinterpret_cast<BF16*>(smem + Q_WGMMA_OFF);
    uint64_t* mbar = reinterpret_cast<uint64_t*>(smem + MBAR_OFF);
    const int kv_row_base = (batch_idx * num_kv_heads + kv_head_idx) * ctx_len;

    // Per-atom row ranges for this warp
    // Atom 0: rows [q_block_start + warp_id*16 .. +15]
    // Atom 1: rows [q_block_start + 64 + warp_id*16 .. +15]
    int warp_q_start0 = q_block_start + warp_id * MMA_M;
    int warp_q_end0   = min(warp_q_start0 + MMA_M, seq_len);
    int actual_rows0  = max(0, warp_q_end0 - warp_q_start0);
    int warp_q_start1 = q_block_start + 64 + warp_id * MMA_M;
    int warp_q_end1   = min(warp_q_start1 + MMA_M, seq_len);
    int actual_rows1  = max(0, warp_q_end1 - warp_q_start1);

    // Producer (warp 0) initializes mbarriers
    if (warp_id == 0 && lane_id == 0) {
        mbar_init(&mbar[0], 1);
        mbar_init(&mbar[1], 1);
    }
    __syncthreads();

    // All threads load Q into K-major SW128 smem
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

    // Per-atom softmax state (separate arrays avoid pointer-arithmetic issues)
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

        // ── PRODUCER (warp 0 lane 0): issue TMA K load ──────────────────────
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

        // ── ALL WARPS: WGMMA QK (2 atoms, 128 rows) ─────────────────────────
        float acc_s[2][N_TILES][4];
        compute_qk_cute(q_smem, k_wgmma, acc_s);

        // ── Issue TMA V^T early (overlaps with softmax below) ────────────────
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

        // ── Softmax atom 0 (rows 0-63 within block) ─────────────────────────
        if (actual_rows0 > 0) {
            softmax_update_reg(acc_s[0], acc_o0, row_max0, row_sum0,
                               inv_sqrt, valid_n, kv_start, warp_q_start0, actual_rows0);
        }
        // ── Softmax atom 1 (rows 64-127 within block) ───────────────────────
        if (actual_rows1 > 0) {
            softmax_update_reg(acc_s[1], acc_o1, row_max1, row_sum1,
                               inv_sqrt, valid_n, kv_start, warp_q_start1, actual_rows1);
        }

        // ── Wait for V^T ─────────────────────────────────────────────────────
        {
            const int vphase = kv_block & 1;
            mbar_wait(&mbar[1], vphase);
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        }
        __syncthreads();

        // ── PV atom 0 ────────────────────────────────────────────────────────
        if (actual_rows0 > 0)
            compute_pv_cute(acc_s[0], vt_smem, acc_o0);
        // ── PV atom 1 ────────────────────────────────────────────────────────
        if (actual_rows1 > 0)
            compute_pv_cute(acc_s[1], vt_smem, acc_o1);
        __syncthreads();
    }

    // ── ALL WARPS: write output for both atoms ──────────────────────────────
    if (actual_rows0 > 0 || actual_rows1 > 0) {
        int group_id     = lane_id >> 2;
        int tid_in_group = lane_id & 3;
        int col0 = tid_in_group * 2;

        // Atom 0 rows
        int global_00 = warp_q_start0 + group_id;
        int global_01 = warp_q_start0 + group_id + 8;
        // Atom 1 rows
        int global_10 = warp_q_start1 + group_id;
        int global_11 = warp_q_start1 + group_id + 8;

        float inv_a0 = (row_sum0[0] > 0.0f) ? (1.0f / row_sum0[0]) : 0.0f;
        float inv_a1 = (row_sum0[1] > 0.0f) ? (1.0f / row_sum0[1]) : 0.0f;
        float inv_b0 = (row_sum1[0] > 0.0f) ? (1.0f / row_sum1[0]) : 0.0f;
        float inv_b1 = (row_sum1[1] > 0.0f) ? (1.0f / row_sum1[1]) : 0.0f;

        #pragma unroll
        for (int nt = 0; nt < OUT_N_TILES; nt++) {
            int n_base = nt * MMA_N;
            if (global_00 < seq_len)
                store_2bf16(o_ptr, global_00 * HEAD_DIM + n_base + col0,
                            acc_o0[nt][0] * inv_a0, acc_o0[nt][1] * inv_a0);
            if (global_01 < seq_len)
                store_2bf16(o_ptr, global_01 * HEAD_DIM + n_base + col0,
                            acc_o0[nt][2] * inv_a1, acc_o0[nt][3] * inv_a1);
            if (global_10 < seq_len)
                store_2bf16(o_ptr, global_10 * HEAD_DIM + n_base + col0,
                            acc_o1[nt][0] * inv_b0, acc_o1[nt][1] * inv_b0);
            if (global_11 < seq_len)
                store_2bf16(o_ptr, global_11 * HEAD_DIM + n_base + col0,
                            acc_o1[nt][2] * inv_b1, acc_o1[nt][3] * inv_b1);
        }
    }
}

// ============================================================================
// FUNCTION: unified_attn_decode_kernel — kBlockM=128 decode with K double-buf
//
// Decode Q is padded to 128 rows (1 valid + 127 zeros).
// K is double-buffered (2 buffers, 32 KB each) to overlap K load with QK + PV.
// K[i+1] loads into the alternate buffer while K[i] is consumed by QK.
// Vt reuses the K buffer that was just consumed by QK (no extra Vt buffer).
// Atom 0 row 0 (warp 0, group_id 0) contains the valid query.
// Atom 1 has no valid rows (rows 64-127 beyond seq_len=1).
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

    // Decode uses double-buffered K + separate mbar layout
    BF16*          q_smem   = reinterpret_cast<BF16*>(smem + DECODE_Q_OFF);
    BF16*          k_buf0   = reinterpret_cast<BF16*>(smem + DECODE_K0_OFF);
    BF16*          k_buf1   = reinterpret_cast<BF16*>(smem + DECODE_K1_OFF);
    uint64_t*      mbar     = reinterpret_cast<uint64_t*>(smem + DECODE_MBAR_OFF);
    const int kv_row_base   = (batch_idx * num_kv_heads + kv_head_idx) * ctx_len;

    if (tid == 0) {
        mbar_init(&mbar[0], 1);  // K buffer 0
        mbar_init(&mbar[1], 1);  // K buffer 1
        mbar_init(&mbar[2], 1);  // Vt
    }

    // Load Q into K-major SW128 smem (128 rows, row 0 valid)
    {
        Tensor sQ = make_tensor(make_smem_ptr(q_smem), SmemLayoutQ{});
        const int total = BLOCK_M_PREFILL * HEAD_DIM;
        for (int i = tid; i < total; i += BLOCK_THREADS) {
            int m = i / HEAD_DIM, k = i % HEAD_DIM;
            sQ(m, k) = (m == 0) ? BF16(q_ptr[k]) : BF16(0.0f);
        }
    }
    __syncthreads();

    // Per-atom softmax state (atom 0 has row 0, atom 1 unused)
    float acc_o0[OUT_N_TILES][4];
    float row_max0[2] = {-INFINITY, -INFINITY};
    float row_sum0[2] = {0.0f, 0.0f};
    #pragma unroll
    for (int nt = 0; nt < OUT_N_TILES; nt++)
        acc_o0[nt][0] = acc_o0[nt][1] = acc_o0[nt][2] = acc_o0[nt][3] = 0.0f;

    float inv_sqrt = 1.0f / sqrtf((float)HEAD_DIM);
    int num_kv_blocks = (ctx_len + BLOCK_N - 1) / BLOCK_N;
    const int k_bytes = BLOCK_N * HEAD_DIM * sizeof(BF16);
    const int vt_bytes = HEAD_DIM * BLOCK_N * sizeof(BF16);

    // ── Prologue: preload K[0] into buffer 0 ──────────────────────────────
    if (num_kv_blocks > 0 && tid == 0) {
        mbar_arrive_tx(&mbar[0], k_bytes);
        Tensor mK = tma_k.get_tma_tensor(make_shape(total_kv_rows, Int<HEAD_DIM>{}));
        Tensor sK0 = make_tensor(make_smem_ptr(k_buf0), SmemLayoutKW{});
        Tensor mK_off0 = domain_offset(make_coord(kv_row_base, 0), mK);
        Tensor gK0 = local_tile(mK_off0, Shape<Int<BLOCK_N>, Int<HEAD_DIM>>{},
                                make_coord(0, 0));
        auto [tKgK0, tKsK0] = tma_partition(tma_k, Int<0>{}, Layout<_1>{},
                                             group_modes<0,2>(sK0),
                                             group_modes<0,2>(gK0));
        copy(tma_k.with(mbar[0]), tKgK0, tKsK0);
    }

    // ── Main KV loop ──────────────────────────────────────────────────────
    for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        int cur_buf = kv_block & 1;                         // K buffer with K[kv_block]
        int next_buf = (kv_block + 1) & 1;                  // K buffer for K[kv_block+1]
        int k_phase = (kv_block / 2) & 1;                   // phase for cur_buf mbarrier
        int kv_start = kv_block * BLOCK_N;
        int kv_end   = min(kv_start + BLOCK_N, ctx_len);
        int valid_n  = kv_end - kv_start;

        // Wait for K[kv_block] to complete
        {
            uint64_t* k_bar = (cur_buf == 0) ? &mbar[0] : &mbar[1];
            mbar_wait(k_bar, k_phase);
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        }
        __syncthreads();

        // WGMMA QK
        float acc_s[2][N_TILES][4];
        compute_qk_cute(q_smem, cur_buf == 0 ? k_buf0 : k_buf1, acc_s);

        // Issue K[kv_block+1] into alternate buffer (overlaps with Vt + softmax + PV)
        if (kv_block + 1 < num_kv_blocks && tid == 0) {
            uint64_t* next_k_bar = (next_buf == 0) ? &mbar[0] : &mbar[1];
            mbar_arrive_tx(next_k_bar, k_bytes);
            Tensor mK = tma_k.get_tma_tensor(make_shape(total_kv_rows, Int<HEAD_DIM>{}));
            Tensor sK_next = make_tensor(
                make_smem_ptr(next_buf == 0 ? k_buf0 : k_buf1), SmemLayoutKW{});
            Tensor mK_off_next = domain_offset(
                make_coord(kv_row_base + (kv_block + 1) * BLOCK_N, 0), mK);
            Tensor gK_next = local_tile(mK_off_next,
                Shape<Int<BLOCK_N>, Int<HEAD_DIM>>{}, make_coord(0, 0));
            auto [tKgK_n, tKsK_n] = tma_partition(tma_k, Int<0>{}, Layout<_1>{},
                                                   group_modes<0,2>(sK_next),
                                                   group_modes<0,2>(gK_next));
            copy(tma_k.with(*next_k_bar), tKgK_n, tKsK_n);
        }

        // Issue Vt[kv_block] into cur_buf (overwrites K which QK just consumed)
        {
            if (tid == 0) {
                mbar_arrive_tx(&mbar[2], vt_bytes);
                Tensor mVt = tma_vt.get_tma_tensor(
                    make_shape(Int<HEAD_DIM>{}, total_kv_rows));
                Tensor sVt_cur = make_tensor(
                    make_smem_ptr(cur_buf == 0 ? k_buf0 : k_buf1), SmemLayoutVt{});
                Tensor mVt_off = domain_offset(
                    make_coord(0, kv_row_base + kv_start), mVt);
                Tensor gVt = local_tile(mVt_off,
                    Shape<Int<HEAD_DIM>, Int<BLOCK_N>>{}, make_coord(0, 0));
                auto [tVgVt, tVsVt] = tma_partition(tma_vt, Int<0>{}, Layout<_1>{},
                                                     group_modes<0,2>(sVt_cur),
                                                     group_modes<0,2>(gVt));
                copy(tma_vt.with(mbar[2]), tVgVt, tVsVt);
            }
        }

        // Softmax (runs while Vt and next K TMA are in flight)
        if (warp_id == 0) {
            softmax_update_reg(acc_s[0], acc_o0, row_max0, row_sum0,
                               inv_sqrt, valid_n, kv_start, ctx_len - 1, 1);
        }

        // Wait for Vt to complete
        {
            const int v_phase = kv_block & 1;
            mbar_wait(&mbar[2], v_phase);
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        }
        __syncthreads();

        // PV atom 0 (reads Vt from cur_buf)
        compute_pv_cute(acc_s[0], cur_buf == 0 ? k_buf0 : k_buf1, acc_o0);
        __syncthreads();
    }

    // ── Write output for row 0 ────────────────────────────────────────────
    {
        // Use a safe scratch location: part of the mbar region that's done
        float* scratch = reinterpret_cast<float*>(smem + DECODE_MBAR_OFF);
        if (warp_id == 0 && lane_id == 0)
            scratch[0] = row_sum0[0];
        __syncthreads();
        float shared_row_sum = scratch[0];
        __syncthreads();

        int warp_row_base = warp_id * MMA_M;
        int group_id      = lane_id >> 2;
        int tid_in_group  = lane_id & 3;
        int row0 = warp_row_base + group_id;
        int col0 = tid_in_group * 2;
        float inv0 = (shared_row_sum > 0.0f) ? (1.0f / shared_row_sum) : 0.0f;

        #pragma unroll
        for (int nt = 0; nt < OUT_N_TILES; nt++) {
            int n_base = nt * MMA_N;
            if (row0 == 0)
                store_2bf16(o_ptr, n_base + col0,
                            acc_o0[nt][0] * inv0, acc_o0[nt][1] * inv0);
        }
    }
}

// ============================================================================
// PyTorch bindings (unchanged)
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
                         cudaFuncAttributeMaxDynamicSharedMemorySize, DECODE_SMEM_TOTAL);
    unified_attn_decode_kernel<<<dim3(1, batch * num_heads), BLOCK_THREADS, DECODE_SMEM_TOTAL>>>(
        (const __nv_bfloat16*)Q.data_ptr(),
        tma_k,
        tma_vt,
        (__nv_bfloat16*)O.data_ptr(),
        ctx_len, num_heads, num_kv_heads, total_kv_rows);
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("unified_prefill", &unified_prefill, "kBlockM=128 unified prefill (kBlockM=128, AtomLayout<2,1,1>)");
    m.def("unified_decode",  &unified_decode,  "kBlockM=128 unified decode");
}
