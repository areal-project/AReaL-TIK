// Iteration 118 — aligned-long causal block-count specialization
//
// Base: exact promoted iteration 116, SHA-256
// 4aa38c51152e848d81cb799800d2c6e91a2993f3f451e0bbab8a839c697a8cfb.
// AsyncKWithL2 is dispatched only for an N128-aligned full-query prefix whose
// rows are all covered by ctx_len.  For the fixed M64/N128 causal geometry,
// CTA q_block_idx therefore retains exactly (q_block_idx >> 1) + 1 KV blocks.
// Propagate that semantic dispatch proof into the long instantiations so they
// avoid the runtime q-block end, context minimum, and ceil-div extent chain.
// The generic bound, block traversal, terminal classification, arithmetic,
// staging, output, tail, decode, host dispatch, shared memory, and API remain
// otherwise exact iteration 116.

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

// Mathematical constant for exp2f trick: exp(x) = exp2(x * LOG2E)
constexpr float LOG2E = 1.442695040888963407359924681001892137426646f;

// Smem layout — SEPARATE K and V ping-pong buffers (fixes K/V overwrite bug)
// K and V must be separate: loading V[i] must not overwrite K[i] that Group B still needs
constexpr int KV_SMEM_PAD    = 8;                                    // bf16 padding per row
constexpr int K_SMEM_STRIDE  = HEAD_DIM + KV_SMEM_PAD;              // 136 bf16 per row (K)
constexpr int V_SMEM_STRIDE  = BLOCK_N  + KV_SMEM_PAD;              // 136 bf16 per row (V)
static_assert((V_SMEM_STRIDE * 2) % 16 == 0,
              "each padded V row must preserve 16-byte ldmatrix alignment");
static_assert(MMA_N * 2 == 16,
              "each output N tile must be one aligned 8xBF16 row segment");
constexpr int K_BUF_BYTES    = BLOCK_N * K_SMEM_STRIDE * 2;         // 34816 B per K buffer
constexpr int V_BUF_BYTES    = BLOCK_N * V_SMEM_STRIDE * 2;         // 34816 B per V buffer
constexpr int Q_TILE_BYTES   = MMA_M * HEAD_DIM * 2;                //  4096 B
constexpr int PV_BUF_COLS    = 24;                                   // padded: 24*2=48B, ldmatrix-aligned
constexpr int PV_BUF_BYTES   = MMA_M * PV_BUF_COLS * 2;             //   768 B (16×24 bf16)
constexpr int PER_WARP_BYTES = Q_TILE_BYTES + PV_BUF_BYTES;         //  4864 B

// 4 buffers: 2 K (ping-pong) + 2 V (ping-pong)
constexpr int K_BUF_0_OFF    = 0;
constexpr int K_BUF_1_OFF    = K_BUF_BYTES;                         // 34816
constexpr int V_BUF_0_OFF    = K_BUF_BYTES * 2;                     // 69632
constexpr int V_BUF_1_OFF    = K_BUF_BYTES * 2 + V_BUF_BYTES;      // 104448
constexpr int WARP_BASE      = K_BUF_BYTES * 2 + V_BUF_BYTES * 2;  // 139264
constexpr int SMEM_BYTES     = WARP_BASE + NUM_WARPS * PER_WARP_BYTES;  // 158720 B = 155 KB

// Prefill-only 09i layout. Keep the four-buffer constants above unchanged for
// the iteration-12 decode code, but launch prefill with one shared K/V tile.
constexpr int PREFILL_KV_BUF_OFF = K_BUF_0_OFF;                         // 0
constexpr int PREFILL_WARP_BASE  = K_BUF_BYTES;                         // 34816
constexpr int PREFILL_SMEM_BYTES =
    PREFILL_WARP_BASE + NUM_WARPS * PER_WARP_BYTES;                     // 54272 B
static_assert(K_BUF_BYTES == V_BUF_BYTES,
              "single prefill K/V buffer requires identical padded extents");

constexpr int W_Q_TILE_OFF   = 0;
constexpr int W_PV_BUF_OFF   = W_Q_TILE_OFF + Q_TILE_BYTES;

// Long-prefill output staging.  One unused uint4 at the end of every row
// rotates its shared-memory bank phase by four banks.  The last 256 bytes
// intentionally cross from the dead q_tile into the contiguous dead pv_buf.
constexpr int PREFILL_OUTPUT_UINT4_WORDS =
    sizeof(uint4) / sizeof(uint32_t);                                  // 4
constexpr int PREFILL_OUTPUT_PADDED_ROW_VECS = OUT_N_TILES + 1;        // 17
constexpr int PREFILL_OUTPUT_PADDED_ROW_WORDS =
    PREFILL_OUTPUT_PADDED_ROW_VECS * PREFILL_OUTPUT_UINT4_WORDS;        // 68
constexpr int PREFILL_OUTPUT_PADDED_ROWS = MMA_M;                      // 16
constexpr int PREFILL_OUTPUT_PADDED_VECS =
    PREFILL_OUTPUT_PADDED_ROWS * PREFILL_OUTPUT_PADDED_ROW_VECS;        // 272
constexpr int PREFILL_OUTPUT_PADDED_BYTES =
    PREFILL_OUTPUT_PADDED_VECS * sizeof(uint4);                         // 4352
constexpr int PREFILL_OUTPUT_GATHER_ROW_STEP_VECS =
    2 * PREFILL_OUTPUT_PADDED_ROW_VECS;                                 // 34

constexpr bool prefill_output_layout_is_bounded_bijective() {
    bool seen[PREFILL_OUTPUT_PADDED_VECS] = {};
    int visited = 0;
    for (int row = 0; row < PREFILL_OUTPUT_PADDED_ROWS; row++) {
        for (int seg = 0; seg < OUT_N_TILES; seg++) {
            const int svec = row * PREFILL_OUTPUT_PADDED_ROW_VECS + seg;
            if (svec < 0 || svec >= PREFILL_OUTPUT_PADDED_VECS ||
                seen[svec]) {
                return false;
            }
            seen[svec] = true;
            visited++;
        }
    }
    if (visited != PREFILL_OUTPUT_PADDED_ROWS * OUT_N_TILES) {
        return false;
    }
    for (int svec = 0; svec < PREFILL_OUTPUT_PADDED_VECS; svec++) {
        const bool is_payload =
            (svec % PREFILL_OUTPUT_PADDED_ROW_VECS) < OUT_N_TILES;
        if (seen[svec] != is_payload) {
            return false;
        }
    }
    return true;
}

// Each unrolled scalar scatter instruction covers all 32 banks exactly once.
// row0 and row1 are separate instructions, so prove the two cohorts separately.
constexpr bool prefill_output_scatter_bank_cohorts_are_bijective() {
    for (int seg = 0; seg < OUT_N_TILES; seg++) {
        for (int row_half = 0; row_half < 2; row_half++) {
            bool bank_seen[WARP_SIZE] = {};
            for (int lane = 0; lane < WARP_SIZE; lane++) {
                const int row = (lane >> 2) + row_half * 8;
                const int tid_in_group = lane & 3;
                const int svec =
                    row * PREFILL_OUTPUT_PADDED_ROW_VECS + seg;
                const int word =
                    svec * PREFILL_OUTPUT_UINT4_WORDS + tid_in_group;
                const int bank = word % WARP_SIZE;
                if (bank_seen[bank]) {
                    return false;
                }
                bank_seen[bank] = true;
            }
            for (int bank = 0; bank < WARP_SIZE; bank++) {
                if (!bank_seen[bank]) {
                    return false;
                }
            }
        }
    }
    return true;
}

// The gather covers each logical row/segment and each contiguous global vector
// exactly once.  For every LDS.128 eight-lane transaction cohort, its 8x4
// words likewise cover all 32 shared-memory banks exactly once.
constexpr bool prefill_output_gather_is_bijective_and_bank_clean() {
    bool output_seen[PREFILL_OUTPUT_PADDED_ROWS * OUT_N_TILES] = {};
    for (int i = 0; i < PREFILL_OUTPUT_PADDED_ROWS / 2; i++) {
        for (int lane = 0; lane < WARP_SIZE; lane++) {
            const int row = i * 2 + (lane >> 4);
            const int seg = lane & 15;
            const int gather_base =
                (lane & 15) + (lane >> 4) * PREFILL_OUTPUT_PADDED_ROW_VECS;
            const int svec =
                gather_base + i * PREFILL_OUTPUT_GATHER_ROW_STEP_VECS;
            const int logical_vec = row * OUT_N_TILES + seg;
            const int global_vec = i * WARP_SIZE + lane;
            if (svec != row * PREFILL_OUTPUT_PADDED_ROW_VECS + seg ||
                svec < 0 || svec >= PREFILL_OUTPUT_PADDED_VECS ||
                logical_vec != global_vec || output_seen[logical_vec]) {
                return false;
            }
            output_seen[logical_vec] = true;
        }
        for (int cohort = 0; cohort < WARP_SIZE / 8; cohort++) {
            bool bank_seen[WARP_SIZE] = {};
            for (int lane_in_cohort = 0; lane_in_cohort < 8;
                 lane_in_cohort++) {
                const int lane = cohort * 8 + lane_in_cohort;
                const int gather_base =
                    (lane & 15) +
                    (lane >> 4) * PREFILL_OUTPUT_PADDED_ROW_VECS;
                const int svec =
                    gather_base + i * PREFILL_OUTPUT_GATHER_ROW_STEP_VECS;
                for (int word = 0; word < PREFILL_OUTPUT_UINT4_WORDS;
                     word++) {
                    const int bank =
                        (svec * PREFILL_OUTPUT_UINT4_WORDS + word) %
                        WARP_SIZE;
                    if (bank_seen[bank]) {
                        return false;
                    }
                    bank_seen[bank] = true;
                }
            }
            for (int bank = 0; bank < WARP_SIZE; bank++) {
                if (!bank_seen[bank]) {
                    return false;
                }
            }
        }
    }
    for (int i = 0; i < PREFILL_OUTPUT_PADDED_ROWS * OUT_N_TILES; i++) {
        if (!output_seen[i]) {
            return false;
        }
    }
    return true;
}

static_assert(PREFILL_OUTPUT_UINT4_WORDS == 4 &&
              PREFILL_OUTPUT_PADDED_ROW_VECS == 17 &&
              PREFILL_OUTPUT_GATHER_ROW_STEP_VECS == 34,
              "padded output indexing assumes 16 payload vectors plus one pad");
static_assert(PREFILL_OUTPUT_PADDED_BYTES == 4352 &&
              PREFILL_OUTPUT_PADDED_BYTES ==
                  Q_TILE_BYTES + MMA_M * sizeof(uint4),
              "padded output must occupy q_tile plus exactly 256 pv_buf bytes");
static_assert(W_PV_BUF_OFF == W_Q_TILE_OFF + Q_TILE_BYTES &&
              PREFILL_OUTPUT_PADDED_BYTES - Q_TILE_BYTES <= PV_BUF_BYTES &&
              W_Q_TILE_OFF + PREFILL_OUTPUT_PADDED_BYTES <= PER_WARP_BYTES,
              "padded output must remain in contiguous dead q_tile and pv_buf");
static_assert((PREFILL_WARP_BASE + W_Q_TILE_OFF) % alignof(uint4) == 0 &&
              PER_WARP_BYTES % alignof(uint4) == 0 &&
              PREFILL_OUTPUT_PADDED_ROW_WORDS * sizeof(uint32_t) %
                      alignof(uint4) ==
                  0,
              "padded output base, warp stride, and rows must be uint4 aligned");
static_assert(prefill_output_layout_is_bounded_bijective(),
              "padded row-major scatter must be bounded and bijective");
static_assert(prefill_output_scatter_bank_cohorts_are_bijective(),
              "each padded scalar-scatter cohort must be bank conflict free");
static_assert(prefill_output_gather_is_bijective_and_bank_clean(),
              "padded uint4 gather must be bijective and bank conflict free");

// Decode split-kernel shared-memory footprints.
constexpr int DECODE_SCORE_SMEM_BYTES = K_BUF_BYTES + Q_TILE_BYTES;
constexpr int DECODE_D8_V_STAGES      = 4;
constexpr int DECODE_D8_V_STAGE_BYTES = BLOCK_N * MMA_N * 2;  // [128, 8] bf16
constexpr int DECODE_D8_V_BYTES       =
    DECODE_D8_V_STAGES * DECODE_D8_V_STAGE_BYTES;
constexpr int DECODE_D8_SMEM_BYTES    =
    DECODE_D8_V_BYTES + NUM_WARPS * PV_BUF_BYTES;  // 8192 + 3072 = 11264 B

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

// FUNCTION: ldmatrix_b_x2
// Load the low/high K halves of one row-major [8, 16] BF16 tile. For
// lane = 4*g+t, r0 packs K[g,2*t],K[g,2*t+1] and r1 packs
// K[g,2*t+8],K[g,2*t+9], exactly matching mma.m16n8k16 B operands.
__device__ __forceinline__
void ldmatrix_b_x2(uint32_t& r0, uint32_t& r1,
                   const __nv_bfloat16* smem_row_ptr) {
    const uint32_t addr =
        static_cast<uint32_t>(__cvta_generic_to_shared(smem_row_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(r0), "=r"(r1)
        : "r"(addr)
    );
}

// FUNCTION: ldmatrix_b_x4
// Load the low/high K halves for two adjacent row-major [8, 16] BF16 tiles.
// r0/r1 are the m16n8k16 B fragment for N tile nt; r2/r3 are the
// corresponding fragment for N tile nt+1.
__device__ __forceinline__
void ldmatrix_b_x4(uint32_t& r0, uint32_t& r1,
                   uint32_t& r2, uint32_t& r3,
                   const __nv_bfloat16* smem_row_ptr) {
    const uint32_t addr =
        static_cast<uint32_t>(__cvta_generic_to_shared(smem_row_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(addr)
    );
}

// FUNCTION: ldmatrix_b_trans_x2
// Load two row-major 8x8 BF16 matrices and return their transposed fragments.
// For lane = 4*g+t, r0 packs V[2*t,g],V[2*t+1,g] and r1 packs
// V[2*t+8,g],V[2*t+9,g], exactly matching mma.m16n8k16 B operands.
__device__ __forceinline__
void ldmatrix_b_trans_x2(uint32_t& r0, uint32_t& r1,
                         const __nv_bfloat16* smem_row_ptr) {
    const uint32_t addr =
        static_cast<uint32_t>(__cvta_generic_to_shared(smem_row_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(r0), "=r"(r1)
        : "r"(addr)
    );
}

// One naturally aligned 16-byte SM80 global-to-shared transfer. A zero source
// size makes the final partial KV block zero-fill its inactive V rows.
__device__ __forceinline__
void cp_async_cg_16(void* smem_ptr, const void* global_ptr, bool predicate) {
    const uint32_t smem_addr =
        static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    const int src_bytes = predicate ? 16 : 0;
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
        :: "r"(smem_addr), "l"(global_ptr), "r"(src_bytes)
        : "memory"
    );
}

// FUNCTION: cp_async_cg_l2_16_full
// One naturally aligned, unpredicated SM80 global-to-shared transfer. The
// L2::128B qualifier is a cache-prefetch-size hint; it does not change the
// 16-byte copy size. This wrapper deliberately has no compiler memory clobber:
// cp.async ordering is established explicitly by commit_group/wait_group and
// the caller's existing CTA barrier.
__device__ __forceinline__
void cp_async_cg_l2_16_full(void* smem_ptr, const void* global_ptr) {
    const uint32_t smem_addr =
        static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
        :: "r"(smem_addr), "l"(global_ptr)
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
            // CHANGED: use padded stride K_SMEM_STRIDE=136 to eliminate bank conflicts
            __nv_bfloat16 v0 = k_smem[n * K_SMEM_STRIDE + k0];
            __nv_bfloat16 v1 = k_smem[n * K_SMEM_STRIDE + k1];
            __nv_bfloat16 v8 = k_smem[n * K_SMEM_STRIDE + k8];
            __nv_bfloat16 v9 = k_smem[n * K_SMEM_STRIDE + k9];
            uint32_t b0 = (reinterpret_cast<uint16_t&>(v1) << 16) | reinterpret_cast<uint16_t&>(v0);
            uint32_t b1 = (reinterpret_cast<uint16_t&>(v9) << 16) | reinterpret_cast<uint16_t&>(v8);
            mma_bf16(acc_s[nt][0], acc_s[nt][1], acc_s[nt][2], acc_s[nt][3],
                     a0, a1, a2, a3, b0, b1,
                     acc_s[nt][0], acc_s[nt][1], acc_s[nt][2], acc_s[nt][3]);
        }
    }
}

// FUNCTION: compute_qk_reg_live4
// Producer-only QK path for four packed GQA rows. ldmatrix.x4 still receives
// sixteen valid, aligned logical row addresses per K half, but address-provider
// lanes for inactive rows alias initialized rows 0..3 instead of reading the
// unstaged rows 4..15. MMA row independence keeps live c0/c1 exact.
__device__ __forceinline__ void compute_qk_reg_live4(
    const __nv_bfloat16* __restrict__ q_tile,
    const __nv_bfloat16* __restrict__ k_smem,
    float acc_s[N_TILES][4]
) {
    const int lane_id = threadIdx.x % WARP_SIZE;

    #pragma unroll
    for (int nt = 0; nt < N_TILES; nt++)
        acc_s[nt][0] = acc_s[nt][1] = acc_s[nt][2] = acc_s[nt][3] = 0.0f;

    #pragma unroll
    for (int kt = 0; kt < K_TILES; kt++) {
        uint32_t a0, a1, a2, a3;
        {
            // Lanes providing logical rows 0..3 retain their exact addresses.
            // Every other logical row reads a duplicate initialized live row.
            const int safe_row = lane_id & 3;
            const int col = (lane_id / MMA_M) * 8 + kt * MMA_K;
            ldmatrix_a(
                a0, a1, a2, a3,
                q_tile + safe_row * HEAD_DIM + col);
        }
        #pragma unroll
        for (int nt = 0; nt < N_TILES; nt++) {
            const int n_col  = nt * MMA_N;
            const int n      = n_col + (lane_id >> 2);
            const int k_base = (lane_id & 3) * 2;
            const int k0 = kt * MMA_K + k_base;
            const int k1 = k0 + 1;
            const int k8 = k0 + 8;
            const int k9 = k8 + 1;
            __nv_bfloat16 v0 = k_smem[n * K_SMEM_STRIDE + k0];
            __nv_bfloat16 v1 = k_smem[n * K_SMEM_STRIDE + k1];
            __nv_bfloat16 v8 = k_smem[n * K_SMEM_STRIDE + k8];
            __nv_bfloat16 v9 = k_smem[n * K_SMEM_STRIDE + k9];
            const uint32_t b0 =
                (reinterpret_cast<uint16_t&>(v1) << 16) |
                 reinterpret_cast<uint16_t&>(v0);
            const uint32_t b1 =
                (reinterpret_cast<uint16_t&>(v9) << 16) |
                 reinterpret_cast<uint16_t&>(v8);
            mma_bf16(acc_s[nt][0], acc_s[nt][1],
                     acc_s[nt][2], acc_s[nt][3],
                     a0, a1, a2, a3, b0, b1,
                     acc_s[nt][0], acc_s[nt][1],
                     acc_s[nt][2], acc_s[nt][3]);
        }
    }
}

// FUNCTION: compute_qk_reg_ldmatrix
__device__ __forceinline__ void compute_qk_reg_ldmatrix(
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
            // FA2 tiles Q as two physical [MMA_M,64] pages. Logical
            // 8-BF16 segment j of row r is stored at j ^ (r & 7).
            int q_seg = kt * 2 + lane_id / MMA_M;
            int q_offset =
                ((q_seg & 8) << 7) +
                (row << 6) +
                (((q_seg ^ row) & 7) << 3);
            ldmatrix_a(a0, a1, a2, a3, q_tile + q_offset);
        }
        #pragma unroll
        for (int nt = 0; nt < N_TILES; nt += 2) {
            // x4 matrix roles by address-provider lanes:
            //   0..7: nt low-K, 8..15: nt high-K,
            //   16..23: nt+1 low-K, 24..31: nt+1 high-K.
            // Non-transposed row-major K is the column-major K^T fragment
            // required by mma.m16n8k16.row.col.
            const int matrix = lane_id >> 3;
            const int matrix_row =
                (nt + (matrix >> 1)) * MMA_N + (lane_id & 7);
            const int matrix_col = kt * MMA_K + (matrix & 1) * 8;
            uint32_t b00, b01, b10, b11;
            ldmatrix_b_x4(
                b00, b01, b10, b11,
                k_smem + matrix_row * K_SMEM_STRIDE + matrix_col);
            mma_bf16(acc_s[nt][0], acc_s[nt][1], acc_s[nt][2], acc_s[nt][3],
                     a0, a1, a2, a3, b00, b01,
                     acc_s[nt][0], acc_s[nt][1], acc_s[nt][2], acc_s[nt][3]);
            mma_bf16(acc_s[nt + 1][0], acc_s[nt + 1][1],
                     acc_s[nt + 1][2], acc_s[nt + 1][3],
                     a0, a1, a2, a3, b10, b11,
                     acc_s[nt + 1][0], acc_s[nt + 1][1],
                     acc_s[nt + 1][2], acc_s[nt + 1][3]);
        }
    }
}

// FUNCTION: compute_qk_reg_ldmatrix_half_terminal
// A 128-aligned complete M64 query tile can reach only columns 0..63 of its
// terminal N128 key block. Across the CTA, four M16 warp tiles therefore form
// an M64xN64 QK operation. Keep the full score fragment for the unchanged
// softmax interface, but materialize the unreachable upper half as -infinity.
__device__ __forceinline__ void compute_qk_reg_ldmatrix_half_terminal(
    const __nv_bfloat16* __restrict__ q_tile,
    const __nv_bfloat16* __restrict__ k_smem,
    float acc_s[N_TILES][4]
) {
    constexpr int HALF_N_TILES = N_TILES / 2;
    static_assert(N_TILES == 16 && HALF_N_TILES == 8,
                  "half-terminal QK requires the M64/N128 geometry");

    int lane_id = threadIdx.x % WARP_SIZE;

    #pragma unroll
    for (int nt = 0; nt < HALF_N_TILES; nt++)
        acc_s[nt][0] = acc_s[nt][1] = acc_s[nt][2] = acc_s[nt][3] = 0.0f;
    #pragma unroll
    for (int nt = HALF_N_TILES; nt < N_TILES; nt++)
        acc_s[nt][0] = acc_s[nt][1] = acc_s[nt][2] = acc_s[nt][3] = -INFINITY;

    #pragma unroll
    for (int kt = 0; kt < K_TILES; kt++) {
        uint32_t a0, a1, a2, a3;
        {
            int row = lane_id % MMA_M;
            int q_seg = kt * 2 + lane_id / MMA_M;
            int q_offset =
                ((q_seg & 8) << 7) +
                (row << 6) +
                (((q_seg ^ row) & 7) << 3);
            ldmatrix_a(a0, a1, a2, a3, q_tile + q_offset);
        }
        #pragma unroll
        for (int nt = 0; nt < HALF_N_TILES; nt += 2) {
            const int matrix = lane_id >> 3;
            const int matrix_row =
                (nt + (matrix >> 1)) * MMA_N + (lane_id & 7);
            const int matrix_col = kt * MMA_K + (matrix & 1) * 8;
            uint32_t b00, b01, b10, b11;
            ldmatrix_b_x4(
                b00, b01, b10, b11,
                k_smem + matrix_row * K_SMEM_STRIDE + matrix_col);
            mma_bf16(acc_s[nt][0], acc_s[nt][1],
                     acc_s[nt][2], acc_s[nt][3],
                     a0, a1, a2, a3, b00, b01,
                     acc_s[nt][0], acc_s[nt][1],
                     acc_s[nt][2], acc_s[nt][3]);
            mma_bf16(acc_s[nt + 1][0], acc_s[nt + 1][1],
                     acc_s[nt + 1][2], acc_s[nt + 1][3],
                     a0, a1, a2, a3, b10, b11,
                     acc_s[nt + 1][0], acc_s[nt + 1][1],
                     acc_s[nt + 1][2], acc_s[nt + 1][3]);
        }
    }
}

// ============================================================================
// Softmax — register-resident
// ============================================================================
// FUNCTION: softmax_update_reg
template <bool FullPhysicalN128 = false>
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

    float bmax_raw0 = -INFINITY, bmax_raw1 = -INFINITY;
    #pragma unroll
    for (int nt = 0; nt < N_TILES; nt++) {
        int n_base = nt * MMA_N;
        int n0 = n_base + col0, n1 = n_base + col1;
        bool m00 = (!FullPhysicalN128 && n0 >= valid_n) ||
                   (kv_start + n0 > q_pos0) || (row0 >= actual_rows);
        bool m01 = (!FullPhysicalN128 && n1 >= valid_n) ||
                   (kv_start + n1 > q_pos0) || (row0 >= actual_rows);
        bool m10 = (!FullPhysicalN128 && n0 >= valid_n) ||
                   (kv_start + n0 > q_pos1) || (row1 >= actual_rows);
        bool m11 = (!FullPhysicalN128 && n1 >= valid_n) ||
                   (kv_start + n1 > q_pos1) || (row1 >= actual_rows);
        acc_s[nt][0] = m00 ? -INFINITY : acc_s[nt][0];
        acc_s[nt][1] = m01 ? -INFINITY : acc_s[nt][1];
        acc_s[nt][2] = m10 ? -INFINITY : acc_s[nt][2];
        acc_s[nt][3] = m11 ? -INFINITY : acc_s[nt][3];
        bmax_raw0 = fmaxf(
            bmax_raw0, fmaxf(acc_s[nt][0], acc_s[nt][1]));
        bmax_raw1 = fmaxf(
            bmax_raw1, fmaxf(acc_s[nt][2], acc_s[nt][3]));
    }
    bmax_raw0 = quad_max(bmax_raw0);
    bmax_raw1 = quad_max(bmax_raw1);

    // Keep row_max in iteration 103's scaled-natural domain.  Since
    // inv_sqrt is positive, reducing before this one scale selects the same
    // value as scaling all scores before the max reduction.
    const float bmax0 = bmax_raw0 * inv_sqrt;
    const float bmax1 = bmax_raw1 * inv_sqrt;

    float new_max0 = fmaxf(row_max[0], bmax0);
    float new_max1 = fmaxf(row_max[1], bmax1);
    float rescale0 = isinf(new_max0) ? 1.0f : fast_exp2_ftz((row_max[0] - new_max0) * LOG2E);
    float rescale1 = isinf(new_max1) ? 1.0f : fast_exp2_ftz((row_max[1] - new_max1) * LOG2E);
    row_max[0] = new_max0;
    row_max[1] = new_max1;

    #pragma unroll
    for (int nt = 0; nt < OUT_N_TILES; nt++) {
        acc_o[nt][0] *= rescale0;  acc_o[nt][1] *= rescale0;
        acc_o[nt][2] *= rescale1;  acc_o[nt][3] *= rescale1;
    }
    row_sum[0] *= rescale0;
    row_sum[1] *= rescale1;

    const float softmax_scale_log2 = inv_sqrt * LOG2E;
    const float new_max0_log2 = new_max0 * LOG2E;
    const float new_max1_log2 = new_max1 * LOG2E;
    float bsum0 = 0.0f, bsum1 = 0.0f;
    #pragma unroll
    for (int nt = 0; nt < N_TILES; nt++) {
        float p0 = isinf(acc_s[nt][0]) ? 0.0f : fast_exp2_ftz(
            fmaf(acc_s[nt][0], softmax_scale_log2, -new_max0_log2));
        float p1 = isinf(acc_s[nt][1]) ? 0.0f : fast_exp2_ftz(
            fmaf(acc_s[nt][1], softmax_scale_log2, -new_max0_log2));
        float p2 = isinf(acc_s[nt][2]) ? 0.0f : fast_exp2_ftz(
            fmaf(acc_s[nt][2], softmax_scale_log2, -new_max1_log2));
        float p3 = isinf(acc_s[nt][3]) ? 0.0f : fast_exp2_ftz(
            fmaf(acc_s[nt][3], softmax_scale_log2, -new_max1_log2));
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
// FUNCTION: compute_pv_reg_per_ktile
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
            int col = (lane_id / MMA_M) * 8;  // col in [0, 8] for pv_buf
            ldmatrix_a(a0, a1, a2, a3, pv_buf + row * PV_BUF_COLS + col);
        }

        // MMA for all output N-tiles using this K-tile's weights
        #pragma unroll
        for (int nt = 0; nt < OUT_N_TILES; nt++) {
            int n_col  = nt * MMA_N;
            // x2 needs row addresses from lanes 0..15. Mirror them into
            // lanes 16..31 as valid addresses even though SM80 ignores
            // the upper half.
            const int v_row = kt * MMA_K + (lane_id & 15);
            uint32_t b0, b1;
            ldmatrix_b_trans_x2(
                b0, b1,
                v_smem + v_row * V_SMEM_STRIDE + n_col);
            mma_bf16(acc_o[nt][0], acc_o[nt][1], acc_o[nt][2], acc_o[nt][3],
                     a0, a1, a2, a3, b0, b1,
                     acc_o[nt][0], acc_o[nt][1], acc_o[nt][2], acc_o[nt][3]);
        }
    }
}

// FUNCTION: compute_pv_reg_per_ktile_nt0_preload
// FA2-style low-register latency scheduling for normal full-query blocks.
// While the warp's probability stores are becoming visible, preload only the
// immutable V fragment for output tile 0.  Keeping a single B fragment live
// across __syncwarp avoids extending all sixteen V-fragment live ranges and
// preserves the original kt/nt MMA order and accumulator recurrence.
__device__ __forceinline__ void compute_pv_reg_per_ktile_nt0_preload(
    float acc_s[N_TILES][4],
    __nv_bfloat16* __restrict__ pv_buf,
    const __nv_bfloat16* __restrict__ v_smem,
    float acc_o[OUT_N_TILES][4]
) {
    int lane_id = threadIdx.x % WARP_SIZE;

    for (int kt = 0; kt < KV_K_TILES; kt++) {
        write_ktile_to_smem(acc_s, kt, pv_buf);

        // V is immutable for the whole PV phase and occupies a disjoint
        // shared-memory region from this warp's pv_buf stores.
        const int v_row = kt * MMA_K + (lane_id & 15);
        uint32_t b0_nt0, b1_nt0;
        ldmatrix_b_trans_x2(
            b0_nt0, b1_nt0,
            v_smem + v_row * V_SMEM_STRIDE);

        __syncwarp();

        uint32_t a0, a1, a2, a3;
        {
            int row = lane_id % MMA_M;
            int col = (lane_id / MMA_M) * 8;
            ldmatrix_a(a0, a1, a2, a3,
                       pv_buf + row * PV_BUF_COLS + col);
        }

        mma_bf16(acc_o[0][0], acc_o[0][1],
                 acc_o[0][2], acc_o[0][3],
                 a0, a1, a2, a3, b0_nt0, b1_nt0,
                 acc_o[0][0], acc_o[0][1],
                 acc_o[0][2], acc_o[0][3]);

        #pragma unroll
        for (int nt = 1; nt < OUT_N_TILES; nt++) {
            int n_col = nt * MMA_N;
            uint32_t b0, b1;
            ldmatrix_b_trans_x2(
                b0, b1,
                v_smem + v_row * V_SMEM_STRIDE + n_col);
            mma_bf16(acc_o[nt][0], acc_o[nt][1],
                     acc_o[nt][2], acc_o[nt][3],
                     a0, a1, a2, a3, b0, b1,
                     acc_o[nt][0], acc_o[nt][1],
                     acc_o[nt][2], acc_o[nt][3]);
        }
    }
}

// FUNCTION: compute_pv_reg_per_ktile_half_terminal
// The half-terminal softmax makes probability columns 64..127 exact zero.
// Consume only K tiles 0..3, preserving the original per-tile shared write,
// ldmatrix, output-tile order, and mma.sync order for all reachable columns.
__device__ __forceinline__ void compute_pv_reg_per_ktile_half_terminal(
    float acc_s[N_TILES][4],
    __nv_bfloat16* __restrict__ pv_buf,
    const __nv_bfloat16* __restrict__ v_smem,
    float acc_o[OUT_N_TILES][4]
) {
    constexpr int HALF_KV_K_TILES = KV_K_TILES / 2;
    static_assert(KV_K_TILES == 8 && HALF_KV_K_TILES == 4,
                  "half-terminal PV requires the M64/N128 geometry");

    int lane_id = threadIdx.x % WARP_SIZE;

    for (int kt = 0; kt < HALF_KV_K_TILES; kt++) {
        write_ktile_to_smem(acc_s, kt, pv_buf);
        __syncwarp();

        uint32_t a0, a1, a2, a3;
        {
            int row = lane_id % MMA_M;
            int col = (lane_id / MMA_M) * 8;
            ldmatrix_a(a0, a1, a2, a3,
                       pv_buf + row * PV_BUF_COLS + col);
        }

        #pragma unroll
        for (int nt = 0; nt < OUT_N_TILES; nt++) {
            int n_col = nt * MMA_N;
            const int v_row = kt * MMA_K + (lane_id & 15);
            uint32_t b0, b1;
            ldmatrix_b_trans_x2(
                b0, b1,
                v_smem + v_row * V_SMEM_STRIDE + n_col);
            mma_bf16(acc_o[nt][0], acc_o[nt][1],
                     acc_o[nt][2], acc_o[nt][3],
                     a0, a1, a2, a3, b0, b1,
                     acc_o[nt][0], acc_o[nt][1],
                     acc_o[nt][2], acc_o[nt][3]);
        }
    }
}

// FUNCTION: pack_bf16x2_rn
// PTX cvt.bf16x2 stores its first FP32 source in the high 16 bits and its
// second source in the low 16 bits. Pass (high, low) to produce the logical
// low-address element in bits 0..15, matching iteration 50's validated store.
__device__ __forceinline__ uint32_t pack_bf16x2_rn(float low, float high) {
    uint32_t packed;
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;\n"
        : "=r"(packed)
        : "f"(low), "f"(high));
    return packed;
}

// ============================================================================
// Prefill kernel
// ============================================================================
// FUNCTION: unified_attn_prefill_kernel
template <bool FullQueryBlocks, bool AsyncKWithL2 = false,
          bool AsyncVWithL2 = false>
__global__ __launch_bounds__(BLOCK_THREADS, 3)
void unified_attn_prefill_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    int seq_len, int ctx_len, int num_heads, int num_kv_heads
) {
    extern __shared__ char smem[];

    // A 128-thread step covers eight rows of sixteen uint4 vectors. The
    // contiguous source advances by 128 uint4; eight padded shared rows
    // advance by 8 * (136 BF16 / 8 BF16 per uint4) = 136 uint4.
    constexpr int BF16_PER_UINT4 = 8;
    constexpr int PREFILL_STAGE_VECS_PER_ROW =
        HEAD_DIM / BF16_PER_UINT4;
    constexpr int PREFILL_STAGE_ROWS_PER_STEP =
        BLOCK_THREADS / PREFILL_STAGE_VECS_PER_ROW;
    constexpr int PREFILL_STAGE_STEPS =
        BLOCK_N / PREFILL_STAGE_ROWS_PER_STEP;
    constexpr int PREFILL_STAGE_SRC_STEP_VECS = BLOCK_THREADS;
    constexpr int PREFILL_STAGE_DST_STEP_VECS =
        PREFILL_STAGE_ROWS_PER_STEP *
        (K_SMEM_STRIDE / BF16_PER_UINT4);
    static_assert(K_SMEM_STRIDE == V_SMEM_STRIDE,
                  "K/V induction requires identical padded row strides");
    static_assert(K_SMEM_STRIDE % BF16_PER_UINT4 == 0,
                  "each padded shared row must contain whole uint4 vectors");
    static_assert(PREFILL_STAGE_STEPS == 16 &&
                  PREFILL_STAGE_SRC_STEP_VECS == 128 &&
                  PREFILL_STAGE_DST_STEP_VECS == 136,
                  "prefill induction must cover the 128x128 tile exactly");

    int batch_idx   = int(blockIdx.z);
    int head_idx    = int(blockIdx.y);
    int q_block_idx = int(blockIdx.x);
    int kv_head_idx = head_idx / (num_heads / num_kv_heads);

    int q_block_start = q_block_idx * BLOCK_M_PREFILL;
    if constexpr (!FullQueryBlocks) {
        if (q_block_start >= seq_len) return;
    }

    const __nv_bfloat16* q_ptr = Q + ((batch_idx * num_heads    + head_idx)    * seq_len) * HEAD_DIM;
    const __nv_bfloat16* k_ptr = K + ((batch_idx * num_kv_heads + kv_head_idx) * ctx_len) * HEAD_DIM;
    const __nv_bfloat16* v_ptr = V + ((batch_idx * num_kv_heads + kv_head_idx) * ctx_len) * HEAD_DIM;
    __nv_bfloat16*        o_ptr = O + ((batch_idx * num_heads    + head_idx)    * seq_len) * HEAD_DIM;

    int tid     = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    char* warp_smem = smem + PREFILL_WARP_BASE + warp_id * PER_WARP_BYTES;
    __nv_bfloat16* q_tile =
        reinterpret_cast<__nv_bfloat16*>(warp_smem + W_Q_TILE_OFF);
    __nv_bfloat16* pv_buf =
        reinterpret_cast<__nv_bfloat16*>(warp_smem + W_PV_BUF_OFF);
    __nv_bfloat16* kv_smem =
        reinterpret_cast<__nv_bfloat16*>(smem + PREFILL_KV_BUF_OFF);

    int warp_q_start = q_block_start + warp_id * MMA_M;
    int actual_rows = MMA_M;
    if constexpr (!FullQueryBlocks) {
        int warp_q_end = min(warp_q_start + MMA_M, seq_len);
        actual_rows = max(0, warp_q_end - warp_q_start);
    }

    if constexpr (FullQueryBlocks) {
        // FA2 assigns eight 128-bit lanes to one 64-column row page.
        // Each instruction covers four rows; the paired access covers the
        // corresponding rows eight positions later in the same page.
        constexpr int Q_VECS_PER_ROW = HEAD_DIM / 8;
        constexpr int Q_PAGE_VECS_PER_ROW = 64 / 8;
        constexpr int Q_PAGE_VECS = MMA_M * Q_PAGE_VECS_PER_ROW;
        constexpr int Q_ROW8_SRC_DELTA = 8 * Q_VECS_PER_ROW;
        constexpr int Q_ROW8_DST_DELTA = 8 * Q_PAGE_VECS_PER_ROW;
        static_assert(Q_VECS_PER_ROW == 16 &&
                      Q_PAGE_VECS_PER_ROW == 8 &&
                      Q_PAGE_VECS == 128 &&
                      Q_ROW8_SRC_DELTA == 128 &&
                      Q_ROW8_DST_DELTA == 64,
                      "FA2 fast-Q page geometry must remain M16xK128");
        const uint4* q_src_vec = reinterpret_cast<const uint4*>(
            q_ptr + warp_q_start * HEAD_DIM);
        uint4* q_dst_vec = reinterpret_cast<uint4*>(q_tile);
        const int row4 = lane_id >> 3;
        const int seg_in_page = lane_id & 7;
        #pragma unroll
        for (int page = 0; page < 2; page++) {
            #pragma unroll
            for (int half = 0; half < 2; half++) {
                const int row = row4 + half * 4;
                const int src_vi =
                    row * Q_VECS_PER_ROW +
                    page * Q_PAGE_VECS_PER_ROW + seg_in_page;
                const int dst_vi =
                    page * Q_PAGE_VECS +
                    row * Q_PAGE_VECS_PER_ROW +
                    (seg_in_page ^ (row & 7));
                uint4 data = q_src_vec[src_vi];
                q_dst_vec[dst_vi] = data;
                data = q_src_vec[src_vi + Q_ROW8_SRC_DELTA];
                q_dst_vec[dst_vi + Q_ROW8_DST_DELTA] = data;
            }
        }
    } else {
        for (int i = lane_id; i < MMA_M * HEAD_DIM; i += WARP_SIZE) {
            int row = i / HEAD_DIM, col = i % HEAD_DIM;
            q_tile[i] = (row < actual_rows)
                        ? q_ptr[(warp_q_start + row) * HEAD_DIM + col]
                        : __float2bfloat16(0.0f);
        }
    }

    float acc_o[OUT_N_TILES][4];
    float row_max[2] = {-INFINITY, -INFINITY};
    float row_sum[2] = {0.0f, 0.0f};
    #pragma unroll
    for (int nt = 0; nt < OUT_N_TILES; nt++)
        acc_o[nt][0] = acc_o[nt][1] = acc_o[nt][2] = acc_o[nt][3] = 0.0f;

    float inv_sqrt = 1.0f / sqrtf((float)HEAD_DIM);

    // Iteration-14 causal upper bound, unchanged.
    int q_block_end;
    if constexpr (FullQueryBlocks) {
        q_block_end = q_block_start + BLOCK_M_PREFILL;
    } else {
        q_block_end = min(q_block_start + BLOCK_M_PREFILL, seq_len);
    }
    const int causal_kv_end = min(ctx_len, q_block_end);
    int num_kv_blocks = (causal_kv_end + BLOCK_N - 1) / BLOCK_N;
    if constexpr (AsyncKWithL2) {
        static_assert(FullQueryBlocks,
                      "aligned long bound requires full query blocks");
        static_assert(BLOCK_M_PREFILL == 64 && BLOCK_N == 128,
                      "aligned long bound requires the M64/N128 geometry");
        num_kv_blocks = (q_block_idx >> 1) + 1;
    }

    // Lockstep 09i schedule: all four warps process the same retained block.
    // The barriers make the one padded tile safe to reuse first for K, then V.
    for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        int kv_start = kv_block * BLOCK_N;
        int valid_n = BLOCK_N;
        if constexpr (!AsyncKWithL2) {
            valid_n = min(kv_start + BLOCK_N, ctx_len) - kv_start;
        }

        // Cooperative K load into the shared padded K/V tile. The long-context
        // specialization is dispatched only when every physical N128 tile is
        // complete, so all 2048 source uint4 addresses are valid and naturally
        // 16-byte aligned. It preserves the synchronous path's exact src +128,
        // dst +136 uint4 induction and drains the group before the old barrier.
        {
            const uint4* src_base =
                reinterpret_cast<const uint4*>(
                    k_ptr + kv_start * HEAD_DIM);
            int src_vi = tid;
            uint4* dst_it = reinterpret_cast<uint4*>(kv_smem)
                + (tid / PREFILL_STAGE_VECS_PER_ROW)
                    * (K_SMEM_STRIDE / BF16_PER_UINT4)
                + (tid % PREFILL_STAGE_VECS_PER_ROW);
            if constexpr (AsyncKWithL2) {
                static_assert(FullQueryBlocks,
                              "async long-K path requires full query blocks");
                for (int step = 0; step < PREFILL_STAGE_STEPS; step++) {
                    cp_async_cg_l2_16_full(dst_it, src_base + src_vi);
                    src_vi += PREFILL_STAGE_SRC_STEP_VECS;
                    // Do not form an unused shared pointer after the final copy.
                    if (step + 1 < PREFILL_STAGE_STEPS) {
                        dst_it += PREFILL_STAGE_DST_STEP_VECS;
                    }
                }
                asm volatile("cp.async.commit_group;\n");
                asm volatile("cp.async.wait_group 0;\n");
            } else {
                int src_row = tid / PREFILL_STAGE_VECS_PER_ROW;
                for (int step = 0; step < PREFILL_STAGE_STEPS; step++) {
                    uint4 data = make_uint4(0, 0, 0, 0);
                    if (src_row < valid_n) {
                        data = src_base[src_vi];
                    }
                    *dst_it = data;
                    src_vi += PREFILL_STAGE_SRC_STEP_VECS;
                    src_row += PREFILL_STAGE_ROWS_PER_STEP;
                    // Do not form an unused shared pointer after the final store.
                    if (step + 1 < PREFILL_STAGE_STEPS) {
                        dst_it += PREFILL_STAGE_DST_STEP_VECS;
                    }
                }
            }
        }
        __syncthreads();  // K is visible; Q tiles are also complete.

        float acc_s[N_TILES][4];
        if constexpr (FullQueryBlocks) {
            // For even M64 query tiles, the last retained N128 block begins
            // at q_block_start and causal visibility ends at column 63.
            // Requiring the exact block start also protects contexts shorter
            // than q_block_start from pruning an earlier fully visible block.
            const bool causal_half_terminal =
                (q_block_start % BLOCK_N == 0) &&
                (kv_block + 1 == num_kv_blocks) &&
                (kv_start == q_block_start);
            if (causal_half_terminal) {
                compute_qk_reg_ldmatrix_half_terminal(
                    q_tile, kv_smem, acc_s);
            } else {
                compute_qk_reg_ldmatrix(q_tile, kv_smem, acc_s);
            }
            if constexpr (AsyncKWithL2) {
                softmax_update_reg<true>(
                    acc_s, acc_o, row_max, row_sum,
                    inv_sqrt, BLOCK_N, kv_start, warp_q_start, MMA_M);
            } else {
                softmax_update_reg<false>(
                    acc_s, acc_o, row_max, row_sum,
                    inv_sqrt, valid_n, kv_start, warp_q_start, MMA_M);
            }
        } else if (actual_rows > 0) {
            compute_qk_reg_ldmatrix(q_tile, kv_smem, acc_s);
            softmax_update_reg<false>(
                acc_s, acc_o, row_max, row_sum,
                inv_sqrt, valid_n, kv_start, warp_q_start, actual_rows);
        }
        __syncthreads();  // Every warp has finished reading K.

        // Reuse the same padded tile for V only after the QK barrier.
        {
            const uint4* src_base =
                reinterpret_cast<const uint4*>(
                    v_ptr + kv_start * HEAD_DIM);
            int src_vi = tid;
            uint4* dst_it = reinterpret_cast<uint4*>(kv_smem)
                + (tid / PREFILL_STAGE_VECS_PER_ROW)
                    * (V_SMEM_STRIDE / BF16_PER_UINT4)
                + (tid % PREFILL_STAGE_VECS_PER_ROW);
            if constexpr (AsyncVWithL2) {
                static_assert(FullQueryBlocks,
                              "async long-V path requires full query blocks");
                static_assert(AsyncKWithL2,
                              "async long-V path requires async long-K staging");
                for (int step = 0; step < PREFILL_STAGE_STEPS; step++) {
                    cp_async_cg_l2_16_full(dst_it, src_base + src_vi);
                    src_vi += PREFILL_STAGE_SRC_STEP_VECS;
                    // Do not form an unused shared pointer after the final copy.
                    if (step + 1 < PREFILL_STAGE_STEPS) {
                        dst_it += PREFILL_STAGE_DST_STEP_VECS;
                    }
                }
                asm volatile("cp.async.commit_group;\n");
                asm volatile("cp.async.wait_group 0;\n");
            } else {
                int src_row = tid / PREFILL_STAGE_VECS_PER_ROW;
                for (int step = 0; step < PREFILL_STAGE_STEPS; step++) {
                    uint4 data = make_uint4(0, 0, 0, 0);
                    if (src_row < valid_n) {
                        data = src_base[src_vi];
                    }
                    *dst_it = data;
                    src_vi += PREFILL_STAGE_SRC_STEP_VECS;
                    src_row += PREFILL_STAGE_ROWS_PER_STEP;
                    // Do not form an unused shared pointer after the final store.
                    if (step + 1 < PREFILL_STAGE_STEPS) {
                        dst_it += PREFILL_STAGE_DST_STEP_VECS;
                    }
                }
            }
        }
        __syncthreads();  // V is visible.

        if constexpr (FullQueryBlocks) {
            const bool causal_half_terminal =
                (q_block_start % BLOCK_N == 0) &&
                (kv_block + 1 == num_kv_blocks) &&
                (kv_start == q_block_start);
            if (causal_half_terminal) {
                compute_pv_reg_per_ktile_half_terminal(
                    acc_s, pv_buf, kv_smem, acc_o);
            } else {
                compute_pv_reg_per_ktile_nt0_preload(
                    acc_s, pv_buf, kv_smem, acc_o);
            }
        } else if (actual_rows > 0) {
            compute_pv_reg_per_ktile(acc_s, pv_buf, kv_smem, acc_o);
        }
        if (kv_block + 1 < num_kv_blocks) {
            __syncthreads();  // Every warp has finished V before the next K load.
        }
    }

    if constexpr (FullQueryBlocks) {
        int group_id     = lane_id >> 2;
        int tid_in_group = lane_id & 3;
        int row0 = group_id, row1 = group_id + 8;
        int col0 = tid_in_group * 2, col1 = col0 + 1;
        // FA2 builds SM80 with --use_fast_math and softmax.h performs this
        // division only at final normalization.  Match that bounded use with
        // __fdividef while retaining the explicit positive-sum fallback.
        float inv0 = (row_sum[0] > 0.0f)
            ? __fdividef(1.0f, row_sum[0]) : 0.0f;
        float inv1 = (row_sum[1] > 0.0f)
            ? __fdividef(1.0f, row_sum[1]) : 0.0f;

        if constexpr (AsyncKWithL2) {
            // Padded row-major shared-output recoalescing.  q_tile and pv_buf
            // are both dead after the final PV and contiguous within this
            // warp, so the 4352-byte layout safely spans q_tile plus the first
            // 256 bytes of pv_buf.  The +1-vector row pad makes the scalar
            // scatter and uint4 gather bank-clean.
            static_assert(MMA_M == 16 && OUT_N_TILES == 16 &&
                          MMA_N == 8 && WARP_SIZE == 32,
                          "padded output epilogue requires M16xN128");
            static_assert(Q_TILE_BYTES == MMA_M * OUT_N_TILES * sizeof(uint4),
                          "q_tile must hold the unpadded vectorized output");

            uint4* out_smem =
                reinterpret_cast<uint4*>(warp_smem + W_Q_TILE_OFF);
            uint32_t* words = reinterpret_cast<uint32_t*>(out_smem);
            uint32_t* w0 =
                words + row0 * PREFILL_OUTPUT_PADDED_ROW_WORDS + tid_in_group;
            uint32_t* w1 =
                w0 + 8 * PREFILL_OUTPUT_PADDED_ROW_WORDS;
            #pragma unroll
            for (int seg = 0; seg < OUT_N_TILES; seg++) {
                const uint32_t packed = pack_bf16x2_rn(
                    acc_o[seg][0] * inv0, acc_o[seg][1] * inv0);
                w0[seg * PREFILL_OUTPUT_UINT4_WORDS] = packed;
                const uint32_t packed_row1 = pack_bf16x2_rn(
                    acc_o[seg][2] * inv1, acc_o[seg][3] * inv1);
                w1[seg * PREFILL_OUTPUT_UINT4_WORDS] = packed_row1;
            }
            __syncwarp();

            uint4* o_output_vec = reinterpret_cast<uint4*>(
                o_ptr + warp_q_start * HEAD_DIM);
            const int gather_base =
                (lane_id & 15) +
                (lane_id >> 4) * PREFILL_OUTPUT_PADDED_ROW_VECS;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                const uint4 data = out_smem[
                    gather_base + i * PREFILL_OUTPUT_GATHER_ROW_STEP_VECS];
                o_output_vec[i * WARP_SIZE + lane_id] = data;
            }
        } else {
            #pragma unroll
            for (int nt = 0; nt < OUT_N_TILES; nt++) {
                int n_base = nt * MMA_N;
                {
                    const uint32_t packed = pack_bf16x2_rn(
                        acc_o[nt][0] * inv0, acc_o[nt][1] * inv0);
                    reinterpret_cast<uint32_t*>(
                        o_ptr + (warp_q_start + row0) * HEAD_DIM +
                        n_base)[tid_in_group] = packed;
                }
                {
                    const uint32_t packed = pack_bf16x2_rn(
                        acc_o[nt][2] * inv1, acc_o[nt][3] * inv1);
                    reinterpret_cast<uint32_t*>(
                        o_ptr + (warp_q_start + row1) * HEAD_DIM +
                        n_base)[tid_in_group] = packed;
                }
            }
        }
    } else if (actual_rows > 0) {
        int group_id     = lane_id >> 2;
        int tid_in_group = lane_id & 3;
        int row0 = group_id, row1 = group_id + 8;
        int col0 = tid_in_group * 2, col1 = col0 + 1;
        float inv0 = (row_sum[0] > 0.0f)
            ? __fdividef(1.0f, row_sum[0]) : 0.0f;
        float inv1 = (row_sum[1] > 0.0f)
            ? __fdividef(1.0f, row_sum[1]) : 0.0f;

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
// Guarded partial-query tail kernel
// ============================================================================
// FUNCTION: unified_attn_prefill_tail_kernel
__global__ __launch_bounds__(BLOCK_THREADS, 3)
void unified_attn_prefill_tail_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    int seq_len, int ctx_len, int num_heads, int num_kv_heads,
    int q_block_base
) {
    constexpr bool FullQueryBlocks = false;
    extern __shared__ char smem[];

    int batch_idx   = int(blockIdx.z);
    int head_idx    = int(blockIdx.y);
    int q_block_idx = q_block_base + int(blockIdx.x);
    int kv_head_idx = head_idx / (num_heads / num_kv_heads);

    int q_block_start = q_block_idx * BLOCK_M_PREFILL;
    if constexpr (!FullQueryBlocks) {
        if (q_block_start >= seq_len) return;
    }

    const __nv_bfloat16* q_ptr = Q + ((batch_idx * num_heads    + head_idx)    * seq_len) * HEAD_DIM;
    const __nv_bfloat16* k_ptr = K + ((batch_idx * num_kv_heads + kv_head_idx) * ctx_len) * HEAD_DIM;
    const __nv_bfloat16* v_ptr = V + ((batch_idx * num_kv_heads + kv_head_idx) * ctx_len) * HEAD_DIM;
    __nv_bfloat16*        o_ptr = O + ((batch_idx * num_heads    + head_idx)    * seq_len) * HEAD_DIM;

    int tid     = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    char* warp_smem = smem + PREFILL_WARP_BASE + warp_id * PER_WARP_BYTES;
    __nv_bfloat16* q_tile =
        reinterpret_cast<__nv_bfloat16*>(warp_smem + W_Q_TILE_OFF);
    __nv_bfloat16* pv_buf =
        reinterpret_cast<__nv_bfloat16*>(warp_smem + W_PV_BUF_OFF);
    __nv_bfloat16* kv_smem =
        reinterpret_cast<__nv_bfloat16*>(smem + PREFILL_KV_BUF_OFF);

    int warp_q_start = q_block_start + warp_id * MMA_M;
    int actual_rows = MMA_M;
    if constexpr (!FullQueryBlocks) {
        int warp_q_end = min(warp_q_start + MMA_M, seq_len);
        actual_rows = max(0, warp_q_end - warp_q_start);
    }

    if constexpr (FullQueryBlocks) {
        constexpr int Q_VECS = MMA_M * HEAD_DIM / 2;
        const uint32_t* q_src_vec = reinterpret_cast<const uint32_t*>(
            q_ptr + warp_q_start * HEAD_DIM);
        uint32_t* q_dst_vec = reinterpret_cast<uint32_t*>(q_tile);
        for (int vi = lane_id; vi < Q_VECS; vi += WARP_SIZE) {
            uint32_t data = q_src_vec[vi];
            q_dst_vec[vi] = data;
        }
    } else {
        for (int i = lane_id; i < MMA_M * HEAD_DIM; i += WARP_SIZE) {
            int row = i / HEAD_DIM, col = i % HEAD_DIM;
            q_tile[i] = (row < actual_rows)
                        ? q_ptr[(warp_q_start + row) * HEAD_DIM + col]
                        : __float2bfloat16(0.0f);
        }
    }

    float acc_o[OUT_N_TILES][4];
    float row_max[2] = {-INFINITY, -INFINITY};
    float row_sum[2] = {0.0f, 0.0f};
    #pragma unroll
    for (int nt = 0; nt < OUT_N_TILES; nt++)
        acc_o[nt][0] = acc_o[nt][1] = acc_o[nt][2] = acc_o[nt][3] = 0.0f;

    float inv_sqrt = 1.0f / sqrtf((float)HEAD_DIM);

    // Iteration-14 causal upper bound, unchanged.
    int q_block_end;
    if constexpr (FullQueryBlocks) {
        q_block_end = q_block_start + BLOCK_M_PREFILL;
    } else {
        q_block_end = min(q_block_start + BLOCK_M_PREFILL, seq_len);
    }
    const int causal_kv_end = min(ctx_len, q_block_end);
    int num_kv_blocks = (causal_kv_end + BLOCK_N - 1) / BLOCK_N;

    // Lockstep 09i schedule: all four warps process the same retained block.
    // The barriers make the one padded tile safe to reuse first for K, then V.
    for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        int kv_start = kv_block * BLOCK_N;
        int kv_end   = min(kv_start + BLOCK_N, ctx_len);
        int valid_n  = kv_end - kv_start;

        // Cooperative K load into the shared padded K/V tile.
        {
            const int vecs_per_row = HEAD_DIM / 8;
            const int vec_valid = valid_n * vecs_per_row;
            const int vec_total = BLOCK_N * vecs_per_row;
            const uint4* src_vec =
                reinterpret_cast<const uint4*>(k_ptr + kv_start * HEAD_DIM);
            for (int vi = tid; vi < vec_total; vi += BLOCK_THREADS) {
                const int n      = vi / vecs_per_row;
                const int vi_row = vi % vecs_per_row;
                uint4 data = (vi < vec_valid)
                    ? src_vec[vi] : make_uint4(0, 0, 0, 0);
                reinterpret_cast<uint4*>(
                    kv_smem + n * K_SMEM_STRIDE)[vi_row] = data;
            }
        }
        __syncthreads();  // K is visible; Q tiles are also complete.

        float acc_s[N_TILES][4];
        if constexpr (FullQueryBlocks) {
            compute_qk_reg(q_tile, kv_smem, acc_s);
            softmax_update_reg(acc_s, acc_o, row_max, row_sum,
                               inv_sqrt, valid_n, kv_start,
                               warp_q_start, MMA_M);
        } else if (actual_rows > 0) {
            compute_qk_reg(q_tile, kv_smem, acc_s);
            softmax_update_reg(acc_s, acc_o, row_max, row_sum,
                               inv_sqrt, valid_n, kv_start,
                               warp_q_start, actual_rows);
        }
        __syncthreads();  // Every warp has finished reading K.

        // Reuse the same padded tile for V only after the QK barrier.
        {
            const int vecs_per_row = HEAD_DIM / 8;
            const int vec_valid = valid_n * vecs_per_row;
            const int vec_total = BLOCK_N * vecs_per_row;
            const uint4* src_vec =
                reinterpret_cast<const uint4*>(v_ptr + kv_start * HEAD_DIM);
            for (int vi = tid; vi < vec_total; vi += BLOCK_THREADS) {
                const int n      = vi / vecs_per_row;
                const int vi_row = vi % vecs_per_row;
                uint4 data = (vi < vec_valid)
                    ? src_vec[vi] : make_uint4(0, 0, 0, 0);
                reinterpret_cast<uint4*>(
                    kv_smem + n * V_SMEM_STRIDE)[vi_row] = data;
            }
        }
        __syncthreads();  // V is visible.

        if constexpr (FullQueryBlocks) {
            compute_pv_reg_per_ktile(acc_s, pv_buf, kv_smem, acc_o);
        } else if (actual_rows > 0) {
            compute_pv_reg_per_ktile(acc_s, pv_buf, kv_smem, acc_o);
        }
        __syncthreads();  // Every warp has finished V before the next K load.
    }

    if constexpr (FullQueryBlocks) {
        int group_id     = lane_id >> 2;
        int tid_in_group = lane_id & 3;
        int row0 = group_id, row1 = group_id + 8;
        int col0 = tid_in_group * 2, col1 = col0 + 1;
        float inv0 = (row_sum[0] > 0.0f)
            ? __fdividef(1.0f, row_sum[0]) : 0.0f;
        float inv1 = (row_sum[1] > 0.0f)
            ? __fdividef(1.0f, row_sum[1]) : 0.0f;

        #pragma unroll
        for (int nt = 0; nt < OUT_N_TILES; nt++) {
            int n_base = nt * MMA_N;
            {
                uint32_t packed = uint32_t(__bfloat16_as_ushort(
                    __float2bfloat16(acc_o[nt][0] * inv0)));
                packed |= uint32_t(__bfloat16_as_ushort(
                    __float2bfloat16(acc_o[nt][1] * inv0))) << 16;
                reinterpret_cast<uint32_t*>(
                    o_ptr + (warp_q_start + row0) * HEAD_DIM +
                    n_base)[tid_in_group] = packed;
            }
            {
                uint32_t packed = uint32_t(__bfloat16_as_ushort(
                    __float2bfloat16(acc_o[nt][2] * inv1)));
                packed |= uint32_t(__bfloat16_as_ushort(
                    __float2bfloat16(acc_o[nt][3] * inv1))) << 16;
                reinterpret_cast<uint32_t*>(
                    o_ptr + (warp_q_start + row1) * HEAD_DIM +
                    n_base)[tid_in_group] = packed;
            }
        }
    } else if (actual_rows > 0) {
        int group_id     = lane_id >> 2;
        int tid_in_group = lane_id & 3;
        int row0 = group_id, row1 = group_id + 8;
        int col0 = tid_in_group * 2, col1 = col0 + 1;
        float inv0 = (row_sum[0] > 0.0f)
            ? __fdividef(1.0f, row_sum[0]) : 0.0f;
        float inv1 = (row_sum[1] > 0.0f)
            ? __fdividef(1.0f, row_sum[1]) : 0.0f;

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
// FUNCTION: unified_attn_decode_score_producer
template <bool FullN128>
__global__ void unified_attn_decode_score_producer(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    float* __restrict__ scores,
    float* __restrict__ block_maxima,
    int ctx_len, int num_heads, int num_kv_heads, int num_kv_blocks
) {
    extern __shared__ char smem[];

    const int kv_block      = blockIdx.x;
    const int kv_head_batch = blockIdx.y;
    const int batch_idx     = kv_head_batch / num_kv_heads;
    const int kv_head_idx   = kv_head_batch % num_kv_heads;
    const int q_head_base   = kv_head_idx * 4;
    const int tid           = threadIdx.x;
    const int warp_id       = tid / WARP_SIZE;
    const int lane_id       = tid % WARP_SIZE;

    const __nv_bfloat16* q_ptr =
        Q + (batch_idx * num_heads + q_head_base) * HEAD_DIM;
    const __nv_bfloat16* k_ptr =
        K + ((batch_idx * num_kv_heads + kv_head_idx) * ctx_len) * HEAD_DIM;
    __nv_bfloat16* k_smem = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* q_tile =
        reinterpret_cast<__nv_bfloat16*>(smem + K_BUF_BYTES);

    // Pack only the four live GQA queries. compute_qk_reg_live4 aliases all
    // inactive logical MMA rows back to this defined shared-memory region.
    const uint4* q_src_vec = reinterpret_cast<const uint4*>(q_ptr);
    uint4* q_dst_vec = reinterpret_cast<uint4*>(q_tile);
    constexpr int Q_VECS_PER_ROW = HEAD_DIM / 8;
    constexpr int Q_VALID_VECS = 4 * Q_VECS_PER_ROW;
    if (tid < Q_VALID_VECS) {
        q_dst_vec[tid] = q_src_vec[tid];
    }

    const int kv_start = kv_block * BLOCK_N;
    const int valid_n  = FullN128
        ? BLOCK_N
        : min(BLOCK_N, ctx_len - kv_start);
    constexpr int K_VECS_PER_ROW = HEAD_DIM / 8;
    constexpr int K_VEC_TOTAL = BLOCK_N * K_VECS_PER_ROW;
    const uint4* k_src_vec =
        reinterpret_cast<const uint4*>(k_ptr + kv_start * HEAD_DIM);
    if constexpr (FullN128) {
        for (int vi = tid; vi < K_VEC_TOTAL; vi += BLOCK_THREADS) {
            const int n      = vi / K_VECS_PER_ROW;
            const int vi_row = vi % K_VECS_PER_ROW;
            reinterpret_cast<uint4*>(
                k_smem + n * K_SMEM_STRIDE)[vi_row] = k_src_vec[vi];
        }
    } else {
        const int vec_valid = valid_n * K_VECS_PER_ROW;
        for (int vi = tid; vi < K_VEC_TOTAL; vi += BLOCK_THREADS) {
            const int n      = vi / K_VECS_PER_ROW;
            const int vi_row = vi % K_VECS_PER_ROW;
            const uint4 data = (vi < vec_valid)
                ? k_src_vec[vi]
                : make_uint4(0, 0, 0, 0);
            reinterpret_cast<uint4*>(
                k_smem + n * K_SMEM_STRIDE)[vi_row] = data;
        }
    }
    __syncthreads();

    // One warp evaluates the four packed rows. Publish raw QK scores and
    // scale only the post-quad maximum so consumers retain the established
    // scaled-natural online-softmax recurrence.
    if (warp_id == 0) {
        float acc_s[N_TILES][4];
        compute_qk_reg_live4(q_tile, k_smem, acc_s);
        const float inv_sqrt = 1.0f / sqrtf((float)HEAD_DIM);

        const int query_row = lane_id >> 2;
        const int col_pair  = lane_id & 3;
        float block_max = -INFINITY;
        if (query_row < 4) {
            const int q_head = q_head_base + query_row;
            float* score_ptr = scores +
                (((batch_idx * num_heads + q_head) * num_kv_blocks + kv_block)
                 * BLOCK_N);
            #pragma unroll
            for (int nt = 0; nt < N_TILES; nt++) {
                const int n0 = nt * MMA_N + col_pair * 2;
                const int n1 = n0 + 1;
                const float raw0 = acc_s[nt][0];
                const float raw1 = acc_s[nt][1];
                score_ptr[n0] = raw0;
                score_ptr[n1] = raw1;

                if constexpr (FullN128) {
                    block_max = fmaxf(
                        block_max, fmaxf(raw0, raw1));
                } else {
                    // Match the consumer's exact nt-major masked sequence.
                    const bool mask0 = (n0 >= valid_n) ||
                                       (kv_start + n0 > ctx_len - 1);
                    const bool mask1 = (n1 >= valid_n) ||
                                       (kv_start + n1 > ctx_len - 1);
                    const float score0 = mask0 ? -INFINITY : raw0;
                    const float score1 = mask1 ? -INFINITY : raw1;
                    block_max = fmaxf(
                        block_max, fmaxf(score0, score1));
                }
            }
        }
        // All lanes execute the same xor-1/xor-2 quad reduction. Quads 0..3
        // are the packed heads; the remaining zero-query quads stay -inf.
        block_max = quad_max(block_max);
        block_max *= inv_sqrt;
        if (query_row < 4 && col_pair == 0) {
            const int q_head = q_head_base + query_row;
            block_maxima[
                (batch_idx * num_heads + q_head) * num_kv_blocks + kv_block] =
                block_max;
        }
    }
}

// FUNCTION: decode_exp_sum
// A finite producer maximum proves that this full score block contains no
// +infinity.  Negative infinity is still safe in the check-free path because
// ex2.approx.ftz.f32(-Inf) is exactly +0; NaNs propagate in both paths.  Keep
// the checked instantiation for generic tails and non-finite full blocks so
// the public API retains its original non-finite-input behavior.
template <bool CheckInf>
__device__ __forceinline__ float decode_exp_sum(
    float live_scores[N_TILES][2], float new_max
) {
    // live_scores are raw QK values while new_max deliberately stays in the
    // scaled-natural domain used by the unchanged consumer recurrence.
    const float inv_sqrt = 1.0f / sqrtf((float)HEAD_DIM);
    const float softmax_scale_log2 = inv_sqrt * LOG2E;
    const float new_max_log2 = new_max * LOG2E;
    float block_sum = 0.0f;
    #pragma unroll
    for (int nt = 0; nt < N_TILES; nt++) {
        float p0;
        float p1;
        if constexpr (CheckInf) {
            p0 = isinf(live_scores[nt][0])
                ? 0.0f
                : fast_exp2_ftz(fmaf(
                    live_scores[nt][0], softmax_scale_log2,
                    -new_max_log2));
            p1 = isinf(live_scores[nt][1])
                ? 0.0f
                : fast_exp2_ftz(fmaf(
                    live_scores[nt][1], softmax_scale_log2,
                    -new_max_log2));
        } else {
            p0 = fast_exp2_ftz(fmaf(
                live_scores[nt][0], softmax_scale_log2, -new_max_log2));
            p1 = fast_exp2_ftz(fmaf(
                live_scores[nt][1], softmax_scale_log2, -new_max_log2));
        }
        live_scores[nt][0] = p0;
        live_scores[nt][1] = p1;
        block_sum += p0 + p1;
    }
    return block_sum;
}

// FUNCTION: unified_attn_decode_d8_consumer
template <bool FullN128>
__global__ void unified_attn_decode_d8_consumer(
    const float* __restrict__ scores,
    const float* __restrict__ block_maxima,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    int ctx_len, int num_heads, int num_kv_heads, int num_kv_blocks
) {
    extern __shared__ char smem[];

    const int output_tile   = blockIdx.x;
    const int kv_head_batch = blockIdx.y;
    const int batch_idx     = kv_head_batch / num_kv_heads;
    const int kv_head_idx   = kv_head_batch % num_kv_heads;
    const int tid           = threadIdx.x;
    const int warp_id       = tid / WARP_SIZE;
    const int lane_id       = tid % WARP_SIZE;
    const int q_head        = kv_head_idx * 4 + warp_id;
    const int head_batch_idx = batch_idx * num_heads + q_head;

    const __nv_bfloat16* v_ptr =
        V + ((batch_idx * num_kv_heads + kv_head_idx) * ctx_len) * HEAD_DIM;
    __nv_bfloat16* o_ptr =
        O + (batch_idx * num_heads + q_head) * HEAD_DIM;

    // DECODE_D8_SMEM_BYTES deliberately still reserves each warp's old
    // PV_BUF_BYTES region so this iteration isolates instruction savings from
    // any occupancy change; only the consumer stops touching that region.

    // Four CTA-uniform cp.async groups seed the compact-V ring. Each thread
    // owns one naturally aligned D8 row vector. Invalid tail threads retain a
    // valid aligned address while src_bytes=0 requests hardware zero-fill.
    const int prologue_groups = min(DECODE_D8_V_STAGES, num_kv_blocks);
    for (int stage = 0; stage < prologue_groups; stage++) {
        const int stage_start = stage * BLOCK_N;
        __nv_bfloat16* stage_dst = reinterpret_cast<__nv_bfloat16*>(
            smem + stage * DECODE_D8_V_STAGE_BYTES);
        const uint4* stage_src_base = reinterpret_cast<const uint4*>(
            v_ptr + stage_start * HEAD_DIM + output_tile * MMA_N);
        if constexpr (FullN128) {
            cp_async_cg_16(
                stage_dst + tid * MMA_N,
                stage_src_base + tid * OUT_N_TILES, true);
        } else {
            const int stage_valid =
                min(BLOCK_N, ctx_len - stage_start);
            const bool valid_copy = tid < stage_valid;
            const uint4* safe_stage_src = valid_copy
                ? stage_src_base + tid * OUT_N_TILES
                : stage_src_base;
            cp_async_cg_16(
                stage_dst + tid * MMA_N,
                safe_stage_src, valid_copy);
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");
    }

    float acc_o[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float row_max = -INFINITY;
    float row_sum = 0.0f;
    const int group_id = lane_id >> 2;
    const int col_pair = lane_id & 3;
    const bool live_row_lane = (group_id == 0);
    const int col0 = col_pair * 2;
    const int col1 = col0 + 1;

    for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        const int kv_start = kv_block * BLOCK_N;
        const int valid_n  = FullN128
            ? BLOCK_N
            : min(BLOCK_N, ctx_len - kv_start);

        // The current block is always the oldest outstanding copy group.
        // Wait only far enough to expose it, leaving newer ring stages in
        // flight under this block's scalar softmax and kt0..7 PV traversal.
        const int pending_groups =
            min(DECODE_D8_V_STAGES, num_kv_blocks - kv_block);
        if (pending_groups >= 4) {
            asm volatile("cp.async.wait_group 3;\n" ::: "memory");
        } else if (pending_groups == 3) {
            asm volatile("cp.async.wait_group 2;\n" ::: "memory");
        } else if (pending_groups == 2) {
            asm volatile("cp.async.wait_group 1;\n" ::: "memory");
        } else {
            asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        }
        __syncthreads();
        const __nv_bfloat16* v_smem =
            reinterpret_cast<const __nv_bfloat16*>(
                smem + (kv_block & (DECODE_D8_V_STAGES - 1)) *
                       DECODE_D8_V_STAGE_BYTES);

        // Only lanes 0..3 own the single live decode row. Retain its two
        // score components in the original nt-major order, cutting the score
        // fragment from 64 to 32 FP32 registers per thread at source level.
        float live_scores[N_TILES][2];
        const float* score_ptr = scores +
            (((batch_idx * num_heads + q_head) * num_kv_blocks + kv_block)
             * BLOCK_N);

        // Preserve the exact score loads and masks needed by the exponent
        // pass. The producer has already executed this block's identical
        // nt-major fmaxf sequence and xor-1/xor-2 quad reduction.
        if (live_row_lane) {
            #pragma unroll
            for (int nt = 0; nt < N_TILES; nt++) {
                const int n_base = nt * MMA_N;
                const int n0 = n_base + col0;
                const int n1 = n_base + col1;
                if constexpr (FullN128) {
                    // n0 is even and every score block starts on a 512-byte
                    // boundary, so this adjacent pair is naturally aligned.
                    const float2 score_pair =
                        *reinterpret_cast<const float2*>(score_ptr + n0);
                    live_scores[nt][0] = score_pair.x;
                    live_scores[nt][1] = score_pair.y;
                } else {
                    const float s0 = score_ptr[n0];
                    const float s1 = score_ptr[n1];
                    const bool mask0 = (n0 >= valid_n) ||
                                       (kv_start + n0 > ctx_len - 1);
                    const bool mask1 = (n1 >= valid_n) ||
                                       (kv_start + n1 > ctx_len - 1);
                    live_scores[nt][0] = mask0 ? -INFINITY : s0;
                    live_scores[nt][1] = mask1 ? -INFINITY : s1;
                }
            }
        }
        const float block_max = block_maxima[
            head_batch_idx * num_kv_blocks + kv_block];

        float new_max = -INFINITY;
        if (live_row_lane) {
            new_max = fmaxf(row_max, block_max);
            const float rescale = isinf(new_max)
                ? 1.0f : fast_exp2_ftz((row_max - new_max) * LOG2E);
            row_max = new_max;

            // Only d0/d1 belong to row zero. The other MMA accumulator rows
            // remain exactly zero and do not require online rescaling.
            acc_o[0] *= rescale;
            acc_o[1] *= rescale;
            row_sum *= rescale;
        }

        float block_sum = 0.0f;
        if (live_row_lane) {
            if constexpr (FullN128) {
                block_sum = isfinite(block_max)
                    ? decode_exp_sum<false>(live_scores, new_max)
                    : decode_exp_sum<true>(live_scores, new_max);
            } else {
                block_sum = decode_exp_sum<true>(live_scores, new_max);
            }
        }
        block_sum = quad_sum(block_sum);
        if (live_row_lane) {
            row_sum += block_sum;
        }

        // NVIDIA's row-major m16n8k16 A mapping assigns packed registers as:
        //   a0: row g,   K[0:1]   a1: row g+8, K[0:1]
        //   a2: row g,   K[8:9]   a3: row g+8, K[8:9]
        // for lane group g=lane>>2 (the lane-in-group selects the K pair).
        // Decode has only row zero, so group zero packs a0/a2 and every other
        // packed operand is the exact uint32 BF16 +0,+0 bit pattern.
        for (int kt = 0; kt < KV_K_TILES; kt++) {
            const int nt0 = kt * 2;
            const int nt1 = nt0 + 1;
            const uint32_t a0 = live_row_lane
                ? pack_bf16x2_rn(
                    live_scores[nt0][0], live_scores[nt0][1])
                : 0u;
            const uint32_t a1 = 0u;
            const uint32_t a2 = live_row_lane
                ? pack_bf16x2_rn(
                    live_scores[nt1][0], live_scores[nt1][1])
                : 0u;
            const uint32_t a3 = 0u;

            const int k_base = (lane_id % 4) * 2;
            const int k0 = kt * MMA_K + k_base;
            const int k1 = k0 + 1;
            const int k8 = k0 + 8;
            const int k9 = k8 + 1;
            const int n  = lane_id / 4;
            __nv_bfloat16 e0 = v_smem[k0 * MMA_N + n];
            __nv_bfloat16 e1 = v_smem[k1 * MMA_N + n];
            __nv_bfloat16 e8 = v_smem[k8 * MMA_N + n];
            __nv_bfloat16 e9 = v_smem[k9 * MMA_N + n];
            const uint32_t b0 =
                (reinterpret_cast<uint16_t&>(e1) << 16) |
                 reinterpret_cast<uint16_t&>(e0);
            const uint32_t b1 =
                (reinterpret_cast<uint16_t&>(e9) << 16) |
                 reinterpret_cast<uint16_t&>(e8);
            mma_bf16(acc_o[0], acc_o[1], acc_o[2], acc_o[3],
                     a0, a1, a2, a3, b0, b1,
                     acc_o[0], acc_o[1], acc_o[2], acc_o[3]);
        }
        // All four warps have finished reading the current stage. Refill that
        // same ring slot with V[b+4], keeping at most four committed groups.
        __syncthreads();
        const int refill_block = kv_block + DECODE_D8_V_STAGES;
        if (refill_block < num_kv_blocks) {
            const int refill_start = refill_block * BLOCK_N;
            __nv_bfloat16* refill_dst = reinterpret_cast<__nv_bfloat16*>(
                smem + (kv_block & (DECODE_D8_V_STAGES - 1)) *
                       DECODE_D8_V_STAGE_BYTES);
            const uint4* refill_src_base =
                reinterpret_cast<const uint4*>(
                    v_ptr + refill_start * HEAD_DIM +
                    output_tile * MMA_N);
            if constexpr (FullN128) {
                cp_async_cg_16(
                    refill_dst + tid * MMA_N,
                    refill_src_base + tid * OUT_N_TILES, true);
            } else {
                const int refill_valid =
                    min(BLOCK_N, ctx_len - refill_start);
                const bool valid_copy = tid < refill_valid;
                const uint4* safe_refill_src = valid_copy
                    ? refill_src_base + tid * OUT_N_TILES
                    : refill_src_base;
                cp_async_cg_16(
                    refill_dst + tid * MMA_N,
                    safe_refill_src, valid_copy);
            }
            asm volatile("cp.async.commit_group;\n" ::: "memory");
        }
    }

    if (live_row_lane) {
        const float inv0 = (row_sum > 0.0f)
            ? __fdividef(1.0f, row_sum) : 0.0f;
        const int n_base = output_tile * MMA_N;
        o_ptr[n_base + col0] = __float2bfloat16(acc_o[0] * inv0);
        o_ptr[n_base + col1] = __float2bfloat16(acc_o[1] * inv0);
    }
}

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
    __nv_bfloat16* q_tile   = reinterpret_cast<__nv_bfloat16*>(w0_smem + W_Q_TILE_OFF);
    __nv_bfloat16* pv_buf   = reinterpret_cast<__nv_bfloat16*>(w0_smem + W_PV_BUF_OFF);
    // Decode uses separate K and V buffers (no staggering, just slot 0 of each)
    __nv_bfloat16* k_smem_p = reinterpret_cast<__nv_bfloat16*>(smem + K_BUF_0_OFF);
    __nv_bfloat16* v_smem_p = reinterpret_cast<__nv_bfloat16*>(smem + V_BUF_0_OFF);

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

        // Load K into padded smem (same as prefill)
        {
            const int vecs_per_row = HEAD_DIM / 8;
            const int vec_valid = valid_n * vecs_per_row;
            const int vec_total = BLOCK_N * vecs_per_row;
            const uint4* src_vec = reinterpret_cast<const uint4*>(k_ptr + kv_start * HEAD_DIM);
            for (int vi = tid; vi < vec_total; vi += BLOCK_THREADS) {
                const int n      = vi / vecs_per_row;
                const int vi_row = vi % vecs_per_row;
                uint4 data = (vi < vec_valid) ? src_vec[vi] : make_uint4(0, 0, 0, 0);
                reinterpret_cast<uint4*>(k_smem_p + n * K_SMEM_STRIDE)[vi_row] = data;
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
                reinterpret_cast<uint4*>(v_smem_p + n * V_SMEM_STRIDE)[vi_row] = data;
            }
        }
        __syncthreads();  // sync #3

        if (warp_id == 0) {
            compute_pv_reg_per_ktile(acc_s, pv_buf, v_smem_p, acc_o);
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
// FUNCTION: unified_prefill
torch::Tensor unified_prefill(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                               int num_heads, int num_kv_heads) {
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda());
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous());
    TORCH_CHECK(Q.dtype() == torch::kBFloat16);
    TORCH_CHECK(Q.size(3) == HEAD_DIM);
    int batch = Q.size(0), seq_len = Q.size(2), ctx_len = K.size(2);
    auto O = torch::zeros_like(Q);
    const int full_q_blocks = seq_len / BLOCK_M_PREFILL;
    const int tail_rows = seq_len % BLOCK_M_PREFILL;

    if (full_q_blocks > 0) {
        const int full_q_rows = full_q_blocks * BLOCK_M_PREFILL;
        const bool aligned_full_prefix =
            full_q_rows % BLOCK_N == 0 && ctx_len >= full_q_rows;
        if (full_q_rows >= 8192 && aligned_full_prefix) {
            cudaFuncSetAttribute(unified_attn_prefill_kernel<true, true, true>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 PREFILL_SMEM_BYTES);
            unified_attn_prefill_kernel<true, true, true><<<
                dim3(full_q_blocks, num_heads, batch),
                BLOCK_THREADS, PREFILL_SMEM_BYTES>>>(
                (const __nv_bfloat16*)Q.data_ptr(),
                (const __nv_bfloat16*)K.data_ptr(),
                (const __nv_bfloat16*)V.data_ptr(),
                (__nv_bfloat16*)O.data_ptr(),
                seq_len, ctx_len, num_heads, num_kv_heads);
        } else if (full_q_rows >= 4096 && aligned_full_prefix) {
            cudaFuncSetAttribute(unified_attn_prefill_kernel<true, true, false>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 PREFILL_SMEM_BYTES);
            unified_attn_prefill_kernel<true, true, false><<<
                dim3(full_q_blocks, num_heads, batch),
                BLOCK_THREADS, PREFILL_SMEM_BYTES>>>(
                (const __nv_bfloat16*)Q.data_ptr(),
                (const __nv_bfloat16*)K.data_ptr(),
                (const __nv_bfloat16*)V.data_ptr(),
                (__nv_bfloat16*)O.data_ptr(),
                seq_len, ctx_len, num_heads, num_kv_heads);
        } else {
            cudaFuncSetAttribute(unified_attn_prefill_kernel<true, false, false>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 PREFILL_SMEM_BYTES);
            unified_attn_prefill_kernel<true, false, false><<<
                dim3(full_q_blocks, num_heads, batch),
                BLOCK_THREADS, PREFILL_SMEM_BYTES>>>(
                (const __nv_bfloat16*)Q.data_ptr(),
                (const __nv_bfloat16*)K.data_ptr(),
                (const __nv_bfloat16*)V.data_ptr(),
                (__nv_bfloat16*)O.data_ptr(),
                seq_len, ctx_len, num_heads, num_kv_heads);
        }
    }

    if (tail_rows > 0) {
        cudaFuncSetAttribute(unified_attn_prefill_tail_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             PREFILL_SMEM_BYTES);
        unified_attn_prefill_tail_kernel<<<
            dim3(1, num_heads, batch),
            BLOCK_THREADS, PREFILL_SMEM_BYTES>>>(
            (const __nv_bfloat16*)Q.data_ptr(), (const __nv_bfloat16*)K.data_ptr(),
            (const __nv_bfloat16*)V.data_ptr(), (__nv_bfloat16*)O.data_ptr(),
            seq_len, ctx_len, num_heads, num_kv_heads, full_q_blocks);
    }
    return O;
}

// FUNCTION: unified_decode
torch::Tensor unified_decode(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                              int num_heads, int num_kv_heads) {
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda());
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous());
    TORCH_CHECK(Q.size(2) == 1);
    TORCH_CHECK(Q.size(3) == HEAD_DIM);
    TORCH_CHECK(num_heads == 4 * num_kv_heads,
                "split D8 decode requires exactly 4:1 GQA");
    int batch = Q.size(0), ctx_len = K.size(2);
    int num_kv_blocks = (ctx_len + BLOCK_N - 1) / BLOCK_N;
    auto O = torch::zeros_like(Q);
    auto scores = torch::empty(
        {batch, num_heads, num_kv_blocks, BLOCK_N},
        Q.options().dtype(torch::kFloat32));
    auto block_maxima = torch::empty(
        {batch, num_heads, num_kv_blocks},
        Q.options().dtype(torch::kFloat32));

    const bool full_n128 = (ctx_len % BLOCK_N) == 0;
    if (full_n128) {
        cudaFuncSetAttribute(unified_attn_decode_score_producer<true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             DECODE_SCORE_SMEM_BYTES);
        unified_attn_decode_score_producer<true><<<
            dim3(num_kv_blocks, batch * num_kv_heads),
            BLOCK_THREADS, DECODE_SCORE_SMEM_BYTES>>>(
            (const __nv_bfloat16*)Q.data_ptr(),
            (const __nv_bfloat16*)K.data_ptr(),
            scores.data_ptr<float>(),
            block_maxima.data_ptr<float>(),
            ctx_len, num_heads, num_kv_heads, num_kv_blocks);

        unified_attn_decode_d8_consumer<true><<<
            dim3(OUT_N_TILES, batch * num_kv_heads),
            BLOCK_THREADS, DECODE_D8_SMEM_BYTES>>>(
            scores.data_ptr<float>(),
            block_maxima.data_ptr<float>(),
            (const __nv_bfloat16*)V.data_ptr(),
            (__nv_bfloat16*)O.data_ptr(),
            ctx_len, num_heads, num_kv_heads, num_kv_blocks);
    } else {
        cudaFuncSetAttribute(unified_attn_decode_score_producer<false>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             DECODE_SCORE_SMEM_BYTES);
        unified_attn_decode_score_producer<false><<<
            dim3(num_kv_blocks, batch * num_kv_heads),
            BLOCK_THREADS, DECODE_SCORE_SMEM_BYTES>>>(
            (const __nv_bfloat16*)Q.data_ptr(),
            (const __nv_bfloat16*)K.data_ptr(),
            scores.data_ptr<float>(),
            block_maxima.data_ptr<float>(),
            ctx_len, num_heads, num_kv_heads, num_kv_blocks);

        unified_attn_decode_d8_consumer<false><<<
            dim3(OUT_N_TILES, batch * num_kv_heads),
            BLOCK_THREADS, DECODE_D8_SMEM_BYTES>>>(
            scores.data_ptr<float>(),
            block_maxima.data_ptr<float>(),
            (const __nv_bfloat16*)V.data_ptr(),
            (__nv_bfloat16*)O.data_ptr(),
            ctx_len, num_heads, num_kv_heads, num_kv_blocks);
    }
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("unified_prefill", &unified_prefill, "Unified prefill (09k: staggered warp-group pipeline)");
    m.def("unified_decode",  &unified_decode,  "Unified decode  (09k: staggered warp-group pipeline)");
}
