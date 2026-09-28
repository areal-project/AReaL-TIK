// Iteration 235 — Repo220 long-prefill one-shot Q descriptor-prefetch elision
// Exact base: kernel_iter220_h200_repo212_b14_l1024_acquire_sync_elision.cu
// Base SHA-256: 185953cf40b3a9d1a9e0bc2a0746b3931de8995f53c63e1756407abb2571f2c5
//
// Repo220 makes thread 0 execute three tensor-map descriptor-prefetch hints
// before initializing the transaction barriers and releasing setup.  The Q
// tensor map is immutable and shared by every CTA in the launch, and each CTA
// uses it for only one 16-KiB Q transaction.  For aligned long prefill
// <true,true> only, omit that one advisory Q-descriptor prefetch while keeping
// the repeated-loop K and Vt descriptor prefetches at their exact Repo220
// sites.  This removes one UTMACCTL-style setup instruction from every long
// CTA and avoids amplifying the same one-shot descriptor hint across 2,048,
// 4,096, or 8,192 CTAs at 4K, 8K, or 16K respectively.
//
// Static correctness proof: tensor-map prefetch is a nonblocking performance
// hint, not a data, transaction, or ordering operation.  Q's unchanged CuTe
// copy still consumes the identical tma_q descriptor, writes the identical
// shared tile, completes the same Q_WGMMA_BYTES on q_tma_mbar, and remains
// gated by the same all-thread acquire wait before QK.  K/V descriptors, all
// actual Q/K/V TMA instructions and coordinates, mbarrier init/arrival/phase/
// wait accounting, CTA rendezvous, WGMMA, shared-memory aliasing, floating-
// point values/order, output stores, dynamic shared memory, and launch geometry
// are exact Repo220.
//
// VtUnderQkExtract is selected only by aligned self-attention with seq_len >=
// 4096, so configured source-affected rows are exactly prefill 4K/8K/16K.
// Aligned 1K <true,false> and generic/tail <false,false> retain Q/K/V descriptor
// prefetches byte-for-byte.  Repo220's promoted ConsumerLocalAcquire B1/B4
// exact-L1024 decode specializations, host dispatch, and every other decode
// path remain source-identical.
//
// Reference: Hopper PTX tensor-map descriptor prefetch is advisory; the actual
// cp.async.bulk.tensor instruction remains the descriptor consumer.  FA3/CuTe
// uses prefetch_tma_descriptor as producer-side latency preparation, which
// makes a single-use, massively duplicated Q hint separable from K/V's loop-
// repeated descriptors.  This candidate tests that H200 descriptor residency
// and launch-wide reuse make the repeated Q hint more expensive than useful.
//
// History audit: Repo85 cached host descriptor construction; Repo139 spread
// all three prefetch/init operations across warp leaders; Repo188 moved Q TMA
// relative to the setup rendezvous; Repo227 hoisted K/V descriptor handles;
// queued Repo233 moves K/V prefetch hints under Q.  No retained H20/H200
// snapshot or ledger entry elides only the long path's one-shot Q descriptor
// prefetch.  This candidate combines none of those mechanisms.
//
// Risk: a CTA whose Q descriptor is not already resident can expose descriptor
// fetch latency at its actual Q TMA, so the candidate can regress startup even
// though the hint is redundant after launch-wide warmup.  Resource/SASS audit
// must confirm exactly the intended long-symbol hint removal, no spill, and no
// occupancy-class change before timing.
//
// Iteration 220 — Repo212-rebased B1/B4 L1024 acquire-sync elision
// Exact base: kernel_iter212_h200_long_prefill_head_coordinate_under_q_tma.cu
// Base SHA-256: c2e2e860902bbb3dd12838b1cf32f8b37585ad78a15d4210256d857fbf809e2f
//
// The exact-L1024 short-decode kernel makes all 128 threads execute the K
// transaction-barrier acquire wait before its aligned QK WGMMA, and likewise
// makes all 128 threads execute the V acquire wait before aligned PV WGMMA.
// A completed acquire wait already makes the corresponding TMA writes visible
// to each consumer.  Compile out only the two immediately following full-CTA
// rendezvous for batch 1 and batch 4.  The post-QK barrier that protects K
// shared memory from the aliased V overwrite and the post-PV barrier that
// protects V from the next K overwrite remain unchanged on every KV block.
//
// The host selects a dedicated <true> specialization only for B1/B4 at exactly
// 1024 tokens; B8/B16 and arbitrary other batches retain <false> with both
// rendezvous.  Thus configured source-affected rows are decode B1/L1024 and
// B4/L1024.  Their B1/B4 4K/8K rows retain Repo212's common full-N128 QK
// producer plus generic D16 or stage-owned D32 consumer, respectively.
// Prefill and all long-decode device/dispatch source remain unchanged.
//
// TMA descriptors, issue ownership/order, coordinates and bytes; mbarrier
// initialization, arrivals, phases and waits; Q staging; WGMMA slices;
// softmax/PV floating-point order; shared-memory layout; output stores; and
// launch geometry are unchanged.  This extends the promoted consumer-local
// acquire-wait reasoning of Repo166 (long prefill) and Repo183 (long-decode K
// producer) to the previously untouched exact-L1024 short-decode symbol.
// Repo189 instead removed statically true nonempty guards from long consumers;
// no prior ledger or retained snapshot elides these exact-short acquire syncs.
// Repo108 changed short K/V issue scaffolding, Repo114 vectorized Q staging,
// Repo132 compiled out exact-N128 tails, and Repo147 changed only L1024 V's
// tensor-map L2 policy; each retained both post-acquire rendezvous.
// Emerging Repo219 changes only the exact-N128 long producer by forming its
// score/max workspace coordinate under split K TMA; Repo220 changes neither
// that producer nor its 4K/8K dispatch and shares no semantic device hunk.
//
// Risk: ptxas may have hidden part of these barriers behind the eight-block
// short pipeline, or the extra specialization/host branch may offset the saved
// rendezvous at launch-dominated B1.  Correctness risk is low because all
// acquire waits and both shared-slot reuse barriers are retained.
//
// Iteration 212 — long-prefill head/KV coordinate derivation under Q TMA
// Exact base: kernel_iter191_h200_direct_split_consumer_coordinates.cu
// Base SHA-256: 732eecbaeb96af9eaacba63436f2e1775edb14ec689169f057632efe56f93e09
//
// Repo191 derives batch_idx, head_idx, kv_head_idx, the output-head pointer,
// and kv_row_base before descriptor setup and before issuing the CTA's one-shot
// 16-KiB Q TMA.  None of those values participates in Q's tensor-map address:
// that path uses head_batch_idx directly.  For aligned long prefill, move the
// unchanged derivation bundle to immediately after the Q issue and before the
// same all-thread Q completion wait, so the transfer can cover its runtime
// integer quotient/remainder and address arithmetic.
//
// The exact expressions and their order inside the bundle are unchanged.
// Q is still issued by thread 0 only; every thread then derives identical
// per-CTA values and executes the original acquire wait before any QK WGMMA or
// K/V TMA use.  The output pointer remains live through the mainloop exactly
// as in Repo191 rather than being deferred to the epilogue.
//
// VtUnderQkExtract selects the relocation only for aligned long prefill
// <true,true>, affecting configured 4K/8K/16K.  Aligned 1K and generic
// prefill execute the bundle at Repo191's prologue site; all decode code is
// untouched.  Descriptor prefetch and Q/K/V coordinates, TMA issues and byte
// counts, barrier initialization/arrivals/phases/waits, CTA rendezvous,
// WGMMA, floating-point arithmetic, stores, shared memory, and launch geometry
// are otherwise unchanged.
//
// History audit: Repo139 distributed descriptor/barrier setup; Repo188 moved
// the Q issue relative to its setup rendezvous; Repo194 pre-armed K phases;
// Repo200/202 changed per-block V-coordinate operands; Repo208 moved the
// output-accumulator clear under Q; and Repo210 deferred only the output-head
// pointer until after the mainloop.  Repo212 changes none of those mechanisms:
// it retains the complete coordinate bundle and relocates it only across the
// already-open Q transaction.  No prior ledger entry schedules the GQA/KV-row
// derivation inside that transfer window.
//
// Risk: ptxas may already reduce part of the quotient/remainder bundle or the
// Q transfer may cover it only at 4K.  The source introduces no new value and
// keeps the REG168 mainloop live ranges and three-CTA residency hypothesis.
//
// Iteration 191 — Repo186-rebased direct split-output consumer coordinates
// Exact base: kernel_iter186_h200_stage_owner_mbar_init.cu
// Base SHA-256: 61d9a4818890408ac9354a9a9b0fc0709f181573815b9ab8d14ce717d9a99afe
// Rebased delta: original Repo191 direct consumer-coordinate specialization
//
// Preserve Repo185's long-prefill V-barrier pre-arming and Repo186's
// stage-owner decode V-barrier initialization byte-for-byte.  unified_decode
// reaches the D16/D32 score consumers only inside its exact q_heads_per_kv == 4
// branch.  Consequently each KV head has exactly one four-head output group:
// the D16 launch has grid.x == 8 and D32 has grid.x == 4.  Decode those proven
// launch coordinates directly: blockIdx.x is the output-D split and head_idx
// is kv_head_idx * 4.  This removes constant head-group extraction and
// gridDim.x arithmetic from every long-decode consumer CTA.
//
// Grid shape/order, pointers, workspace/V coordinates, TMA/barriers,
// score/softmax/PV arithmetic, stores, producer, short decode, and prefill are
// unchanged.  This applies H20 Iteration 38's successful direct-coordinate
// principle without changing the launch grid as rejected H200 Iteration 89.
//
// Iteration 186 — Repo185-rebased stage-owner V-mbarrier initialization
// Exact base: kernel_iter185_h200_prefill_v_mbar_prearm_under_k_tma.cu
// Base SHA-256: f28f1497b14e43b75d48336c5969b95b45210420de7548d00e87ce724578bd72
// Rebased delta: original Repo186 D16/D32 consumer barrier initialization
// Original candidate SHA-256: 2f93ae0beb756b09634b80964c5a8d0dbde4e82726ef236584b18d4080e8c82d
//
// Preserve Repo185's promoted long-prefill V-barrier pre-arming and common
// long-decode producer byte-for-byte.  Iteration 156 distributed long-decode
// V-TMA prologue issues and refills to stable stage-owner warp leaders, but
// warp 0 still serially initializes every ring barrier.  Complete the same
// stage-local ownership during consumer setup: each D16 warp leader initializes
// its one four-stage slot, while the two B4 D32 owners initialize their two
// slots in parallel.  The existing CTA fence remains between initialization
// and every arrive/TMA use.  Descriptor prefetch, barrier values, stages,
// phases, TMA bytes and coordinates, arithmetic, output stores, and launch
// geometry are unchanged.  B8/B16 keep the original separately compiled
// single-thread D32 setup specialization.
// Reference: PTX ISA mbarrier.init shared::cta semantics and the promoted
// Iteration-156 stage-owned V-TMA schedule.
//
// Iteration 185 — Repo183-rebased pre-arm long-prefill V barriers under K TMA
// Exact base: kernel_iter183_h200_decode_producer_k_acquire_sync_elision.cu
// Base SHA-256: 91658e675a1eb0634aedb3f763fd11fcd80ac98589e5fc6b6b82ffac423f813f
// Rebased delta: original Repo185 V expect-tx relocation
// Original candidate SHA-256: 25d749124fb116cfa395fef4ac59a53ddc2931dec5df7f1db7c733e31326c35b
//
// Preserve Repo183's promoted decode path byte-for-byte.  In aligned 4K+
// prefill, move each unchanged 16-KiB V transaction expectation from directly
// before V issue to immediately after the corresponding K TMA issue.  The
// existing K completion wait can hide this bookkeeping.  Descriptors,
// coordinates, destinations, aggregate bytes, barrier count/phases, waits,
// TMA issuer ownership, WGMMA, arithmetic, and stores remain exact.
// Reference: Hopper mbarrier.arrive.expect_tx accounting and FA3 producer-side
// acquire/issue separation in its SM90 TMA pipeline.
//
// Iteration 183 — long-decode producer K acquire-sync elision
// Exact base: kernel_iter178_h200_decode_k_split_warp_evict_first.cu
// Base SHA-256: 02bcb040bf32e5ad6dfd933dcf6ce13d98b280638c3d77d3b2d20a5b119195e6
//
// The common four-head long-decode producer already makes every thread
// execute the count-two K transaction-barrier acquire wait after the two
// split-warp native-box TMA issues.  A completed wait makes both K boxes
// visible to that thread, and the following QK WGMMA is warpgroup-aligned.
// Remove only the redundant full-CTA rendezvous immediately after that
// acquire.  The pre-issue Q-staging barrier remains, and this one-shot K
// tile is never reused, so TMA bytes, barrier count, QK/score arithmetic,
// workspace ABI, consumer kernels, dispatch, and output bits are unchanged.
// Reference: PTX ISA 8.7 sections 9.7.13.15.16 (mbarrier acquire wait)
// and 9.7.15.7.3 (aligned warpgroup wait/participation).
//
// Iteration 178 — Repo171-rebased split-warp decode K-TMA issuers
// Exact base: kernel_iter171_h200_prefill_sync_elision_decode_k_evict_first.cu
// Base SHA-256: 23010ee3dd0b5a54a4eda5822ff47863c41b73bae4563f073f71c94ae158293b
// Rebased delta: kernel_iter167_h200_decode_k_split_warp_issuers.cu
// Delta-source SHA-256: 04f3b624a990132bd8865029cd0373a734dd571e9a34d63f5174ac8d02632ace
//
// Preserve Repo171's promoted L2::evict_first policy on both native K boxes,
// but assign their otherwise independent issues to separate warp leaders.
// Thread 0 owns the low-feature box and thread 32 owns the high-feature box;
// each contributes one 16-KiB transaction arrival to a count-two K-ready
// barrier.  The existing all-thread acquire wait still gates QK on completion
// of both boxes.  Shared destinations, tensor coordinates, cache hint, Q
// staging, QK arithmetic, score/max workspace bits, every consumer, and all
// prefill code remain exact.
//
// Iteration 171 — Iteration-166 rebase with streaming decode K-TMA eviction
// Exact base: kernel_iter166_h200_long_prefill_mbar_acquire_sync_elision.cu
// Base SHA-256: 6265e80cdbc4b460e2deeea2d91a57febe59380711d2783dae2c2cfb2c611a0c
// Rebased delta: kernel_iter169_h200_long_decode_k_tma_evict_first.cu
// Delta-source SHA-256: 3b7e66022ca53f28df192b9c2b6ff53e17bb1cabce5f79244039b39a39d14acf
//
// Preserve Iteration 166's long-prefill consumer-local acquire-wait sync
// elision byte-for-byte and rebase Iteration 169's orthogonal decode-only
// policy on top.  Every configured long-decode route shares the four-head QK
// producer, whose K tile is consumed exactly once and never read by a score
// consumer.  Mark only its two unchanged native 16-KiB K TMA boxes
// L2::evict_first so the dependent score/max workspace can retain L2 capacity.
// Descriptor and tensor coordinates, low-before-high issue order, bytes,
// transaction-barrier contract, QK arithmetic, workspace bits, consumers,
// all prefill instructions, and every other route remain exact.
//
// Iteration 166 — long-prefill consumer-local mbarrier acquire sync elision
// Exact base: kernel_iter156_h200_stage_owned_v_tma_issuers.cu
// Base SHA-256: 704513a57681ecc30fa0b646e1bcfcb11f6e038dc5e5ac03340eabd18e1d04b0
//
// Every thread in the long aligned prefill specialization already executes
// the Q, K, and V transaction-barrier wait.  A completed acquire wait makes
// the corresponding TMA writes visible to that consumer, and the immediately
// following QK/PV WGMMA is warpgroup-aligned.  Therefore the extra full-CTA
// rendezvous after those waits neither publishes data nor releases a reused
// stage.  Compile them out only for <true, true>, removing one setup barrier
// and two barriers per retained KV block.  The post-QK and post-PV barriers
// that protect K/V shared-slot reuse remain unchanged, as do every TMA issue,
// barrier phase, WGMMA, score, softmax, reduction, and output operation.
// Reference: PTX ISA 8.7 sections 9.7.13.15.16 (mbarrier acquire wait) and
// 9.7.15.7.3 (aligned warpgroup wait/participation).
//
// Iteration 156 — stage-owned long-decode V-TMA issuers
// Exact base: kernel_iter150_h200_d32_shared_block_max.cu
// Base SHA-256: 0705f6728fd7cddf2ed163ad13460a2952a44e25ce543a00806a7fede38f4ded
//
// Assign each existing V-ring stage to the lane-zero thread of the matching
// warp.  The two D32 stage leaders and four D16 stage leaders can therefore
// issue their independent prologue TMA transfers concurrently; steady-state
// reuse is rearmed by the same stage owner after the unchanged WGMMA wait.
// Every tensor-map coordinate, byte count, barrier slot/phase, score load,
// producer-shared block maximum, online-softmax operation, PV K16 slice, and
// output store remains in its promoted order.  This targets only the weak B4
// D32 and B1 D16 long-decode routes; B8/B16 select the original D32 issuer
// specialization, and Iteration 150's shared-maximum optimization is intact.
//
// Iteration 150 — D32 consumers reuse the producer's exact block maximum
// Exact base: kernel_iter137_h200_batch1_4k_generic_d16_consumer.cu
//
// Batch-4/8/16 long decode launches four sibling D32 consumers for every
// (batch, KV head).  Iteration 137 makes each sibling rescan the same 2-KiB
// score block and repeat the same ascending fmax/quad_max tree.  The four-head
// producer already contains that exact tree for its D16 sideband.  Publish it
// for D32 as well, then let every D32 sibling consume the shared FP32 maximum.
// Score masking, exponentials, sums, output rescaling, P conversion, eight
// ascending PV K16 slices, and output stores stay in their original order.
// Only the six configured B4/B8/B16 x 4K/8K rows select D32, though arbitrary
// long/tail D32 calls remain correct through the producer's masked sideband.
//
// Iteration 137 — batch-1/4K generic D16 consumer routing
// Exact base: kernel_iter135_h200_dual_lane_split_tma_prefill.cu
// SHA-256: 861b8f9bad1854d3cd422d6a31c08af45d8a991917c601e306369dcd1522fe6a
// Isolate the remaining batch-1/4K D16 schedule choice: retain the promoted
// exact-N128 score producer, but select the existing generic D16 consumer as
// Iteration 128 already does at batch-1/8K.  Device code, grids, descriptors,
// transfers, and arithmetic are unchanged; only one configured host route
// chooses a different already-compiled bitwise-equivalent symbol.
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
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <optional>
#include <vector>
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

using TmaQ = decltype(make_tma_atom(
    SM90_TMA_LOAD{},
    make_tensor(static_cast<BF16 const*>(nullptr),
                make_shape(1, Int<HEAD_DIM>{}),
                make_stride(Int<HEAD_DIM>{}, Int<1>{})),
    SmemLayoutQ{},
    Shape<Int<BLOCK_M_PREFILL>, Int<HEAD_DIM>>{}));

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

// Iteration 55 long-decode consumer: split the output feature dimension into
// two independent 64-column tiles.  The score/P operand remains [64,128], but
// each CTA loads only V^T[d_half * 64 : (d_half + 1) * 64, 0:128] and emits
// one disjoint half of O.  The full [128,total_kv_rows] gmem view is retained
// so blockIdx.x can select either half through a first-mode coordinate offset.
constexpr int DECODE_D_SPLIT = HEAD_DIM / 2;
using SmemLayoutVt64 = decltype(tile_to_shape(
    SmemLayoutAtomVt{},
    Shape<Int<DECODE_D_SPLIT>, Int<BLOCK_N>>{},
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

using TmaVt64 = decltype(make_tma_atom(
    SM90_TMA_LOAD{},
    make_tensor(static_cast<BF16 const*>(nullptr),
                make_shape(Int<HEAD_DIM>{}, 1),
                make_stride(Int<1>{}, Int<HEAD_DIM>{})),
    SmemLayoutVt64{},
    Shape<Int<DECODE_D_SPLIT>, Int<BLOCK_N>>{}));

// Iteration 56 deepens the verified output-d split to four independent
// 32-column tiles.  Keeping this beside the D64 definitions preserves the
// promoted Iteration-55 path as a compile-time fallback/reference.
constexpr int DECODE_D_QUARTER = HEAD_DIM / 4;
// SW128's BF16 atom is 64 rows wide and therefore cannot tile a 32-row
// first mode.  SW64's BF16 atom is [32,8], exactly matching the D-quarter.
using SmemLayoutAtomVt32 = decltype(GMMA::Layout_MN_SW64_Atom<BF16>{});
using SmemLayoutVt32 = decltype(tile_to_shape(
    SmemLayoutAtomVt32{},
    Shape<Int<DECODE_D_QUARTER>, Int<BLOCK_N>>{},
    Step<_2, _1>{}));

using TmaVt32 = decltype(make_tma_atom(
    SM90_TMA_LOAD{},
    make_tensor(static_cast<BF16 const*>(nullptr),
                make_shape(Int<HEAD_DIM>{}, 1),
                make_stride(Int<1>{}, Int<HEAD_DIM>{})),
    SmemLayoutVt32{},
    Shape<Int<DECODE_D_QUARTER>, Int<BLOCK_N>>{}));

// Iteration 57 deepens the output split once more: eight independent
// 16-column consumers cover one four-head group.  A BF16 MN-SW32 atom is
// exactly [16,8], so the [16,128] V^T box is both a legal single TMA tile and
// the native B layout for the N=16 RS WGMMA opcode.
constexpr int DECODE_D_EIGHTH = HEAD_DIM / 8;
using SmemLayoutAtomVt16 = decltype(GMMA::Layout_MN_SW32_Atom<BF16>{});
using SmemLayoutVt16 = decltype(tile_to_shape(
    SmemLayoutAtomVt16{},
    Shape<Int<DECODE_D_EIGHTH>, Int<BLOCK_N>>{},
    Step<_2, _1>{}));

using TmaVt16 = decltype(make_tma_atom(
    SM90_TMA_LOAD{},
    make_tensor(static_cast<BF16 const*>(nullptr),
                make_shape(Int<HEAD_DIM>{}, 1),
                make_stride(Int<1>{}, Int<HEAD_DIM>{})),
    SmemLayoutVt16{},
    Shape<Int<DECODE_D_EIGHTH>, Int<BLOCK_N>>{}));

// WGMMA PV: P[64,128] (K-major, in registers RS) x Vt[128,128] (MN-major, in smem SS)
// -> O[64,128] accumulator
// RS = P stays in registers (no smem write needed)
// MN = Vt is MN-major (BLOCK_N is fast dimension)
using TiledMmaPV = decltype(make_tiled_mma(
    SM90_64x128x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::MN>{}));

using TiledMmaPV64 = decltype(make_tiled_mma(
    SM90_64x64x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::MN>{}));

using TiledMmaPV32 = decltype(make_tiled_mma(
    SM90_64x32x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::MN>{}));

using TiledMmaPV16 = decltype(make_tiled_mma(
    SM90_64x16x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::MN>{}));

constexpr int N_TILES    = BLOCK_N  / MMA_N;   // 16
constexpr int K_TILES    = HEAD_DIM / MMA_K;   // 8
constexpr int KV_K_TILES = BLOCK_N  / MMA_K;   // 8
constexpr int OUT_N_TILES= HEAD_DIM / MMA_N;   // 16
constexpr int OUT_N_TILES_D64 = DECODE_D_SPLIT / MMA_N;  // 8
constexpr int VT_D64_SMEM_BYTES =
    DECODE_D_SPLIT * BLOCK_N * sizeof(BF16);              // 16 KiB
constexpr int OUT_N_TILES_D32 = DECODE_D_QUARTER / MMA_N; // 4
constexpr int VT_D32_SMEM_BYTES =
    DECODE_D_QUARTER * BLOCK_N * sizeof(BF16);             // 8 KiB
constexpr int OUT_N_TILES_D16 = DECODE_D_EIGHTH / MMA_N;   // 2
constexpr int VT_D16_SMEM_BYTES =
    DECODE_D_EIGHTH * BLOCK_N * sizeof(BF16);               // 4 KiB

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
constexpr int SMEM_BYTES     = WARP_BASE + NUM_WARPS * PER_WARP_BYTES;  // 52240 B

constexpr int W_Q_TILE_OFF   = 0;
constexpr int W_PV_BUF_OFF   = W_Q_TILE_OFF + Q_TILE_BYTES;

// Extra smem for WGMMA: Q in K-major SW128 layout
// K smem is now K-major SW128 [BLOCK_N=128, HEAD_DIM=128] = 32KB (same size as 10c split-half)
// placed at K_SMEM_OFF=0, reusing the same slot — no extra K buffer needed
constexpr int Q_WGMMA_BYTES  = 64  * 128 * 2;   // 16384 B  [BLOCK_M=64,  HEAD_DIM=128]
constexpr int Q_TMA_MBAR_OFF   = SMEM_BYTES;                       // 52240
constexpr int Q_TMA_MBAR_BYTES = int(sizeof(uint64_t));            //     8
constexpr int Q_WGMMA_OFF      =
    ((Q_TMA_MBAR_OFF + Q_TMA_MBAR_BYTES + 127) / 128) * 128;       // 52352
constexpr int SMEM_BYTES_10H   = Q_WGMMA_OFF + Q_WGMMA_BYTES;      // 68736 B (~67 KB)

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

// Long-decode staged-P actor pipeline: QK WG, PV WG, and one softmax warp.
constexpr int DECODE_ACTOR_THREADS       = 9 * WARP_SIZE;
constexpr int DECODE_ACTOR_WG_THREADS    = 4 * WARP_SIZE;
constexpr int DECODE_ACTOR_FRAGMENT_VALS = N_TILES * 4;
constexpr int DECODE_ACTOR_SHARED_LANES  = WARP_SIZE;
constexpr int DECODE_SCORE_STAGE_BYTES =
    DECODE_ACTOR_SHARED_LANES * DECODE_ACTOR_FRAGMENT_VALS * sizeof(float);
constexpr int DECODE_P_STAGE_BYTES =
    DECODE_ACTOR_SHARED_LANES * DECODE_ACTOR_FRAGMENT_VALS * sizeof(BF16);
constexpr int DECODE_SCALE_STAGE_BYTES =
    DECODE_ACTOR_SHARED_LANES * 2 * sizeof(float);
constexpr int DECODE_ACTOR_SCORE_OFF = SMEM_BYTES_DECODE_PIPE;
constexpr int DECODE_ACTOR_P_OFF =
    DECODE_ACTOR_SCORE_OFF + 2 * DECODE_SCORE_STAGE_BYTES;
constexpr int DECODE_ACTOR_SCALE_OFF =
    DECODE_ACTOR_P_OFF + 2 * DECODE_P_STAGE_BYTES;
constexpr int DECODE_ACTOR_NORM_OFF =
    DECODE_ACTOR_SCALE_OFF + 2 * DECODE_SCALE_STAGE_BYTES;
constexpr int DECODE_ACTOR_MBAR_OFF =
    DECODE_ACTOR_NORM_OFF + DECODE_ACTOR_SHARED_LANES * sizeof(float);
constexpr int DECODE_ACTOR_MBAR_COUNT = 9;
constexpr int SMEM_BYTES_DECODE_ACTOR =
    DECODE_ACTOR_MBAR_OFF + DECODE_ACTOR_MBAR_COUNT * sizeof(uint64_t);

constexpr int SCORE_FULL_MBAR  = 0;  // [0,1]
constexpr int SCORE_EMPTY_MBAR = 2;  // [2,3]
constexpr int P_FULL_MBAR      = 4;  // [4,5]
constexpr int P_EMPTY_MBAR     = 6;  // [6,7]
constexpr int NORM_READY_MBAR  = 8;

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

// Long-decode producer-only TMA variant.  The caller supplies CUTLASS's
// architecture-defined opaque policy and reuses it for both native K boxes,
// preserving their original issue order.
__device__ __forceinline__ void tma_load_2d_l2_cache_hint(
    const CUtensorMap* __restrict__ desc,
    __nv_bfloat16* __restrict__ smem_dst,
    uint64_t* __restrict__ mbar,
    int coord_col,
    int coord_row,
    uint64_t l2_cache_policy
) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.L2::cache_hint "
        "[%0], [%1, {%3, %4}], [%2], %5;\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem_dst)),
           "l"(desc),
           "r"((uint32_t)__cvta_generic_to_shared(mbar)),
           "r"(coord_col), "r"(coord_row),
           "l"(l2_cache_policy)
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
// FUNCTION: compute_qk_cute
template <bool IssueVtAfterQK = false>
__device__ __forceinline__ void compute_qk_cute(
    const BF16* __restrict__ q_smem,
    const BF16* __restrict__ k_wgmma,
    float acc_s[N_TILES][4],
    const CUtensorMap* __restrict__ vt_desc = nullptr,
    BF16* __restrict__ vt_smem = nullptr,
    uint64_t* __restrict__ vt_mbar = nullptr,
    int v_global_row = 0
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

    if constexpr (IssueVtAfterQK) {
        static_assert(KV_HALF_BYTES * 2 ==
                          HEAD_DIM * BLOCK_N * sizeof(BF16),
                      "early Vt issue assumes two 16-KiB boxes");
        __syncthreads();
        // This V phase was pre-armed by the same two lanes immediately after
        // the current K boxes entered the TMA path.  Preserve native-box
        // issuer ownership and coordinates here.
        if (threadIdx.x < 2) {
            const int half = int(threadIdx.x);
            auto* vt_native =
                reinterpret_cast<__nv_bfloat16*>(vt_smem);
            tma_load_2d(vt_desc,
                        vt_native + half * KV_HI_OFF, vt_mbar,
                        half * HALF_DIM, v_global_row);
        }
    }
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
    cutlass::Array<float, N_TILES * 4>& acc_s,
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
        float s0 = acc_s[nt * 4 + 0] * inv_sqrt;
        float s1 = acc_s[nt * 4 + 1] * inv_sqrt;
        float s2 = acc_s[nt * 4 + 2] * inv_sqrt;
        float s3 = acc_s[nt * 4 + 3] * inv_sqrt;
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
        acc_s[nt * 4 + 0] = m00 ? -INFINITY : s0;
        acc_s[nt * 4 + 1] = m01 ? -INFINITY : s1;
        acc_s[nt * 4 + 2] = m10 ? -INFINITY : s2;
        acc_s[nt * 4 + 3] = m11 ? -INFINITY : s3;
        bmax0 = fmaxf(bmax0, fmaxf(acc_s[nt * 4 + 0], acc_s[nt * 4 + 1]));
        bmax1 = fmaxf(bmax1, fmaxf(acc_s[nt * 4 + 2], acc_s[nt * 4 + 3]));
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
            p0 = isinf(acc_s[nt * 4 + 0]) ? 0.0f
                                      : fast_exp2_ftz((acc_s[nt * 4 + 0] - new_max0) * LOG2E);
            p1 = isinf(acc_s[nt * 4 + 1]) ? 0.0f
                                      : fast_exp2_ftz((acc_s[nt * 4 + 1] - new_max0) * LOG2E);
            p2 = isinf(acc_s[nt * 4 + 2]) ? 0.0f
                                      : fast_exp2_ftz((acc_s[nt * 4 + 2] - new_max1) * LOG2E);
            p3 = isinf(acc_s[nt * 4 + 3]) ? 0.0f
                                      : fast_exp2_ftz((acc_s[nt * 4 + 3] - new_max1) * LOG2E);
        } else {
            p0 = fast_exp2_ftz((acc_s[nt * 4 + 0] - new_max0) * LOG2E);
            p1 = fast_exp2_ftz((acc_s[nt * 4 + 1] - new_max0) * LOG2E);
            p2 = fast_exp2_ftz((acc_s[nt * 4 + 2] - new_max1) * LOG2E);
            p3 = fast_exp2_ftz((acc_s[nt * 4 + 3] - new_max1) * LOG2E);
        }
        acc_s[nt * 4 + 0] = p0;
        acc_s[nt * 4 + 1] = p1;
        acc_s[nt * 4 + 2] = p2;
        acc_s[nt * 4 + 3] = p3;
        bsum0 += p0 + p1;
        bsum1 += p2 + p3;
    }
    row_sum[0] += bsum0;
    row_sum[1] += bsum1;
}

// D32-only counterpart of softmax_update_reg_deferred_o_rescale.  The score
// producer has already executed this helper's exact ascending pair-fmax scan
// and quad_max tree once for each of the four live rows.  Accept that rounded
// FP32 maximum as an input while retaining the original mask, exp2, sum, and
// deferred-output-rescale order.  The permanently masked second row remains
// materialized so the D32 fragment/state shape is unchanged.
template <bool IsFirst, bool CheckTail>
__device__ __forceinline__ void
softmax_update_reg_precomputed_block_max(
    cutlass::Array<float, N_TILES * 4>& acc_s,
    float row_max[2],
    float row_sum[2],
    float output_rescale[2],
    float precomputed_block_max,
    int valid_n,
    int actual_rows
) {
    int lane_id = threadIdx.x % WARP_SIZE;
    int group_id = lane_id >> 2;
    int col_pair = lane_id & 3;
    int row0 = group_id;
    int row1 = group_id + 8;
    int col0 = col_pair * 2;
    int col1 = col0 + 1;

    #pragma unroll
    for (int nt = 0; nt < N_TILES; ++nt) {
        int n_base = nt * MMA_N;
        int n0 = n_base + col0;
        int n1 = n0 + 1;
        float s0 = acc_s[nt * 4 + 0];
        float s1 = acc_s[nt * 4 + 1];
        float s2 = acc_s[nt * 4 + 2];
        float s3 = acc_s[nt * 4 + 3];
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
        acc_s[nt * 4 + 0] = m00 ? -INFINITY : s0;
        acc_s[nt * 4 + 1] = m01 ? -INFINITY : s1;
        acc_s[nt * 4 + 2] = m10 ? -INFINITY : s2;
        acc_s[nt * 4 + 3] = m11 ? -INFINITY : s3;
    }

    float new_max0 = fmaxf(row_max[0], precomputed_block_max);
    float new_max1 = fmaxf(row_max[1], -INFINITY);
    float rescale0 = 1.0f;
    float rescale1 = 1.0f;
    if constexpr (!IsFirst) {
        rescale0 = isinf(new_max0)
            ? 1.0f
            : fast_exp2_ftz((row_max[0] - new_max0) * LOG2E);
        rescale1 = isinf(new_max1)
            ? 1.0f
            : fast_exp2_ftz((row_max[1] - new_max1) * LOG2E);
    }
    row_max[0] = new_max0;
    row_max[1] = new_max1;
    output_rescale[0] = rescale0;
    output_rescale[1] = rescale1;

    if constexpr (!IsFirst) {
        row_sum[0] *= rescale0;
        row_sum[1] *= rescale1;
    }

    float bsum0 = 0.0f;
    float bsum1 = 0.0f;
    #pragma unroll
    for (int nt = 0; nt < N_TILES; ++nt) {
        float p0 = isinf(acc_s[nt * 4 + 0])
            ? 0.0f
            : fast_exp2_ftz(
                (acc_s[nt * 4 + 0] - new_max0) * LOG2E);
        float p1 = isinf(acc_s[nt * 4 + 1])
            ? 0.0f
            : fast_exp2_ftz(
                (acc_s[nt * 4 + 1] - new_max0) * LOG2E);
        float p2 = isinf(acc_s[nt * 4 + 2])
            ? 0.0f
            : fast_exp2_ftz(
                (acc_s[nt * 4 + 2] - new_max1) * LOG2E);
        float p3 = isinf(acc_s[nt * 4 + 3])
            ? 0.0f
            : fast_exp2_ftz(
                (acc_s[nt * 4 + 3] - new_max1) * LOG2E);
        acc_s[nt * 4 + 0] = p0;
        acc_s[nt * 4 + 1] = p1;
        acc_s[nt * 4 + 2] = p2;
        acc_s[nt * 4 + 3] = p3;
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
template <bool AlignedSelf, bool VtUnderQkExtract = false>
__global__
__launch_bounds__(BLOCK_THREADS)
void unified_attn_prefill_kernel(
    const __nv_bfloat16* __restrict__ Q,
    CUTLASS_GRID_CONSTANT TmaQ  const tma_q,   // cute TMA for Q → K-major SW128
    CUTLASS_GRID_CONSTANT TmaK  const tma_k,   // cute TMA for K → K-major SW128
    CUTLASS_GRID_CONSTANT TmaVt const tma_vt,  // cute TMA for V^T → MN-major SW128
    __nv_bfloat16* __restrict__ O,
    int seq_len, int ctx_len, int num_heads, int num_kv_heads,
    int total_q_rows, int total_kv_rows
) {
    static_assert(!VtUnderQkExtract || AlignedSelf,
                  "early Vt issue is qualified only for aligned prefill");
    extern __shared__ char smem[];

    // FA3-style causal LPT exposure: adjacent x blocks retain GQA K/V locality,
    // while reversed y exposes the longest causal query tiles first.
    int head_batch_idx = blockIdx.x;
    int q_block_idx    = int(gridDim.y) - 1 - int(blockIdx.y);
    int batch_idx;
    int head_idx;
    int kv_head_idx;

    int q_block_start = q_block_idx * BLOCK_M_PREFILL;
    if constexpr (!AlignedSelf) {
        if (q_block_start >= seq_len) return;
    }

    __nv_bfloat16* o_ptr;
    int kv_row_base;
    if constexpr (!VtUnderQkExtract) {
        batch_idx   = head_batch_idx / num_heads;
        head_idx    = head_batch_idx % num_heads;
        kv_head_idx = head_idx / (num_heads / num_kv_heads);
        o_ptr = O +
            ((batch_idx * num_heads + head_idx) * seq_len) * HEAD_DIM;
        kv_row_base =
            (batch_idx * num_kv_heads + kv_head_idx) * ctx_len;
    }

    int tid     = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    char* warp_smem = smem + WARP_BASE + warp_id * PER_WARP_BYTES;
    __nv_bfloat16* pv_buf  = reinterpret_cast<__nv_bfloat16*>(warp_smem + W_PV_BUF_OFF);
    BF16*          k_wgmma = reinterpret_cast<BF16*>(smem + K_SMEM_OFF);  // K-major SW128; reused as Vt after QK
    BF16*          vt_smem = reinterpret_cast<BF16*>(smem + K_SMEM_OFF);  // MN-major SW128 Vt (same slot as K)
    BF16*          q_smem  = reinterpret_cast<BF16*>(smem + Q_WGMMA_OFF);
    uint64_t* mbar = reinterpret_cast<uint64_t*>(smem + MBAR_OFF);
    uint64_t* q_tma_mbar = reinterpret_cast<uint64_t*>(smem + Q_TMA_MBAR_OFF);

    int warp_q_start = q_block_start + warp_id * MMA_M;
    int actual_rows;
    if constexpr (AlignedSelf) {
        actual_rows = MMA_M;
    } else {
        int warp_q_end = min(warp_q_start + MMA_M, seq_len);
        actual_rows = max(0, warp_q_end - warp_q_start);
    }

    // Producer (warp 0) prefetches TMA descriptors and initializes mbarriers.
    // Long prefill assigns the two native K/V boxes to two producer lanes, so
    // each streaming barrier expects their two independent arrivals.  The Q,
    // aligned-1K, and generic paths retain their original count-one contract.
    if (warp_id == 0 && lane_id == 0) {
        // The long grid reuses one immutable Q tensor map across every CTA.
        // Its actual TMA still resolves the descriptor; omit only the
        // advisory per-CTA prefetch from the one-shot long-Q path.
        if constexpr (!VtUnderQkExtract) {
            cute::prefetch_tma_descriptor(tma_q.get_tma_descriptor());
        }
        cute::prefetch_tma_descriptor(tma_k.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_vt.get_tma_descriptor());
        mbar_init(q_tma_mbar, 1);
        if constexpr (VtUnderQkExtract) {
            mbar_init(&mbar[0], 2);
            mbar_init(&mbar[1], 2);
        } else {
            mbar_init(&mbar[0], 1);
            mbar_init(&mbar[1], 1);
        }
    }
    __syncthreads();

    // Stage the full Q tile with one TMA transaction.  The flattened global
    // view keeps head tiles contiguous; tail rows are masked by actual_rows.
    if (tid == 0) {
        mbar_arrive_tx(q_tma_mbar, Q_WGMMA_BYTES);
        int q_global_row = head_batch_idx * seq_len + q_block_start;
        Tensor mQ = tma_q.get_tma_tensor(
            make_shape(total_q_rows, Int<HEAD_DIM>{}));
        Tensor sQ = make_tensor(make_smem_ptr(q_smem), SmemLayoutQ{});
        Tensor mQ_off = domain_offset(make_coord(q_global_row, 0), mQ);
        Tensor gQ = local_tile(
            mQ_off, Shape<Int<BLOCK_M_PREFILL>, Int<HEAD_DIM>>{},
            make_coord(0, 0));
        auto [tQgQ, tQsQ] = tma_partition(
            tma_q, Int<0>{}, Layout<_1>{},
            group_modes<0,2>(sQ), group_modes<0,2>(gQ));
        copy(tma_q.with(*q_tma_mbar), tQgQ, tQsQ);
    }
    // Q's global coordinate uses head_batch_idx directly.  Derive the
    // independent head/GQA coordinates while that one-shot TMA is in flight.
    if constexpr (VtUnderQkExtract) {
        batch_idx   = head_batch_idx / num_heads;
        head_idx    = head_batch_idx % num_heads;
        kv_head_idx = head_idx / (num_heads / num_kv_heads);
        o_ptr = O +
            ((batch_idx * num_heads + head_idx) * seq_len) * HEAD_DIM;
        kv_row_base =
            (batch_idx * num_kv_heads + kv_head_idx) * ctx_len;
    }
    mbar_wait(q_tma_mbar, 0);
    // The long path has one acquire wait per consumer thread.  It needs no
    // second CTA-wide publication step; other specializations retain the
    // promoted rendezvous verbatim.
    if constexpr (!VtUnderQkExtract) {
        __syncthreads();
    }

    float acc_o[OUT_N_TILES][4];
    float row_max[2] = {-INFINITY, -INFINITY};
    float row_sum[2] = {0.0f, 0.0f};
    #pragma unroll
    for (int nt = 0; nt < OUT_N_TILES; nt++)
        acc_o[nt][0] = acc_o[nt][1] = acc_o[nt][2] = acc_o[nt][3] = 0.0f;

    float inv_sqrt = 1.0f / sqrtf((float)HEAD_DIM);

    // FA3 BlockMN-style causal upper bound: this Q tile cannot attend to any
    // complete KV block strictly to its right.  In the aligned self-attention
    // specialization BLOCK_N == 2 * BLOCK_M_PREFILL, so every M64 query tile
    // lies wholly inside one physical N128 KV block and its last retained KV
    // block is exactly q_block_idx / 2.  The generic expression is unchanged.
    static_assert(BLOCK_N == 2 * BLOCK_M_PREFILL,
                  "aligned self-attention specialization assumes N128/M64");
    int num_kv_blocks;
    if constexpr (AlignedSelf) {
        num_kv_blocks = (q_block_idx >> 1) + 1;
    } else {
        const int q_block_end = min(q_block_start + BLOCK_M_PREFILL, seq_len);
        const int causal_kv_end =
            max(0, min(ctx_len, q_block_end + ctx_len - seq_len));
        num_kv_blocks = (causal_kv_end + BLOCK_N - 1) / BLOCK_N;
    }

    for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        int kv_start = kv_block * BLOCK_N;
        int valid_n;
        if constexpr (AlignedSelf) {
            // ctx_len is a positive multiple of BLOCK_N, so every physically
            // loaded tile is full.  Causal visibility remains handled below.
            valid_n = BLOCK_N;
        } else {
            int kv_end = min(kv_start + BLOCK_N, ctx_len);
            valid_n = kv_end - kv_start;
        }

        // ── PRODUCER role (warp 0): issue TMA K load ─────────────────────────
        // The aligned tile is exactly two native 16-KiB SW128 boxes.  Issue
        // those boxes directly from the CuTe-owned descriptor so ptxas need
        // not materialize the tensor/view/partition scaffolding in this loop.
        if constexpr (AlignedSelf) {
            static_assert(TmaK::NumValSrc * sizeof(BF16) == KV_HALF_BYTES,
                          "aligned K direct issue requires a 16-KiB TMA box");
            if constexpr (VtUnderQkExtract) {
                // Lanes 0/1 issue the original low/high K boxes, then pre-arm
                // this iteration's count-two V barrier beneath the K wait.
                if (warp_id == 0 && lane_id < 2) {
                    const int half = lane_id;
                    const int k_global_row = kv_row_base + kv_start;
                    auto* k_native =
                        reinterpret_cast<__nv_bfloat16*>(k_wgmma);
                    mbar_arrive_tx(&mbar[0], KV_HALF_BYTES);
                    tma_load_2d(tma_k.get_tma_descriptor(),
                                k_native + half * KV_HI_OFF, &mbar[0],
                                half * HALF_DIM, k_global_row);
                    mbar_arrive_tx(&mbar[1], KV_HALF_BYTES);
                }
            } else if (warp_id == 0 && lane_id == 0) {
                const int k_bytes =
                    BLOCK_N * HEAD_DIM * sizeof(BF16);
                const int k_global_row = kv_row_base + kv_start;
                auto* k_native =
                    reinterpret_cast<__nv_bfloat16*>(k_wgmma);
                mbar_arrive_tx(&mbar[0], k_bytes);
                tma_load_2d(tma_k.get_tma_descriptor(),
                            k_native + KV_LO_OFF, &mbar[0],
                            0, k_global_row);
                tma_load_2d(tma_k.get_tma_descriptor(),
                            k_native + KV_HI_OFF, &mbar[0],
                            HALF_DIM, k_global_row);
            }
        } else {
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
        }
        // All warps wait for K
        {
            const int kphase = kv_block & 1;
            mbar_wait(&mbar[0], kphase);
        }
        if constexpr (!VtUnderQkExtract) {
            __syncthreads();
        }

        // ── ALL WARPS: WGMMA QK (.aligned requires all 128 threads) ──────────
        float acc_s[N_TILES][4];
        if constexpr (VtUnderQkExtract) {
            static_assert(TmaVt::NumValSrc * sizeof(BF16) == KV_HALF_BYTES,
                          "early Vt direct issue requires a 16-KiB box");
            compute_qk_cute<true>(
                q_smem, k_wgmma, acc_s,
                tma_vt.get_tma_descriptor(), vt_smem, &mbar[1],
                kv_row_base + kv_start);
        } else {
            compute_qk_cute(q_smem, k_wgmma, acc_s);
        }

        // QK has retired for the complete warpgroup, so K smem can become the
        // V^T destination.  Issue V before scalar softmax and retain the same
        // completion wait before PV, hiding the transfer without reordering
        // any score, probability, or output-accumulator arithmetic.
        if constexpr (!VtUnderQkExtract) {
          __syncthreads();
          if constexpr (AlignedSelf) {
            static_assert(TmaVt::NumValSrc * sizeof(BF16) == KV_HALF_BYTES,
                          "aligned Vt direct issue requires a 16-KiB TMA box");
            if (warp_id == 0 && lane_id == 0) {
                const int vt_bytes = HEAD_DIM * BLOCK_N * sizeof(BF16);
                const int v_global_row = kv_row_base + kv_start;
                auto* vt_native =
                    reinterpret_cast<__nv_bfloat16*>(vt_smem);
                mbar_arrive_tx(&mbar[1], vt_bytes);
                tma_load_2d(tma_vt.get_tma_descriptor(),
                            vt_native + KV_LO_OFF, &mbar[1],
                            0, v_global_row);
                tma_load_2d(tma_vt.get_tma_descriptor(),
                            vt_native + KV_HI_OFF, &mbar[1],
                            HALF_DIM, v_global_row);
            }
          } else {
            if (warp_id == 0 && lane_id == 0) {
                const int vt_bytes = HEAD_DIM * BLOCK_N * sizeof(BF16);
                mbar_arrive_tx(&mbar[1], vt_bytes);
                Tensor mVt = tma_vt.get_tma_tensor(
                    make_shape(Int<HEAD_DIM>{}, total_kv_rows));
                Tensor sVt = make_tensor(
                    make_smem_ptr(vt_smem), SmemLayoutVt{});
                Tensor mVt_off = domain_offset(
                    make_coord(0, kv_row_base + kv_start), mVt);
                Tensor gVt = local_tile(
                    mVt_off, Shape<Int<HEAD_DIM>, Int<BLOCK_N>>{},
                    make_coord(0, 0));
                auto [tVgVt, tVsVt] = tma_partition(
                    tma_vt, Int<0>{}, Layout<_1>{},
                    group_modes<0,2>(sVt), group_modes<0,2>(gVt));
                copy(tma_vt.with(mbar[1]), tVgVt, tVsVt);
            }
          }
        }

        // ALL warps run softmax for their own rows.  The aligned path proves
        // every physical Q/K tile is full while retaining the exact causal
        // updater on the final retained KV block.  The generic dispatch below
        // is the Iteration-92 source verbatim.
        if constexpr (AlignedSelf) {
            const bool needs_causal_mask =
                kv_block + 1 == num_kv_blocks;
            if (kv_block == 0 && needs_causal_mask) {
                softmax_update_reg<true, true, true, false>(
                    acc_s, acc_o, row_max, row_sum, inv_sqrt, BLOCK_N,
                    kv_start, warp_q_start, MMA_M);
            } else if (kv_block == 0) {
                softmax_update_reg<false, false, true, false>(
                    acc_s, acc_o, row_max, row_sum, inv_sqrt, BLOCK_N,
                    kv_start, warp_q_start, MMA_M);
            } else if (needs_causal_mask) {
                softmax_update_reg<true, true, false, false>(
                    acc_s, acc_o, row_max, row_sum, inv_sqrt, BLOCK_N,
                    kv_start, warp_q_start, MMA_M);
            } else {
                softmax_update_reg<false, false, false, false>(
                    acc_s, acc_o, row_max, row_sum, inv_sqrt, BLOCK_N,
                    kv_start, warp_q_start, MMA_M);
            }
        } else {
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
        }

        // All warps wait for V^T
        {
            const int vphase = kv_block & 1;
            mbar_wait(&mbar[1], vphase);
        }
        if constexpr (!VtUnderQkExtract) {
            __syncthreads();
        }

        // ── ALL WARPS: WGMMA PV (.aligned requires all 128 threads) ──────────
        if constexpr (AlignedSelf) {
            if (kv_block == 0) {
                compute_pv_cute<true>(acc_s, vt_smem, acc_o);
            } else {
                compute_pv_cute<false>(acc_s, vt_smem, acc_o);
            }
        } else {
            if (actual_rows > 0) {
                if (kv_block == 0) {
                    compute_pv_cute<true>(acc_s, vt_smem, acc_o);
                } else {
                    compute_pv_cute<false>(acc_s, vt_smem, acc_o);
                }
            }
        }
        __syncthreads();
    }

    // ── ALL WARPS: write output for their own rows ────────────────────────────
    if constexpr (AlignedSelf) {
        int group_id     = lane_id >> 2;
        int tid_in_group = lane_id & 3;
        int row0 = group_id, row1 = group_id + 8;
        const bool odd_pair_lane = (tid_in_group & 1) != 0;
        const int store_row = odd_pair_lane ? row1 : row0;
        const int store_col = (tid_in_group >> 1) * 4;
        float final_sum0 = quad_sum(row_sum[0]);
        float final_sum1 = quad_sum(row_sum[1]);
        float inv0 = (final_sum0 > 0.0f) ? (1.0f / final_sum0) : 0.0f;
        float inv1 = (final_sum1 > 0.0f) ? (1.0f / final_sum1) : 0.0f;

        #pragma unroll
        for (int nt = 0; nt < OUT_N_TILES; nt++) {
            int n_base = nt * MMA_N;
            __nv_bfloat16 out00 = __float2bfloat16(acc_o[nt][0] * inv0);
            __nv_bfloat16 out01 = __float2bfloat16(acc_o[nt][1] * inv0);
            __nv_bfloat16 out10 = __float2bfloat16(acc_o[nt][2] * inv1);
            __nv_bfloat16 out11 = __float2bfloat16(acc_o[nt][3] * inv1);
            __nv_bfloat162 packed0 = __halves2bfloat162(out00, out01);
            __nv_bfloat162 packed1 = __halves2bfloat162(out10, out11);
            uint32_t pair0 = __BFLOAT162_TO_CUI(packed0);
            uint32_t pair1 = __BFLOAT162_TO_CUI(packed1);
            uint32_t send = odd_pair_lane ? pair0 : pair1;
            uint32_t recv = __shfl_xor_sync(0xffffffffu, send, 1);
            uint32_t lo = odd_pair_lane ? recv : pair0;
            uint32_t hi = odd_pair_lane ? pair1 : recv;
            uint2 out4 = make_uint2(lo, hi);
            uint2* dst = reinterpret_cast<uint2*>(
                o_ptr + (warp_q_start + store_row) * HEAD_DIM +
                n_base + store_col);
            *dst = out4;
        }
    } else if (actual_rows > 0) {
        int group_id     = lane_id >> 2;
        int tid_in_group = lane_id & 3;
        int row0 = group_id, row1 = group_id + 8;
        const bool odd_pair_lane = (tid_in_group & 1) != 0;
        const int store_row = odd_pair_lane ? row1 : row0;
        const int store_col = (tid_in_group >> 1) * 4;
        float final_sum0 = quad_sum(row_sum[0]);
        float final_sum1 = quad_sum(row_sum[1]);
        float inv0 = (final_sum0 > 0.0f) ? (1.0f / final_sum0) : 0.0f;
        float inv1 = (final_sum1 > 0.0f) ? (1.0f / final_sum1) : 0.0f;

        #pragma unroll
        for (int nt = 0; nt < OUT_N_TILES; nt++) {
            int n_base = nt * MMA_N;
            __nv_bfloat16 out00 = __float2bfloat16(acc_o[nt][0] * inv0);
            __nv_bfloat16 out01 = __float2bfloat16(acc_o[nt][1] * inv0);
            __nv_bfloat16 out10 = __float2bfloat16(acc_o[nt][2] * inv1);
            __nv_bfloat16 out11 = __float2bfloat16(acc_o[nt][3] * inv1);
            __nv_bfloat162 packed0 = __halves2bfloat162(out00, out01);
            __nv_bfloat162 packed1 = __halves2bfloat162(out10, out11);
            uint32_t pair0 = __BFLOAT162_TO_CUI(packed0);
            uint32_t pair1 = __BFLOAT162_TO_CUI(packed1);
            uint32_t send = odd_pair_lane ? pair0 : pair1;
            uint32_t recv = __shfl_xor_sync(0xffffffffu, send, 1);
            uint32_t lo = odd_pair_lane ? recv : pair0;
            uint32_t hi = odd_pair_lane ? pair1 : recv;
            uint2 out4 = make_uint2(lo, hi);

            if (store_row < actual_rows) {
                uint2* dst = reinterpret_cast<uint2*>(
                    o_ptr + (warp_q_start + store_row) * HEAD_DIM +
                    n_base + store_col);
                *dst = out4;
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
    // Stage one aligned 128-bit segment per participating thread, matching the
    // verified prefill-Q path.  Each logical SW128 row is a sequence of sixteen
    // aligned eight-BF16 segments, so this replaces 256 scalar load/store pairs
    // with 32 vector pairs without changing a single Q bit or WGMMA coordinate.
    // Leave the other 62 rows untouched because their outputs are discarded.
    {
        Tensor sQ = make_tensor(make_smem_ptr(q_smem), SmemLayoutQ{});
        constexpr int ELEMS_PER_VEC = sizeof(uint4) / sizeof(BF16);
        constexpr int VECS_PER_ROW  = HEAD_DIM / ELEMS_PER_VEC;
        constexpr int VEC_TOTAL     = DECODE_HEADS_PER_CTA * VECS_PER_ROW;
        if (tid < VEC_TOTAL) {
            int vi = tid;
            int row = vi / VECS_PER_ROW;
            int vec_in_row = vi % VECS_PER_ROW;
            int k = vec_in_row * ELEMS_PER_VEC;
            const uint4* src = reinterpret_cast<const uint4*>(
                q_ptr + row * HEAD_DIM);
            uint4 q_vec = src[vec_in_row];
            uint4* dst = reinterpret_cast<uint4*>(&sQ(row, k));
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
    int num_kv_blocks = (ctx_len + BLOCK_N - 1) / BLOCK_N;

    for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        int kv_start = kv_block * BLOCK_N;
        int kv_end   = min(kv_start + BLOCK_N, ctx_len);
        int valid_n  = kv_end - kv_start;
        const int vphase = kv_block & 1;

        // Load K directly through the CuTe-owned descriptor.  TmaK's native
        // box is one 128x64 BF16 SW128 half-tile, so two explicit boxes cover
        // the same 128x128 destination and coordinates as tma_partition/copy.
        {
            const int kphase = kv_block & 1;
            const int k_bytes = BLOCK_N * HEAD_DIM * sizeof(BF16);
            static_assert(TmaK::NumValSrc * sizeof(BF16) == KV_HALF_BYTES,
                          "short-decode K direct issue requires a 16-KiB TMA box");
            if (tid == 0) {
                const int k_global_row = kv_row_base + kv_start;
                auto* k_native =
                    reinterpret_cast<__nv_bfloat16*>(k_wgmma);
                mbar_arrive_tx(&mbar[0], k_bytes);
                tma_load_2d(tma_k.get_tma_descriptor(),
                            k_native + KV_LO_OFF, &mbar[0],
                            0, k_global_row);
                tma_load_2d(tma_k.get_tma_descriptor(),
                            k_native + KV_HI_OFF, &mbar[0],
                            HALF_DIM, k_global_row);
            }
            mbar_wait(&mbar[0], kphase);
        }
        __syncthreads();  // sync #1

        // WGMMA QK — all 128 threads unconditionally
        float acc_s[N_TILES][4];
        compute_qk_cute(q_smem, k_wgmma, acc_s);
        __syncthreads();  // sync #2

        // Issue the same two native 16-KiB V^T boxes directly after QK has
        // finished consuming K smem; retain the original softmax overlap.
        {
            const int vt_bytes = HEAD_DIM * BLOCK_N * sizeof(BF16);
            static_assert(TmaVt::NumValSrc * sizeof(BF16) == KV_HALF_BYTES,
                          "short-decode Vt direct issue requires a 16-KiB TMA box");
            if (tid == 0) {
                const int v_global_row = kv_row_base + kv_start;
                auto* vt_native =
                    reinterpret_cast<__nv_bfloat16*>(vt_smem);
                mbar_arrive_tx(&mbar[1], vt_bytes);
                tma_load_2d(tma_vt.get_tma_descriptor(),
                            vt_native + KV_LO_OFF, &mbar[1],
                            0, v_global_row);
                tma_load_2d(tma_vt.get_tma_descriptor(),
                            vt_native + KV_HI_OFF, &mbar[1],
                            HALF_DIM, v_global_row);
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
        int col0 = tid_in_group * 2;
        __nv_bfloat16* packed_o_ptr = o_ptr + packed_row * HEAD_DIM;

        #pragma unroll
        for (int nt = 0; nt < OUT_N_TILES; nt++) {
            int n_base = nt * MMA_N;
            // col0 is even, HEAD_DIM is 128, and every row begins on a
            // 256-byte boundary.  Preserve the original left-to-right BF16
            // conversion order, then commit the adjacent pair with one
            // naturally aligned 32-bit global store at the old col0 address.
            __nv_bfloat16 out0 =
                __float2bfloat16(acc_o[nt][0] * inv0);
            __nv_bfloat16 out1 =
                __float2bfloat16(acc_o[nt][1] * inv0);
            uint32_t packed_out = __BFLOAT162_TO_CUI(
                __halves2bfloat162(out0, out1));
            *reinterpret_cast<uint32_t*>(
                packed_o_ptr + n_base + col0) = packed_out;
        }
    }
}

// FUNCTION: unified_attn_decode_full_n128_kernel
// L1024-only short-decode symbol.  It preserves the original two-head CTA,
// vectorized Q staging, direct K/V TMA issue, softmax/PV order, and epilogue,
// while compiling out the unreachable partial-N128 branches.
template <bool ConsumerLocalAcquire>
__global__ void unified_attn_decode_full_n128_kernel(
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
    // Stage one aligned 128-bit segment per participating thread, matching the
    // verified prefill-Q path.  Each logical SW128 row is a sequence of sixteen
    // aligned eight-BF16 segments, so this replaces 256 scalar load/store pairs
    // with 32 vector pairs without changing a single Q bit or WGMMA coordinate.
    // Leave the other 62 rows untouched because their outputs are discarded.
    {
        Tensor sQ = make_tensor(make_smem_ptr(q_smem), SmemLayoutQ{});
        constexpr int ELEMS_PER_VEC = sizeof(uint4) / sizeof(BF16);
        constexpr int VECS_PER_ROW  = HEAD_DIM / ELEMS_PER_VEC;
        constexpr int VEC_TOTAL     = DECODE_HEADS_PER_CTA * VECS_PER_ROW;
        if (tid < VEC_TOTAL) {
            int vi = tid;
            int row = vi / VECS_PER_ROW;
            int vec_in_row = vi % VECS_PER_ROW;
            int k = vec_in_row * ELEMS_PER_VEC;
            const uint4* src = reinterpret_cast<const uint4*>(
                q_ptr + row * HEAD_DIM);
            uint4 q_vec = src[vec_in_row];
            uint4* dst = reinterpret_cast<uint4*>(&sQ(row, k));
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
    int num_kv_blocks = (ctx_len + BLOCK_N - 1) / BLOCK_N;

    for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        int kv_start = kv_block * BLOCK_N;
        // Iteration 132 is launched only for L1024: all eight N128 tiles
        // are physically full, so valid_n is a compile-time tile constant.
        constexpr int valid_n = BLOCK_N;
        const int vphase = kv_block & 1;

        // Load K directly through the CuTe-owned descriptor.  TmaK's native
        // box is one 128x64 BF16 SW128 half-tile, so two explicit boxes cover
        // the same 128x128 destination and coordinates as tma_partition/copy.
        {
            const int kphase = kv_block & 1;
            const int k_bytes = BLOCK_N * HEAD_DIM * sizeof(BF16);
            static_assert(TmaK::NumValSrc * sizeof(BF16) == KV_HALF_BYTES,
                          "short-decode K direct issue requires a 16-KiB TMA box");
            if (tid == 0) {
                const int k_global_row = kv_row_base + kv_start;
                auto* k_native =
                    reinterpret_cast<__nv_bfloat16*>(k_wgmma);
                mbar_arrive_tx(&mbar[0], k_bytes);
                tma_load_2d(tma_k.get_tma_descriptor(),
                            k_native + KV_LO_OFF, &mbar[0],
                            0, k_global_row);
                tma_load_2d(tma_k.get_tma_descriptor(),
                            k_native + KV_HI_OFF, &mbar[0],
                            HALF_DIM, k_global_row);
            }
            mbar_wait(&mbar[0], kphase);
        }
        if constexpr (!ConsumerLocalAcquire) {
            __syncthreads();  // sync #1: retained outside selected B1/B4
        }

        // WGMMA QK — all 128 threads unconditionally
        float acc_s[N_TILES][4];
        compute_qk_cute(q_smem, k_wgmma, acc_s);
        __syncthreads();  // sync #2

        // Issue the same two native 16-KiB V^T boxes directly after QK has
        // finished consuming K smem; retain the original softmax overlap.
        {
            const int vt_bytes = HEAD_DIM * BLOCK_N * sizeof(BF16);
            static_assert(TmaVt::NumValSrc * sizeof(BF16) == KV_HALF_BYTES,
                          "short-decode Vt direct issue requires a 16-KiB TMA box");
            if (tid == 0) {
                const int v_global_row = kv_row_base + kv_start;
                auto* vt_native =
                    reinterpret_cast<__nv_bfloat16*>(vt_smem);
                mbar_arrive_tx(&mbar[1], vt_bytes);
                tma_load_2d(tma_vt.get_tma_descriptor(),
                            vt_native + KV_LO_OFF, &mbar[1],
                            0, v_global_row);
                tma_load_2d(tma_vt.get_tma_descriptor(),
                            vt_native + KV_HI_OFF, &mbar[1],
                            HALF_DIM, v_global_row);
            }
        }
        if (warp_id == 0) {
            if (kv_block == 0) {
                softmax_update_reg<false, true, true, false>(
                    acc_s, acc_o, row_max, row_sum,
                    inv_sqrt, valid_n, kv_start, ctx_len - 1,
                    DECODE_HEADS_PER_CTA);
            } else {
                softmax_update_reg<false, true, false, false>(
                    acc_s, acc_o, row_max, row_sum,
                    inv_sqrt, valid_n, kv_start, ctx_len - 1,
                    DECODE_HEADS_PER_CTA);
            }
        }
        {
            mbar_wait(&mbar[1], vphase);
        }
        if constexpr (!ConsumerLocalAcquire) {
            __syncthreads();  // sync #3: retained outside selected B1/B4
        }

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
        int col0 = tid_in_group * 2;
        __nv_bfloat16* packed_o_ptr = o_ptr + packed_row * HEAD_DIM;

        #pragma unroll
        for (int nt = 0; nt < OUT_N_TILES; nt++) {
            int n_base = nt * MMA_N;
            // col0 is even, HEAD_DIM is 128, and every row begins on a
            // 256-byte boundary.  Preserve the original left-to-right BF16
            // conversion order, then commit the adjacent pair with one
            // naturally aligned 32-bit global store at the old col0 address.
            __nv_bfloat16 out0 =
                __float2bfloat16(acc_o[nt][0] * inv0);
            __nv_bfloat16 out1 =
                __float2bfloat16(acc_o[nt][1] * inv0);
            uint32_t packed_out = __BFLOAT162_TO_CUI(
                __halves2bfloat162(out0, out1));
            *reinterpret_cast<uint32_t*>(
                packed_o_ptr + n_base + col0) = packed_out;
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
    // Completed QK scores are copied here before the same tSrS fragment is
    // reused for the bounded one-block lookahead.  The flat owning array also
    // backs the FP32-to-BF16 P conversion without aliasing a pending QK group.
    cutlass::Array<float, N_TILES * 4> acc_s;

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
    // tOrP_acc supplies only the proven accumulator-to-RS layout.  Conversion
    // below reads the independent owning acc_s directly, never this tSrS alias
    // while a lookahead QK is pending.
    auto tOrP_acc = make_tensor(tSrS.data(), tOrP_layout);
    auto tOrP = make_tensor_like<BF16>(tOrP_acc);
    // The first PV's first K-slice overwrites the known-zero O accumulator.
    // The assignment inside each K loop leaves subsequent blocks at ScaleOut::One.
    tiled_mma_pv.accumulate_ = GMMA::ScaleOut::Zero;

    auto convert_scores_to_p = [&]() {
        using From_t = float;
        using To_t = BF16;
        constexpr int N = decltype(size(tOrP_acc))::value;
        static_assert(N == N_TILES * 4);
        cutlass::NumericArrayConverter<To_t, From_t, N> cvt;
        auto dst = recast<cutlass::Array<To_t, N>>(tOrP);
        dst[0] = cvt(acc_s);
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
            acc_s[nt * 4 + 0] = tSrS(nt * 4 + 0, 0, 0);
            acc_s[nt * 4 + 1] = tSrS(nt * 4 + 1, 0, 0);
            acc_s[nt * 4 + 2] = tSrS(nt * 4 + 2, 0, 0);
            acc_s[nt * 4 + 3] = tSrS(nt * 4 + 3, 0, 0);
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
        convert_scores_to_p();
    }
    // K0 can now be reused for K2; V0 remains live for PV0.
    __syncthreads();

    // Bounded two-score-block lookahead.  The runtime backedge is reached only
    // after wait<0>, avoiding Iteration 47's compiler-serialized loop-carried
    // accumulator queue.  Inside each explicit pair, QK(i+1) overlaps
    // softmax(i), and PV(i) overlaps softmax(i+1).  The clean-entry invariant
    // is: P(i-1) is ready, its output_rescale is pending, and no WGMMA group is
    // outstanding.
    int score_block = 1;
    for (; score_block + 1 < num_kv_blocks; score_block += 2) {
        int first = score_block;
        int second = first + 1;
        int prev = first - 1;
        int first_k_stage = first & 1;
        int first_k_phase = (first >> 1) & 1;
        int prev_v_stage = prev & 1;
        int prev_v_phase = (prev >> 1) & 1;

        mbar_wait(&mbar[first_k_stage], first_k_phase);
        mbar_wait(&mbar[2 + prev_v_stage], prev_v_phase);
        if (tid == 0) {
            issue_k(second);
            issue_v(first);
        }
        __syncthreads();

        BF16* k_first = reinterpret_cast<BF16*>(
            smem + (first_k_stage == 0 ? DECODE_K_STAGE0_OFF
                                       : DECODE_K_STAGE1_OFF));
        Tensor sKFirst = make_tensor(make_smem_ptr(k_first), SmemLayoutKW{});
        Tensor tCsKFirst = thr_mma_qk.partition_B(sKFirst);
        auto tCrKFirst = thr_mma_qk.make_fragment_B(tCsKFirst);
        clear(tSrS);
        warpgroup_fence_operand(tSrS);
        warpgroup_arrive();
        gemm(tiled_mma_qk, tCrQ, tCrKFirst, tSrS);
        warpgroup_commit_batch();
        warpgroup_fence_operand(tSrS);

        // Consume the scale produced by score block first-1 before PV(first-1).
        if (warp_id == 0) {
            #pragma unroll
            for (int nt = 0; nt < OUT_N_TILES; ++nt) {
                tOrO(nt * 4 + 0, 0, 0) *= output_rescale[0];
                tOrO(nt * 4 + 1, 0, 0) *= output_rescale[0];
                tOrO(nt * 4 + 2, 0, 0) *= output_rescale[1];
                tOrO(nt * 4 + 3, 0, 0) *= output_rescale[1];
            }
        }

        BF16* v_prev = reinterpret_cast<BF16*>(
            smem + (prev_v_stage == 0 ? DECODE_V_STAGE0_OFF
                                      : DECODE_V_STAGE1_OFF));
        Tensor sVtPrev = make_tensor(make_smem_ptr(v_prev), SmemLayoutVt{});
        Tensor tCsVtPrev = thr_mma_pv.partition_B(sVtPrev);
        auto tCrVtPrev = thr_mma_pv.make_fragment_B(tCsVtPrev);
        warpgroup_fence_operand(tOrP);
        warpgroup_fence_operand(tOrO);
        warpgroup_arrive();
        #pragma unroll
        for (int k = 0; k < size<2>(tOrP); ++k) {
            cute::gemm(tiled_mma_pv, tOrP(_,_,k), tCrVtPrev(_,_,k), tOrO);
            tiled_mma_pv.accumulate_ = GMMA::ScaleOut::One;
        }
        warpgroup_commit_batch();
        warpgroup_fence_operand(tOrO);
        warpgroup_fence_operand(tOrP);

        // Retire QK(first), keep PV(prev), and preserve its scores outside
        // tSrS before tSrS becomes the QK(second) destination.
        warpgroup_wait<1>();
        warpgroup_fence_operand(tSrS);
        #pragma unroll
        for (int nt = 0; nt < N_TILES; ++nt) {
            acc_s[nt * 4 + 0] = tSrS(nt * 4 + 0, 0, 0);
            acc_s[nt * 4 + 1] = tSrS(nt * 4 + 1, 0, 0);
            acc_s[nt * 4 + 2] = tSrS(nt * 4 + 2, 0, 0);
            acc_s[nt * 4 + 3] = tSrS(nt * 4 + 3, 0, 0);
        }

        // QK(first) has retired, so its K stage can prefetch the clean-entry K
        // block for the next pair while K(second) is consumed here.
        if (tid == 0 && first + 2 < num_kv_blocks) {
            issue_k(first + 2);
        }
        int second_k_stage = second & 1;
        int second_k_phase = (second >> 1) & 1;
        mbar_wait(&mbar[second_k_stage], second_k_phase);
        __syncthreads();

        BF16* k_second = reinterpret_cast<BF16*>(
            smem + (second_k_stage == 0 ? DECODE_K_STAGE0_OFF
                                        : DECODE_K_STAGE1_OFF));
        Tensor sKSecond = make_tensor(make_smem_ptr(k_second), SmemLayoutKW{});
        Tensor tCsKSecond = thr_mma_qk.partition_B(sKSecond);
        auto tCrKSecond = thr_mma_qk.make_fragment_B(tCsKSecond);
        clear(tSrS);
        warpgroup_fence_operand(tSrS);
        warpgroup_arrive();
        gemm(tiled_mma_qk, tCrQ, tCrKSecond, tSrS);
        warpgroup_commit_batch();
        warpgroup_fence_operand(tSrS);

        // Scalar softmax(first) now overlaps both the older PV(prev) and the
        // newer QK(second).  first cannot be the partial tail of this pair.
        if (warp_id == 0) {
            softmax_update_reg_deferred_o_rescale<false, true, false, false>(
                acc_s, row_max, row_sum, output_rescale,
                inv_sqrt, BLOCK_N, first * BLOCK_N, ctx_len - 1,
                DECODE_HEADS_PER_CTA);
        }

        // Retire PV(prev), retain QK(second).  V(prev)'s stage is now free for
        // V(second); V(first) has transferred in the opposite stage.
        warpgroup_wait<1>();
        warpgroup_fence_operand(tOrO);
        warpgroup_fence_operand(tOrP);
        if (tid == 0) {
            issue_v(second);
        }
        int first_v_stage = first & 1;
        int first_v_phase = (first >> 1) & 1;
        mbar_wait(&mbar[2 + first_v_stage], first_v_phase);
        __syncthreads();

        // Apply scale(first), convert P(first), and enqueue PV(first) behind
        // QK(second), preserving scale(O_previous) + P(first) * V(first).
        if (warp_id == 0) {
            #pragma unroll
            for (int nt = 0; nt < OUT_N_TILES; ++nt) {
                tOrO(nt * 4 + 0, 0, 0) *= output_rescale[0];
                tOrO(nt * 4 + 1, 0, 0) *= output_rescale[0];
                tOrO(nt * 4 + 2, 0, 0) *= output_rescale[1];
                tOrO(nt * 4 + 3, 0, 0) *= output_rescale[1];
            }
        }
        convert_scores_to_p();

        BF16* v_first = reinterpret_cast<BF16*>(
            smem + (first_v_stage == 0 ? DECODE_V_STAGE0_OFF
                                       : DECODE_V_STAGE1_OFF));
        Tensor sVtFirst = make_tensor(make_smem_ptr(v_first), SmemLayoutVt{});
        Tensor tCsVtFirst = thr_mma_pv.partition_B(sVtFirst);
        auto tCrVtFirst = thr_mma_pv.make_fragment_B(tCsVtFirst);
        warpgroup_fence_operand(tOrP);
        warpgroup_fence_operand(tOrO);
        warpgroup_arrive();
        #pragma unroll
        for (int k = 0; k < size<2>(tOrP); ++k) {
            cute::gemm(tiled_mma_pv, tOrP(_,_,k), tCrVtFirst(_,_,k), tOrO);
            tiled_mma_pv.accumulate_ = GMMA::ScaleOut::One;
        }
        warpgroup_commit_batch();
        warpgroup_fence_operand(tOrO);
        warpgroup_fence_operand(tOrP);

        // QK(second) is now the older group.  Expose it while PV(first) stays
        // outstanding and overlap that PV with softmax(second).
        warpgroup_wait<1>();
        warpgroup_fence_operand(tSrS);
        #pragma unroll
        for (int nt = 0; nt < N_TILES; ++nt) {
            acc_s[nt * 4 + 0] = tSrS(nt * 4 + 0, 0, 0);
            acc_s[nt * 4 + 1] = tSrS(nt * 4 + 1, 0, 0);
            acc_s[nt * 4 + 2] = tSrS(nt * 4 + 2, 0, 0);
            acc_s[nt * 4 + 3] = tSrS(nt * 4 + 3, 0, 0);
        }

        int second_kv_start = second * BLOCK_N;
        int second_valid_n = min(BLOCK_N, ctx_len - second_kv_start);
        if (warp_id == 0) {
            if (second_valid_n == BLOCK_N) {
                softmax_update_reg_deferred_o_rescale<false, true, false, false>(
                    acc_s, row_max, row_sum, output_rescale,
                    inv_sqrt, second_valid_n, second_kv_start, ctx_len - 1,
                    DECODE_HEADS_PER_CTA);
            } else {
                softmax_update_reg_deferred_o_rescale<false, true, false, true>(
                    acc_s, row_max, row_sum, output_rescale,
                    inv_sqrt, second_valid_n, second_kv_start, ctx_len - 1,
                    DECODE_HEADS_PER_CTA);
            }
        }

        // Close the asynchronous region before the only runtime backedge.
        // P(second) is converted after PV(first) releases the shared BF16 P
        // registers; scale(second) remains deferred for the next pair/tail.
        warpgroup_wait<0>();
        warpgroup_fence_operand(tSrS);
        warpgroup_fence_operand(tOrO);
        warpgroup_fence_operand(tOrP);
        convert_scores_to_p();
        __syncthreads();
    }

    // If the number of post-prologue score blocks is odd, process the final
    // score with Iteration 46's safe QK/PV pair.  The preceding bounded pair
    // has already prefetched K(last) and V(last-1).
    if (score_block < num_kv_blocks) {
        int last_score = score_block;
        int prev = last_score - 1;
        int k_stage = last_score & 1;
        int k_phase = (last_score >> 1) & 1;
        int v_stage = prev & 1;
        int v_phase = (prev >> 1) & 1;
        mbar_wait(&mbar[k_stage], k_phase);
        mbar_wait(&mbar[2 + v_stage], v_phase);
        if (tid == 0) {
            issue_v(last_score);
        }
        __syncthreads();

        BF16* k_last = reinterpret_cast<BF16*>(
            smem + (k_stage == 0 ? DECODE_K_STAGE0_OFF : DECODE_K_STAGE1_OFF));
        Tensor sKLast = make_tensor(make_smem_ptr(k_last), SmemLayoutKW{});
        Tensor tCsKLast = thr_mma_qk.partition_B(sKLast);
        auto tCrKLast = thr_mma_qk.make_fragment_B(tCsKLast);
        clear(tSrS);
        warpgroup_fence_operand(tSrS);
        warpgroup_arrive();
        gemm(tiled_mma_qk, tCrQ, tCrKLast, tSrS);
        warpgroup_commit_batch();
        warpgroup_fence_operand(tSrS);

        if (warp_id == 0) {
            #pragma unroll
            for (int nt = 0; nt < OUT_N_TILES; ++nt) {
                tOrO(nt * 4 + 0, 0, 0) *= output_rescale[0];
                tOrO(nt * 4 + 1, 0, 0) *= output_rescale[0];
                tOrO(nt * 4 + 2, 0, 0) *= output_rescale[1];
                tOrO(nt * 4 + 3, 0, 0) *= output_rescale[1];
            }
        }

        BF16* v_prev = reinterpret_cast<BF16*>(
            smem + (v_stage == 0 ? DECODE_V_STAGE0_OFF : DECODE_V_STAGE1_OFF));
        Tensor sVtPrev = make_tensor(make_smem_ptr(v_prev), SmemLayoutVt{});
        Tensor tCsVtPrev = thr_mma_pv.partition_B(sVtPrev);
        auto tCrVtPrev = thr_mma_pv.make_fragment_B(tCsVtPrev);
        warpgroup_fence_operand(tOrP);
        warpgroup_fence_operand(tOrO);
        warpgroup_arrive();
        #pragma unroll
        for (int k = 0; k < size<2>(tOrP); ++k) {
            cute::gemm(tiled_mma_pv, tOrP(_,_,k), tCrVtPrev(_,_,k), tOrO);
            tiled_mma_pv.accumulate_ = GMMA::ScaleOut::One;
        }
        warpgroup_commit_batch();
        warpgroup_fence_operand(tOrO);
        warpgroup_fence_operand(tOrP);

        warpgroup_wait<1>();
        warpgroup_fence_operand(tSrS);
        #pragma unroll
        for (int nt = 0; nt < N_TILES; ++nt) {
            acc_s[nt * 4 + 0] = tSrS(nt * 4 + 0, 0, 0);
            acc_s[nt * 4 + 1] = tSrS(nt * 4 + 1, 0, 0);
            acc_s[nt * 4 + 2] = tSrS(nt * 4 + 2, 0, 0);
            acc_s[nt * 4 + 3] = tSrS(nt * 4 + 3, 0, 0);
        }

        int kv_start = last_score * BLOCK_N;
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

        warpgroup_wait<0>();
        warpgroup_fence_operand(tSrS);
        warpgroup_fence_operand(tOrO);
        warpgroup_fence_operand(tOrP);
        convert_scores_to_p();
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

// FUNCTION: unified_attn_decode_four_head_qk_kernel
// One CTA owns a complete (batch, KV head, KV block) tile.  For the benchmark's
// 4:1 GQA ratio this reuses the 32-KiB K tile across all four query heads,
// halving both producer CTAs and K traffic versus iteration 52.  Iteration 58
// places the four live Q rows at the first M row of each WGMMA warp stripe
// (0/16/32/48).  Local lanes 0..3 in every warp then publish one compact
// float2, retaining the exact 16-pair score-buffer ABI consumed by Iteration
// 57.  K, Q, and the transaction barrier stay packed contiguously.
// Iteration 126 separately compiles the exact-N128 case so the post-Iteration
// 63 producer can hoist final-tile valid_n arithmetic and predicates out of
// every configured 4K/8K CTA.  The generic symbol retains arbitrary tails.
// Iteration 130 specialized whether the selected consumer needed the former
// D16-only block-maximum sideband.  Iteration 150 intentionally re-enables
// that exact sideband for D32 so four sibling consumers share one max scan.
// Iteration 171 gives only this common producer's single-use K stream an
// evict-first TMA policy, leaving every consumer and every arithmetic edge
// untouched.
// Iteration 178 assigns those two disjoint hinted boxes to separate warp
// leaders and joins them with the promoted count-two transaction contract.
template <bool FullKvTiles, bool PublishBlockMax = true>
__global__ void unified_attn_decode_four_head_qk_kernel(
    const __nv_bfloat16* __restrict__ Q,
    CUTLASS_GRID_CONSTANT TmaK const tma_k,
    float* __restrict__ score_buffer,
    float* __restrict__ block_max_buffer,
    int ctx_len, int num_heads, int num_kv_heads,
    int total_kv_rows, int num_kv_blocks
) {
    extern __shared__ char smem[];
    constexpr int GQA_HEADS_PER_KV = 4;
    constexpr int LIVE_SCORE_LANES = 16;
    constexpr int LIVE_SCORE_COMPONENTS = 2;
    constexpr int SCORE_VALUES_PER_BLOCK =
        N_TILES * LIVE_SCORE_LANES * LIVE_SCORE_COMPONENTS;
    constexpr int PRODUCER_K_OFF = 0;
    constexpr int PRODUCER_Q_OFF = KV_SMEM_BYTES;
    constexpr int PRODUCER_MBAR_OFF = PRODUCER_Q_OFF + Q_WGMMA_BYTES;

    int kv_block = int(blockIdx.x);
    int kv_head_idx = int(blockIdx.y);
    int batch_idx = int(blockIdx.z);
    int head_idx = kv_head_idx * GQA_HEADS_PER_KV;
    int tid = int(threadIdx.x);

    const __nv_bfloat16* q_ptr = Q +
        (batch_idx * num_heads + head_idx) * HEAD_DIM;
    int kv_row_base =
        (batch_idx * num_kv_heads + kv_head_idx) * ctx_len;
    int kv_start = kv_block * BLOCK_N;

    BF16* k_smem = reinterpret_cast<BF16*>(smem + PRODUCER_K_OFF);
    BF16* q_smem = reinterpret_cast<BF16*>(smem + PRODUCER_Q_OFF);
    uint64_t* mbar = reinterpret_cast<uint64_t*>(smem + PRODUCER_MBAR_OFF);

    if (tid == 0) {
        cute::prefetch_tma_descriptor(tma_k.get_tma_descriptor());
        mbar_init(&mbar[0], 2);
    }
    {
        Tensor sQ = make_tensor(make_smem_ptr(q_smem), SmemLayoutQ{});
        constexpr int PACKED_Q_ELEMS = GQA_HEADS_PER_KV * HEAD_DIM;
        for (int idx = tid; idx < PACKED_Q_ELEMS; idx += BLOCK_THREADS) {
            int head = idx / HEAD_DIM;
            int k = idx % HEAD_DIM;
            sQ(head * MMA_M, k) = BF16(q_ptr[head * HEAD_DIM + k]);
        }
    }
    __syncthreads();

    if (tid == 0 || tid == WARP_SIZE) {
        static_assert(TmaK::NumValSrc * sizeof(BF16) == KV_HALF_BYTES,
                      "four-head K direct issue requires a 16-KiB TMA box");
        // Each warp leader owns one native box and independently contributes
        // its 16-KiB expectation to the count-two K-ready barrier.
        // K is single-use in this producer.  Prefer eviction over the score
        // and block-max workspace published after QK, which four/eight sibling
        // output-D CTAs consume in the dependent kernel.
        constexpr uint64_t k_l2_evict_first_policy =
            static_cast<uint64_t>(TMA::CacheHintSm90::EVICT_FIRST);
        const int half = tid / WARP_SIZE;
        mbar_arrive_tx(&mbar[0], KV_HALF_BYTES);
        auto* k_native = reinterpret_cast<__nv_bfloat16*>(k_smem);
        const int k_global_row = kv_row_base + kv_start;
        tma_load_2d_l2_cache_hint(
            tma_k.get_tma_descriptor(),
            k_native + half * KV_HI_OFF, &mbar[0],
            half * HALF_DIM, k_global_row, k_l2_evict_first_policy);
    }
    mbar_wait(&mbar[0], 0);
    // Every producer thread has completed the acquire wait.  QK is
    // warpgroup-aligned and this one-shot K tile is not reused, so a
    // second CTA-wide publication/reuse rendezvous is unnecessary.

    float acc_s[N_TILES][4];
    compute_qk_cute(q_smem, k_smem, acc_s);

    // Iteration 61 performs the exact FP32 attention-score scaling once in
    // the producer instead of repeating it in all eight D16 consumers.  The
    // rounded FP32 product is stored in the unchanged compact workspace, so
    // the consumer observes the same value at the same softmax boundary.
    float inv_sqrt = 1.0f / sqrtf((float)HEAD_DIM);
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int kv_linear = batch_idx * num_kv_heads + kv_head_idx;
    int valid_n;
    if constexpr (FullKvTiles) {
        valid_n = BLOCK_N;
    } else {
        valid_n = min(BLOCK_N, ctx_len - kv_start);
    }
    float block_max = -INFINITY;
    uint64_t workspace_l2_policy;
    float workspace_l2_fraction = 1.0f;
    asm volatile(
        "createpolicy.fractional.L2::evict_last.b64 %0, %1;\n"
        : "=l"(workspace_l2_policy)
        : "f"(workspace_l2_fraction)
        : "memory");
    if (lane_id < GQA_HEADS_PER_KV) {
        int compact_lane = warp_id * GQA_HEADS_PER_KV + lane_id;
        float* block_scores = score_buffer +
            (static_cast<int64_t>(kv_linear) * num_kv_blocks + kv_block) *
                SCORE_VALUES_PER_BLOCK;
        float2* block_score_pairs = reinterpret_cast<float2*>(block_scores);
        #pragma unroll
        for (int nt = 0; nt < N_TILES; ++nt) {
            float s0 = acc_s[nt][0] * inv_sqrt;
            float s1 = acc_s[nt][1] * inv_sqrt;
            union {
                float2 value;
                uint64_t bits;
            } score_word;
            score_word.value = make_float2(s0, s1);
            float2* score_addr =
                &block_score_pairs[
                    nt * LIVE_SCORE_LANES + compact_lane];
            asm volatile(
                "st.global.L2::cache_hint.b64 [%0], %1, %2;\n"
                :
                : "l"(score_addr), "l"(score_word.bits),
                  "l"(workspace_l2_policy)
                : "memory");

            if constexpr (PublishBlockMax) {
                // Reproduce the D16 consumer's exact scaled, tail-masked
                // maximum scan before hoisting it across all D16 slices.
                int n_base = nt * MMA_N;
                int n0 = n_base + lane_id * 2;
                int n1 = n0 + 1;
                float max_s0 = s0;
                float max_s1 = s1;
                if constexpr (!FullKvTiles) {
                    max_s0 = n0 >= valid_n ? -INFINITY : s0;
                    max_s1 = n1 >= valid_n ? -INFINITY : s1;
                }
                block_max = fmaxf(block_max, fmaxf(max_s0, max_s1));
            }
        }
    }
    if constexpr (PublishBlockMax) {
        // Identical four-lane xor-1/xor-2 tree to update_one_live_row.  Each
        // warp publishes one FP32 maximum per KV block for D16 consumers.
        block_max = quad_max(block_max);
        if (lane_id == 0) {
            float* block_maxes = block_max_buffer +
                (static_cast<int64_t>(kv_linear) * num_kv_blocks + kv_block) *
                    GQA_HEADS_PER_KV;
            float* block_max_addr = &block_maxes[warp_id];
            uint32_t block_max_bits = __float_as_uint(block_max);
            asm volatile(
                "st.global.L2::cache_hint.b32 [%0], %1, %2;\n"
                :
                : "l"(block_max_addr), "r"(block_max_bits),
                  "l"(workspace_l2_policy)
                : "memory");
        }
    }
}

// FUNCTION: unified_attn_decode_actor_score_consumer_kernel
// QK-free long consumer.  One physical warpgroup owns RS-PV and a fifth warp
// owns the exact scalar softmax recurrence.  The scalar warp restores all four
// packed GQA rows in ascending KV-block order, then hands
// BF16 P and deferred O-rescale values to PV through a two-stage count-one
// PFull/PEmpty mbarrier ring.  Two independent V TMA stages let scalar softmax
// overlap the preceding PV without changing block, softmax, or PV order.
__global__ void unified_attn_decode_actor_score_consumer_kernel(
    const float* __restrict__ score_buffer,
    CUTLASS_GRID_CONSTANT TmaVt const tma_vt,
    __nv_bfloat16* __restrict__ O,
    int ctx_len, int num_heads, int num_kv_heads,
    int total_kv_rows, int num_kv_blocks
) {
    extern __shared__ char smem[];
    constexpr int DECODE_HEADS_PER_CTA = 4;
    constexpr int PACKED_SCORE_LANES = 16;
    constexpr int LIVE_SCORE_COMPONENTS = 2;
    constexpr int SCORE_VALUES_PER_BLOCK =
        N_TILES * PACKED_SCORE_LANES * LIVE_SCORE_COMPONENTS;
    constexpr int CONSUMER_THREADS = 5 * WARP_SIZE;
    constexpr int PV_WG_THREADS = 4 * WARP_SIZE;
    constexpr int FRAGMENT_VALS = N_TILES * 4;
    constexpr int LIVE_P_LANES = 16;
    constexpr int LIVE_P_COMPONENTS = 2;
    constexpr int V_STAGE0_OFF = 0;
    constexpr int V_STAGE1_OFF = KV_SMEM_BYTES;
    constexpr int TMA_MBAR_OFF = 2 * KV_SMEM_BYTES;
    constexpr int TMA_MBAR_BYTES = 2 * sizeof(uint64_t);
    constexpr int P_STAGE_BYTES =
        N_TILES * LIVE_P_LANES * LIVE_P_COMPONENTS * sizeof(BF16);
    constexpr int SCALE_STAGE_BYTES =
        LIVE_P_LANES * sizeof(float);
    constexpr int P_RING_OFF = TMA_MBAR_OFF + TMA_MBAR_BYTES;
    constexpr int SCALE_RING_OFF = P_RING_OFF + 2 * P_STAGE_BYTES;
    constexpr int NORM_OFF = SCALE_RING_OFF + 2 * SCALE_STAGE_BYTES;
    constexpr int ACTOR_MBAR_OFF = NORM_OFF + LIVE_P_LANES * sizeof(float);
    constexpr int P_FULL_MBAR = 0;
    constexpr int P_EMPTY_MBAR = 2;
    constexpr int NORM_READY_MBAR = 4;
    constexpr int ACTOR_MBAR_COUNT = 5;

    int head_group_in_kv = int(blockIdx.x);
    int kv_head_idx = int(blockIdx.y);
    int batch_idx = int(blockIdx.z);
    int head_group_idx = kv_head_idx * int(gridDim.x) + head_group_in_kv;
    int head_idx = head_group_idx * DECODE_HEADS_PER_CTA;
    int tid = int(threadIdx.x);

    __nv_bfloat16* o_ptr = O +
        (batch_idx * num_heads + head_idx) * HEAD_DIM;
    int kv_row_base =
        (batch_idx * num_kv_heads + kv_head_idx) * ctx_len;
    int kv_linear = batch_idx * num_kv_heads + kv_head_idx;
    const float* kv_scores = score_buffer +
        static_cast<int64_t>(kv_linear) * num_kv_blocks *
        SCORE_VALUES_PER_BLOCK;
    uint64_t* tma_mbar = reinterpret_cast<uint64_t*>(smem + TMA_MBAR_OFF);
    BF16* p_ring = reinterpret_cast<BF16*>(smem + P_RING_OFF);
    float* scale_ring = reinterpret_cast<float*>(smem + SCALE_RING_OFF);
    float* norm_smem = reinterpret_cast<float*>(smem + NORM_OFF);
    uint64_t* actor_mbar = reinterpret_cast<uint64_t*>(smem + ACTOR_MBAR_OFF);

    if (tid == 0) {
        cute::prefetch_tma_descriptor(tma_vt.get_tma_descriptor());
        mbar_init(&tma_mbar[0], 1);
        mbar_init(&tma_mbar[1], 1);
        #pragma unroll
        for (int i = 0; i < ACTOR_MBAR_COUNT; ++i) {
            mbar_init(&actor_mbar[i], 1);
        }
    }
    __syncthreads();

    auto issue_v = [&](int kv_block) {
        int stage = kv_block & 1;
        int kv_start = kv_block * BLOCK_N;
        BF16* v_stage = reinterpret_cast<BF16*>(
            smem + (stage == 0 ? V_STAGE0_OFF : V_STAGE1_OFF));
        constexpr int VT_BYTES = HEAD_DIM * BLOCK_N * sizeof(BF16);
        mbar_arrive_tx(&tma_mbar[stage], VT_BYTES);
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
        copy(tma_vt.with(tma_mbar[stage]), tVgVt, tVsVt);
    };

    if (tid < PV_WG_THREADS) {
        int local_tid = tid;
        TiledMmaPV tiled_mma_pv;
        ThrMMA thr_mma_pv = tiled_mma_pv.get_thread_slice(local_tid);
        auto tOrO = partition_fragment_C(
            tiled_mma_pv, Shape<Int<BLOCK_M_PREFILL>, Int<HEAD_DIM>>{});
        clear(tOrO);
        auto tOrP_layout =
            convert_layout_acc_Aregs<TiledMmaPV>(tOrO.layout());
        auto tOrP = make_tensor<BF16>(tOrP_layout);
        auto p_regs = recast<cutlass::Array<BF16, FRAGMENT_VALS>>(tOrP);
        tiled_mma_pv.accumulate_ = GMMA::ScaleOut::Zero;

        if (local_tid == 0) {
            if (num_kv_blocks > 0) issue_v(0);
            if (num_kv_blocks > 1) issue_v(1);
        }

        for (int kv_block = 0; kv_block < num_kv_blocks; ++kv_block) {
            int stage = kv_block & 1;
            int phase = (kv_block >> 1) & 1;
            if (local_tid == 0) {
                mbar_wait(&actor_mbar[P_FULL_MBAR + stage], phase);
            }
            cutlass::arch::NamedBarrier::sync(PV_WG_THREADS, 0);

            // Only rows 0..3 are live.  Zero the complete RS fragment first,
            // then restore its two live components for lanes 0..15 from the
            // compact [stage][nt][lane][component] P ring.
            #pragma unroll
            for (int frag = 0; frag < FRAGMENT_VALS; ++frag) {
                p_regs[0][frag] = BF16(0.0f);
            }
            float scale0 = 1.0f;
            if (local_tid < LIVE_P_LANES) {
                BF16* stage_p = p_ring +
                    stage * N_TILES * LIVE_P_LANES * LIVE_P_COMPONENTS;
                float* stage_scale =
                    scale_ring + stage * LIVE_P_LANES;
                #pragma unroll
                for (int nt = 0; nt < N_TILES; ++nt) {
                    int compact_idx =
                        (nt * LIVE_P_LANES + local_tid) * LIVE_P_COMPONENTS;
                    p_regs[0][nt * 4 + 0] = stage_p[compact_idx + 0];
                    p_regs[0][nt * 4 + 1] = stage_p[compact_idx + 1];
                }
                scale0 = stage_scale[local_tid];
            }
            cutlass::arch::NamedBarrier::sync(PV_WG_THREADS, 0);
            if (local_tid == 0) {
                cutlass::arch::ClusterBarrier::arrive(
                    &actor_mbar[P_EMPTY_MBAR + stage]);
            }

            if (local_tid < LIVE_P_LANES) {
                #pragma unroll
                for (int nt = 0; nt < OUT_N_TILES; ++nt) {
                    tOrO(nt * 4 + 0, 0, 0) *= scale0;
                    tOrO(nt * 4 + 1, 0, 0) *= scale0;
                }
            }

            mbar_wait(&tma_mbar[stage], phase);
            BF16* v_stage = reinterpret_cast<BF16*>(
                smem + (stage == 0 ? V_STAGE0_OFF : V_STAGE1_OFF));
            Tensor sVt = make_tensor(make_smem_ptr(v_stage), SmemLayoutVt{});
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
            warpgroup_wait<0>();
            warpgroup_fence_operand(tOrO);
            warpgroup_fence_operand(tOrP);

            if (local_tid == 0 && kv_block + 2 < num_kv_blocks) {
                issue_v(kv_block + 2);
            }
        }

        if (local_tid == 0) {
            mbar_wait(&actor_mbar[NORM_READY_MBAR], 0);
        }
        cutlass::arch::NamedBarrier::sync(PV_WG_THREADS, 0);
        int packed_row = local_tid >> 2;
        if (local_tid < LIVE_P_LANES &&
            packed_row < DECODE_HEADS_PER_CTA) {
            float final_sum = norm_smem[local_tid];
            float inv = final_sum > 0.0f ? 1.0f / final_sum : 0.0f;
            int col0 = (local_tid & 3) * 2;
            int col1 = col0 + 1;
            __nv_bfloat16* packed_o_ptr = o_ptr + packed_row * HEAD_DIM;
            #pragma unroll
            for (int nt = 0; nt < OUT_N_TILES; ++nt) {
                int n_base = nt * MMA_N;
                packed_o_ptr[n_base + col0] =
                    __float2bfloat16(tOrO(nt * 4 + 0, 0, 0) * inv);
                packed_o_ptr[n_base + col1] =
                    __float2bfloat16(tOrO(nt * 4 + 1, 0, 0) * inv);
            }
        }
    } else if (tid < CONSUMER_THREADS) {
        int lane_id = tid - PV_WG_THREADS;
        float row_max[2] = {-INFINITY, -INFINITY};
        float row_sum[2] = {0.0f, 0.0f};
        float output_rescale[2] = {1.0f, 1.0f};
        cutlass::Array<float, FRAGMENT_VALS> acc_s;
        float inv_sqrt = 1.0f / sqrtf((float)HEAD_DIM);
        using PConverter =
            cutlass::NumericArrayConverter<BF16, float, FRAGMENT_VALS>;
        PConverter convert_p;

        for (int kv_block = 0; kv_block < num_kv_blocks; ++kv_block) {
            #pragma unroll
            for (int frag = 0; frag < FRAGMENT_VALS; ++frag) {
                acc_s[frag] = 0.0f;
            }
            if (lane_id < PACKED_SCORE_LANES) {
                const float* block_scores = kv_scores +
                    static_cast<int64_t>(kv_block) * SCORE_VALUES_PER_BLOCK;
                const float2* score_pairs =
                    reinterpret_cast<const float2*>(block_scores);
                #pragma unroll
                for (int nt = 0; nt < N_TILES; ++nt) {
                    float2 scores =
                        score_pairs[nt * PACKED_SCORE_LANES + lane_id];
                    acc_s[nt * 4 + 0] = scores.x;
                    acc_s[nt * 4 + 1] = scores.y;
                }
            }

            int kv_start = kv_block * BLOCK_N;
            int valid_n = min(BLOCK_N, ctx_len - kv_start);
            if (kv_block == 0) {
                if (valid_n == BLOCK_N) {
                    softmax_update_reg_deferred_o_rescale<
                        false, true, true, false>(
                        acc_s, row_max, row_sum, output_rescale,
                        inv_sqrt, valid_n, kv_start, ctx_len - 1,
                        DECODE_HEADS_PER_CTA);
                } else {
                    softmax_update_reg_deferred_o_rescale<
                        false, true, true, true>(
                        acc_s, row_max, row_sum, output_rescale,
                        inv_sqrt, valid_n, kv_start, ctx_len - 1,
                        DECODE_HEADS_PER_CTA);
                }
            } else if (valid_n == BLOCK_N) {
                softmax_update_reg_deferred_o_rescale<
                    false, true, false, false>(
                    acc_s, row_max, row_sum, output_rescale,
                    inv_sqrt, valid_n, kv_start, ctx_len - 1,
                    DECODE_HEADS_PER_CTA);
            } else {
                softmax_update_reg_deferred_o_rescale<
                    false, true, false, true>(
                    acc_s, row_max, row_sum, output_rescale,
                    inv_sqrt, valid_n, kv_start, ctx_len - 1,
                    DECODE_HEADS_PER_CTA);
            }

            int stage = kv_block & 1;
            if (lane_id == 0 && kv_block >= 2) {
                int empty_phase = ((kv_block - 2) >> 1) & 1;
                mbar_wait(&actor_mbar[P_EMPTY_MBAR + stage], empty_phase);
            }
            __syncwarp();
            if (lane_id < LIVE_P_LANES) {
                cutlass::Array<BF16, FRAGMENT_VALS> p_values =
                    convert_p(acc_s);
                BF16* stage_p = p_ring +
                    stage * N_TILES * LIVE_P_LANES * LIVE_P_COMPONENTS;
                float* stage_scale =
                    scale_ring + stage * LIVE_P_LANES;
                #pragma unroll
                for (int nt = 0; nt < N_TILES; ++nt) {
                    int compact_idx =
                        (nt * LIVE_P_LANES + lane_id) * LIVE_P_COMPONENTS;
                    stage_p[compact_idx + 0] = p_values[nt * 4 + 0];
                    stage_p[compact_idx + 1] = p_values[nt * 4 + 1];
                }
                stage_scale[lane_id] = output_rescale[0];
            }
            __syncwarp();
            if (lane_id == 0) {
                cutlass::arch::ClusterBarrier::arrive(
                    &actor_mbar[P_FULL_MBAR + stage]);
            }
        }

        row_sum[0] = quad_sum(row_sum[0]);
        if (lane_id < LIVE_P_LANES) {
            norm_smem[lane_id] = row_sum[0];
        }
        __syncwarp();
        if (lane_id == 0) {
            cutlass::arch::ClusterBarrier::arrive(
                &actor_mbar[NORM_READY_MBAR]);
        }
    }
}

// FUNCTION: unified_attn_decode_intrawg_score_consumer_kernel
// QK-free long consumer with FA3-style intra-warpgroup overlap.  The complete
// 128-thread warpgroup owns RS-PV.  While PV(i) is outstanding, only warp 0
// restores the exact producer scores and executes the unchanged ascending
// online-softmax update for block i+1 in an FP32 register array that is
// disjoint from both WGMMA operands.  A named-barrier rendezvous followed by
// wait_group 0 closes PV(i) before any thread rescales O or overwrites the
// current RS-P registers.  Consequently no WGMMA group crosses the runtime
// loop backedge and no shared P/scale ring or actor mbarrier is required.
__global__ void unified_attn_decode_intrawg_score_consumer_kernel(
    const float* __restrict__ score_buffer,
    CUTLASS_GRID_CONSTANT TmaVt const tma_vt,
    __nv_bfloat16* __restrict__ O,
    int ctx_len, int num_heads, int num_kv_heads,
    int total_kv_rows, int num_kv_blocks
) {
    extern __shared__ char smem[];
    constexpr int DECODE_HEADS_PER_CTA = 4;
    constexpr int PACKED_SCORE_LANES = 16;
    constexpr int LIVE_SCORE_COMPONENTS = 2;
    constexpr int SCORE_VALUES_PER_BLOCK =
        N_TILES * PACKED_SCORE_LANES * LIVE_SCORE_COMPONENTS;
    constexpr int PV_WG_THREADS = 4 * WARP_SIZE;
    constexpr int FRAGMENT_VALS = N_TILES * 4;
    constexpr int LIVE_OUTPUT_LANES = 16;
    constexpr int V_STAGE0_OFF = 0;
    constexpr int V_STAGE1_OFF = KV_SMEM_BYTES;
    constexpr int TMA_MBAR_OFF = 2 * KV_SMEM_BYTES;

    int head_group_in_kv = int(blockIdx.x);
    int kv_head_idx = int(blockIdx.y);
    int batch_idx = int(blockIdx.z);
    int head_group_idx = kv_head_idx * int(gridDim.x) + head_group_in_kv;
    int head_idx = head_group_idx * DECODE_HEADS_PER_CTA;
    int tid = int(threadIdx.x);
    int warp_id = tid / WARP_SIZE;

    __nv_bfloat16* o_ptr = O +
        (batch_idx * num_heads + head_idx) * HEAD_DIM;
    int kv_row_base =
        (batch_idx * num_kv_heads + kv_head_idx) * ctx_len;
    int kv_linear = batch_idx * num_kv_heads + kv_head_idx;
    const float* kv_scores = score_buffer +
        static_cast<int64_t>(kv_linear) * num_kv_blocks *
        SCORE_VALUES_PER_BLOCK;
    uint64_t* tma_mbar = reinterpret_cast<uint64_t*>(smem + TMA_MBAR_OFF);

    if (tid == 0) {
        cute::prefetch_tma_descriptor(tma_vt.get_tma_descriptor());
        mbar_init(&tma_mbar[0], 1);
        mbar_init(&tma_mbar[1], 1);
    }
    __syncthreads();

    auto issue_v = [&](int kv_block) {
        int stage = kv_block & 1;
        int kv_start = kv_block * BLOCK_N;
        BF16* v_stage = reinterpret_cast<BF16*>(
            smem + (stage == 0 ? V_STAGE0_OFF : V_STAGE1_OFF));
        constexpr int VT_BYTES = HEAD_DIM * BLOCK_N * sizeof(BF16);
        mbar_arrive_tx(&tma_mbar[stage], VT_BYTES);
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
        copy(tma_vt.with(tma_mbar[stage]), tVgVt, tVsVt);
    };

    if (tid == 0) {
        if (num_kv_blocks > 0) issue_v(0);
        if (num_kv_blocks > 1) issue_v(1);
    }

    TiledMmaPV tiled_mma_pv;
    ThrMMA thr_mma_pv = tiled_mma_pv.get_thread_slice(tid);
    auto tOrO = partition_fragment_C(
        tiled_mma_pv, Shape<Int<BLOCK_M_PREFILL>, Int<HEAD_DIM>>{});
    clear(tOrO);
    auto tOrP_layout = convert_layout_acc_Aregs<TiledMmaPV>(tOrO.layout());
    auto tOrP = make_tensor<BF16>(tOrP_layout);
    auto p_regs = recast<cutlass::Array<BF16, FRAGMENT_VALS>>(tOrP);

    // The FP32 score/softmax registers are intentionally independent from the
    // asynchronous WGMMA accumulator and RS-P operands.  Inactive warpgroup
    // rows remain positive zero for every block.
    cutlass::Array<float, FRAGMENT_VALS> next_scores;
    #pragma unroll
    for (int frag = 0; frag < FRAGMENT_VALS; ++frag) {
        next_scores[frag] = 0.0f;
    }
    float row_max[2] = {-INFINITY, -INFINITY};
    float row_sum[2] = {0.0f, 0.0f};
    float output_rescale[2] = {1.0f, 1.0f};
    float inv_sqrt = 1.0f / sqrtf((float)HEAD_DIM);
    using PConverter =
        cutlass::NumericArrayConverter<BF16, float, FRAGMENT_VALS>;
    PConverter convert_p;

    auto load_exact_scores = [&](int kv_block) {
        if (warp_id == 0 && tid < PACKED_SCORE_LANES) {
            const float* block_scores = kv_scores +
                static_cast<int64_t>(kv_block) * SCORE_VALUES_PER_BLOCK;
            const float2* score_pairs =
                reinterpret_cast<const float2*>(block_scores);
            #pragma unroll
            for (int nt = 0; nt < N_TILES; ++nt) {
                float2 scores =
                    score_pairs[nt * PACKED_SCORE_LANES + tid];
                next_scores[nt * 4 + 0] = scores.x;
                next_scores[nt * 4 + 1] = scores.y;
            }
        }
    };

    // Prologue: prepare P(0) while V(0) and V(1) are in flight.  The first
    // online-softmax update is exactly the Iteration-53 actor operation.
    if (num_kv_blocks > 0) {
        load_exact_scores(0);
        if (warp_id == 0) {
            int valid_n = min(BLOCK_N, ctx_len);
            if (valid_n == BLOCK_N) {
                softmax_update_reg_deferred_o_rescale<
                    false, true, true, false>(
                    next_scores, row_max, row_sum, output_rescale,
                    inv_sqrt, valid_n, 0, ctx_len - 1,
                    DECODE_HEADS_PER_CTA);
            } else {
                softmax_update_reg_deferred_o_rescale<
                    false, true, true, true>(
                    next_scores, row_max, row_sum, output_rescale,
                    inv_sqrt, valid_n, 0, ctx_len - 1,
                    DECODE_HEADS_PER_CTA);
            }
        }
        cutlass::arch::NamedBarrier::sync(PV_WG_THREADS, 0);
        p_regs[0] = convert_p(next_scores);
    }

    tiled_mma_pv.accumulate_ = GMMA::ScaleOut::Zero;
    // Every steady-state iteration has a following score tile.  PV(i) is
    // issued, softmax(i+1) runs under it, and wait_group 0 closes the group
    // before the backedge.  The final PV is peeled below.
    for (int kv_block = 0; kv_block + 1 < num_kv_blocks; ++kv_block) {
        int stage = kv_block & 1;
        int phase = (kv_block >> 1) & 1;
        mbar_wait(&tma_mbar[stage], phase);

        BF16* v_stage = reinterpret_cast<BF16*>(
            smem + (stage == 0 ? V_STAGE0_OFF : V_STAGE1_OFF));
        Tensor sVt = make_tensor(make_smem_ptr(v_stage), SmemLayoutVt{});
        Tensor tCsVt = thr_mma_pv.partition_B(sVt);
        auto tCrVt = thr_mma_pv.make_fragment_B(tCsVt);

        // Start PV(i).  Until wait_group 0 below, neither tOrO nor tOrP is
        // referenced by ordinary instructions; warp 0 works only on the
        // separate next_scores/softmax state.
        warpgroup_fence_operand(tOrP);
        warpgroup_fence_operand(tOrO);
        warpgroup_arrive();
        #pragma unroll
        for (int k = 0; k < size<2>(tOrP); ++k) {
            cute::gemm(tiled_mma_pv, tOrP(_,_,k), tCrVt(_,_,k), tOrO);
            tiled_mma_pv.accumulate_ = GMMA::ScaleOut::One;
        }
        warpgroup_commit_batch();

        int next_block = kv_block + 1;
        load_exact_scores(next_block);
        if (warp_id == 0) {
            int kv_start = next_block * BLOCK_N;
            int valid_n = min(BLOCK_N, ctx_len - kv_start);
            if (valid_n == BLOCK_N) {
                softmax_update_reg_deferred_o_rescale<
                    false, true, false, false>(
                    next_scores, row_max, row_sum, output_rescale,
                    inv_sqrt, valid_n, kv_start, ctx_len - 1,
                    DECODE_HEADS_PER_CTA);
            } else {
                softmax_update_reg_deferred_o_rescale<
                    false, true, false, true>(
                    next_scores, row_max, row_sum, output_rescale,
                    inv_sqrt, valid_n, kv_start, ctx_len - 1,
                    DECODE_HEADS_PER_CTA);
            }
        }

        // Re-form the complete warpgroup before retiring PV(i).  This is the
        // only outstanding WGMMA group, and it is always closed before either
        // operand is modified or the runtime loop backedge is taken.
        cutlass::arch::NamedBarrier::sync(PV_WG_THREADS, 0);
        warpgroup_wait<0>();
        warpgroup_fence_operand(tOrO);
        warpgroup_fence_operand(tOrP);

        // PV(i) has released V stage i; begin V(i+2) before the scalar
        // rescale/conversion work that prepares the next issue.
        if (tid == 0 && kv_block + 2 < num_kv_blocks) {
            issue_v(kv_block + 2);
        }
        if (tid < LIVE_OUTPUT_LANES) {
            float scale = output_rescale[0];
            #pragma unroll
            for (int nt = 0; nt < OUT_N_TILES; ++nt) {
                tOrO(nt * 4 + 0, 0, 0) *= scale;
                tOrO(nt * 4 + 1, 0, 0) *= scale;
            }
        }
        p_regs[0] = convert_p(next_scores);
    }

    // Peeled final PV: there is no next softmax tile.  It is issued only
    // after the last steady-state group was fully retired and is itself
    // drained before normalization or output-register reads.
    if (num_kv_blocks > 0) {
        int last_block = num_kv_blocks - 1;
        int stage = last_block & 1;
        int phase = (last_block >> 1) & 1;
        mbar_wait(&tma_mbar[stage], phase);

        BF16* v_stage = reinterpret_cast<BF16*>(
            smem + (stage == 0 ? V_STAGE0_OFF : V_STAGE1_OFF));
        Tensor sVt = make_tensor(make_smem_ptr(v_stage), SmemLayoutVt{});
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
        warpgroup_wait<0>();
        warpgroup_fence_operand(tOrO);
        warpgroup_fence_operand(tOrP);
    }

    if (warp_id == 0) {
        row_sum[0] = quad_sum(row_sum[0]);
    }
    int packed_row = tid >> 2;
    if (tid < LIVE_OUTPUT_LANES && packed_row < DECODE_HEADS_PER_CTA) {
        float final_sum = row_sum[0];
        float inv = final_sum > 0.0f ? 1.0f / final_sum : 0.0f;
        int col0 = (tid & 3) * 2;
        int col1 = col0 + 1;
        __nv_bfloat16* packed_o_ptr = o_ptr + packed_row * HEAD_DIM;
        #pragma unroll
        for (int nt = 0; nt < OUT_N_TILES; ++nt) {
            int n_base = nt * MMA_N;
            packed_o_ptr[n_base + col0] =
                __float2bfloat16(tOrO(nt * 4 + 0, 0, 0) * inv);
            packed_o_ptr[n_base + col1] =
                __float2bfloat16(tOrO(nt * 4 + 1, 0, 0) * inv);
        }
    }
}

// FUNCTION: unified_attn_decode_output_d64_score_consumer_kernel
// Iteration 55 splits each four-head long-decode consumer into two CTAs over
// disjoint 64-column output halves.  Both CTAs replay the identical producer
// scores and ascending online-softmax recurrence, so there is no cross-CTA
// reduction.  The only changed mathematical tile is PV's N dimension:
// [64,128] P x [128,64] V -> [64,64] O.  The Iteration-54 schedule is kept:
// PV(i) remains pending while warp 0 computes exact softmax(i+1), followed by
// a full-warpgroup rendezvous and wait_group 0 before O/P registers are read.
__global__ void unified_attn_decode_output_d64_score_consumer_kernel(
    const float* __restrict__ score_buffer,
    CUTLASS_GRID_CONSTANT TmaVt64 const tma_vt64,
    __nv_bfloat16* __restrict__ O,
    int ctx_len, int num_heads, int num_kv_heads,
    int total_kv_rows, int num_kv_blocks
) {
    extern __shared__ char smem[];
    constexpr int DECODE_HEADS_PER_CTA = 4;
    constexpr int PACKED_SCORE_LANES = 16;
    constexpr int LIVE_SCORE_COMPONENTS = 2;
    constexpr int SCORE_VALUES_PER_BLOCK =
        N_TILES * PACKED_SCORE_LANES * LIVE_SCORE_COMPONENTS;
    constexpr int PV_WG_THREADS = 4 * WARP_SIZE;
    constexpr int FRAGMENT_VALS = N_TILES * 4;
    constexpr int LIVE_OUTPUT_LANES = 16;
    constexpr int V_STAGE0_OFF = 0;
    constexpr int V_STAGE1_OFF = VT_D64_SMEM_BYTES;
    constexpr int TMA_MBAR_OFF = 2 * VT_D64_SMEM_BYTES;

    // x is [four-head group, output-d half], with d_half as the minor
    // coordinate.  For the configured 4:1 GQA case gridDim.x == 2, expanding
    // the long consumer grid from eight CTAs to sixteen CTAs.
    int split_group_in_kv = int(blockIdx.x);
    int d_half = split_group_in_kv & 1;
    int head_group_in_kv = split_group_in_kv >> 1;
    int head_groups_per_kv = int(gridDim.x) >> 1;
    int kv_head_idx = int(blockIdx.y);
    int batch_idx = int(blockIdx.z);
    int head_group_idx =
        kv_head_idx * head_groups_per_kv + head_group_in_kv;
    int head_idx = head_group_idx * DECODE_HEADS_PER_CTA;
    int d_base = d_half * DECODE_D_SPLIT;
    int tid = int(threadIdx.x);
    int warp_id = tid / WARP_SIZE;

    __nv_bfloat16* o_ptr = O +
        (batch_idx * num_heads + head_idx) * HEAD_DIM;
    int kv_row_base =
        (batch_idx * num_kv_heads + kv_head_idx) * ctx_len;
    int kv_linear = batch_idx * num_kv_heads + kv_head_idx;
    const float* kv_scores = score_buffer +
        static_cast<int64_t>(kv_linear) * num_kv_blocks *
        SCORE_VALUES_PER_BLOCK;
    uint64_t* tma_mbar =
        reinterpret_cast<uint64_t*>(smem + TMA_MBAR_OFF);

    if (tid == 0) {
        cute::prefetch_tma_descriptor(tma_vt64.get_tma_descriptor());
        mbar_init(&tma_mbar[0], 1);
        mbar_init(&tma_mbar[1], 1);
    }
    __syncthreads();

    auto issue_v = [&](int kv_block) {
        int stage = kv_block & 1;
        int kv_start = kv_block * BLOCK_N;
        BF16* v_stage = reinterpret_cast<BF16*>(
            smem + (stage == 0 ? V_STAGE0_OFF : V_STAGE1_OFF));
        constexpr int VT_BYTES =
            DECODE_D_SPLIT * BLOCK_N * sizeof(BF16);
        mbar_arrive_tx(&tma_mbar[stage], VT_BYTES);
        Tensor mVt = tma_vt64.get_tma_tensor(
            make_shape(Int<HEAD_DIM>{}, total_kv_rows));
        Tensor sVt =
            make_tensor(make_smem_ptr(v_stage), SmemLayoutVt64{});
        Tensor mVt_off = domain_offset(
            make_coord(d_base, kv_row_base + kv_start), mVt);
        Tensor gVt = local_tile(
            mVt_off,
            Shape<Int<DECODE_D_SPLIT>, Int<BLOCK_N>>{},
            make_coord(0, 0));
        auto [tVgVt, tVsVt] = tma_partition(
            tma_vt64, Int<0>{}, Layout<_1>{},
            group_modes<0,2>(sVt), group_modes<0,2>(gVt));
        copy(tma_vt64.with(tma_mbar[stage]), tVgVt, tVsVt);
    };

    if (tid == 0) {
        if (num_kv_blocks > 0) issue_v(0);
        if (num_kv_blocks > 1) issue_v(1);
    }

    TiledMmaPV64 tiled_mma_pv;
    ThrMMA thr_mma_pv = tiled_mma_pv.get_thread_slice(tid);
    auto tOrO = partition_fragment_C(
        tiled_mma_pv,
        Shape<Int<BLOCK_M_PREFILL>, Int<DECODE_D_SPLIT>>{});
    clear(tOrO);
    // The N=64 C fragment owns only 32 FP32 values/thread, but RS-P still
    // spans K=128 and therefore requires 64 BF16 values/thread (eight K16
    // slices).  Seed the A-register conversion from the proven N=128 C
    // layout; the RS A atom is identical for the N64 and N128 PV opcodes.
    // Deriving this layout from tOrO would silently expose only four K tiles.
    TiledMmaPV full_width_p_layout_mma;
    auto full_width_c_layout_seed = partition_fragment_C(
        full_width_p_layout_mma,
        Shape<Int<BLOCK_M_PREFILL>, Int<HEAD_DIM>>{});
    auto tOrP_layout = convert_layout_acc_Aregs<TiledMmaPV>(
        full_width_c_layout_seed.layout());
    auto tOrP = make_tensor<BF16>(tOrP_layout);
    auto p_regs =
        recast<cutlass::Array<BF16, FRAGMENT_VALS>>(tOrP);

    // Each d-half CTA owns an independent copy of the exact scalar softmax
    // state.  Scores and their operation order are identical between halves;
    // only the disjoint V/O columns differ.
    cutlass::Array<float, FRAGMENT_VALS> next_scores;
    #pragma unroll
    for (int frag = 0; frag < FRAGMENT_VALS; ++frag) {
        next_scores[frag] = 0.0f;
    }
    float row_max[2] = {-INFINITY, -INFINITY};
    float row_sum[2] = {0.0f, 0.0f};
    float output_rescale[2] = {1.0f, 1.0f};
    float inv_sqrt = 1.0f / sqrtf((float)HEAD_DIM);
    using PConverter =
        cutlass::NumericArrayConverter<BF16, float, FRAGMENT_VALS>;
    PConverter convert_p;

    auto load_exact_scores = [&](int kv_block) {
        if (warp_id == 0 && tid < PACKED_SCORE_LANES) {
            const float* block_scores = kv_scores +
                static_cast<int64_t>(kv_block) * SCORE_VALUES_PER_BLOCK;
            const float2* score_pairs =
                reinterpret_cast<const float2*>(block_scores);
            #pragma unroll
            for (int nt = 0; nt < N_TILES; ++nt) {
                float2 scores =
                    score_pairs[nt * PACKED_SCORE_LANES + tid];
                next_scores[nt * 4 + 0] = scores.x;
                next_scores[nt * 4 + 1] = scores.y;
            }
        }
    };

    // Prologue: prepare P(0) while both first V-half stages are in flight.
    if (num_kv_blocks > 0) {
        load_exact_scores(0);
        if (warp_id == 0) {
            int valid_n = min(BLOCK_N, ctx_len);
            if (valid_n == BLOCK_N) {
                softmax_update_reg_deferred_o_rescale<
                    false, true, true, false>(
                    next_scores, row_max, row_sum, output_rescale,
                    inv_sqrt, valid_n, 0, ctx_len - 1,
                    DECODE_HEADS_PER_CTA);
            } else {
                softmax_update_reg_deferred_o_rescale<
                    false, true, true, true>(
                    next_scores, row_max, row_sum, output_rescale,
                    inv_sqrt, valid_n, 0, ctx_len - 1,
                    DECODE_HEADS_PER_CTA);
            }
        }
        cutlass::arch::NamedBarrier::sync(PV_WG_THREADS, 0);
        p_regs[0] = convert_p(next_scores);
    }

    tiled_mma_pv.accumulate_ = GMMA::ScaleOut::Zero;
    for (int kv_block = 0; kv_block + 1 < num_kv_blocks; ++kv_block) {
        int stage = kv_block & 1;
        int phase = (kv_block >> 1) & 1;
        mbar_wait(&tma_mbar[stage], phase);

        BF16* v_stage = reinterpret_cast<BF16*>(
            smem + (stage == 0 ? V_STAGE0_OFF : V_STAGE1_OFF));
        Tensor sVt =
            make_tensor(make_smem_ptr(v_stage), SmemLayoutVt64{});
        Tensor tCsVt = thr_mma_pv.partition_B(sVt);
        auto tCrVt = thr_mma_pv.make_fragment_B(tCsVt);

        warpgroup_fence_operand(tOrP);
        warpgroup_fence_operand(tOrO);
        warpgroup_arrive();
        #pragma unroll
        for (int k = 0; k < size<2>(tOrP); ++k) {
            cute::gemm(
                tiled_mma_pv, tOrP(_,_,k), tCrVt(_,_,k), tOrO);
            tiled_mma_pv.accumulate_ = GMMA::ScaleOut::One;
        }
        warpgroup_commit_batch();

        int next_block = kv_block + 1;
        load_exact_scores(next_block);
        if (warp_id == 0) {
            int kv_start = next_block * BLOCK_N;
            int valid_n = min(BLOCK_N, ctx_len - kv_start);
            if (valid_n == BLOCK_N) {
                softmax_update_reg_deferred_o_rescale<
                    false, true, false, false>(
                    next_scores, row_max, row_sum, output_rescale,
                    inv_sqrt, valid_n, kv_start, ctx_len - 1,
                    DECODE_HEADS_PER_CTA);
            } else {
                softmax_update_reg_deferred_o_rescale<
                    false, true, false, true>(
                    next_scores, row_max, row_sum, output_rescale,
                    inv_sqrt, valid_n, kv_start, ctx_len - 1,
                    DECODE_HEADS_PER_CTA);
            }
        }

        cutlass::arch::NamedBarrier::sync(PV_WG_THREADS, 0);
        warpgroup_wait<0>();
        warpgroup_fence_operand(tOrO);
        warpgroup_fence_operand(tOrP);

        if (tid == 0 && kv_block + 2 < num_kv_blocks) {
            issue_v(kv_block + 2);
        }
        if (tid < LIVE_OUTPUT_LANES) {
            float scale = output_rescale[0];
            #pragma unroll
            for (int nt = 0; nt < OUT_N_TILES_D64; ++nt) {
                tOrO(nt * 4 + 0, 0, 0) *= scale;
                tOrO(nt * 4 + 1, 0, 0) *= scale;
            }
        }
        p_regs[0] = convert_p(next_scores);
    }

    // Peeled final N=64 PV, drained before normalization/output reads.
    if (num_kv_blocks > 0) {
        int last_block = num_kv_blocks - 1;
        int stage = last_block & 1;
        int phase = (last_block >> 1) & 1;
        mbar_wait(&tma_mbar[stage], phase);

        BF16* v_stage = reinterpret_cast<BF16*>(
            smem + (stage == 0 ? V_STAGE0_OFF : V_STAGE1_OFF));
        Tensor sVt =
            make_tensor(make_smem_ptr(v_stage), SmemLayoutVt64{});
        Tensor tCsVt = thr_mma_pv.partition_B(sVt);
        auto tCrVt = thr_mma_pv.make_fragment_B(tCsVt);
        warpgroup_fence_operand(tOrP);
        warpgroup_fence_operand(tOrO);
        warpgroup_arrive();
        #pragma unroll
        for (int k = 0; k < size<2>(tOrP); ++k) {
            cute::gemm(
                tiled_mma_pv, tOrP(_,_,k), tCrVt(_,_,k), tOrO);
            tiled_mma_pv.accumulate_ = GMMA::ScaleOut::One;
        }
        warpgroup_commit_batch();
        warpgroup_wait<0>();
        warpgroup_fence_operand(tOrO);
        warpgroup_fence_operand(tOrP);
    }

    if (warp_id == 0) {
        row_sum[0] = quad_sum(row_sum[0]);
    }
    int packed_row = tid >> 2;
    if (tid < LIVE_OUTPUT_LANES &&
        packed_row < DECODE_HEADS_PER_CTA) {
        float final_sum = row_sum[0];
        float inv = final_sum > 0.0f ? 1.0f / final_sum : 0.0f;
        int col0 = (tid & 3) * 2;
        int col1 = col0 + 1;
        __nv_bfloat16* packed_o_ptr =
            o_ptr + packed_row * HEAD_DIM + d_base;
        #pragma unroll
        for (int nt = 0; nt < OUT_N_TILES_D64; ++nt) {
            int n_base = nt * MMA_N;
            packed_o_ptr[n_base + col0] =
                __float2bfloat16(tOrO(nt * 4 + 0, 0, 0) * inv);
            packed_o_ptr[n_base + col1] =
                __float2bfloat16(tOrO(nt * 4 + 1, 0, 0) * inv);
        }
    }
}

// FUNCTION: unified_attn_decode_output_d32_score_consumer_kernel
// Iteration 105 reactivates Iteration 56's four-way D32 consumer only for the
// batch-16 long-decode dispatch.  The producer introduced in Iteration 61
// stores the exactly rounded, already-scaled FP32 score, so this dormant path
// must consume the workspace with a unit score scale.  Every CTA otherwise
// replays the original ascending online-softmax/PV order and writes one
// disjoint 32-column output quarter.  P deliberately keeps the proven full-
// N128-derived A-register layout: PV N changes accumulator width, but its
// K=128 operand still needs all eight ascending K16 slices (64 BF16 register
// values per thread).
// Iteration 150 consumes the producer's exact four-value block-max sideband,
// replacing four redundant per-sibling fmax/quad_max scans.  All masking,
// exponentiation, summation, and PV arithmetic remains in this kernel.
template <bool StageOwnedIssuers>
__global__ void unified_attn_decode_output_d32_score_consumer_kernel(
    const float* __restrict__ score_buffer,
    const float* __restrict__ block_max_buffer,
    CUTLASS_GRID_CONSTANT TmaVt32 const tma_vt32,
    __nv_bfloat16* __restrict__ O,
    int ctx_len, int num_heads, int num_kv_heads,
    int total_kv_rows, int num_kv_blocks
) {
    extern __shared__ char smem[];
    constexpr int DECODE_HEADS_PER_CTA = 4;
    constexpr int PACKED_SCORE_LANES = 16;
    constexpr int LIVE_SCORE_COMPONENTS = 2;
    constexpr int SCORE_VALUES_PER_BLOCK =
        N_TILES * PACKED_SCORE_LANES * LIVE_SCORE_COMPONENTS;
    constexpr int PV_WG_THREADS = 4 * WARP_SIZE;
    constexpr int FRAGMENT_VALS = N_TILES * 4;
    constexpr int LIVE_OUTPUT_LANES = 16;
    constexpr int V_STAGE0_OFF = 0;
    constexpr int V_STAGE1_OFF = VT_D32_SMEM_BYTES;
    constexpr int TMA_MBAR_OFF = 2 * VT_D32_SMEM_BYTES;

    // x is [four-head group, output-d quarter], with d_quarter as the minor
    // coordinate.  At the benchmark's 4:1 GQA ratio gridDim.x == 4: this is
    // 32 D32 CTAs per batch, or 512 CTAs for the Iteration-105 batch-16 path.
    int d_quarter = int(blockIdx.x);
    int kv_head_idx = int(blockIdx.y);
    int batch_idx = int(blockIdx.z);
    int head_idx = kv_head_idx * DECODE_HEADS_PER_CTA;
    int d_base = d_quarter * DECODE_D_QUARTER;
    int tid = int(threadIdx.x);
    int warp_id = tid / WARP_SIZE;

    __nv_bfloat16* o_ptr = O +
        (batch_idx * num_heads + head_idx) * HEAD_DIM;
    int kv_row_base =
        (batch_idx * num_kv_heads + kv_head_idx) * ctx_len;
    int kv_linear = batch_idx * num_kv_heads + kv_head_idx;
    const float* kv_scores = score_buffer +
        static_cast<int64_t>(kv_linear) * num_kv_blocks *
        SCORE_VALUES_PER_BLOCK;
    const float* kv_block_maxes = block_max_buffer +
        static_cast<int64_t>(kv_linear) * num_kv_blocks *
        DECODE_HEADS_PER_CTA;
    uint64_t* tma_mbar =
        reinterpret_cast<uint64_t*>(smem + TMA_MBAR_OFF);

    if constexpr (StageOwnedIssuers) {
        // Initialization follows the same stable ownership as prologue issue
        // and refill.  The retained CTA fence below orders both disjoint
        // mbarrier.init operations before any arrival or TMA transaction.
        if (tid == 0) {
            cute::prefetch_tma_descriptor(tma_vt32.get_tma_descriptor());
        }
        if (tid == 0 || tid == WARP_SIZE) {
            mbar_init(&tma_mbar[warp_id], 1);
        }
    } else if (tid == 0) {
        // Preserve the B8/B16 single-thread setup specialization verbatim.
        cute::prefetch_tma_descriptor(tma_vt32.get_tma_descriptor());
        mbar_init(&tma_mbar[0], 1);
        mbar_init(&tma_mbar[1], 1);
    }
    __syncthreads();

    auto issue_v = [&](int kv_block) {
        int stage = kv_block & 1;
        int kv_start = kv_block * BLOCK_N;
        BF16* v_stage = reinterpret_cast<BF16*>(
            smem + (stage == 0 ? V_STAGE0_OFF : V_STAGE1_OFF));
        constexpr int VT_BYTES =
            DECODE_D_QUARTER * BLOCK_N * sizeof(BF16);
        static_assert(TmaVt32::NumValSrc * sizeof(BF16) == VT_BYTES,
                      "D32 V direct issue requires one 8-KiB TMA box");
        // Iteration 110: this D32 tile is one native MN-SW64 tensor-map box.
        // Issue it directly into the original two-stage ring while retaining
        // the exact barrier payload, phase, coordinate, and lookahead.
        mbar_arrive_tx(&tma_mbar[stage], VT_BYTES);
        tma_load_2d(tma_vt32.get_tma_descriptor(),
                    reinterpret_cast<__nv_bfloat16*>(v_stage),
                    &tma_mbar[stage], d_base, kv_row_base + kv_start);
    };

    if constexpr (StageOwnedIssuers) {
        // Each barrier slot has one stable warp-leader owner.  The two
        // independent prologue transfers no longer serialize through warp 0;
        // completion order is immaterial because consumption waits per stage.
        if (tid == 0 && num_kv_blocks > 0) {
            issue_v(0);
        } else if (tid == WARP_SIZE && num_kv_blocks > 1) {
            issue_v(1);
        }
    } else if (tid == 0) {
        if (num_kv_blocks > 0) issue_v(0);
        if (num_kv_blocks > 1) issue_v(1);
    }

    TiledMmaPV32 tiled_mma_pv;
    ThrMMA thr_mma_pv = tiled_mma_pv.get_thread_slice(tid);
    auto tOrO = partition_fragment_C(
        tiled_mma_pv,
        Shape<Int<BLOCK_M_PREFILL>, Int<DECODE_D_QUARTER>>{});
    clear(tOrO);
    // N=32 owns 16 FP32 accumulator values/thread.  Do not derive RS-P from
    // that fragment: the A operand remains [64,128] and needs eight K16
    // slices.  The N128 seed is proven by the promoted D64 implementation.
    TiledMmaPV full_width_p_layout_mma;
    auto full_width_c_layout_seed = partition_fragment_C(
        full_width_p_layout_mma,
        Shape<Int<BLOCK_M_PREFILL>, Int<HEAD_DIM>>{});
    auto tOrP_layout = convert_layout_acc_Aregs<TiledMmaPV>(
        full_width_c_layout_seed.layout());
    auto tOrP = make_tensor<BF16>(tOrP_layout);
    auto p_regs =
        recast<cutlass::Array<BF16, FRAGMENT_VALS>>(tOrP);

    cutlass::Array<float, FRAGMENT_VALS> next_scores;
    #pragma unroll
    for (int frag = 0; frag < FRAGMENT_VALS; ++frag) {
        next_scores[frag] = 0.0f;
    }
    float row_max[2] = {-INFINITY, -INFINITY};
    float row_sum[2] = {0.0f, 0.0f};
    float output_rescale[2] = {1.0f, 1.0f};
    using PConverter =
        cutlass::NumericArrayConverter<BF16, float, FRAGMENT_VALS>;
    PConverter convert_p;

    auto load_exact_scores = [&](int kv_block) {
        if (warp_id == 0 && tid < PACKED_SCORE_LANES) {
            const float* block_scores = kv_scores +
                static_cast<int64_t>(kv_block) * SCORE_VALUES_PER_BLOCK;
            const float2* score_pairs =
                reinterpret_cast<const float2*>(block_scores);
            #pragma unroll
            for (int nt = 0; nt < N_TILES; ++nt) {
                float2 scores =
                    score_pairs[nt * PACKED_SCORE_LANES + tid];
                next_scores[nt * 4 + 0] = scores.x;
                next_scores[nt * 4 + 1] = scores.y;
            }
        }
    };

    // The four lanes in each live quad deliberately load the same value, just
    // as the proven D16 consumer does.  The producer stored this value after
    // the identical per-lane scan and xor-1/xor-2 quad tree.  Dead groups keep
    // -inf, matching the former helper's fully masked second half-warp.
    auto load_precomputed_block_max = [&](int kv_block) {
        float block_max = -INFINITY;
        if (warp_id == 0 && tid < PACKED_SCORE_LANES) {
            int packed_head = tid >> 2;
            block_max = kv_block_maxes[
                static_cast<int64_t>(kv_block) * DECODE_HEADS_PER_CTA +
                packed_head];
        }
        return block_max;
    };

    // Prologue: prepare P(0) while the first two V-quarter stages are in
    // flight.  All score and softmax operations intentionally match D64.
    if (num_kv_blocks > 0) {
        load_exact_scores(0);
        if (warp_id == 0) {
            int valid_n = min(BLOCK_N, ctx_len);
            float block_max = load_precomputed_block_max(0);
            if (valid_n == BLOCK_N) {
                softmax_update_reg_precomputed_block_max<true, false>(
                    next_scores, row_max, row_sum, output_rescale,
                    block_max, valid_n, DECODE_HEADS_PER_CTA);
            } else {
                softmax_update_reg_precomputed_block_max<true, true>(
                    next_scores, row_max, row_sum, output_rescale,
                    block_max, valid_n, DECODE_HEADS_PER_CTA);
            }
        }
        cutlass::arch::NamedBarrier::sync(PV_WG_THREADS, 0);
        p_regs[0] = convert_p(next_scores);
    }

    tiled_mma_pv.accumulate_ = GMMA::ScaleOut::Zero;
    for (int kv_block = 0; kv_block + 1 < num_kv_blocks; ++kv_block) {
        int stage = kv_block & 1;
        int phase = (kv_block >> 1) & 1;
        mbar_wait(&tma_mbar[stage], phase);

        BF16* v_stage = reinterpret_cast<BF16*>(
            smem + (stage == 0 ? V_STAGE0_OFF : V_STAGE1_OFF));
        Tensor sVt =
            make_tensor(make_smem_ptr(v_stage), SmemLayoutVt32{});
        Tensor tCsVt = thr_mma_pv.partition_B(sVt);
        auto tCrVt = thr_mma_pv.make_fragment_B(tCsVt);

        warpgroup_fence_operand(tOrP);
        warpgroup_fence_operand(tOrO);
        warpgroup_arrive();
        #pragma unroll
        for (int k = 0; k < size<2>(tOrP); ++k) {
            cute::gemm(
                tiled_mma_pv, tOrP(_,_,k), tCrVt(_,_,k), tOrO);
            tiled_mma_pv.accumulate_ = GMMA::ScaleOut::One;
        }
        warpgroup_commit_batch();

        int next_block = kv_block + 1;
        load_exact_scores(next_block);
        if (warp_id == 0) {
            int kv_start = next_block * BLOCK_N;
            int valid_n = min(BLOCK_N, ctx_len - kv_start);
            float block_max = load_precomputed_block_max(next_block);
            if (valid_n == BLOCK_N) {
                softmax_update_reg_precomputed_block_max<false, false>(
                    next_scores, row_max, row_sum, output_rescale,
                    block_max, valid_n, DECODE_HEADS_PER_CTA);
            } else {
                softmax_update_reg_precomputed_block_max<false, true>(
                    next_scores, row_max, row_sum, output_rescale,
                    block_max, valid_n, DECODE_HEADS_PER_CTA);
            }
        }

        cutlass::arch::NamedBarrier::sync(PV_WG_THREADS, 0);
        warpgroup_wait<0>();
        warpgroup_fence_operand(tOrO);
        warpgroup_fence_operand(tOrP);

        if constexpr (StageOwnedIssuers) {
            if (tid == stage * WARP_SIZE &&
                kv_block + 2 < num_kv_blocks) {
                issue_v(kv_block + 2);
            }
        } else if (tid == 0 && kv_block + 2 < num_kv_blocks) {
            issue_v(kv_block + 2);
        }
        if (tid < LIVE_OUTPUT_LANES) {
            float scale = output_rescale[0];
            #pragma unroll
            for (int nt = 0; nt < OUT_N_TILES_D32; ++nt) {
                tOrO(nt * 4 + 0, 0, 0) *= scale;
                tOrO(nt * 4 + 1, 0, 0) *= scale;
            }
        }
        p_regs[0] = convert_p(next_scores);
    }

    // Peeled final N=32 PV is fully drained before normalization/output reads.
    if (num_kv_blocks > 0) {
        int last_block = num_kv_blocks - 1;
        int stage = last_block & 1;
        int phase = (last_block >> 1) & 1;
        mbar_wait(&tma_mbar[stage], phase);

        BF16* v_stage = reinterpret_cast<BF16*>(
            smem + (stage == 0 ? V_STAGE0_OFF : V_STAGE1_OFF));
        Tensor sVt =
            make_tensor(make_smem_ptr(v_stage), SmemLayoutVt32{});
        Tensor tCsVt = thr_mma_pv.partition_B(sVt);
        auto tCrVt = thr_mma_pv.make_fragment_B(tCsVt);
        warpgroup_fence_operand(tOrP);
        warpgroup_fence_operand(tOrO);
        warpgroup_arrive();
        #pragma unroll
        for (int k = 0; k < size<2>(tOrP); ++k) {
            cute::gemm(
                tiled_mma_pv, tOrP(_,_,k), tCrVt(_,_,k), tOrO);
            tiled_mma_pv.accumulate_ = GMMA::ScaleOut::One;
        }
        warpgroup_commit_batch();
        warpgroup_wait<0>();
        warpgroup_fence_operand(tOrO);
        warpgroup_fence_operand(tOrP);
    }

    if (warp_id == 0) {
        row_sum[0] = quad_sum(row_sum[0]);
    }
    int packed_row = tid >> 2;
    if (tid < LIVE_OUTPUT_LANES &&
        packed_row < DECODE_HEADS_PER_CTA) {
        float final_sum = row_sum[0];
        float inv = final_sum > 0.0f ? 1.0f / final_sum : 0.0f;
        int col0 = (tid & 3) * 2;
        __nv_bfloat16* packed_o_ptr =
            o_ptr + packed_row * HEAD_DIM + d_base;
        #pragma unroll
        for (int nt = 0; nt < OUT_N_TILES_D32; ++nt) {
            int n_base = nt * MMA_N;
            // Each D32 quarter starts at a 64-byte boundary, n_base advances
            // by eight BF16 elements, and col0 is even.  Convert in the same
            // scalar order as the promoted kernel, then pack the adjacent
            // results into one naturally aligned 32-bit global transaction.
            __nv_bfloat16 out0 = __float2bfloat16(
                tOrO(nt * 4 + 0, 0, 0) * inv);
            __nv_bfloat16 out1 = __float2bfloat16(
                tOrO(nt * 4 + 1, 0, 0) * inv);
            uint32_t packed_out = __BFLOAT162_TO_CUI(
                __halves2bfloat162(out0, out1));
            *reinterpret_cast<uint32_t*>(
                packed_o_ptr + n_base + col0) = packed_out;
        }
    }
}

// FUNCTION: unified_attn_decode_output_d16_score_consumer_kernel
// Iteration 58 keeps Iteration 57's D16 split but assigns one packed GQA head
// to the natural first row of each WGMMA warp stripe (M=0/16/32/48).  All four
// warps now replay one exact online-softmax row concurrently; the old warp-0
// path replayed all four rows serially.  Each warp's local lanes 0..3 hold the
// live score/P/output fragments, while every lane still participates in the
// full-mask quad reductions and aligned RS-PV WGMMA.  Iteration 62 expands
// only the V TMA ring from two to four stages, increasing lookahead without
// changing score, softmax, PV, or output arithmetic order.
// Iteration 126 gives batch one's D16 path a separately compiled exact-N128
// symbol.  This hoists its per-block valid_n/tail dispatch while retaining
// every infinity guard, recurrence operation, L2 policy, and V-ring edge.
template <bool FullKvTiles>
__global__ void unified_attn_decode_output_d16_score_consumer_kernel(
    const float* __restrict__ score_buffer,
    const float* __restrict__ block_max_buffer,
    CUTLASS_GRID_CONSTANT TmaVt16 const tma_vt16,
    __nv_bfloat16* __restrict__ O,
    int ctx_len, int num_heads, int num_kv_heads,
    int total_kv_rows, int num_kv_blocks
) {
    extern __shared__ char smem[];
    constexpr int DECODE_HEADS_PER_CTA = 4;
    constexpr int PACKED_SCORE_LANES = 16;
    constexpr int LIVE_SCORE_COMPONENTS = 2;
    constexpr int SCORE_VALUES_PER_BLOCK =
        N_TILES * PACKED_SCORE_LANES * LIVE_SCORE_COMPONENTS;
    constexpr int PV_WG_THREADS = 4 * WARP_SIZE;
    constexpr int FRAGMENT_VALS = N_TILES * 4;
    constexpr int LIVE_OUTPUT_LANES = 16;
    constexpr int V_STAGE0_OFF = 0;
    constexpr int V_STAGE1_OFF = VT_D16_SMEM_BYTES;
    constexpr int V_STAGE2_OFF = 2 * VT_D16_SMEM_BYTES;
    constexpr int V_STAGE3_OFF = 3 * VT_D16_SMEM_BYTES;
    constexpr int V_STAGE_COUNT = 4;
    constexpr int TMA_MBAR_OFF = V_STAGE_COUNT * VT_D16_SMEM_BYTES;
    static_assert(V_STAGE0_OFF == 0 && V_STAGE1_OFF == 4096 &&
                  V_STAGE2_OFF == 8192 && V_STAGE3_OFF == 12288 &&
                  TMA_MBAR_OFF == 16384,
                  "D16 four-stage V ring requires exact 4-KiB spacing");

    // x is [four-head group, output-d eighth], with d_eighth as the minor
    // coordinate.  At the benchmark's 4:1 GQA ratio gridDim.x == 8, expanding
    // the eight-KV-head consumer wave from 32 D32 CTAs to 64 D16 CTAs.
    int d_eighth = int(blockIdx.x);
    int kv_head_idx = int(blockIdx.y);
    int batch_idx = int(blockIdx.z);
    int head_idx = kv_head_idx * DECODE_HEADS_PER_CTA;
    int d_base = d_eighth * DECODE_D_EIGHTH;
    int tid = int(threadIdx.x);
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    __nv_bfloat16* o_ptr = O +
        (batch_idx * num_heads + head_idx) * HEAD_DIM;
    int kv_row_base =
        (batch_idx * num_kv_heads + kv_head_idx) * ctx_len;
    int kv_linear = batch_idx * num_kv_heads + kv_head_idx;
    const float* kv_scores = score_buffer +
        static_cast<int64_t>(kv_linear) * num_kv_blocks *
        SCORE_VALUES_PER_BLOCK;

    // Advisory L1 warmup only: the exact score consumption below retains its
    // promoted L2::evict_last policy and instruction order.  One warp covers
    // the complete 2-KiB block as sixteen independent 128-byte stripes.
    auto prefetch_score_block_l1 = [&](int kv_block) {
        constexpr int SCORE_FLOATS_PER_L1_STRIPE = 128 / sizeof(float);
        static_assert(
            PACKED_SCORE_LANES * SCORE_FLOATS_PER_L1_STRIPE ==
                SCORE_VALUES_PER_BLOCK,
            "warp-0 score prefetch must cover one exact 2-KiB block");
        if (warp_id == 0 && lane_id < PACKED_SCORE_LANES) {
            const float* stripe_addr = kv_scores +
                static_cast<int64_t>(kv_block) * SCORE_VALUES_PER_BLOCK +
                lane_id * SCORE_FLOATS_PER_L1_STRIPE;
            asm volatile(
                "prefetch.global.L1 [%0];\n"
                :
                : "l"(stripe_addr));
        }
    };

    if (num_kv_blocks > 0) prefetch_score_block_l1(0);
    if (num_kv_blocks > 1) prefetch_score_block_l1(1);

    const float* kv_block_maxes = block_max_buffer +
        static_cast<int64_t>(kv_linear) * num_kv_blocks *
        DECODE_HEADS_PER_CTA;
    uint64_t* tma_mbar =
        reinterpret_cast<uint64_t*>(smem + TMA_MBAR_OFF);

    static_assert(V_STAGE_COUNT == NUM_WARPS,
                  "one D16 warp leader must own each V ring stage");
    if (tid == 0) {
        cute::prefetch_tma_descriptor(tma_vt16.get_tma_descriptor());
    }
    if (lane_id == 0) {
        // The retained CTA fence below orders all four disjoint initializers
        // before these same stage owners issue the ring prologue.
        mbar_init(&tma_mbar[warp_id], 1);
    }
    __syncthreads();

    auto issue_v = [&](int kv_block) {
        int stage = kv_block & 3;
        int kv_start = kv_block * BLOCK_N;
        BF16* v_stage = reinterpret_cast<BF16*>(
            smem + stage * VT_D16_SMEM_BYTES);
        constexpr int VT_BYTES =
            DECODE_D_EIGHTH * BLOCK_N * sizeof(BF16);
        static_assert(TmaVt16::NumValSrc * sizeof(BF16) == VT_BYTES,
                      "D16 V direct issue requires one 4-KiB TMA box");
        // The D16 tile is already one native MN-SW32 tensor-map box.  Issue it
        // directly into the selected four-stage ring slot while retaining the
        // original barrier payload, stage phase, coordinate, and lookahead.
        mbar_arrive_tx(&tma_mbar[stage], VT_BYTES);
        tma_load_2d(tma_vt16.get_tma_descriptor(),
                    reinterpret_cast<__nv_bfloat16*>(v_stage),
                    &tma_mbar[stage], d_base, kv_row_base + kv_start);
    };

    // The four stage owners issue the ring prologue independently.  Each
    // writes a disjoint 4-KiB destination and arrives on a disjoint mbarrier.
    if (lane_id == 0 && warp_id < num_kv_blocks) {
        issue_v(warp_id);
    }

    TiledMmaPV16 tiled_mma_pv;
    ThrMMA thr_mma_pv = tiled_mma_pv.get_thread_slice(tid);
    auto tOrO = partition_fragment_C(
        tiled_mma_pv,
        Shape<Int<BLOCK_M_PREFILL>, Int<DECODE_D_EIGHTH>>{});
    clear(tOrO);
    // N=16 owns 8 FP32 accumulator values/thread.  Do not derive RS-P from
    // that fragment: the A operand remains [64,128] and needs eight K16
    // slices.  The N128 seed is proven by the promoted D64 implementation.
    TiledMmaPV full_width_p_layout_mma;
    auto full_width_c_layout_seed = partition_fragment_C(
        full_width_p_layout_mma,
        Shape<Int<BLOCK_M_PREFILL>, Int<HEAD_DIM>>{});
    auto tOrP_layout = convert_layout_acc_Aregs<TiledMmaPV>(
        full_width_c_layout_seed.layout());
    auto tOrP = make_tensor<BF16>(tOrP_layout);
    auto p_regs =
        recast<cutlass::Array<BF16, FRAGMENT_VALS>>(tOrP);

    cutlass::Array<float, FRAGMENT_VALS> next_scores;
    #pragma unroll
    for (int frag = 0; frag < FRAGMENT_VALS; ++frag) {
        next_scores[frag] = 0.0f;
    }
    float row_max = -INFINITY;
    float row_sum = 0.0f;
    float output_rescale = 1.0f;
    using PConverter =
        cutlass::NumericArrayConverter<BF16, float, FRAGMENT_VALS>;
    PConverter convert_p;
    uint64_t workspace_l2_policy;
    float workspace_l2_fraction = 1.0f;
    asm volatile(
        "createpolicy.fractional.L2::evict_last.b64 %0, %1;\n"
        : "=l"(workspace_l2_policy)
        : "f"(workspace_l2_fraction)
        : "memory");

    auto load_exact_scores = [&](int kv_block) {
        if (lane_id < DECODE_HEADS_PER_CTA) {
            const float* block_scores = kv_scores +
                static_cast<int64_t>(kv_block) * SCORE_VALUES_PER_BLOCK;
            const float2* score_pairs =
                reinterpret_cast<const float2*>(block_scores);
            int compact_lane = warp_id * DECODE_HEADS_PER_CTA + lane_id;
            #pragma unroll
            for (int nt = 0; nt < N_TILES; ++nt) {
                union {
                    float2 value;
                    uint64_t bits;
                } score_word;
                const float2* score_addr =
                    &score_pairs[
                        nt * PACKED_SCORE_LANES + compact_lane];
                asm volatile(
                    "ld.global.L2::cache_hint.b64 %0, [%1], %2;\n"
                    : "=l"(score_word.bits)
                    : "l"(score_addr), "l"(workspace_l2_policy)
                    : "memory");
                float2 scores = score_word.value;
                next_scores[nt * 4 + 0] = scores.x;
                next_scores[nt * 4 + 1] = scores.y;
            }
        }
    };

    // Exact p0/p1 subset of softmax_update_reg_deferred_o_rescale.  Moving
    // each head to a separate warp makes group 0 the sole live row in every
    // warp.  The dependency order for scaling, max, exponentials, sums, and
    // deferred O rescale is identical to the promoted four-row helper; only
    // the permanently masked p2/p3 row and its state are removed.
    auto update_one_live_row = [&](bool is_first, bool check_tail,
                                   int valid_n, int kv_block) {
        int col0 = lane_id * 2;
        int col1 = col0 + 1;
        bool live_lane = lane_id < DECODE_HEADS_PER_CTA;

        // Producer used the identical ascending pair-fmax scan and quad_max
        // tree once for all eight D slices.  Preserve the former lane-group
        // state: only local lanes 0..3 receive the packed head maximum.
        // Iteration 65 starts this unchanged global load before the independent
        // mask/materialization pass so the load latency can overlap that pass.
        float block_max = -INFINITY;
        if (live_lane) {
            const float* block_max_addr =
                &kv_block_maxes[
                    static_cast<int64_t>(kv_block) * DECODE_HEADS_PER_CTA +
                    warp_id];
            uint32_t block_max_bits;
            asm volatile(
                "ld.global.L2::cache_hint.b32 %0, [%1], %2;\n"
                : "=r"(block_max_bits)
                : "l"(block_max_addr), "l"(workspace_l2_policy)
                : "memory");
            block_max = __uint_as_float(block_max_bits);
        }

        #pragma unroll
        for (int nt = 0; nt < N_TILES; ++nt) {
            int n_base = nt * MMA_N;
            int n0 = n_base + col0;
            int n1 = n_base + col1;
            // Producer stored the identically rounded FP32 scaled score.
            float s0 = next_scores[nt * 4 + 0];
            float s1 = next_scores[nt * 4 + 1];
            bool mask0 = !live_lane;
            bool mask1 = !live_lane;
            if constexpr (!FullKvTiles) {
                mask0 = mask0 || (check_tail && n0 >= valid_n);
                mask1 = mask1 || (check_tail && n1 >= valid_n);
            }
            next_scores[nt * 4 + 0] = mask0 ? -INFINITY : s0;
            next_scores[nt * 4 + 1] = mask1 ? -INFINITY : s1;
        }

        float new_max = fmaxf(row_max, block_max);
        float rescale = 1.0f;
        if (!is_first) {
            rescale = isinf(new_max)
                ? 1.0f
                : fast_exp2_ftz((row_max - new_max) * LOG2E);
        }
        row_max = new_max;
        output_rescale = rescale;
        if (!is_first) {
            row_sum *= rescale;
        }

        float block_sum = 0.0f;
        #pragma unroll
        for (int nt = 0; nt < N_TILES; ++nt) {
            float p0 = isinf(next_scores[nt * 4 + 0])
                ? 0.0f
                : fast_exp2_ftz(
                    (next_scores[nt * 4 + 0] - new_max) * LOG2E);
            float p1 = isinf(next_scores[nt * 4 + 1])
                ? 0.0f
                : fast_exp2_ftz(
                    (next_scores[nt * 4 + 1] - new_max) * LOG2E);
            next_scores[nt * 4 + 0] = p0;
            next_scores[nt * 4 + 1] = p1;
            block_sum += p0 + p1;
        }
        row_sum += block_sum;
    };

    // Prologue: prepare P(0) while the first two V-eighth stages are in
    // flight.  All score and softmax operations intentionally match D64.
    if (num_kv_blocks > 0) {
        load_exact_scores(0);
        int valid_n = BLOCK_N;
        bool check_tail = false;
        if constexpr (!FullKvTiles) {
            valid_n = min(BLOCK_N, ctx_len);
            check_tail = valid_n != BLOCK_N;
        }
        update_one_live_row(true, check_tail, valid_n, 0);
        cutlass::arch::NamedBarrier::sync(PV_WG_THREADS, 0);
        p_regs[0] = convert_p(next_scores);
    }

    tiled_mma_pv.accumulate_ = GMMA::ScaleOut::Zero;
    for (int kv_block = 0; kv_block + 1 < num_kv_blocks; ++kv_block) {
        int stage = kv_block & 3;
        int phase = (kv_block >> 2) & 1;
        if (kv_block + 2 < num_kv_blocks) {
            prefetch_score_block_l1(kv_block + 2);
        }
        mbar_wait(&tma_mbar[stage], phase);

        BF16* v_stage = reinterpret_cast<BF16*>(
            smem + stage * VT_D16_SMEM_BYTES);
        Tensor sVt =
            make_tensor(make_smem_ptr(v_stage), SmemLayoutVt16{});
        Tensor tCsVt = thr_mma_pv.partition_B(sVt);
        auto tCrVt = thr_mma_pv.make_fragment_B(tCsVt);

        warpgroup_fence_operand(tOrP);
        warpgroup_fence_operand(tOrO);
        warpgroup_arrive();
        #pragma unroll
        for (int k = 0; k < size<2>(tOrP); ++k) {
            cute::gemm(
                tiled_mma_pv, tOrP(_,_,k), tCrVt(_,_,k), tOrO);
            tiled_mma_pv.accumulate_ = GMMA::ScaleOut::One;
        }
        warpgroup_commit_batch();

        int next_block = kv_block + 1;
        load_exact_scores(next_block);
        int next_kv_start = next_block * BLOCK_N;
        int valid_n = BLOCK_N;
        bool check_tail = false;
        if constexpr (!FullKvTiles) {
            valid_n = min(BLOCK_N, ctx_len - next_kv_start);
            check_tail = valid_n != BLOCK_N;
        }
        update_one_live_row(
            false, check_tail, valid_n, next_block);

        cutlass::arch::NamedBarrier::sync(PV_WG_THREADS, 0);
        warpgroup_wait<0>();
        warpgroup_fence_operand(tOrO);
        warpgroup_fence_operand(tOrP);

        // Re-arm this stage only after the WGMMA group consuming it has been
        // fully retired and both register operands have been fenced.
        if (tid == stage * WARP_SIZE &&
            kv_block + 4 < num_kv_blocks) {
            issue_v(kv_block + 4);
        }
        if (lane_id < DECODE_HEADS_PER_CTA) {
            float scale = output_rescale;
            #pragma unroll
            for (int nt = 0; nt < OUT_N_TILES_D16; ++nt) {
                tOrO(nt * 4 + 0, 0, 0) *= scale;
                tOrO(nt * 4 + 1, 0, 0) *= scale;
            }
        }
        p_regs[0] = convert_p(next_scores);
    }

    // Peeled final N=16 PV is fully drained before normalization/output reads.
    if (num_kv_blocks > 0) {
        int last_block = num_kv_blocks - 1;
        int stage = last_block & 3;
        int phase = (last_block >> 2) & 1;
        mbar_wait(&tma_mbar[stage], phase);

        BF16* v_stage = reinterpret_cast<BF16*>(
            smem + stage * VT_D16_SMEM_BYTES);
        Tensor sVt =
            make_tensor(make_smem_ptr(v_stage), SmemLayoutVt16{});
        Tensor tCsVt = thr_mma_pv.partition_B(sVt);
        auto tCrVt = thr_mma_pv.make_fragment_B(tCsVt);
        warpgroup_fence_operand(tOrP);
        warpgroup_fence_operand(tOrO);
        warpgroup_arrive();
        #pragma unroll
        for (int k = 0; k < size<2>(tOrP); ++k) {
            cute::gemm(
                tiled_mma_pv, tOrP(_,_,k), tCrVt(_,_,k), tOrO);
            tiled_mma_pv.accumulate_ = GMMA::ScaleOut::One;
        }
        warpgroup_commit_batch();
        warpgroup_wait<0>();
        warpgroup_fence_operand(tOrO);
        warpgroup_fence_operand(tOrP);
    }

    // All 32 lanes in every warp execute the full-mask reduction.  Only the
    // local lane-0..3 quad is live, and each warp writes its own packed head.
    row_sum = quad_sum(row_sum);
    if (lane_id < DECODE_HEADS_PER_CTA) {
        float final_sum = row_sum;
        float inv = final_sum > 0.0f ? 1.0f / final_sum : 0.0f;
        int col0 = lane_id * 2;
        __nv_bfloat16* packed_o_ptr =
            o_ptr + warp_id * HEAD_DIM + d_base;
        #pragma unroll
        for (int nt = 0; nt < OUT_N_TILES_D16; ++nt) {
            int n_base = nt * MMA_N;
            // d_base is a multiple of 16 BF16 elements, n_base is a multiple
            // of eight, and col0 is even.  Keep both scalar conversion inputs
            // and their order unchanged while replacing their adjacent stores
            // with one aligned 32-bit transaction at the old col0 address.
            __nv_bfloat16 out0 = __float2bfloat16(
                tOrO(nt * 4 + 0, 0, 0) * inv);
            __nv_bfloat16 out1 = __float2bfloat16(
                tOrO(nt * 4 + 1, 0, 0) * inv);
            uint32_t packed_out = __BFLOAT162_TO_CUI(
                __halves2bfloat162(out0, out1));
            *reinterpret_cast<uint32_t*>(
                packed_o_ptr + n_base + col0) = packed_out;
        }
    }
}

// FUNCTION: unified_attn_decode_staged_p_actor_kernel
// QK warp-group (0..127), PV warp-group (128..255), and warp-8 softmax
// communicate only through two-stage shared score/P rings.  Each WGMMA actor
// drains its own group before publishing or rearming an operand stage.
__global__ void unified_attn_decode_staged_p_actor_kernel(
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
    int kv_head_idx = int(blockIdx.y);
    int batch_idx = int(blockIdx.z);
    int head_pair_idx = kv_head_idx * int(gridDim.x) + head_pair_in_kv;
    int head_idx = head_pair_idx * DECODE_HEADS_PER_CTA;
    const __nv_bfloat16* q_ptr = Q +
        (batch_idx * num_heads + head_idx) * HEAD_DIM;
    __nv_bfloat16* o_ptr = O +
        (batch_idx * num_heads + head_idx) * HEAD_DIM;

    int tid = int(threadIdx.x);
    int lane_id = tid % WARP_SIZE;
    BF16* q_smem = reinterpret_cast<BF16*>(smem + DECODE_Q_OFF);
    uint64_t* tma_mbar = reinterpret_cast<uint64_t*>(smem + DECODE_MBAR_OFF);
    float* score_ring = reinterpret_cast<float*>(
        smem + DECODE_ACTOR_SCORE_OFF);
    BF16* p_ring = reinterpret_cast<BF16*>(smem + DECODE_ACTOR_P_OFF);
    float* scale_ring = reinterpret_cast<float*>(
        smem + DECODE_ACTOR_SCALE_OFF);
    float* norm_smem = reinterpret_cast<float*>(smem + DECODE_ACTOR_NORM_OFF);
    uint64_t* actor_mbar = reinterpret_cast<uint64_t*>(
        smem + DECODE_ACTOR_MBAR_OFF);
    int kv_row_base = (batch_idx * num_kv_heads + kv_head_idx) * ctx_len;
    int num_kv_blocks = (ctx_len + BLOCK_N - 1) / BLOCK_N;

    if (tid == 0) {
        cute::prefetch_tma_descriptor(tma_k.get_tma_descriptor());
        mbar_init(&tma_mbar[0], 1);
        mbar_init(&tma_mbar[1], 1);
        #pragma unroll
        for (int i = 0; i < DECODE_ACTOR_MBAR_COUNT; ++i) {
            mbar_init(&actor_mbar[i], 1);
        }
    } else if (tid == DECODE_ACTOR_WG_THREADS) {
        cute::prefetch_tma_descriptor(tma_vt.get_tma_descriptor());
        mbar_init(&tma_mbar[2], 1);
        mbar_init(&tma_mbar[3], 1);
    }
    {
        Tensor sQ = make_tensor(make_smem_ptr(q_smem), SmemLayoutQ{});
        constexpr int PACKED_Q_ELEMS = DECODE_HEADS_PER_CTA * HEAD_DIM;
        for (int idx = tid; idx < PACKED_Q_ELEMS;
             idx += DECODE_ACTOR_THREADS) {
            int row = idx / HEAD_DIM;
            int k = idx % HEAD_DIM;
            sQ(row, k) = BF16(q_ptr[row * HEAD_DIM + k]);
        }
    }
    __syncthreads();

    if (tid < DECODE_ACTOR_WG_THREADS) {
        int local_tid = tid;
        TiledMmaQK tiled_mma_qk;
        ThrMMA thr_mma_qk = tiled_mma_qk.get_thread_slice(local_tid);
        Tensor sQ = make_tensor(make_smem_ptr(q_smem), SmemLayoutQ{});
        Tensor tCsQ = thr_mma_qk.partition_A(sQ);
        auto tCrQ = thr_mma_qk.make_fragment_A(tCsQ);
        auto tSrS = partition_fragment_C(
            tiled_mma_qk, Shape<Int<BLOCK_M_PREFILL>, Int<BLOCK_N>>{});

        auto issue_k = [&](int kv_block) {
            int stage = kv_block & 1;
            int kv_start = kv_block * BLOCK_N;
            BF16* k_stage = reinterpret_cast<BF16*>(
                smem + (stage == 0 ? DECODE_K_STAGE0_OFF
                                   : DECODE_K_STAGE1_OFF));
            int k_bytes = BLOCK_N * HEAD_DIM * sizeof(BF16);
            mbar_arrive_tx(&tma_mbar[stage], k_bytes);
            Tensor mK = tma_k.get_tma_tensor(
                make_shape(total_kv_rows, Int<HEAD_DIM>{}));
            Tensor sK = make_tensor(make_smem_ptr(k_stage), SmemLayoutKW{});
            Tensor mK_off = domain_offset(
                make_coord(kv_row_base + kv_start, 0), mK);
            Tensor gK = local_tile(
                mK_off, Shape<Int<BLOCK_N>, Int<HEAD_DIM>>{},
                make_coord(0, 0));
            auto [tKgK, tKsK] = tma_partition(
                tma_k, Int<0>{}, Layout<_1>{},
                group_modes<0,2>(sK), group_modes<0,2>(gK));
            copy(tma_k.with(tma_mbar[stage]), tKgK, tKsK);
        };

        if (local_tid == 0) {
            if (num_kv_blocks > 0) issue_k(0);
            if (num_kv_blocks > 1) issue_k(1);
        }

        for (int kv_block = 0; kv_block < num_kv_blocks; ++kv_block) {
            int stage = kv_block & 1;
            int phase = (kv_block >> 1) & 1;
            if (local_tid == 0 && kv_block >= 2) {
                int empty_phase = ((kv_block - 2) >> 1) & 1;
                mbar_wait(&actor_mbar[SCORE_EMPTY_MBAR + stage], empty_phase);
            }
            cutlass::arch::NamedBarrier::sync(
                DECODE_ACTOR_WG_THREADS, 0);
            mbar_wait(&tma_mbar[stage], phase);

            BF16* k_stage = reinterpret_cast<BF16*>(
                smem + (stage == 0 ? DECODE_K_STAGE0_OFF
                                   : DECODE_K_STAGE1_OFF));
            Tensor sK = make_tensor(make_smem_ptr(k_stage), SmemLayoutKW{});
            Tensor tCsK = thr_mma_qk.partition_B(sK);
            auto tCrK = thr_mma_qk.make_fragment_B(tCsK);
            clear(tSrS);
            warpgroup_fence_operand(tSrS);
            warpgroup_arrive();
            gemm(tiled_mma_qk, tCrQ, tCrK, tSrS);
            warpgroup_commit_batch();
            warpgroup_wait<0>();
            warpgroup_fence_operand(tSrS);

            // wait<0> releases this K buffer; rearm it two blocks ahead.
            if (local_tid == 0 && kv_block + 2 < num_kv_blocks) {
                issue_k(kv_block + 2);
            }
            if (local_tid < DECODE_ACTOR_SHARED_LANES) {
                float* stage_scores = score_ring +
                    stage * DECODE_ACTOR_SHARED_LANES *
                            DECODE_ACTOR_FRAGMENT_VALS;
                #pragma unroll
                for (int frag = 0; frag < DECODE_ACTOR_FRAGMENT_VALS; ++frag) {
                    stage_scores[frag * DECODE_ACTOR_SHARED_LANES + local_tid] =
                        tSrS(frag, 0, 0);
                }
            }
            cutlass::arch::NamedBarrier::sync(
                DECODE_ACTOR_WG_THREADS, 0);
            if (local_tid == 0) {
                cutlass::arch::ClusterBarrier::arrive(
                    &actor_mbar[SCORE_FULL_MBAR + stage]);
            }
        }
    } else if (tid < 2 * DECODE_ACTOR_WG_THREADS) {
        int local_tid = tid - DECODE_ACTOR_WG_THREADS;
        TiledMmaPV tiled_mma_pv;
        ThrMMA thr_mma_pv = tiled_mma_pv.get_thread_slice(local_tid);
        auto tOrO = partition_fragment_C(
            tiled_mma_pv, Shape<Int<BLOCK_M_PREFILL>, Int<HEAD_DIM>>{});
        clear(tOrO);
        auto tOrP_layout =
            convert_layout_acc_Aregs<TiledMmaPV>(tOrO.layout());
        auto tOrP = make_tensor<BF16>(tOrP_layout);
        auto p_regs = recast<cutlass::Array<
            BF16, DECODE_ACTOR_FRAGMENT_VALS>>(tOrP);
        tiled_mma_pv.accumulate_ = GMMA::ScaleOut::Zero;

        auto issue_v = [&](int kv_block) {
            int stage = kv_block & 1;
            int kv_start = kv_block * BLOCK_N;
            BF16* v_stage = reinterpret_cast<BF16*>(
                smem + (stage == 0 ? DECODE_V_STAGE0_OFF
                                   : DECODE_V_STAGE1_OFF));
            int vt_bytes = HEAD_DIM * BLOCK_N * sizeof(BF16);
            mbar_arrive_tx(&tma_mbar[2 + stage], vt_bytes);
            Tensor mVt = tma_vt.get_tma_tensor(
                make_shape(Int<HEAD_DIM>{}, total_kv_rows));
            Tensor sVt = make_tensor(make_smem_ptr(v_stage), SmemLayoutVt{});
            Tensor mVt_off = domain_offset(
                make_coord(0, kv_row_base + kv_start), mVt);
            Tensor gVt = local_tile(
                mVt_off, Shape<Int<HEAD_DIM>, Int<BLOCK_N>>{},
                make_coord(0, 0));
            auto [tVgVt, tVsVt] = tma_partition(
                tma_vt, Int<0>{}, Layout<_1>{},
                group_modes<0,2>(sVt), group_modes<0,2>(gVt));
            copy(tma_vt.with(tma_mbar[2 + stage]), tVgVt, tVsVt);
        };

        if (local_tid == 0) {
            if (num_kv_blocks > 0) issue_v(0);
            if (num_kv_blocks > 1) issue_v(1);
        }

        for (int kv_block = 0; kv_block < num_kv_blocks; ++kv_block) {
            int stage = kv_block & 1;
            int phase = (kv_block >> 1) & 1;
            if (local_tid == 0) {
                mbar_wait(&actor_mbar[P_FULL_MBAR + stage], phase);
            }
            cutlass::arch::NamedBarrier::sync(
                DECODE_ACTOR_WG_THREADS, 1);

            float scale0 = 1.0f, scale1 = 1.0f;
            if (local_tid < DECODE_ACTOR_SHARED_LANES) {
                BF16* stage_p = p_ring +
                    stage * DECODE_ACTOR_SHARED_LANES *
                            DECODE_ACTOR_FRAGMENT_VALS;
                float* stage_scale = scale_ring +
                    stage * DECODE_ACTOR_SHARED_LANES * 2;
                #pragma unroll
                for (int frag = 0; frag < DECODE_ACTOR_FRAGMENT_VALS; ++frag) {
                    p_regs[0][frag] =
                        stage_p[frag * DECODE_ACTOR_SHARED_LANES + local_tid];
                }
                scale0 = stage_scale[0 * DECODE_ACTOR_SHARED_LANES + local_tid];
                scale1 = stage_scale[1 * DECODE_ACTOR_SHARED_LANES + local_tid];
            } else {
                #pragma unroll
                for (int frag = 0; frag < DECODE_ACTOR_FRAGMENT_VALS; ++frag) {
                    p_regs[0][frag] = BF16(0.0f);
                }
            }
            cutlass::arch::NamedBarrier::sync(
                DECODE_ACTOR_WG_THREADS, 1);
            if (local_tid == 0) {
                cutlass::arch::ClusterBarrier::arrive(
                    &actor_mbar[P_EMPTY_MBAR + stage]);
            }

            if (local_tid < DECODE_ACTOR_SHARED_LANES) {
                #pragma unroll
                for (int nt = 0; nt < OUT_N_TILES; ++nt) {
                    tOrO(nt * 4 + 0, 0, 0) *= scale0;
                    tOrO(nt * 4 + 1, 0, 0) *= scale0;
                    tOrO(nt * 4 + 2, 0, 0) *= scale1;
                    tOrO(nt * 4 + 3, 0, 0) *= scale1;
                }
            }
            mbar_wait(&tma_mbar[2 + stage], phase);
            BF16* v_stage = reinterpret_cast<BF16*>(
                smem + (stage == 0 ? DECODE_V_STAGE0_OFF
                                   : DECODE_V_STAGE1_OFF));
            Tensor sVt = make_tensor(make_smem_ptr(v_stage), SmemLayoutVt{});
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
            warpgroup_wait<0>();
            warpgroup_fence_operand(tOrO);
            warpgroup_fence_operand(tOrP);

            if (local_tid == 0 && kv_block + 2 < num_kv_blocks) {
                issue_v(kv_block + 2);
            }
        }

        if (local_tid == 0) {
            mbar_wait(&actor_mbar[NORM_READY_MBAR], 0);
        }
        cutlass::arch::NamedBarrier::sync(DECODE_ACTOR_WG_THREADS, 1);
        int packed_row = local_tid >> 2;
        if (local_tid < DECODE_ACTOR_SHARED_LANES &&
            packed_row < DECODE_HEADS_PER_CTA) {
            float final_sum = norm_smem[local_tid];
            float inv = final_sum > 0.0f ? 1.0f / final_sum : 0.0f;
            int col0 = (local_tid & 3) * 2;
            int col1 = col0 + 1;
            __nv_bfloat16* packed_o_ptr = o_ptr + packed_row * HEAD_DIM;
            #pragma unroll
            for (int nt = 0; nt < OUT_N_TILES; ++nt) {
                int n_base = nt * MMA_N;
                packed_o_ptr[n_base + col0] =
                    __float2bfloat16(tOrO(nt * 4 + 0, 0, 0) * inv);
                packed_o_ptr[n_base + col1] =
                    __float2bfloat16(tOrO(nt * 4 + 1, 0, 0) * inv);
            }
        }
    } else {
        float row_max[2] = {-INFINITY, -INFINITY};
        float row_sum[2] = {0.0f, 0.0f};
        float output_rescale[2] = {1.0f, 1.0f};
        cutlass::Array<float, DECODE_ACTOR_FRAGMENT_VALS> acc_s;
        float inv_sqrt = 1.0f / sqrtf((float)HEAD_DIM);
        using PConverter = cutlass::NumericArrayConverter<
            BF16, float, DECODE_ACTOR_FRAGMENT_VALS>;
        PConverter convert_p;

        for (int kv_block = 0; kv_block < num_kv_blocks; ++kv_block) {
            int stage = kv_block & 1;
            int phase = (kv_block >> 1) & 1;
            mbar_wait(&actor_mbar[SCORE_FULL_MBAR + stage], phase);
            float* stage_scores = score_ring +
                stage * DECODE_ACTOR_SHARED_LANES *
                        DECODE_ACTOR_FRAGMENT_VALS;
            #pragma unroll
            for (int frag = 0; frag < DECODE_ACTOR_FRAGMENT_VALS; ++frag) {
                acc_s[frag] =
                    stage_scores[frag * DECODE_ACTOR_SHARED_LANES + lane_id];
            }
            __syncwarp();
            if (lane_id == 0) {
                cutlass::arch::ClusterBarrier::arrive(
                    &actor_mbar[SCORE_EMPTY_MBAR + stage]);
            }

            int kv_start = kv_block * BLOCK_N;
            int valid_n = min(BLOCK_N, ctx_len - kv_start);
            if (kv_block == 0) {
                if (valid_n == BLOCK_N) {
                    softmax_update_reg_deferred_o_rescale<
                        false, true, true, false>(
                        acc_s, row_max, row_sum, output_rescale,
                        inv_sqrt, valid_n, kv_start, ctx_len - 1,
                        DECODE_HEADS_PER_CTA);
                } else {
                    softmax_update_reg_deferred_o_rescale<
                        false, true, true, true>(
                        acc_s, row_max, row_sum, output_rescale,
                        inv_sqrt, valid_n, kv_start, ctx_len - 1,
                        DECODE_HEADS_PER_CTA);
                }
            } else if (valid_n == BLOCK_N) {
                softmax_update_reg_deferred_o_rescale<
                    false, true, false, false>(
                    acc_s, row_max, row_sum, output_rescale,
                    inv_sqrt, valid_n, kv_start, ctx_len - 1,
                    DECODE_HEADS_PER_CTA);
            } else {
                softmax_update_reg_deferred_o_rescale<
                    false, true, false, true>(
                    acc_s, row_max, row_sum, output_rescale,
                    inv_sqrt, valid_n, kv_start, ctx_len - 1,
                    DECODE_HEADS_PER_CTA);
            }

            if (lane_id == 0 && kv_block >= 2) {
                int empty_phase = ((kv_block - 2) >> 1) & 1;
                mbar_wait(&actor_mbar[P_EMPTY_MBAR + stage], empty_phase);
            }
            __syncwarp();
            cutlass::Array<BF16, DECODE_ACTOR_FRAGMENT_VALS> p_values =
                convert_p(acc_s);
            BF16* stage_p = p_ring +
                stage * DECODE_ACTOR_SHARED_LANES *
                        DECODE_ACTOR_FRAGMENT_VALS;
            float* stage_scale = scale_ring +
                stage * DECODE_ACTOR_SHARED_LANES * 2;
            #pragma unroll
            for (int frag = 0; frag < DECODE_ACTOR_FRAGMENT_VALS; ++frag) {
                stage_p[frag * DECODE_ACTOR_SHARED_LANES + lane_id] =
                    p_values[frag];
            }
            stage_scale[0 * DECODE_ACTOR_SHARED_LANES + lane_id] =
                output_rescale[0];
            stage_scale[1 * DECODE_ACTOR_SHARED_LANES + lane_id] =
                output_rescale[1];
            __syncwarp();
            if (lane_id == 0) {
                cutlass::arch::ClusterBarrier::arrive(
                    &actor_mbar[P_FULL_MBAR + stage]);
            }
        }

        row_sum[0] = quad_sum(row_sum[0]);
        norm_smem[lane_id] = row_sum[0];
        __syncwarp();
        if (lane_id == 0) {
            cutlass::arch::ClusterBarrier::arrive(
                &actor_mbar[NORM_READY_MBAR]);
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

// FUNCTION: make_decode_vt_tma_desc_l2_256
// Re-encode the exact full-width V-transpose map used by short decode while
// changing only Hopper's advisory L2-promotion width.  Each direct TMA issue
// remains one 64-feature by 128-token, 16-KiB SW128 native box.
static CUtensorMap make_decode_vt_tma_desc_l2_256(
    const void* global_ptr,
    int num_rows
) {
    static_assert(TmaVt::NumValSrc * sizeof(BF16) == KV_HALF_BYTES,
                  "short Vt L2 policy requires the native 16-KiB box");
    CUtensorMap desc;
    uint64_t global_dim[2] = {
        static_cast<uint64_t>(HEAD_DIM),
        static_cast<uint64_t>(num_rows)
    };
    uint64_t global_stride[1] = {
        static_cast<uint64_t>(HEAD_DIM * sizeof(__nv_bfloat16))
    };
    uint32_t smem_box[2] = {
        static_cast<uint32_t>(HALF_DIM),
        static_cast<uint32_t>(BLOCK_N)
    };
    uint32_t element_stride[2] = {1, 1};

    CUresult result = cuTensorMapEncodeTiled(
        &desc,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2,
        const_cast<void*>(global_ptr),
        global_dim,
        global_stride,
        smem_box,
        element_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(
        result == CUDA_SUCCESS,
        "L1024 decode Vt cuTensorMapEncodeTiled failed with error: ",
        static_cast<int>(result));
    return desc;
}

// FUNCTION: make_vt16_tma_desc_l2_256
// Encode the exact global coordinate space and MN-SW32 shared-memory box used
// by TmaVt16, changing only Hopper's advisory L2-promotion width.  Each TMA
// still copies exactly 16 BF16 features from each of 128 rows (4 KiB total).
static CUtensorMap make_vt16_tma_desc_l2_256(
    const void* global_ptr,
    int num_rows
) {
    CUtensorMap desc;
    uint64_t global_dim[2] = {
        static_cast<uint64_t>(HEAD_DIM),
        static_cast<uint64_t>(num_rows)
    };
    uint64_t global_stride[1] = {
        static_cast<uint64_t>(HEAD_DIM * sizeof(__nv_bfloat16))
    };
    uint32_t smem_box[2] = {
        static_cast<uint32_t>(DECODE_D_EIGHTH),
        static_cast<uint32_t>(BLOCK_N)
    };
    uint32_t element_stride[2] = {1, 1};

    CUresult result = cuTensorMapEncodeTiled(
        &desc,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2,
        const_cast<void*>(global_ptr),
        global_dim,
        global_stride,
        smem_box,
        element_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_32B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(
        result == CUDA_SUCCESS,
        "batch-1 D16 cuTensorMapEncodeTiled failed with error: ",
        static_cast<int>(result));
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
    int total_q_rows = batch * num_heads * seq_len;
    int total_kv_rows = batch * num_kv_heads_actual * ctx_len;

    // cute TMA for Q: flattened [batch * heads * sequence, HEAD_DIM]
    auto dQ = make_stride(Int<HEAD_DIM>{}, Int<1>{});
    Tensor mQ = make_tensor(reinterpret_cast<BF16 const*>(Q.data_ptr()),
                            make_shape(total_q_rows, Int<HEAD_DIM>{}), dQ);
    auto tma_q = make_tma_atom(SM90_TMA_LOAD{}, mQ, SmemLayoutQ{},
                               Shape<Int<BLOCK_M_PREFILL>, Int<HEAD_DIM>>{});

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

    // The configured prefill shapes are self-attention lengths divisible by
    // N128.  This also proves every M64 Q tile and physical N128 K/V tile is
    // full.  All other shapes retain the exact generic Iteration-92 path.
    const bool aligned_self =
        seq_len == ctx_len && seq_len > 0 && seq_len % 128 == 0;
    if (aligned_self) {
        const int nq = seq_len / BLOCK_M_PREFILL;
        if (seq_len >= 4096) {
            static const cudaError_t attr_status =
                cudaFuncSetAttribute(
                    unified_attn_prefill_kernel<true, true>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    SMEM_BYTES_10H);
            TORCH_CHECK(attr_status == cudaSuccess,
                        "long early-Vt prefill dynamic-smem configuration failed: ",
                        cudaGetErrorString(attr_status));
            unified_attn_prefill_kernel<true, true>
                <<<dim3(batch * num_heads, nq), BLOCK_THREADS,
                    SMEM_BYTES_10H>>>(
                    (const __nv_bfloat16*)Q.data_ptr(),
                    tma_q, tma_k, tma_vt,
                    (__nv_bfloat16*)O.data_ptr(),
                    seq_len, ctx_len, num_heads, num_kv_heads,
                    total_q_rows, total_kv_rows);
            return O;
        }
        static const cudaError_t attr_status =
            cudaFuncSetAttribute(
                unified_attn_prefill_kernel<true, false>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                SMEM_BYTES_10H);
        TORCH_CHECK(attr_status == cudaSuccess,
                    "aligned prefill dynamic-smem configuration failed: ",
                    cudaGetErrorString(attr_status));
        unified_attn_prefill_kernel<true, false>
            <<<dim3(batch * num_heads, nq), BLOCK_THREADS, SMEM_BYTES_10H>>>(
                (const __nv_bfloat16*)Q.data_ptr(),
                tma_q, tma_k, tma_vt,
                (__nv_bfloat16*)O.data_ptr(),
                seq_len, ctx_len, num_heads, num_kv_heads,
                total_q_rows, total_kv_rows);
    } else {
        const int nq =
            (seq_len + BLOCK_M_PREFILL - 1) / BLOCK_M_PREFILL;
        static const cudaError_t attr_status =
            cudaFuncSetAttribute(
                unified_attn_prefill_kernel<false, false>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                SMEM_BYTES_10H);
        TORCH_CHECK(attr_status == cudaSuccess,
                    "prefill dynamic-smem configuration failed: ",
                    cudaGetErrorString(attr_status));
        unified_attn_prefill_kernel<false, false>
            <<<dim3(batch * num_heads, nq), BLOCK_THREADS, SMEM_BYTES_10H>>>(
                (const __nv_bfloat16*)Q.data_ptr(),
                tma_q, tma_k, tma_vt,
                (__nv_bfloat16*)O.data_ptr(),
                seq_len, ctx_len, num_heads, num_kv_heads,
                total_q_rows, total_kv_rows);
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
    // immutable launch metadata. Do the same for the tensor maps: repeated
    // benchmark calls use the same K/V storage and encoded flattened shape.
    struct DecodeTmaCacheEntry {
        int device;
        const void* k_ptr;
        const void* v_ptr;
        int total_rows;
        int batch_size;
        int ctx_len;
        TmaK tma_k;
        TmaVt tma_vt;
        TmaVt64 tma_vt64;
        TmaVt32 tma_vt32;
        TmaVt16 tma_vt16;
    };
    static thread_local std::optional<DecodeTmaCacheEntry> tma_cache;

    const int device = Q.get_device();
    const void* k_ptr = K.data_ptr();
    const void* v_ptr = V.data_ptr();
    const bool cache_hit = tma_cache.has_value() &&
        tma_cache->device == device &&
        tma_cache->k_ptr == k_ptr &&
        tma_cache->v_ptr == v_ptr &&
        tma_cache->total_rows == total_kv_rows &&
        tma_cache->batch_size == batch &&
        tma_cache->ctx_len == ctx_len;
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
        // L1024's exact short kernel consumes both adjacent 128-byte feature
        // halves of every V row, and its two sibling head-pair CTAs reuse the
        // same KV stream.  Promote the complete 256-byte row on the first
        // half miss.  The descriptor cache key includes context and batch, so
        // this advisory policy cannot leak into L128 or long-decode routes.
        if (ctx_len == 1024) {
            CUtensorMap promoted_vt =
                make_decode_vt_tma_desc_l2_256(v_ptr, total_kv_rows);
            *const_cast<CUtensorMap*>(
                new_tma_vt.get_tma_descriptor()) = promoted_vt;
        }
        auto new_tma_vt64 = make_tma_atom(
            SM90_TMA_LOAD{}, mVt, SmemLayoutVt64{},
            Shape<Int<DECODE_D_SPLIT>, Int<BLOCK_N>>{});
        auto new_tma_vt32 = make_tma_atom(
            SM90_TMA_LOAD{}, mVt, SmemLayoutVt32{},
            Shape<Int<DECODE_D_QUARTER>, Int<BLOCK_N>>{});
        auto new_tma_vt16 = make_tma_atom(
            SM90_TMA_LOAD{}, mVt, SmemLayoutVt16{},
            Shape<Int<DECODE_D_EIGHTH>, Int<BLOCK_N>>{});
        // A D16 CTA requests one 32-byte slice from every 256-byte V row.
        // At batch one all eight slices launch together but only 64 CTAs are
        // available across H200.  Promote the full row into L2 on the first
        // miss so sibling slices can hit it; other batches and short contexts
        // retain CuTe's original descriptor byte-for-byte.
        if (batch == 1 && ctx_len >= 4096) {
            CUtensorMap promoted_vt16 =
                make_vt16_tma_desc_l2_256(v_ptr, total_kv_rows);
            *const_cast<CUtensorMap*>(
                new_tma_vt16.get_tma_descriptor()) = promoted_vt16;
        }
        tma_cache = DecodeTmaCacheEntry{
            device, k_ptr, v_ptr, total_kv_rows, batch, ctx_len,
            new_tma_k, new_tma_vt, new_tma_vt64, new_tma_vt32,
            new_tma_vt16
        };
    }
    TmaK const& tma_k = tma_cache->tma_k;
    TmaVt const& tma_vt = tma_cache->tma_vt;
    TmaVt32 const& tma_vt32 = tma_cache->tma_vt32;
    TmaVt16 const& tma_vt16 = tma_cache->tma_vt16;

    int head_pairs_per_kv = q_heads_per_kv / DECODE_HEADS_PER_CTA;
    dim3 decode_grid(head_pairs_per_kv, num_kv_heads, batch);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
    if (ctx_len > 1024 && q_heads_per_kv == 4) {
        constexpr int PACKED_SCORE_LANES = 16;
        constexpr int LIVE_SCORE_COMPONENTS = 2;
        constexpr int SCORE_VALUES_PER_BLOCK =
            N_TILES * PACKED_SCORE_LANES * LIVE_SCORE_COMPONENTS;
        constexpr int BLOCK_MAX_VALUES_PER_BLOCK = 4;
        constexpr int PRODUCER_SMEM_BYTES =
            KV_SMEM_BYTES + Q_WGMMA_BYTES + sizeof(uint64_t);
        constexpr int CONSUMER_THREADS = BLOCK_THREADS;
        // The D16 fallback carries four 4-KiB stages.  Batches 4, 8, and 16
        // use the D32 consumer's two 8-KiB stages: both configurations keep P,
        // softmax state, and O in registers and use essentially the same
        // shared-memory footprint.  Batch 4 supplies 128 D32 consumer CTAs,
        // nearly one complete H200 wave, while halving duplicated score and
        // softmax replay relative to its former 256-CTA D16 mapping.
        constexpr int D16_CONSUMER_SMEM_BYTES =
            4 * VT_D16_SMEM_BYTES + 4 * sizeof(uint64_t);
        constexpr int D32_CONSUMER_SMEM_BYTES =
            2 * VT_D32_SMEM_BYTES + 2 * sizeof(uint64_t);
        static_assert(D16_CONSUMER_SMEM_BYTES == 16416,
                      "D16 four-stage V ring must request 16,416 bytes");
        static_assert(D32_CONSUMER_SMEM_BYTES == 16400,
                      "D32 two-stage V ring must request 16,400 bytes");
        const bool use_large_batch_d32 = batch >= 4;
        const bool use_stage_owned_d32 = batch == 4;
        // The configured 4K/8K contexts are exact sequences of N128 tiles.
        // Preserve every promoted batch mapping and select only separately
        // compiled producer/D16 symbols; arbitrary tails remain generic.
        const bool full_kv_tiles = ctx_len % BLOCK_N == 0;
        // Iteration 128 established that the generic D16 compiler schedule is
        // decisively better at B1/8K.  Test the still-unmeasured consumer-only
        // choice at B1/4K while retaining the exact producer at both lengths.
        // All other full-tile D16 callers keep the promoted exact symbol.
        const bool use_full_d16_consumer =
            full_kv_tiles &&
            !(batch == 1 && (ctx_len == 4096 || ctx_len == 8192));
        int num_kv_blocks = (ctx_len + BLOCK_N - 1) / BLOCK_N;
        int64_t required_scores =
            static_cast<int64_t>(batch) * num_kv_heads *
            num_kv_blocks * SCORE_VALUES_PER_BLOCK;
        int64_t required_block_maxes =
            static_cast<int64_t>(batch) * num_kv_heads *
            num_kv_blocks * BLOCK_MAX_VALUES_PER_BLOCK;
        int64_t required_workspace =
            required_scores + required_block_maxes;

        // Retain the workspace across benchmark calls, mirroring the TMA plan
        // cache above.  The workspace is stream-ordered scratch and contains no
        // state between calls.
        struct DecodeScoreCacheEntry {
            int device = -1;
            cudaStream_t stream = nullptr;
            int64_t capacity = 0;
            torch::Tensor storage;
        };
        static thread_local std::vector<DecodeScoreCacheEntry> score_caches;
        DecodeScoreCacheEntry* score_cache = nullptr;
        for (auto& entry : score_caches) {
            if (entry.device == device && entry.stream == stream) {
                score_cache = &entry;
                break;
            }
        }
        if (score_cache == nullptr) {
            score_caches.emplace_back();
            score_cache = &score_caches.back();
            score_cache->device = device;
            score_cache->stream = stream;
        }
        if (!score_cache->storage.defined() ||
            score_cache->capacity < required_workspace) {
            score_cache->storage = torch::empty(
                {required_workspace}, Q.options().dtype(torch::kFloat32));
            score_cache->capacity = required_workspace;
        }
        // Preserve the complete score-workspace prefix and append exactly four
        // FP32 producer maxima per (KV head, KV block).
        float* score_ptr = score_cache->storage.data_ptr<float>();
        float* block_max_ptr = score_ptr + required_scores;

        dim3 qk_grid(num_kv_blocks, num_kv_heads, batch);
        // Batch 4/8/16 has 128/256/512 D32 consumer CTAs on the configured
        // 8-KV-head shape, enough for about one/two/four H200 waves while
        // halving duplicated score/softmax work versus the D16 mapping.
        // Batch 1 retains D16 so its 64-CTA grid preserves useful occupancy.
        dim3 actor_decode_grid(
            (use_large_batch_d32 ? 4 : 8) * (q_heads_per_kv / 4),
            num_kv_heads, batch);
        if (full_kv_tiles) {
            if (use_large_batch_d32) {
                static const cudaError_t qk_attr_status =
                    cudaFuncSetAttribute(
                        unified_attn_decode_four_head_qk_kernel<true>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                        PRODUCER_SMEM_BYTES);
                TORCH_CHECK(
                    qk_attr_status == cudaSuccess,
                    "D32-only four-head QK dynamic-smem configuration failed: ",
                    cudaGetErrorString(qk_attr_status));
                unified_attn_decode_four_head_qk_kernel<true><<<
                    qk_grid, BLOCK_THREADS, PRODUCER_SMEM_BYTES, stream>>>(
                        (const __nv_bfloat16*)Q.data_ptr(),
                        tma_k,
                        score_ptr,
                        block_max_ptr,
                        ctx_len, num_heads, num_kv_heads, total_kv_rows,
                        num_kv_blocks);
            } else {
                static const cudaError_t qk_attr_status =
                    cudaFuncSetAttribute(
                        unified_attn_decode_four_head_qk_kernel<true>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                        PRODUCER_SMEM_BYTES);
                TORCH_CHECK(
                    qk_attr_status == cudaSuccess,
                    "full-N128 four-head QK dynamic-smem configuration failed: ",
                    cudaGetErrorString(qk_attr_status));
                unified_attn_decode_four_head_qk_kernel<true><<<
                    qk_grid, BLOCK_THREADS, PRODUCER_SMEM_BYTES, stream>>>(
                        (const __nv_bfloat16*)Q.data_ptr(),
                        tma_k,
                        score_ptr,
                        block_max_ptr,
                        ctx_len, num_heads, num_kv_heads, total_kv_rows,
                        num_kv_blocks);
            }
        } else {
            static const cudaError_t qk_attr_status =
                cudaFuncSetAttribute(
                    unified_attn_decode_four_head_qk_kernel<false>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    PRODUCER_SMEM_BYTES);
            TORCH_CHECK(qk_attr_status == cudaSuccess,
                        "four-head QK dynamic-smem configuration failed: ",
                        cudaGetErrorString(qk_attr_status));
            unified_attn_decode_four_head_qk_kernel<false><<<
                qk_grid, BLOCK_THREADS, PRODUCER_SMEM_BYTES, stream>>>(
                    (const __nv_bfloat16*)Q.data_ptr(),
                    tma_k,
                    score_ptr,
                    block_max_ptr,
                    ctx_len, num_heads, num_kv_heads, total_kv_rows,
                    num_kv_blocks);
        }
        if (use_large_batch_d32) {
            if (use_stage_owned_d32) {
                static const cudaError_t d32_consumer_attr_status =
                    cudaFuncSetAttribute(
                        unified_attn_decode_output_d32_score_consumer_kernel<true>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                        D32_CONSUMER_SMEM_BYTES);
                TORCH_CHECK(
                    d32_consumer_attr_status == cudaSuccess,
                    "stage-owned output-d32 dynamic-smem configuration failed: ",
                    cudaGetErrorString(d32_consumer_attr_status));
                unified_attn_decode_output_d32_score_consumer_kernel<true><<<
                    actor_decode_grid, CONSUMER_THREADS,
                    D32_CONSUMER_SMEM_BYTES, stream>>>(
                        score_ptr,
                        block_max_ptr,
                        tma_vt32,
                        (__nv_bfloat16*)O.data_ptr(),
                        ctx_len, num_heads, num_kv_heads, total_kv_rows,
                        num_kv_blocks);
            } else {
                static const cudaError_t d32_consumer_attr_status =
                    cudaFuncSetAttribute(
                        unified_attn_decode_output_d32_score_consumer_kernel<false>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                        D32_CONSUMER_SMEM_BYTES);
                TORCH_CHECK(
                    d32_consumer_attr_status == cudaSuccess,
                    "output-d32 score consumer dynamic-smem configuration failed: ",
                    cudaGetErrorString(d32_consumer_attr_status));
                unified_attn_decode_output_d32_score_consumer_kernel<false><<<
                    actor_decode_grid, CONSUMER_THREADS,
                    D32_CONSUMER_SMEM_BYTES, stream>>>(
                        score_ptr,
                        block_max_ptr,
                        tma_vt32,
                        (__nv_bfloat16*)O.data_ptr(),
                        ctx_len, num_heads, num_kv_heads, total_kv_rows,
                        num_kv_blocks);
            }
        } else {
            if (use_full_d16_consumer) {
                static const cudaError_t d16_consumer_attr_status =
                    cudaFuncSetAttribute(
                        unified_attn_decode_output_d16_score_consumer_kernel<true>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                        D16_CONSUMER_SMEM_BYTES);
                TORCH_CHECK(
                    d16_consumer_attr_status == cudaSuccess,
                    "full-N128 output-d16 dynamic-smem configuration failed: ",
                    cudaGetErrorString(d16_consumer_attr_status));
                unified_attn_decode_output_d16_score_consumer_kernel<true><<<
                    actor_decode_grid, CONSUMER_THREADS,
                    D16_CONSUMER_SMEM_BYTES, stream>>>(
                        score_ptr,
                        block_max_ptr,
                        tma_vt16,
                        (__nv_bfloat16*)O.data_ptr(),
                        ctx_len, num_heads, num_kv_heads, total_kv_rows,
                        num_kv_blocks);
            } else {
                static const cudaError_t d16_consumer_attr_status =
                    cudaFuncSetAttribute(
                        unified_attn_decode_output_d16_score_consumer_kernel<false>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                        D16_CONSUMER_SMEM_BYTES);
                TORCH_CHECK(
                    d16_consumer_attr_status == cudaSuccess,
                    "output-d16 score consumer dynamic-smem configuration failed: ",
                    cudaGetErrorString(d16_consumer_attr_status));
                unified_attn_decode_output_d16_score_consumer_kernel<false><<<
                    actor_decode_grid, CONSUMER_THREADS,
                    D16_CONSUMER_SMEM_BYTES, stream>>>(
                        score_ptr,
                        block_max_ptr,
                        tma_vt16,
                        (__nv_bfloat16*)O.data_ptr(),
                        ctx_len, num_heads, num_kv_heads, total_kv_rows,
                        num_kv_blocks);
            }
        }
    } else if (ctx_len > 1024) {
        // Preserve the verified iteration-50 actor path for non-4:1 GQA
        // callers; the four-head producer's packing is intentionally exact for
        // the benchmark's four query heads per KV head.
        static const cudaError_t pipeline_attr_status =
            cudaFuncSetAttribute(unified_attn_decode_staged_p_actor_kernel,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 SMEM_BYTES_DECODE_ACTOR);
        TORCH_CHECK(pipeline_attr_status == cudaSuccess,
                    "pipelined decode dynamic-smem configuration failed: ",
                    cudaGetErrorString(pipeline_attr_status));
        unified_attn_decode_staged_p_actor_kernel<<<
            decode_grid, DECODE_ACTOR_THREADS, SMEM_BYTES_DECODE_ACTOR,
            stream>>>(
                (const __nv_bfloat16*)Q.data_ptr(),
                tma_k,
                tma_vt,
                (__nv_bfloat16*)O.data_ptr(),
                ctx_len, num_heads, num_kv_heads, total_kv_rows);
    } else {
        if (ctx_len == 1024) {
            const bool use_consumer_local_acquire =
                batch == 1 || batch == 4;
            if (use_consumer_local_acquire) {
                static const cudaError_t exact_short_attr_status =
                    cudaFuncSetAttribute(
                        unified_attn_decode_full_n128_kernel<true>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                        SMEM_BYTES_10H);
                TORCH_CHECK(
                    exact_short_attr_status == cudaSuccess,
                    "B1/B4 exact-1K acquire-local dynamic-smem configuration failed: ",
                    cudaGetErrorString(exact_short_attr_status));
                unified_attn_decode_full_n128_kernel<true><<<
                    decode_grid, BLOCK_THREADS, SMEM_BYTES_10H, stream>>>(
                    (const __nv_bfloat16*)Q.data_ptr(),
                    tma_k,
                    tma_vt,
                    (__nv_bfloat16*)O.data_ptr(),
                    ctx_len, num_heads, num_kv_heads, total_kv_rows);
            } else {
                static const cudaError_t exact_short_attr_status =
                    cudaFuncSetAttribute(
                        unified_attn_decode_full_n128_kernel<false>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                        SMEM_BYTES_10H);
                TORCH_CHECK(
                    exact_short_attr_status == cudaSuccess,
                    "exact-1K decode dynamic-smem configuration failed: ",
                    cudaGetErrorString(exact_short_attr_status));
                unified_attn_decode_full_n128_kernel<false><<<
                    decode_grid, BLOCK_THREADS, SMEM_BYTES_10H, stream>>>(
                    (const __nv_bfloat16*)Q.data_ptr(),
                    tma_k,
                    tma_vt,
                    (__nv_bfloat16*)O.data_ptr(),
                    ctx_len, num_heads, num_kv_heads, total_kv_rows);
            }
        } else {
            static const cudaError_t short_attr_status =
                cudaFuncSetAttribute(unified_attn_decode_kernel,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     SMEM_BYTES_10H);
            TORCH_CHECK(short_attr_status == cudaSuccess,
                        "decode dynamic-smem configuration failed: ",
                        cudaGetErrorString(short_attr_status));
            unified_attn_decode_kernel<<<
                decode_grid, BLOCK_THREADS, SMEM_BYTES_10H, stream>>>(
                (const __nv_bfloat16*)Q.data_ptr(),
                tma_k,
                tma_vt,
                (__nv_bfloat16*)O.data_ptr(),
                ctx_len, num_heads, num_kv_heads, total_kv_rows);
        }
    }
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("unified_prefill", &unified_prefill, "Unified prefill (10h: cute TMA K → K-major SW128, WGMMA QK)");
    m.def("unified_decode",  &unified_decode,  "Unified decode  (10h: WGMMA QK, vectorized K/V)");
}
