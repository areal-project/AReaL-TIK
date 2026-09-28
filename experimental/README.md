# Experiments and validation

This directory contains the per-GPU configs, selected attention kernels, source
snapshots, and recorded validation. See the [main README](../README.md) for setup
and kernel usage. Run the commands below from the repository root.

## Files and selected kernels

Each GPU directory has the following layout:

```text
<a100|h20|h200>/
├── config.yaml                # Agent prompts, evaluation contract, build settings
├── validation.json            # Selected kernel's checks and timings
└── kernels/
    ├── unified_attn_ext.cu    # Selected implementation loaded by default
    ├── manifest.json          # Inventory and evidence for all snapshots
    ├── snapshots/             # Versions with recorded correctness passes
    └── snapshots-demo/        # Experimental, pending, or failed candidates
```

`unified_attn_ext.cu` is byte-identical to the selected snapshot named in
`config.yaml`. The manifest indexes all source versions, their hashes, origins,
and individual test scopes; `validation.json` records checks and timings for
the selected kernel. [archive.json](archive.json) summarizes selections and counts.

| GPU config | Selected snapshot | Correctness-qualified snapshots | Experimental snapshots |
| --- | --- | ---: | ---: |
| [H20](h20/config.yaml) | [Candidate 17, current-stream adapter](h20/kernels/snapshots/g6_17_combined_current_stream.cu) | 77 | 114 |
| [H200](h200/config.yaml) | [Candidate 11, combined](h200/kernels/snapshots/g6_11_combined.cu) | 48 | 221 |
| [A100](a100/config.yaml) | [Candidate 16, tail-safe](a100/kernels/snapshots/g6_a16_tail_safe.cu) | 94 | 133 |

Snapshots preserve their original bytes and recorded test scopes. Some earlier
versions passed only a subset of the current checks. Exact duplicate snapshots
are stored once per GPU, with their original locations retained in the manifest.

## Recorded GPU validation

Each selected source has 32 recorded paired cases: 16 prefill and 16 decode.
Both use batches 1/4/8/16. Prefill lengths are 1K/4K/8K/16K; decode contexts are
128/1K/4K/8K. Internal comparisons check exact BF16 storage bits. External
comparisons against FlashAttention and FlashInfer require maximum absolute
error below 0.05, rather than bitwise equality to those libraries.

Expanded suites additionally cover changed-input graph replay and, where
recorded, prompt/answer partition and batch consistency. The exact coverage is
in [H20 validation](h20/validation.json), [H200 validation](h200/validation.json),
and [A100 validation](a100/validation.json).

To verify the downloaded source hashes and saved evidence without a GPU:

```sh
python tools/verify_archive.py
```

This command checks the archive's integrity. It does not execute kernels or
produce new GPU measurements.
It also checks the [249 original evidence records](evidence/README.md) against
their recorded hashes and reproduces the performance figure from its 60 timing
cells. Resolve any manifest's `record_sha256` through [the evidence index](evidence/index.json);
the original workspace paths do not need to exist on your machine.

## A100 correctness repair

The earlier A100 iteration-309 kernel passed its original checks but failed an
expanded prompt-prefix comparison: at batch 1, a 4K prompt evaluated alone could
differ from the same prompt's outputs when 128 answer tokens were appended.
The selected implementation uses a consistent prefill path across lengths, which
changes some iteration-309 prefill bits while retaining the tested decode bits.

A later check exposed an incomplete-tail mismatch in an intermediate reference
at batch 8, a 16K prompt, and four answer tokens. The selected A16 kernel evaluates
partial 64-token tails through the decode path token by token. Its recorded
validation passes all 64 partition cases and 36 decode batch-consistency cases.
The fallback adds launches and copies for partial tails; that cost is outside
the aligned performance grid.

The bitwise requirement applies between the corrected implementation's prefill
and decode paths on the recorded cases. It does not require reproducing the
earlier inconsistent prefill outputs.

## Running evaluation

Execute the [GPU evaluator](../tools/evaluate.py) on the corresponding device:

```sh
python tools/evaluate.py --gpu h200 --output runs/h200-evaluation.json
```

Use `--gpu h20` or `--gpu a100` for the other devices. The evaluator records
bitwise and external numerical checks, latency measurements, baseline selections,
source hashes, and any failures in JSON. Existing result
files are never overwritten. To test another candidate against the selected
reference, add `--source /path/to/candidate.cu`. Baseline and timing helpers are
in [tools/baselines.py](../tools/baselines.py).
Every screened FlashInfer route must have a finite numerical error below the
configured threshold. The report records each route's error and screening time,
and separately checks the Hopper FA3 route used for the auxiliary comparison.

For optimization, pass a frozen run config plus explicit `--source`, `--root`,
and `--incumbent` paths. Add `--root-report` with that run's successful root
qualification report to check each workload's slowdown cap against the original
root timings. The evaluator verifies complete coverage and matching config,
root-source, and tool hashes. It reports weighted candidate/incumbent objectives,
`admissible`, and `promotion_eligible`; it does not modify search records or kernels.
Without `--root-report`, the two eligibility fields are null because there is no
frozen timing anchor. Any failed evaluation sets `promotion_eligible` to false.
Fresh reports also record the actual dependency/compiler/driver identities and
CUTLASS header hash. A later audit must match its root's measured environment;
changing the environment requires a separate run and new root qualification.

The recorded benchmark grid and the release evaluator have different coverage:

| Evaluation | Prefill | Policy-update forward | Decode | Total timing cases per GPU |
| --- | ---: | ---: | ---: | ---: |
| Recorded GPU benchmark | 16 | 0 | 16 | 32 |
| Release evaluator | 16 | 16 | 16 | 48 |

The extra policy-update-forward cases evaluate attention over each prompt plus
128 response tokens. They are attention-only workloads, not full-model training
steps. No completed GPU qualification is available for the expanded release
evaluator. Fresh GPU validation is deferred; its CPU checks and archived results
do not establish a new GPU pass or add new performance measurements.

Before an optimization run, the [skill](../.agents/skills/areal-tik/SKILL.md)
instructs the agent to evaluate its selected starting kernel (the "root") on the
target GPU. Search proceeds only after the complete configured evaluation passes.
CPU tests cover report completeness, weighted comparisons, slowdown limits, source
binding, and result handling; they do not establish GPU numerical correctness:

```sh
python -B -m unittest discover -s tests -v
```

## Prompt-driven optimization runs

The skill is the entry point; the role prompts remain in each GPU's `config.yaml`.
The existing coding agent follows those prompts and writes the optimization IR
directly to `state.json`, with immutable actions, audit evidence, and lineage
decisions in the run directory. The skill documents their record format.
`tools/evaluate.py` supplies independent measurements and eligibility checks.

Use a new directory under `runs/` for each experiment. Freeze the config and
evaluator tools before root qualification, preserve all source versions and failed
attempts, and retain real process exit codes and timestamps in the run logs.
The agent executes the evaluator with the configured shell timeout. An interrupted
or timed-out audit remains unverified even if a partial JSON report exists.
When resuming, reconcile saved records and hashes before starting another attempt.

The four roles are prompt instructions for the agent, and the IR is a documented
record format. Budget tracking and state updates are performed by that agent;
account-quota enforcement depends on what its host exposes.

## Kernel interface

`tools.load_kernel.load_kernel()` returns a checked forward-only interface.
Q/K/V must be contiguous CUDA BF16 tensors in BHSD layout on the same device,
with matching batch sizes, matching K/V shapes, and the configured head counts
and dimension. Empty dimensions are rejected. Prefill requires equal Q/KV
sequence lengths; decode requires one query token. Only the documented workload
grid has recorded correctness coverage.

The loader binds each extension instance to the current CUDA device and restores
the caller's device after each invocation. For another GPU, load under
`with torch.cuda.device(index):`. Per-device extension instances isolate the
archived kernels' static launch settings. Launches use PyTorch's current stream.
These checks do not change the archived CUDA sources or their arithmetic.

This interface does not implement autograd. With gradient tracking enabled,
inputs requiring gradients are rejected; use `torch.no_grad()` for forward-only
evaluation. A full training integration must supply its own backward path.

## Adding another GPU

Reuse the role prompts and start a new GPU directory/config. Adapt `gpu`, `build`,
the initial source/hash, numerical workloads, baselines, and architecture-specific
prompt guidance. Freeze representative workload weights before starting the run.

A B200 config alone is not sufficient today. Add the device to the supported-GPU
checks in `tools/load_kernel.py` and `tools/evaluate.py`, select compatible prefill
and decode baseline adapters in `tools/baselines.py`, and update architecture-
dependent checks in the evaluator. Its current FA3/FA2 dispatch and Hopper-route
requirement are specific to the released devices. Do not copy those assumptions
to a new architecture without qualifying its baselines.

Then run a complete root audit before any search or performance claim. Record
the environment, raw-bit tests, prompt/answer partitions, graph replay, CUDA
dispatch, and baseline errors. Keep new results in a separate run directory;
the existing GPU archives remain the historical release results.

## Timing and figure data

Operator timings use matched CUDA-graph replay for the candidate and baselines.
Compilation, warmup, graph capture, FlashInfer planning, and KV packing occur
outside the timed region. These measurements describe attention execution,
rather than end-to-end RL training throughput.

The main README's performance figure preserves the six attention values adopted
in the manuscript. Values are averaged baseline-to-AReaL-TIK latency ratios;
above 1 means faster. Each prefill bar averages 4 configurations: batch 1 with
sequence lengths 1K/4K/8K/16K. Each decode bar averages 16 configurations: all
combinations of batches 1/4/8/16 and context lengths 128/1K/4K/8K. The figure
covers 20 configurations per GPU, or 60 across A100, H20, and H200.

The full 32-case validation grids include all four prefill batch sizes, so their
aggregates differ from the figure's. H20's figure values also come from a separate
recorded run. See the [figure CSV](figures/attention_performance.csv) and
[source provenance](figures/provenance.json) for the exact data. Plot data and
provenance are kept in `figures/`; README figures and their PDF versions are
kept together in `assets/` at the repository root.
The [individual timing cells](figures/attention_performance_cells.csv) contain all
60 observations, including that separate H20 run. Each row identifies its source
and original evidence hash. `tools/verify_archive.py` checks every latency against
that original record and recomputes the six displayed aggregates.

The performance SVG embeds Times New Roman glyph outlines so the chart stays
sharp when resized and does not depend on fonts installed on the reader's device.
