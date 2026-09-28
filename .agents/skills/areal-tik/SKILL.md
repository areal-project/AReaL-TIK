---
name: areal-tik
description: Optimize AReaL-TIK's bitwise-aligned CUDA attention kernels on H20, H200, or A100 using the repository's GPU prompts, evaluator, and persistent search records. Use for kernel optimization or resuming a search, not ordinary documentation edits.
---

# AReaL-TIK

Run the optimization protocol through the current coding agent. The GPU config
contains the authoritative prompts; this skill supplies the entry point and
the format of the records shared between roles. No model API client or separate
agent runtime is needed in this repository.

## Select and prepare

1. Locate the AReaL-TIK checkout containing this skill, `experimental/`, and
   `tools/evaluate.py`. All command paths are relative to that checkout unless
   made absolute. Use the GPU and run directory requested by the user. If the GPU
   is unspecified, inspect the available device; ask only if the target remains
   ambiguous. Work on one GPU architecture at a time.
2. Read the selected config: [H20](../../../experimental/h20/config.yaml),
   [H200](../../../experimental/h200/config.yaml), or
   [A100](../../../experimental/a100/config.yaml). Read `prompts.agent_task`,
   `prompts.agent_task_epoch`, and the four role prompts before starting. Use
   [experimental/README.md](../../../experimental/README.md) for dependency and
   validation context when needed.
3. For a new run, use an unused directory under `runs/` unless the user specifies
   another location. Copy the chosen config
   to `config.yaml`, and copy its selected source to
   `kernels/snapshots-demo/root.cu`. Verify the source against `kernel.sha256`.
   Resolve any workload, coefficient, or budget changes requested by the user
   before freezing the run config. Retain the published unit-frequency weights
   when no representative frequencies are supplied; do not invent paper weights.
   Record the config hash and hashes of
   every file in `tools.evaluate.TOOL_FILES`, including the environment recorder.
4. Create the records below with `root` unverified, `current_best: null`, and empty
   frontier, archive, and opportunities. Read any existing run before resuming;
   never reinitialize it. Check its frozen config, tool, and source hashes against
   the saved records. Resume only compatible record formats and frozen tools;
   preserve incompatible runs and start a new one. A changed evaluation contract
   requires a new run.

The config's `{repo_dir}`, `{run_dir}`, `{gpu}`, `{arch}`,
`{max_compile_retries}`, and `{benchmark_timeout}` placeholders refer to the
checkout, isolated run, selected GPU, CUDA architecture, and configured limits.
Substitute them when following the prompts; quote paths when running commands.
This substitution is an instruction to the agent, not a rendering program.

## Execute the prompts

Follow Initialization once, then Action → Audit → Lineage Management. One
persistent agent can perform the logical roles. If delegation is available and
useful, workers may propose changes, but serialize writes to the shared records.

Use the config's evaluator commands on the target Linux GPU. Apply
`limits.benchmark_timeout` to the whole evaluator process with `timeout`, keep
stdout/stderr in a new attempt log, and retain the actual exit status. Check that
both output paths are unused before starting; enable shell noclobber for log
redirection. A timeout,
interruption, failed process, or incomplete report leaves the attempt unverified.
The scripts in `tools/` compile kernels and measure correctness and performance;
the agent interprets the results and updates records according to the prompts.

For root qualification, evaluate the frozen root as candidate, root, and
incumbent. Only after a complete successful audit may it become `current_best`
and enter the frontier. Root qualification uses successful verification, not
`promotion_eligible`, which is null without an earlier root timing anchor. Retain
this audit as `audits/root.json`; its per-workload latencies anchor slowdown limits
throughout the run.

For subsequent audits, pass `--root-report` with that original audit and use an
immutable copy of the current best as `--incumbent`. The evaluator returns
`verified`, weighted objectives, `admissible`, and `promotion_eligible` without
editing search state. Promote only when the process succeeds, the hashes match,
and `promotion_eligible` is true. Correct but slower candidates remain available
for exploration. Never replace evaluation evidence with an agent's assertion.
The evaluator also binds subsequent audits to the root's measured environment.
An environment change requires a new run and root qualification.

## Persistent optimization IR

Maintain these files directly with the agent's normal filesystem tools:

```text
runs/<gpu>-<experiment>/
├── config.yaml                # Frozen prompts, workload and numerical contract
├── state.json                 # Current optimization IR
├── events.jsonl               # Append-only stage and decision history
├── CODEX_OPTIMIZATION_LOG.md   # Measured progress, failures, and costs
├── actions/<id>.json          # Hypothesis and concrete change
├── audits/<id>.json           # Evaluator output; never manually edit
├── audits/<id>.log            # Compiler/runtime output, including failures
├── lineage/<id>.json          # Continue/Merge/Archive/Abandon decision
├── kernels/snapshots-demo/    # Every attempted immutable source
├── kernels/snapshots/         # Copies with complete passing audits in this run
└── best.cu                    # Copy of current_best after qualification/promotion
```

`state.json` is a JSON object with these fields:

| Field | Contents |
| --- | --- |
| `schema` | `areal-tik-ir-v2` (agent-maintained records) |
| `gpu`, `created_at`, `updated_at` | Device and UTC timestamps |
| `config_sha256`, `tool_sha256` | Frozen config hash and tool-path-to-hash map |
| `graph` | Implementation ID → source path, SHA-256, parent IDs, verified flag, action/audit paths, branch status, objective and promotion decision |
| `root`, `root_audit`, `current_best` | Root ID, qualification report, and best verified ID; no best before qualification |
| `frontier`, `archive` | Verified implementation IDs with/without an immediate follow-up |
| `opportunities` | Opportunity ID → parents, mechanism, workloads, evidence, rationale, and pending/consumed/withdrawn status |
| `pending`, `stage` | Incomplete attempt and next role, allowing interrupted work to resume |
| `kappa`, `audited_iterations` | Consecutive non-promoting attempts and total completed candidate attempts; exclude root qualification |
| `budget` | Agreed run limits and provider-measured token usage, or null when unavailable |

Use unique IDs and paths relative to the run directory. Every source version is
immutable, including failed attempts; repairs receive new IDs. Audit paths always
refer to reports for the exact source/config hashes. Additional profiling evidence
may be attached to the corresponding action or audit record without altering the
evaluator's JSON.

An action record includes `id`, `opportunity`, `parents`, `source`,
`source_sha256`, `description`, `functions_changed`, `expected_effect`,
`numerical_risks`, `falsification`, and `started_at`. A lineage record includes
`operation`, `target`, `rationale`, `evidence`, `opportunities`, and `timestamp`.
Record all contributing parents for a merge. Unverified sources remain in the
graph as evidence and cannot be parents or members of the verified frontier/archive.

After every stage, append an event containing its timestamp, implementation ID,
artifact paths/hashes, outcome, process exit code when applicable, and measured
elapsed time, then checkpoint state.
Write state through a temporary file followed by replacement to avoid partial
JSON. On resume, reconcile the checkpoint with immutable actions, reports, and
events before repeating work; do not count or promote the same attempt twice.
An interrupted audit without a completed successful process record needs a new
audit attempt and a new output path. Preserve its partial report and log.

## Stop and report

Honor the user's scope and the config's iteration, patience, timeout, and budget
limits. Respect the configured quota reserve through the host agent's budget
controls; do not claim to enforce account quota that the host does not expose.
Count failures and record actual stage/iteration timestamps. Provider token counts
must be measured or marked unavailable, never estimated from file size or time.

Stop at the configured limit or when no supported follow-up remains. Distinguish
convergence, opportunity exhaustion, budget exhaustion, and environment failures.
Report the best verified source, root/incumbent/baseline comparisons, per-workload
tradeoffs, correctness coverage, failures, elapsed time, and available token costs.
Preserve the packaged kernels and recorded paper results; this skill writes new
experiments under `runs/`.
