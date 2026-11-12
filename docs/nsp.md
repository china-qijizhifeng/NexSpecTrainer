# nsp: training and evaluation runs

Install the repository package to get `nsp`, or use `python -m specforge.nsp.cli`
from this checkout. Control/observation imports Click and the standard library,
not torch or CUDA. Training workers use the normal NexSpecTrainer dependencies.
This version supports Linux/macOS process supervision and single-node trainers.

```bash
nsp train -c train.yaml --plan
nsp train -c train.yaml --name experiment --request-id unique-request --watch
nsp list --kind train --status running --json
nsp show RUN --json
nsp watch RUN --until terminal --timeout 3600
nsp logs RUN --follow
nsp checkpoints RUN --json
nsp stop RUN
nsp train --resume RUN@step-100 --watch
nsp calibrate -c calibration.yaml --plan
nsp calibrate -c calibration.yaml --watch
```

`NSP_ROOT` or global `--root` selects the durable store; default:
`~/.local/share/nsp`. Use a persistent shared volume inside scheduler containers.
Training creates a new output directory under the attempt; it does not overwrite
the recipe's old output directory. Configuration overrides use the existing
`section.field=value` syntax. `--plan` does not create a Run or launch processes.

A submission starts an independent supervisor and returns a Run ID. `--watch`
then observes it; Ctrl-C detaches without stopping work. `--foreground` waits,
requests stop on Ctrl-C/TERM, and is intended as a scheduler job payload. An SSH
profile transports the same CLI to an already provisioned execution host:

```yaml
# ~/.config/nsp/profiles.yaml (or NSP_PROFILES)
yicloud-h200:
  host: allocated-gpu-host
  cwd: /workspace/NexSpecTrainer
  python: /opt/venv/bin/python
  root: /shared/nsp
```

```bash
nsp --profile yicloud-h200 train -c /shared/train.yaml --plan
nsp --profile yicloud-h200 train -c /shared/train.yaml --watch
nsp --profile yicloud-h200 list
```

All paths in a profile command are remote paths. This is SSH transport, **not a
YiCloud/Northjob resource allocator**. Submit the platform job using its normal
contract and put `nsp train --foreground` in its payload. Do not invoke nsp inside
every torchrun rank. Multi-node platform submission, a global cross-profile
registry, rank log filtering and automatic checkpoint evaluation are follow-up
work; this version rejects multi-node/distributed-worker submission before launch.

## Durable execution

Each Run has a stable ID, original intent, request ID, attempts, resolved config,
a source snapshot (including uncommitted Python changes), source digest, worker
log, metrics/event streams and completion receipts. Workers import the frozen
package rather than a mutable checkout. Model/data paths still refer to the
configured storage. Credentials should come from environment variables, not
literal recipe values. Store directories are private to the submitting user; observe with that same user
or use an explicitly configured group-accessible storage policy.

Retry a network-uncertain submission with the same `--request-id` and same intent;
a different intent using that ID fails. Resume requires a terminal training Run
and creates a new attempt with full checkpoint state. New experimental settings
belong to a new Run using `model.draft_checkpoint_path` for weights-only loading.

`checkpoints` lists completed publications. `RUN@latest` is resolved once at eval
submission. Eval/resume retain their own immutable shard links (copying across
filesystems) under the same rotation lock as CheckpointManager, so later training
rotation cannot remove their input. These snapshots retain storage until the Run
is explicitly removed by its owner; there is no automatic garbage collector.

A run succeeds only after its worker exits successfully and publishes completion.
Training additionally records the final positive optimizer step and matching
completed checkpoint. Unreachable/stale observation is separate from execution
failure. Killing a supervisor externally may leave a stale Run; `stop` never kills
a recycled PID. `stop` is a durable request processed by a live supervisor, with a
120-second graceful drain followed by process-group kill.

Exit codes: submission 0 means accepted; `watch --until terminal` / foreground
returns 0 on success, 1 on worker failure, 2 on invalid CLI/config, 3 on observation
timeout, 4 on cancellation. JSON snapshots are versioned; `watch --jsonl` streams
snapshots including observation timestamps. All other commands accept `--json`
where applicable. Logs remain plain text.

## Standalone evaluation

```yaml
# eval.yaml
recipe: /shared/train.yaml
features: /shared/validation-features
metrics: [loss, accuracy, prefix-al]
group_by: ctx-len
ctx_origin: record-start
ctx_bins: [0, 4096, 8192, 16384, 32768, 65536, 131072, inf]
seed: 42
anchor_chunk_size: 32
```

```bash
nsp eval RUN@step-100 -c eval.yaml --plan
nsp eval RUN@step-100 -c eval.yaml --metrics loss,accuracy,prefix-al --watch
nsp results EVAL_RUN
nsp results EVAL_RUN --json
```

Eval loads only draft weights and frozen target parts: no optimizer, scheduler,
training loop, or fabricated training eval interval. A recipe supplies model and
objective settings; features are a separate fixed validation split. Relative
recipe/features/prompt paths are relative to eval.yaml; model paths within the
recipe retain the training invocation's working-directory interpretation.
Native checkpoint provenance is compared with the assembled model; pretrained
exports without provenance report it unavailable. Native checkpoints supply the
objective's global step and schedule horizon. For external exports with scheduled
objectives, supply `global_step` and `total_steps` explicitly.

| Selection | Output and population |
| --- | --- |
| `loss` | Additive objective numerator / its actual weight denominator |
| `accuracy` | Correct predictions / valid predicted slots; anchor excluded |
| `prefix-al` | Offline `1 + leading matches`, over complete-horizon anchors |
| `al` | Actual committed output tokens / speculative verify cycles |

Overall always accompanies bins. Counts pool before division. Zero-denominator
bins contain JSON null, rendered `— (n=0)`. Offline and serving stages keep separate
populations, definitions and results. No simulated metric substitutes for AL.

Offline evaluation enumerates every eligible anchor once, chunking computation
without changing its supervision mask. `ctx_len = anchor index + 1` for a complete
record beginning at its original prefix origin; each block belongs to its anchor's
bin. This measures **record/target prefix length**, not the draft's private sliding
attention window. `ctx_origin: record-start` explicitly attests this relationship.
Sliced/packed captures with lost origins cannot claim dynamic ctx bins. A record
longer than recipe `data.max_length` fails rather than being silently truncated.
The feature path/size/mtime inventory is saved and rechecked after evaluation;
it is a metadata fingerprint, not a full content hash.

Current offline capabilities:

- DFlash (including DFlash2 selector), DSpark, native DeepSeek V4.1: loss,
  accuracy, prefix AL, overall and anchor ctx bins. Plain DFlash uses unary
  matches; DFlash2 accuracy uses the self-conditioned selector. DSpark unary
  accuracy follows the existing reference-conditioned logits head; prefix AL
  counts only the prefix before the first mismatch. Reference is recorded tokens.
- EAGLE3: overall loss and accuracy pooled across draft positions; reference is
  target argmax, explicitly labeled. MTP and Domino: overall additive metrics.
- Other algorithms, LK lambda, D-PACE and unsupported combinations fail rather
  than borrowing a different definition. Single-device offline eval only; no
  padded DP sampling or FSDP inference path is introduced in this version.

`prefix-al` excludes incomplete horizons, including short/masked blocks; results
include eligible and full-horizon anchor counts. Choosing it does not change the
loss/accuracy population. Requesting a metric with no observations fails the eval
and preserves its partial result. Numerical summation uses float64 counts.

## Serving AL

Provide a dedicated SGLang server already loaded with the exact selected draft:

```yaml
metrics: [al]
serving:
  url: http://gpu-server:30000
  model: target-model-name
  prompts: /shared/heldout-prompts.jsonl
  temperature: 0
  top_p: 1
  max_new_tokens: 128
  timeout: 300
  # api_key_env: SGLANG_API_KEY
```

Each JSONL prompt contains `messages`, and optionally `id`. The endpoint's
`/get_server_info` must attest `speculative_draft_model_path` matching the fixed
checkpoint (the endpoint/field are configurable via `info_path` and `checkpoint_field`). Only this
eval's response counters are aggregated; there is no shared-server counter delta.
Requests use fixed seed and sampling settings. `serving.concurrency` controls
bounded concurrent requests (default 1), and `warmup_requests` are executed
before the measurement window. `protocol: native` accepts frozen `input_ids`
and retains exact `output_ids`; `protocol: openai` accepts `messages`.

Add `performance` to measure total output tokens / HTTP measurement wall time
and nonstreaming request p50/p95 latency. This does not measure TTFT or TPOT.
A target-only endpoint uses `metrics: [performance]` and
`checkpoint_field: model_path`; it does not need speculative counters.
`expected_server_args` pins additional resolved serving arguments.

Serving-only evaluations freeze prompts and retain checkpoint metadata
fingerprints instead of copying all weights. The weight inventory checks
filenames, sizes and modification times; it is not a full shard content hash.
Offline evaluation retains the complete checkpoint snapshot.

Overall uses `choices[0].meta_info.spec_verify_ct` and usage completion tokens.
This includes actual emitted correction/bonus tokens after EOS/budget truncation;
`1 + accepted_draft/verify` can differ and is not substituted. Missing counters,
HTTP errors and mismatched server identity fail the stage, preserving receipts.

Dynamic ctx bins require this additional response telemetry:

```json
{"spec_verify_cycles": [
  {"cycle_id": 0, "ctx_len": 4095, "committed_output_tokens": 3},
  {"cycle_id": 1, "ctx_len": 4098, "committed_output_tokens": 1}
]}
```

Cycles must be unique per request and emitted counts must sum to completion tokens.
Each cycle goes to the bin of its starting ctx length. Ordinary request-level
SGLang counters and histograms do **not** provide this information. This PR does
not modify external SGLang to add telemetry: unsupported endpoints fail grouped
AL instead of manufacturing bins from prompt length. Both offline and serving
metrics can be requested in one eval config; `--metrics all` includes loss,
accuracy, prefix-al and al. Add performance explicitly.

## Hybrid calibration

`nsp calibrate -c calibration.yaml` creates a separate durable Run (`ca_...`)
and uses the same list/show/watch/logs/results/stop commands. It freezes the
trajectory input, fits the request-local n-gram logistic estimator on CPU and
publishes its coefficients, source identities and held-out fit metrics.
Every trajectory must include observed greedy sampling receipts; warmup or
nonzero-temperature traces are rejected. Splitting is by trace, and labels
stop at the first candidate mismatch.

The complete calibration and native V4.1 workflow is in
[hybrid-spark.md](hybrid-spark.md). A protocol fixture establishes the Run and
measurement contract; model output equivalence and speedups require actual
native GPU serving evidence.

## Agent workflow

Use `.agents/skills/use-nsp/SKILL.md` for training/evaluation operations. The older
`training-run` skill now holds environment/data/checkpoint guidance and delegates
task lifecycle to nsp. Existing `specforge` entrypoints remain internal execution
and explicit differential-test baselines, with their original foreground behavior.
