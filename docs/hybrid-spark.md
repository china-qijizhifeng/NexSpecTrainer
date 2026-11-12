# Hybrid Spark on native V4.1

The runtime bundle is in `patches/sglang/v41-hybrid-spark-38d9e4a3fe`.
Keep the official W128 MTP frozen. The verifier has room for anchor + 15
drafts, while MTP still predicts five. GPU suffix search uses request-local
token history with deterministic longest/latest tie breaking, runner-up
agreement and backward copy length capped at 256. It cannot inspect another
request, unaccepted drafts, padding, or future output.

`specforge.hybrid.ngram.scan` dispatches to the explicit CPU reference or two
Triton kernels (tile top-two, then per-request reduction/gather). The selection
module implements conditional-to-survival conversion, contains merging, net
chain value, valid lengths, and prefix-preserving batch top-k allocation.

## Calibration

Generate target-only greedy trajectories through native `nsp eval`. Input JSONL
contains fixed `input_ids`; request receipts also retain exact `output_ids`.
Fit on a separate split before measuring Hybrid:

```yaml
traces: /shared/nsp/runs/ev_ID/attempts/0001/requests.jsonl
temperature: 0
provenance:
  target: sha256-of-official-target-config-json
  tokenizer: sha256-of-official-tokenizer-json
  runtime: pinned-SGLang-and-overlay-sha256
  dataset: calibration-prompt-set-sha256
ridge: 0.001
validation_fraction: 0.2
seed: 42
```

```bash
nsp --root /shared/nsp calibrate -c calibration.yaml --plan
nsp --root /shared/nsp calibrate -c calibration.yaml --foreground
nsp --root /shared/nsp results ca_ID --json
```

This trains only 22 logistic coefficients, including one intercept per draft
position. Traces are frozen with the Run. The artifact records n-gram rules,
input identity, split, log-loss and Brier score. It does not recalibrate the
native MTP confidence head or establish generalization on editing benchmarks.

## Evaluation

```yaml
metrics: [al, performance]
serving:
  url: http://127.0.0.1:30000
  protocol: native
  model: deepseek-v4.1-flash
  checkpoint_field: model_path
  prompts: /shared/heldout-input-ids.jsonl
  temperature: 0
  max_new_tokens: 256
  concurrency: 4
  warmup_requests: 2
  expected_server_args:
    speculative_algorithm: DSPARK
    speculative_num_draft_tokens: 16
```

Run target-only, baseline DSPARK, and Hybrid sequentially on the same hardware.
Compare token IDs first, prove actual commits longer than six emitted tokens,
and only then compare throughput. Target-only requests use `[performance]`;
they never fabricate AL. Performance covers nonstreaming end-to-end HTTP wall
time, excludes warmup, and includes output-token throughput and request latency.
AL remains total emitted tokens / actual verifier cycles.

Serving-only evals freeze prompts and record an external checkpoint metadata
identity (config/index content hashes and weight-file size/mtime inventory).
They do not copy the server's large model directory. This inventory is not a
full weight-shard content hash. Offline evaluation keeps its existing complete
checkpoint snapshot behavior.

The adapter checks target config/tokenizer hashes against the fitted artifact.
The first runtime version accepts greedy targets only; nonzero temperature
requires separate calibration and sampling-distribution acceptance.

The engineering acceptance harness and receipts are in
`acceptance/hybrid-spark-v41-20261112`. Kernel checks and protocol fixtures do
not replace native-model E2E, distribution checks at nonzero temperature, or
representative calibrated throughput benchmarks.
