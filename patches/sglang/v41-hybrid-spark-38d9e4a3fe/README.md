# Native DeepSeek V4.1 Hybrid Spark

This bundle targets SGLang `38d9e4a3fe4a493b0ae468477a274e83defda81a`.
It separates the unchanged five-query native MTP from the sixteen-query target
verification capacity. It also includes the historical native draft window/ring
fix, so W128 uses the pool's actual attention window. Install this bundle on
the pristine image source, before other experimental instrumentation.

```bash
python -m specforge.hybrid.runtime_patch --root /sgl-workspace/sglang \
  --bundle patches/sglang/v41-hybrid-spark-38d9e4a3fe
# Repeat with --execute after the all-file preflight passes.
```

Launch the ordinary native DSPARK server with the official checkpoint, block
size 5, and `SGLANG_DSPARK_USE_GREEDY_DRAFT=1`. Set
`SGLANG_DSPARK_HYBRID_CONFIG` to an absolute JSON path. The resolver selects
verify width 16; do not change the MTP checkpoint block size. NexSpecTrainer
must be available on PYTHONPATH for every worker rank.

```json
{
  "schema_version": 1,
  "model_horizon": 5,
  "min_match": 5,
  "max_match": 15,
  "max_draft": 15,
  "mode": "hybrid",
  "contains": true,
  "slot_price": 0.0,
  "calibrator": "/shared/nsp/runs/ca_ID/attempts/0001/ngram-calibrator.json"
}
```

`mode: mtp` measures the additional scan/history overhead while forcing the
original MTP chain; `mode: ngram` uses an anchor-only step if no match exists.
`extra_budget_per_request` optionally fixes a batch-wide extra-token budget
(the count excludes anchors). Without a profiled SPS table this is a fixed
policy, not a measured adaptive-verification speedup. `uncalibrated_acceptance`
may replace `calibrator` only for explicit smoke tests.

The adapter accepts only native V4.1, single-node tensor parallelism, and no
attention DP or disaggregated prefill. Draft graphs remain five-wide. Folded
target acceptance is disabled because its buffers assume a fixed native chain;
the ordinary exact target verifier and committed-hidden injection are retained.
Request history reads only prompt + committed output + current anchor once.
Selection and budget allocation stay on the GPU. Existing delayed confidence
relay carries *survival directly*, and the budget planner does not cumprod it
again. Dynamic slot price uses the preceding host decision and profiled marginal
step cost. Calibration and performance claims require real serving validation.

Use `nsp eval` with `protocol: native`, `metrics: [al, performance]`, bounded
`concurrency`, and fixed `input_ids`. Target-only uses `[performance]` and
`checkpoint_field: model_path`. `nsp calibrate` fits the n-gram estimator from
T=0 request receipts; it splits by trace and labels only positions through the
first rejection. See `docs/hybrid-spark.md` for the workflow and evidence.
