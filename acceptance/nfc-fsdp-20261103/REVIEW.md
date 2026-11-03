# Independent review record

Scope: NexSpecTrainer native FSDP1/BF16Optimizer checkpoint integration against
base `01ff8af94a447041e9377590ee3db5754b083523`, with NFC dependency
`79d838fc19ba1f179fc6c1dc36147e5183d82ac8`.

Three independent read-only review rounds completed on 2026-11-03, within the
six-round limit for this change size. Review agents could inspect the source,
contracts, tests and acceptance evidence; implementation self-review was not
used as a substitute.

| Round | Findings and resolution | Independent validation |
| --- | --- | --- |
| 1 | P1: JSON round-tripping lost tuple-valued provider contracts. Replaced with typed Core common metadata and a checksum-bound completion marker. P2: permissive weights-only loading could leave missing export keys uninitialized. Export now requires every declared weight key. Added regressions for both. | 64 passed, 2 skipped |
| 2 | Online resume's rank-0 global-step probe still assumed legacy `.pt` files. It now reads native metadata locally without invoking the collective resume reader. Added a regression. Reviewer confirmed the fix and no remaining findings. | 38 passed |
| 3 | Checked native shards, full training-state restoration, collective error propagation, provider contracts, publication, Trainer/controller/manager and launch/export/warm-start callers. No new provable P1/P2 blockers; earlier fixes remain effective. | 38 passed; diff check passed |

The final reviewer read the 8-H100 r5 acceptance receipts without launching
another GPU job. Approval applies to the documented fixed-world-size FSDP1,
BF16Optimizer and pinned Torch 2.10/mcore runtime scope. It does not establish
full-size V4.1 checkpoint throughput or complete Torch 2.13 GPU compatibility.
