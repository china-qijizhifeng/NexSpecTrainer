# Independent review record

Scope: NexSpecTrainer native FSDP1/BF16Optimizer checkpoint integration against
base `01ff8af94a447041e9377590ee3db5754b083523`, initially with NFC dependency
`79d838fc19ba1f179fc6c1dc36147e5183d82ac8`.

Six independent read-only review rounds completed on 2026-11-03, within the
six-round limit for this change size. Review agents could inspect the source,
contracts, tests and acceptance evidence; implementation self-review was not
used as a substitute.

| Round | Findings and resolution | Independent validation |
| --- | --- | --- |
| 1 | P1: JSON round-tripping lost tuple-valued provider contracts. Replaced with typed Core common metadata and a checksum-bound completion marker. P2: permissive weights-only loading could leave missing export keys uninitialized. Export now requires every declared weight key. Added regressions for both. | 64 passed, 2 skipped |
| 2 | Online resume's rank-0 global-step probe still assumed legacy `.pt` files. It now reads native metadata locally without invoking the collective resume reader. Added a regression. Reviewer confirmed the fix and no remaining findings. | 38 passed |
| 3 | Checked native shards, full training-state restoration, collective error propagation, provider contracts, publication, Trainer/controller/manager and launch/export/warm-start callers. No new provable P1/P2 blockers; earlier fixes remain effective. | 38 passed; diff check passed |
| 4 | Async integration: found P2 rollback best comparisons against an invalidated future best and P2 missing final-directory durability barriers. Both fixed with regressions. Confirmed NFC timeout retry/error ordering and cache invalidation; independently inspected 199 GB async driver. | 37 passed; diff check passed |
| 5 | Final async revision: verified umask-preserving private directory creation (027 → 750), the earlier best/fsync fixes, and source-release/wait/fit-drain documentation. No unresolved P1/P2. | 38 passed; diff check passed |
| 6 | Final NFC dependency update to `be06ae228600d8c2eed3c6a15c740163392a8587`: exact-storage CPU alias and explicit source device preserve schema, layout/cache signatures, lifetime and the collective capture fence. No new P1/P2. Actual NativeState/BF16Optimizer lazy/initialized state covered 48 tiles, source mutation and restore. | Trainer 37 passed, 2 skipped; NFC 16 passed, 1 skipped |

The third reviewer read the 8-H100 r5 acceptance receipts without launching
another GPU job. Approval applies to the documented fixed-world-size FSDP1,
BF16Optimizer and pinned Torch 2.10/mcore runtime scope. It does not establish
full-size V4.1 checkpoint throughput or complete Torch 2.13 GPU compatibility.

Rounds 4–5 independently reviewed the asynchronous implementation. GPU and
real-scale measurements are separate dynamic acceptance evidence; the reviewer
did not treat an uncompleted GPU job as a passing test.

The sixth reviewer had no Core/CUDA locally. Core descriptors were dataclass
substitutes and the clone guard was taken from the inspected deployed source.
Torch 2.13 native DCP saving failed two existing `data.names` checks in files
unchanged by the dependency update; repeated runs stopped. This is an existing
compatibility limit, not native Torch 2.13 acceptance. GPU benchmark receipts
are separate evidence and are not inferred from the independent CPU review.
