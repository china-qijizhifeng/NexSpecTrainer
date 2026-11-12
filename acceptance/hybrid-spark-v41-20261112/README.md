# Native V4.1 Hybrid Spark acceptance

This suite targets SGLang `38d9e4a3fe4a493b0ae468477a274e83defda81a`
and the official DeepSeek V4.1 Flash-1017 model. H100/H200 kernel gates passed;
native-model output equivalence is pending. All GPU acceptance runs have
immutable managed Run records. Serving evaluation and calibration use `nsp`.

## Completed gates

- CPU scan passed 135 independent cases for both prefix-length dtypes;
  chain selection passed 54 scenarios and 331 prefix-budget allocations.
  The native CPU cycle observer also rejected four corrupted receipts.
  See [cpu-protocol/manifest.json](cpu-protocol/manifest.json).
- Detached `nsp` Runs passed the real HTTP protocol fixture with eight measured
  requests, two separate warmups and peak concurrency four. The serving stage
  retained external checkpoint identity without copying weights. Missing cycle
  telemetry, malformed token output, warmup calibration and sampled calibration
  failed as expected. These fixtures establish protocol behavior, not model
  throughput or calibration quality. The corrected native-seed fixture is in
  `cpu-protocol/native-seed-corrected/`; nsp tests passed 16/16 after the fix.
  `native-sampling-contract.json` records the pinned SGLang parser's rejection
  of `seed` and successful parsing of all 41 corrected request settings.
- H100 n-gram scan, selection, history writes and CUDA graph replay passed on BP Dev;
  see the [independent H100 report](northjob/README.md).
- YiCloud H200 module Run `1aeaa3c1-a709-4045-b45f-9e65662e3a2f`,
  Job `j-1794475750-48378`, source `3c6626db72e6688943457a38043de4102f73a747`:
  parent and sole replica Succeeded; managed terminal reconciliation recorded
  `2026-11-12T09:33:13.838922+00:00`. Evidence: `kernel-artifacts/`.
- The H200 module gate passed 135 independent scan fixtures for both int32/int64
  prefix lengths, 12 graph replays, 54 selection cases and four history-write
  replays. Eager warmed scan calls measured about 0.036–0.037 ms at 8K/32K/64K
  histories and batches 1/4; these timings include eager calls and allocation
  after compilation, rather than isolated kernel execution or serving throughput.
- The native SGLang accept/commit gate passed 256 cases per prefix dtype plus
  four fused-argmax cases, with a maximum commit of 16. Ring addresses passed
  51 prefix/commit cases at ring sizes 128 and 1024. This checks real kernels
  with synthetic logits; it does not establish model-output or KV-content parity.
- The runtime patch applied all nine pinned files and official config resolution
  kept the trained MTP horizon at five with target verify capacity sixteen.
  This Run used patch SHA-256
  `c82a768532e7fb72d1e8c0353465785fb6fe880cba67a28b62c3f9ee45b9b669`.
  The current patch adds fixed-budget and topology guards; the tested scan,
  selection and native accept kernels are byte-identical.
- Official tokenizer CPU preflight Job `j-1794475888-503043` and its sole replica
  Succeeded. Native `encode_simple_chat(spec="dsv41")` produced 24 independent
  calibration and 17 measured prompts, including chat copies, short EOS/repetition,
  thinking mode, and exact lengths 4095/4096. See `tokenizer-preflight.json`.
- Fixed-budget precedence passed 32 configuration assertions;
  see `fixed-budget-precedence.json`. The latest nine-file patch was also
  validated, applied and checked for idempotence against pristine source;
  see `pristine9-preflight.json`.
- Native worker lifecycle passed the CPU regression in
  [pool_lifecycle_acceptance.py](pool_lifecycle_acceptance.py): the real patched
  constructor defers history allocation, and the actual worker pool hook binds
  the scheduler-owned pool after draft allocation. Disabled Hybrid and ranks
  without draft ownership skip allocation. GPU kernels are not exercised here.
  See [pool-lifecycle-cpu.json](pool-lifecycle-cpu.json).

## Full-model gate

Managed Run `58cc695e-ee65-4046-8df4-2955afa66755`, Job `j-1794476150-416069`,
uses source `7018bd1bc9073be03dc7052bae62ebbc9bd6948f` in an independent clean
shared checkout. It requests one replica with eight H200, 64 CPU and 640 GiB,
TP8/EP8, in `fdj-infra/nex_bp`. Submission was verified at
`2026-11-12T09:35:50Z`. It failed at 17:59 CST because target-only cold startup
exceeded the harness's 1200-second readiness limit. It never entered model
evaluation; `results` was empty. Parent and sole replica reached `Failed`, and
the managed reconciler recorded resource release. This is a startup failure,
not evidence about Hybrid output correctness. See
[full-r1-failure.json](full-r1-failure.json) and
[full-r1-evidence/](full-r1-evidence/) for the immutable configuration, terminal
receipts, log excerpt and complete-log SHA-256.

The second attempt is Run `aa335087-ea2c-4099-aaed-f8da736e4e8b`, Job
`j-1794477911-83691`, source `0fdd86ea741c0f6970bbbfb4a8f2255c30cf4f95`.
It keeps the same eight-GPU allocation and comparison gates, increases each
server readiness limit to 5400 seconds, and uniformly enables fast DeepGEMM
warmup and persistent kernel caches. All runtime outputs are outside its new
immutable source checkout. Target readiness succeeded in 800.42 seconds. The
first calibration warmup then returned HTTP 400 before any measured request;
the original client did not preserve the response body. The pinned native
parser reproduces the cause: `/generate` takes `sampling_seed`, while nsp sent
`seed`. The fix maps the native field while keeping OpenAI `seed`, preserves
bounded HTTP error bodies, and passed the strict protocol fixture and 16 nsp
tests. Parent and every replica of the GPU Job, controller and log relay are
terminal. See [full-r2-failure.json](full-r2-failure.json) and
[full-r2-evidence/](full-r2-evidence/).

The third attempt uses the corrected request contract: Run
`c675617d-beb9-40e1-9ff3-470baea4ea45`, Job `j-1794480511-583250`, immutable
source `8225a6c`, submitted at `2026-11-12T10:48:31Z`. Target and DSPARK completed
all 17 requests: 16 matched exactly, while `reasoning-no-match` diverged at
output index 51 (target token 666 versus DSPARK 42499). Hybrid then failed at
startup because history allocation read a request pool before the scheduler
created it. Parent and all replicas, controller and observer are terminal.
The fix binds history in the native pool hook. The next comparison also fixes
all server random seeds at 42: the third attempt used different server seeds,
despite identical request sampling seeds. Neither change establishes the cause
of the reasoning divergence; all three output equality gates remain required.
See [full-r3-failure.json](full-r3-failure.json) and
[full-r3-reasoning-and-suite-comparison.json](full-r3-reasoning-and-suite-comparison.json).

The next harness includes two serial repeats of the reasoning request and two
one-token continuations of its target prefix in separate `nsp` diagnostic Runs
after each measured window. Prefix replay uses prefill, so it does not establish
the original decode/verify logits or their margin. The diagnostic helper passed
four real HTTP requests through detached `nsp` in
[diagnostics-cpu-preflight.json](diagnostics-cpu-preflight.json); the existing
store, metric and serving tests passed 15/15 in
[nsp-lifecycle-regression.xml](nsp-lifecycle-regression.xml).

The target-only, DSPARK and Hybrid servers execute sequentially in the
same allocation with the same immutable token IDs and T=0. Every server uses
the pinned runtime bundle; DSPARK leaves Hybrid disabled but retains the shared
native draft window/ring correction. Calibration uses the
separate 24-prompt population. The Hybrid engineering gate fixes fifteen extra
slots per request, records every native verify cycle through an independent CPU
observer, requires a commit beyond the five-token MTP horizon followed by another
decode cycle, and checks output IDs across all variants. Performance results are
interpreted only after correctness passes. No AV throughput, general model
quality, fitted calibration quality or production deployment claim follows from
this small engineering population.

## Environment and transport

Image digest:
`registry.gate.yicloud.com.cn/fdj-infra/nexspec-sglang:v41-c6da252c6837-20261104@sha256:03a08910ead2720094402c9bc36fa896611a60e48b6ffb026f0ca6be3a9fd298`.
Official model path:
`/gpfs/shared-model-weight/models/deepseek-ai/DeepSeek-V4.1-Flash-1017`.
Toolchain: clean YiCloud skill commit `f97044f244c6bce3dc45e4dcdf568adf92562d21`.
Credentials remain in the platform-injected files or local external credential file;
no credential contents were transported or included in evidence.

Three immutable toolchain parts were transported by CPU Jobs
`j-1794475002-529064`, `j-1794475002-502553`, `j-1794475002-511439`;
all parents and replicas Succeeded. The initial assemble controller
`j-1794475339-953544` failed before GPU submission because a shallow bundle was
cloned without restoring its shallow boundary. The replacement
`j-1794475741-316771` explicitly unbundled the objects, restored `.git/shallow`,
verified the clean checkout and submitted/reconciled the H200 gate.
Failure receipts are preserved. The full-model controller is
`j-1794476141-919557`; it allocates TP8 only after reading both successful kernel
gates. It keeps the managed reconciler alive until parent and all replicas end.
