# Native V4.1 Hybrid Spark acceptance

The implementation is tested against SGLang `38d9e4a3fe4a493b0ae468477a274e83defda81a`
and the official DeepSeek V4.1 Flash-1017 model. All GPU acceptance runs have immutable
managed Run records. Serving evaluation and calibration use `nsp`.

## Completed gates

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

## Full-model gate

Managed Run `58cc695e-ee65-4046-8df4-2955afa66755`, Job `j-1794476150-416069`,
uses source `7018bd1bc9073be03dc7052bae62ebbc9bd6948f` in an independent clean
shared checkout. It requests one replica with eight H200, 64 CPU and 640 GiB,
TP8/EP8, in `fdj-infra/nex_bp`. Submission was verified at
`2026-11-12T09:35:50Z`. Its target-only server was still in cold startup at
17:56 CST. Output equivalence and the full-model gate have not yet passed.

The target-only, original DSPARK and Hybrid servers execute sequentially in the
same allocation with the same immutable token IDs and T=0. Calibration uses the
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
