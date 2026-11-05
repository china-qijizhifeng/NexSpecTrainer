# V4.1 Flash H100 acceptance

Inference optimization is accepted for the measured workloads. Full-prompt
training capture has not passed long-input numerical acceptance. All new
runtime optimization flags default off; this archive does not enable them in
training or modify the default SGLang dependency.

| Workload, same 64,326-token prompt | Cold TTFT median |
| --- | ---: |
| Same-node baseline | 3.977456s |
| Full computation, CP8, 16K prefill chunks | 1.504059s |
| Compact128, CP8 | 1.200400s |
| Compact128, CP8, W4A8, reusable workspace | 1.034914s |

Each cold request flushes KV and the CUDA allocator and checks cached_tokens=0.
These numbers are inference TTFT, not hidden-state publication time. No task
quality scoring was performed. Generation TPS comparisons with different
prompt-logprob contracts are kept separate in `comparison.json`.

The capture plugin publishes only from attention TP0/CP0. Real HTTP → SGLang
observer → SpecForge → Mooncake returned every requested prompt row, with
BF16 aux layers37/38/39, final hidden, exact token IDs/loss masks and finite
values through16,385 tokens. Full hidden mode disables Compact; the plugin
requires unchunked prefill. Three important numerical findings remain:

- Cross-process comparisons at16,385 tokens failed in runs012/015.
- With DeepSelect disabled, same-process16,384/16,385-token repeated captures
  still differed. Thus disabling DeepSelect alone is not a validated fix.
- All1,920 workspace-on/off comparisons on the same actual model layer inputs
  were bit exact across8 EP ranks (run016). This establishes the tested MoE
  scratch behavior, not whole-model hidden equivalence.

Run017 separately reproduces nondeterministic tie selection in DeepSelect's
cluster path. It does not establish the cause of the remaining whole-model
variation. No64K full-hidden capture timing or success is claimed. Internal
MoE token chunking was removed; the proposed unused CP gather elimination
was never installed. Both are absent from the committed runtime patch.

Validation includes154 namespace/dispatch tests plus2 subtests, kernel and
Graph replay checks,22 workspace GPU cases,7 full-capture configuration
cases, and34 plugin tests plus26 subtests. `patch-acceptance.json` records
application, all42 resulting file hashes and reverse restoration against the
exact SGLang base revision.

`requests/` preserves the original immutable input manifests. `repro/` stores
exact harness text as evidence, not standalone runnable tests: it depends on
the pinned image and the existing baseline harness under
`/gpfs/users/linlizhi/dspark-test-20261104-r10`. Complete numbered lab inputs
and logs remain at `/gpfs/users/linlizhi/v41-h100-2s-20261105`. No prompt or
hidden tensor payloads are committed here.

Worker exit was0, the exact owned job
`job-58a7093f827c-20261105131049` reached Completed, and it was deleted.
`cleanup/after.json` is the verified empty alias lookup. The detailed report
and structured receipts retain failures as well as accepted results.
