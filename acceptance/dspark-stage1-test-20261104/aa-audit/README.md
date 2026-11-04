# Target-only A/A failure audit

The node040 serving-image import succeeded and the server became ready. The
r3 worker then stopped at the target-only A/A gate, before any test request.
The exact failed job was recorded and deleted; the original attempt must not
be restarted. The imported image and accepted step25/118 exports are retained.

Two identical validation requests with 54,235 input tokens generated the same
61 tokens through the first EOS (ID 1, zero-based position 60). Their forced
256-token outputs first differ at zero-based position 77. Both requests had
zero cached tokens and zero retractions. Replaying every captured raw SSE
event reproduces the saved output and its disagreement. Two short diagnostic
prompts passed A/A; the remaining validation prompts were not run after the
failure. There are no test scores.

The original `ignore_eos=True` protocol still fails its full-output A/A gate.
Agreement through EOS does not establish deterministic kernels or repair the
previous AR-versus-DSPARK discrepancy. The numerical source of the post-EOS
difference remains unknown. The pinned serving source rejects the global
deterministic-inference flag with the `dsv4` attention backend; do not blindly
enable it or substitute MegaMoE flags for the actual SM90 W4A16 path.

The user has been asked to choose between EOS-aware generation capped at
1,024 tokens and retaining forced 1,024-token generation. The recorded test
inputs remain unchanged and unseen by inference. A new protocol needs a new
attempt and a passing validation gate; no GPU rerun is launched by this audit.

`audit_failure.py` reads the private captured results, frozen validation input,
and target `config.json`. It only checks JSON and event contracts; it never
sends generation requests or reads model shards. The published receipt contains
hashes, counts and boundaries, without raw prompts or generated text.

```sh
python3 acceptance/dspark-stage1-test-20261104/aa-audit/check_audit.py
```

This CPU check covers EOS boundaries, mismatches before EOS, absent/multiple
EOS IDs, raw cumulative/delta events and corrupt-prefix rejection. It cannot
substitute for a GPU validation rerun. `static-review.json` records the exact
pinned source, inspected call chains, findings and falsifiable hypotheses.
