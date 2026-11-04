# dSpark validation-only numerical probe r1

Read plan.json, static-contract-review.json and CONTINUATION.md. This is a single-use independent investigation of r7 pre-EOS A/A variability. No held-out test is accessed. Fixed source commit38d9e4a, original serving image, scheduler-selected8H100/64CPU/512Gi, WeaverRSI/proj-32. Worker4h/controller6h.

Exactly24 validation requests, native top10/two-candidate logprob evidence and one supported allreduce-fusion intervention. All request inputs and raw responses stay private. Finite agreement does not certify deterministic kernels or a fix, and prefix-prefill differs from incremental decode.

Run run_cpu_checks.py on Linux before freezing; controller.py --check is read-only. launch_controller.py can run ONCE after frozen execution SHA, archived bytes, source identity and API checks pass. After any launch/started/submit intent exists, never rerun it. Observe with monitor.py; stop on new runtime/count errors, preserve receipt, delete exact owned job and confirm alias absent.

Original r7 and all previous evaluation/training tasks must remain stopped. Keep original r7 protocol/gates/input selection and860execution manifest unchanged. Preserve baseline A/B and checkpoints. All authored scripts and non-sensitive evidence must be published to GitHub main and verified against API SHA and parent ancestry.

Permission amendment: control-only metadata is explicitly0644 for a separate SSH controller UID; raw inference outputs and SSE are exclusive0600. Init077 is preserved. Linux regression reproduces oldr1 failure and exercises all writers under077/022 before this independent submission.
