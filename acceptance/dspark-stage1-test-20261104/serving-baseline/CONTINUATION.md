# Baseline serving continuation

Read RUN.md, RECIPE-REVIEW.md and candidate-profile.json first. This serving deployment is independent of the frozen held-out test-r4; preserve both.

Only replica-a was submitted, once: alias dspark-v41-baseline-a-20261104-r1,
job job-a87130e9ff3d-20261104122759, WeaverRSI/proj-32, node205, 8H100.
submit-intent/response/job receipts are in remote replica-a/. No serving controller exists; no submission retry is permitted.
At 04:28 UTC the exact job was Pending0/1 but its init was running: actual 8 NVIDIA H100 80GB HBM3 passed, archive full-SHA check underway.
Platform Pending can include init. Follow image-import/result.json and worker phase, not the historical job.json status.
The 1500s import deadline starts at ctr invocation, after archive verification, not at submission.

The CPU-only image probe job-bbcdb77ae0a8-20261104121421 was cancelled exactly while Pending with no worker output. Its alias is absent; receipts in image-probe/. It used 0 GPUs and must not be resubmitted. Its inspection is now part of the serving worker preflight.

Run local monitor.py to save a read-only snapshot and local JSON receipts. It does not submit, delete or read private keys.
No replica-b submission, NodePort Service, router job or NorthGate channel has been created yet.
Baseline source/recipe are published on GitHub main; verify the current github-main-publication.json/API instead of assuming a local commit is published.
monitor.py is deliberately outside the 14-file worker execution manifest; adding a monitor does not change the frozen worker files.
Any future change to running worker code/config requires a new attempt identity after required static review/regression, not mutation/restart.

After replica-a worker_ready, first inspect actual source version/backend/parser and installed router help/source (copied under replica-a/router-source/).
Current worker_ready only means authentication/models/plain chat/nonstream/SSE smoke passed. It is not full service acceptance.
Before starting replica-b, validate actual reasoning fields, tool-call JSON and result resubmission in stream and nonstream, native EOS/finish/spec counters, finite stats, a bounded concurrent sample and a long-prefix request.
Any new runtime/count/parser error stops that class of dynamic validation. Fully audit configuration/contracts/call chain/similar paths and batch-fix/regress before a new attempt; do not keep probing errors with blind reruns.
Keep lossless_greedy_verified=false and no synthetic acceptance. Do not claim old FP4 AR/verify numerics fixed.

Replica-b's reviewed YAML is frozen, but it has NOT been submitted. Its planned node207 must be rechecked (Ready, schedulable, node_shcheduleable=true, no reservation annotation, generic GPU8 capacity); actual H100 checked in init. If node changed, do not overwrite frozen YAML or take another job's resources. Use a justified new identity/config instead.
Before submitting a planned worker, verify_execution(root), current WeaverRSI/proj-32, exact alias absent and no prior submit-intent. Then create its intent exclusively with fsync, submit exactly once, save response and exact job identity. Never resubmit an uncertain result.
Both workers have authenticated port8000, no hostNetwork, separate read-only official-model mount and private cache. Key lives ONLY remotely in private/worker-key, mode0600. Do not print/copy it.

Northjob network route contract was checked from current frontend source: GET /api/v1/service/list, POST /api/v1/service/create, GET /api/v1/service/detail, DELETE /api/v1/service/delete. Use remote-only Northjob credentials as in the GPU skill.
Create only unique owned Service(s) selecting this task's exact labels; preserve all existing Services. Let Kubernetes allocate NodePort rather than grabbing an occupied one.
Prefer the actual pinned SGLang router with cache-aware routing and both worker/API authentication, after inspecting its installed options. Do not assume the newest router docs match the installed package. Preserve health checks; do not copy benchmark retries/synthetic options or install latest ad hoc.
If the router is absent, first resolve a pinned compatible gateway distribution and validate it; do not replace it with an untested custom stream proxy.

NorthGate checkout /Users/shom/Downloads/NorthGate. Route/DTO questions start with docs/api-reference.json per CLAUDE.md.
NorthGate https://northgate.xiaobei.top, user267 (linlizhi-data). Existing access token is in ~/Library/Application Support/Northgate Monitor/config.json under api_key; keep it private.
Onboarding auth headers are Authorization: the stored access token, X-NorthGate-User:267. Admin channel API is unavailable; no admin permission is needed for user onboarding.
GET /api/onboarding/models; POST /api/onboarding/check (preconditions only); POST /api/onboarding/model (upsert owned target).
Target model deepseek-v4.1-flash-dspark-baseline, type openai, groups [token-bp-self], upstream_model_name same as served model.
Only register after actual tested reachable upstream endpoint exists. Preserve current Nemotron channel6245 and all unrelated models.
Finally verify NorthGate /v1/models visibility and real chat/stream/reasoning/tool-result roundtrip, response request/model headers and error boundaries. Save redacted publication/acceptance receipts; don't commit raw generated content.

If the serving job fails or reaches lease end, save terminal receipt before exact identity deletion and verify alias absence; don't clean a healthy deployed service as if it were a one-shot benchmark.
The seven-day resource lease must be reported as a service limit once published. GPFS must retain850GB headroom.
All further fixes/operational scripts require necessary checks and GitHub main push/API SHA+ancestry verification. No force push. Simple temporary operations are exempt from independent review per Downloads AGENTS.
