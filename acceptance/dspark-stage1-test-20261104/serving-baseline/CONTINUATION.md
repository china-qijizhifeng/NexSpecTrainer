# Baseline serving continuation

Latest verified state, 2026-11-04 06:45 UTC: both workers are Running1/1.
Replica-b reached worker_ready at06:38:01 UTC and passed the same full API
acceptance at06:40:41 UTC (14 positive requests and6 missing/invalid-key checks).
Actual node207 has8H100; full archive/import/CRI and environment passed. Its
resolved configuration and environment match the accepted replica-a; see
replica-b/resolved-config-acceptance.json. Native greeting again returned10
tokens/EOS1, verify3/proposed15/correct7, and long-prefix cache reused12288 tokens.
This is functional acceptance, not a performance optimum or lossless result.

Replica-b now has its own NodePort Service:
svc-53d971bd3abc4057-20261104143955, alias dspark-v41-replica-b-20261104-r1,
NodePort30392, ClusterIP10.233.39.37, selector exactly task app+replica-b.
Detail resolves only job-ac6e7833c28c-20261104130228-worker-0 on node207.
http://10.51.6.31:30392 passed actual API acceptance. Both existing replica
submission intents and original14-file manifest remain unchanged; never resubmit.

No router job/Service or NorthGate model exists yet. Next prepare a separate
CPU-only router job on an eligible node with the verified cached image; read the
actual node/project constraints again before submission. Use installed router0.3.2,
cache_aware/history_backend none, separate control-plane admin key, health checks
enabled and automatic retries disabled. Freeze a separate router execution
manifest without changing the running workers. Keep keys out of argv and logs.

Read-only metadata inspection found both authenticated worker server-info
responses echo the caller's already-known worker key. Raw responses were not
saved; saved server-info receipts are redacted. The pinned Rust public metadata
proxy forwards caller Authorization rather than injecting the worker key; keep
that boundary and verify missing/invalid-key rejection in the actual router.
Never print or archive raw metadata with credentials. A separate router admin
key must not appear in any inference or metadata response.

For actual routing evidence, both workers expose sglang:num_requests_total by
is_streaming on authenticated /metrics. Snapshot both counters around isolated
router requests to identify the selected worker, then verify repeated fresh long
prefixes stay on that worker and reuse cache. Pinned cache_aware implementation
uses cached-prefix affinity when match_rate exceeds threshold; a cold miss uses
minimum live load with random tie-breaking, so do not require alternate cold
prefixes to deterministically select the other worker. Source pins are under
router-audit/; the Rust wheel commit still is not proved by Python equality.

The independent test-r4 remained Pending0/1 with its exact controller verified.
GPFS free was about1.25TB at06:45 UTC, above850GB; use the latest snapshot.

Read RUN.md, RECIPE-REVIEW.md and candidate-profile.json first. This serving deployment is independent of the frozen held-out test-r4; preserve both.

Replica-a was submitted once: alias dspark-v41-baseline-a-20261104-r1,
job job-a87130e9ff3d-20261104122759, WeaverRSI/proj-32, node205, 8H100.
submit-intent/response/job receipts are in remote replica-a/. No serving controller exists; no submission retry is permitted.
At 04:56:28 UTC it reached worker_ready. Its actual 8 NVIDIA H100 80GB HBM3, full archive SHA/import/CRI, environment and model identity preflight passed. Initial compilation/autotune/graph capture completed; a transient readiness503 during startup resolved before smoke and is not an observed inference failure.
Platform Pending can include init. Follow image-import/result.json and worker phase, not the historical job.json status.
The 1500s import deadline starts at ctr invocation, after archive verification, not at submission.

The CPU-only image probe job-bbcdb77ae0a8-20261104121421 was cancelled exactly while Pending with no worker output. Its alias is absent; receipts in image-probe/. It used 0 GPUs and must not be resubmitted. Its inspection is now part of the serving worker preflight.

Run local monitor.py to save a read-only snapshot and local JSON receipts. It does not submit, delete or read private keys.
Replica-a full API acceptance passed at 04:59:36 UTC: 14 positive synthetic requests plus missing/invalid-key rejection checks, reasoning and tool/result roundtrips in stream/nonstream, native EOS/spec counters, concurrency2 and >8192-token long-prefix/cache. Native greeting returned10 tokens including EOS1, verify3/proposed15/correct7; cached long-prefix request reported12288 cached tokens. This is functional acceptance, not a throughput optimum or held-out-test result. See replica-a/api-acceptance.json and resolved-config-acceptance.json; raw synthetic responses remain remote and are not committed.

Own replica-a NodePort Service: svc-3c8d5ec190774fa2-20261104125851, alias dspark-v41-replica-a-20261104-r1, NodePort30813, ClusterIP10.233.20.28; selector exactly app=dspark-v41-baseline-20261104-r1, replica=replica-a. Service detail resolves only exact job-a87130e9ff3d-20261104122759-worker-0. http://10.51.6.31:30813 passed actual acceptance. All8 pre-existing owned Services were preserved. Platform lists are paginated; use creator=linlizhi/page_size100 and require observed count==total, never assume default page20 is a complete list.

Replica-b was submitted once at 05:02:28 UTC after first-worker acceptance and refreshed WeaverRSI/proj-32/node207 Ready/allowed-label/no-reservation checks. Exact alias dspark-v41-baseline-b-20261104-r1, job-ac6e7833c28c-20261104130228, node207, 8GPU. replica-b/submit-intent.json, response, job and submission-preflight exist. Do not run submit_replica_b.py again or restart it. Generic resource capacity does not prove H100; actual init must check8H100 before importing. Queue/init is normal. No replica-b Service, router job/Service or NorthGate channel has been created yet.
Baseline source/recipe are published on GitHub main; verify the current github-main-publication.json/API instead of assuming a local commit is published.
monitor.py is deliberately outside the 14-file worker execution manifest; adding a monitor does not change the frozen worker files.
Any future change to running worker code/config requires a new attempt identity after required static review/regression, not mutation/restart.

After replica-a worker_ready, first inspect actual source version/backend/parser and installed router help/source (copied under replica-a/router-source/).
Current worker_ready only means authentication/models/plain chat/nonstream/SSE smoke passed. It is not full service acceptance.
Before starting replica-b, validate actual reasoning fields, tool-call JSON and result resubmission in stream and nonstream, native EOS/finish/spec counters, finite stats, a bounded concurrent sample and a long-prefix request.
Any new runtime/count/parser error stops that class of dynamic validation. Fully audit configuration/contracts/call chain/similar paths and batch-fix/regress before a new attempt; do not keep probing errors with blind reruns.
Keep lossless_greedy_verified=false and no synthetic acceptance. Do not claim old FP4 AR/verify numerics fixed.

Replica-b's reviewed YAML is frozen and now submitted. Node207 was rechecked Ready, schedulable, node_shcheduleable=true, no reservation annotation, generic GPU8 capacity. Actual H100 is checked in init. Do not overwrite frozen YAML or take another job's resources.
Before submitting a planned worker, verify_execution(root), current WeaverRSI/proj-32, exact alias absent and no prior submit-intent. Then create its intent exclusively with fsync, submit exactly once, save response and exact job identity. Never resubmit an uncertain result.
Both workers have authenticated port8000, no hostNetwork, separate read-only official-model mount and private cache. Key lives ONLY remotely in private/worker-key, mode0600. Do not print/copy it.

Northjob network route contract was checked from current frontend source: GET /api/v1/service/list, POST /api/v1/service/create, GET /api/v1/service/detail, DELETE /api/v1/service/delete. Use remote-only Northjob credentials as in the GPU skill.
Create only unique owned Service(s) selecting this task's exact labels; preserve all existing Services. Let Kubernetes allocate NodePort rather than grabbing an occupied one.
Prefer the actual pinned SGLang router with cache-aware routing and both worker/API authentication, after inspecting its installed options. Do not assume the newest router docs match the installed package. Preserve health checks; do not copy benchmark retries/synthetic options or install latest ad hoc.
The installed router is0.3.2; its Python router.py is byte-identical to the pinned SGLang commit's sgl-model-gateway source. Static Rust audit confirms global api_key protects inference routes despite the CLI help describing worker auth; initial worker registration also inherits it. Configure a separate control-plane admin key, cache_aware, history_backend=none, and bounded/no automatic retries for acceptance. Public models/server-info routes forward the caller's Authorization header to the protected worker; actual missing/invalid-key checks still required. Do not assume the Rust wheel build identity from Python equality alone. Static audit artifacts are local router-audit/; router runtime is not yet validated or launched.
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
