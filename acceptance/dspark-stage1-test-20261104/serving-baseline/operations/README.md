# Worker acceptance and guarded expansion

These operational scripts belong to the authorized official Flash-1017/DSPARK
service, independently of the frozen test-r4 run. They do not restart workers or
training and do not alter the original 14-file worker execution manifest.

- `platform_ops.py` uses the current frontend's Service contract, checks complete
  owner-filtered pagination, creates only an exact task/replica selector with a
  platform-allocated NodePort, and preserves existing Services. Creation is
  guarded by an exclusive intent; uncertain results are never resubmitted.
- `accept_worker.py` performs one bounded synthetic acceptance with remote-only
  credentials: missing/invalid-key rejection, models, reasoning, tool calls and
  tool-result resubmission in stream/nonstream, native EOS and raw DSPARK counts,
  concurrency2, and a long prefix with actual cache reuse. It retains private raw
  responses remotely and publishes only hashes and metrics. No held-out inputs or
  generated responses are committed.
- `submit_replica_b.py` checks actual first-worker acceptance, the frozen worker
  manifest, current project/queue/node eligibility and exact job identity. It
  creates the second replica's intent exclusively and submits once. It must never
  be rerun after its intent exists.
- `check_operations.py` checks EOS/count boundaries, final-block raw draft counts,
  incomplete/foreign Service lists, and actual HTTP SSE completion on CPU. Run
  `python3 check_operations.py` from this directory.
- `accept_replica_config.py RUN_ROOT` checks the second worker's saved resolved
  configuration and environment against the accepted first worker. Both actual
  node identities, GPU lists, versions, protobuf backend, serving source hashes
  and official model identity are checked.
- `inspect_metadata_boundary.py RUN_ROOT` reads authenticated metadata without
  saving raw responses or credentials. It records whether server-info echoes the
  caller's existing worker key and whether per-worker request counters are
  available for subsequent router affinity acceptance.

`replica-a-api-acceptance.json` records functional acceptance at 04:59:36 UTC.
The 14 positive requests and six missing/invalid-key checks passed. A greeting
returned10 tokens including EOS1, verify3, proposed15 and correct7; a long-prefix
request reused12288 cached tokens. These small requests do not establish maximum
throughput, an optimal topology, 128K load acceptance, or lossless equivalence.
`replica-a-resolved-config-acceptance.json` records the actual backend, parsers,
CPU Engram, Radix and decode graph configuration. Bounded decoder SWA replay is
enabled; numerical equivalence remains unverified.

Replica-a's exact owned NodePort30813 selects only its job; all8 prior owned
Services were preserved. Replica-b was submitted at05:02:28 UTC as
job-ac6e7833c28c-20261104130228 on node207. Its initialization and full acceptance
passed at06:40:41 UTC: the same14 positive requests and6 missing/invalid-key
checks passed, including12288 cached tokens. Its actual environment and resolved
configuration match replica-a. Its own NodePort30392 selects only its exact job;
the Service identity and detail receipts are included here. No router or
NorthGate model was published by these scripts. Router source audit findings are static; the installed Rust
runtime, authentication, two-worker health and cache routing still require
validation before NorthGate publication.

Both authenticated workers' raw server-info responses echo the already-known
caller key. Saved server-info receipts are redacted, and the metadata audit saves
only booleans and request counters. The pinned Rust public metadata path forwards
caller Authorization; it does not inject the private worker key. Preserve that
boundary and check the actual router's missing/invalid-key rejection and separate
control-plane key before publication. Cold misses in the pinned cache-aware
implementation choose minimum live load with random tie-breaking; cache affinity
acceptance must use repeated fresh prefixes rather than assume alternating cold
requests deterministically use different workers.

The execution and formatted archive SHA256 values are recorded separately in
`source-provenance.json`; non-import AST equality is checked. Formatting the
archive does not rewrite executed sources, run manifests or historical receipts.
