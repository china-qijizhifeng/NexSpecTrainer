# Validation-only numerical probe r1

This independent attempt diagnoses r7 pre-EOS A/A failure; it does not run held-out test or certify a fix. Read plan.json and probe_contract.py. Source r7 stays failed and deleted; never restart its job/controller. Frozen r7 manifest SHA 38a080efa6a37603725d8e0f0cbce67dc884ee8cd97b03ecf0950922880ed4e2 must not change.

Single8GPU64CPU512Gi with no fixed node, WeaverRSI/proj-32. Init and regular verify actual8H100, fixed image and serving source pins. All outputs private, only SHA/count/logprob margins in summaries. Read monitor.py only for observation. Root local /Users/shom/Downloads/dspark-aa-numerics-20261104-r1, remote /gpfs/users/linlizhi/dspark-aa-numerics-20261104-r1, SSH linlizhi-bp-dev.

Exactly24 validation requests: each baseline / no-fused-allreduce mode has6 recorded history prompts(max1024),4 failing-input decode probes(max32),2 common19token-prefix prefill probes(max1). Same EOS1/temp0/ignore_eosFalse. Capture native top10 and both historical differing-token logprobs. The sole server intervention is --enforce-disable-flashinfer-allreduce-fusion. Prefix-prefill differs from incremental decode; instrumentation can perturb execution. Do not call a finite agreement determinism or a fix. No test tuning or gate change.

CPU alias guard initially rejected stale YAML metadata copied from r7; no new GPU submission occurred. Both metadata fields fixed, controller/init/worker paths and mounts statically checked, alias/root rejections added. Keep original failed CPU receipt. Batch regress all HTTP/SSE/count/numeric/identity/lifecycle paths before dynamics. Freeze only after formatting and passing CPU; scripts and archive must be byte-identical.

Launch once only after --check; launch/started/submit intents prevent replay. New runtime errors stop without retries. Numerical disagreement is expected diagnostic evidence and does not itself mean transport failure. Worker4h, controller6h; preserve receipt then exact-owned-identity deletion and alias absent. If controller exits with job live, observe read-only until terminal and clean exact job; never relaunch.

Formal held-out evaluation still requires all10 A/A and later988 requests with frozen r7 protocol, but no held-out job is running. No checkpoint deletion, no touching baseline A/B or others; Harbor already done. Publish all new scripts and non-sensitive evidence to GitHub main, verify API SHA and ancestry. Quiet normal queue/health; notify new meaningful result/failure/resource/user decision only. Automation active until task completed.

Status: submitted once; observation only.

Submitted once at2026-11-04 11:47:01UTC. Exact alias denovo-dspark-aa-numerics-20261104-r1, job job-fe890712eecc-20261104194701. Controller PID21679/start_ticks2351947779, exact command /usr/bin/python3 -u /gpfs/users/linlizhi/dspark-aa-numerics-20261104-r1/controller.py followed by one space. Launch/started/submit intents exist: NEVER invoke launch_controller.py or submit again. Latest11:47UTC Pending0/1, controller verified, no worker/init output yet; normal queue. Frozen33-entry manifest SHA41ee3f34b201dd873401110b81c0412df20ff4bf4d558a7d877e2d325864c24f.
