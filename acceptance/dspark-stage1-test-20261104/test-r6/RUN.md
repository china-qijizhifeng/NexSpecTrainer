# Test r6: scheduler-selected eight H100s

User requested removal of fixed-node scheduling. The r5 scheduler explicitly rejected node090 for insufficient memory against the unchanged768Gi request. Remove hostname selection, keep the allowed-node label and existing Weaver tolerations; do not bypass reserved partitions. Scheduler chooses the node; init and regular worker must independently verify actual8H100. Fixed OCI image import, source/model hashes, EOS protocol, ten A/A diagnostics and988 held-out requests are unchanged.

Only migrate while exact r5 remains Pending0/1 with no init/worker output. Pin its verified controller with pidfd, stop it once, then recheck state, save cancellation receipts and cancel the exact job before launching r6 once. Never restart an old controller or modify an old execution manifest. Preserve serving replicas and all active data/checkpoint/cache dependencies. Temporary50GB reserve and60GB submit/cache threshold remain authorized.
