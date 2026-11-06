# Recorded GPU acceptance

Pinned image: `registry.gate.yicloud.com.cn/fdj-infra/nexspec-sglang:v41-c6da252c6837-20261104@sha256:03a08910ead2720094402c9bc36fa896611a60e48b6ffb026f0ca6be3a9fd298`.
Runtime: SGLang `38d9e4a3`, Hopper, TP8/EP8, CP8 with attention TP1.

- Short captures: 4096, 4097, 14226, 15871 and 15872 tokens match original TP tensor hashes. Replays and native readback are exact as recorded in the short acceptance manifest.
- Before the long-length fix, 131072-token final hidden relative L2 was 0.3350898767.
- Operator job `j-1793940418-267260`, source `4796b7e300a89a5eca919cd0ab793e6fbfebb2cd`: real captured residuals cyclically extended to 65537/131072 rows. Old coefficient paths differ; all eight corrected logical shards match their full reference and repeat exactly.
- Full-model job `j-1793940683-594812`, source `dd82e377c2e3f46081446b889f23993edcd83173`: two full 131072-token CP captures match the immutable TP reference in every tensor. All eight ranks retain parameter pointers/versions before and after capture. Native Brotli10/Zstd roundtrip is exact.
- Both diagnostic jobs finished Succeeded with zero active replicas.

The exact 128K input is a diagnostic prefix of a longer real record. The initial
128K failure halted its planned 130999-token full-record and repeat tests; these
must not be counted as accepted. Full-dataset CP remains unreleased.

The JSON files are recorded experimental evidence, not a CI run on this PR.
Some historical receipt paths refer to the original private experiment archive.
The implementation files are unchanged from the validated source except for
repository formatting; source identities are recorded in `source-identity.json`.
