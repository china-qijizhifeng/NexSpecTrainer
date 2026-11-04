# Harbor image publication

The user authorized uploading existing NexSpecTrainer images using a two-line private credential file. Authentication uses the HTTPS registry token endpoint and checks pull/push grants for both destination repositories before any upload. Credentials remain in memory and an isolated0600 temporary Docker configuration, removed when the publishing process exits. Credentials are not included in commands or receipts.

The historical training image is bp/nexspec-trainer:v41-weaver-torch210-20261103-r1. Its config identity is da11e080cc11b041ecb97342c80da74ff66d48b612faa7d4e5a8ca7c0b093336, matching the retained cpu100 build. This upload preserves that historical runtime; it does not establish compatibility with subsequent NFC/source changes.

The serving image is the existing lmsysorg/sglang manifest c6da252c6837bb062184b01c7aa2df7a2aadb1ed4944f236d7027174c73c45fa, config31d58781877bdcd322d5fcc7a858d427db54f5cbf5a9e6342c5fafc01bc67c6e, copied to bp/nexspec-sglang:v41-c6da252c6837-20261104. The frozen r6 worker continues using its original image; publication does not restart it.

Each receipt verifies the destination tag and pinned digest return identical raw manifests, the config blob SHA matches the original local image identity, and all rootfs diff IDs match. Existing destination tags with a different config are rejected. The publisher never rebuilds an image or overwrites an unrelated tag. Uploads are single-use and are not automatically retried after failure.

An initial management API check returned422 before any push. Static authentication review replaced that unrelated user-management check with the registry's advertised repository-token contract. Both required repository pull/push grants then passed. Final completion requires both image receipts and result.json with passed=true.

Both publications completed and passed registry read-back. The publisher exited and its temporary authentication directory was removed. Registry manifest encodings differ from the source SGLang reference, while the config SHA and all70 rootfs diff IDs match the original image.

| Image | Verified immutable reference |
| --- | --- |
| Historical training runtime | harbor.local.clusters/bp/nexspec-trainer@sha256:f2e360fea06554224cf3070d8c5c65bba15762ffad7cc6a7f7d7a3719297c960 |
| Existing serving runtime | harbor.local.clusters/bp/nexspec-sglang@sha256:03a08910ead2720094402c9bc36fa896611a60e48b6ffb026f0ca6be3a9fd298 |
