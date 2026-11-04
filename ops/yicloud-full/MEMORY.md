# Longest full-trajectory memory acceptance

R2 (`j-1793816845-824844`) passed all eight ranks' full/SWA pool checks but
failed during the 234048-token forward. CUDA mapping exhausted memory on all
ranks before the Engram gather sort raised invalid-device errors. Job and sole
replica are Failed and terminal reconciliation completed. No captured result.

R3 reuses the exact bounded MoE implementation validated on the same SGLang
38d9e4a3fe4a493b0ae468477a274e83defda81a on BP H100 at 16K and 64K.
Source SHA256: 59822eabdae34639937138efb5d77e4d3b278e210ab7c6c8f290f62aba42af27.
Those complete outputs were bitwise equal, and 64K allocated peak decreased
12.27 GiB. That evidence does not establish 234K correctness or memory fit.

Router weights/IDs are computed once, local CUTLASS calls use <=8192 rows,
collectives stay outside this loop. Full attention, CPU Engram, capture layers,
KV pool capacity and official input encoding are unchanged. Graph execution
remains disabled. Reject unsupported activation quantization and routing.
Probe-only allocation tracing observes Engram and MoE entry/exit without
copying tensors or changing output. Whole longest output must pass native
reader, shape, finite, token/mask checks before full-data submission.

CPU contract regression uses the pinned image's actual standard and packed
route definitions, 8192+1 token tail, unchanged IDs/weights and exact output.
Prior GPU parity evidence is in speculative-decoding-training-research/v41/
prefill-long-20261103/evidence/moe-parity16k.json and moe-parity64k.json.
