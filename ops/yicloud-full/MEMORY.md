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


R3 reached MoE layer39 but failed with CUDA illegal access/TMA700, peak allocated
103892217344 bytes. All replicas terminal Failed. Standalone job
j-1793818857-148111 reproduced illegal access in the original mhc_combine at
234048 rows, directly synchronized after the call. Widened combine + stats
partial kernels match row-tiled originals bitwise at8193,104858,234048; all
three existing packed stream-mean aux slots also match. The general combine
fallback runs above65536 rows on Hopper; int32 row*20480 overflows at104858.
This is a proven operator bug; full-model attribution remains pending.

Static audit found the same issue in Engram gate (token*25600 and row*5120).
R4 widens these indices too, and runs GPU differential before model load for
masked/unmasked gate plus unchanged adjacent TileLang post. BF16 arithmetic
and accumulation order stay unchanged. The original stream-mean and broadcast
kernels already widen their row offsets; small stats arrays do not exceed int32.
Bounded MoE keeps local token products below the boundary. Probe diagnostics
synchronize each boundary to localize any remaining asynchronous CUDA failure.
