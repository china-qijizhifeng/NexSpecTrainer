# All-anchor AdamW LR experiments — November 6, 2026

| Run | Platform | Progress / outcome | Last observed validation |
|---|---|---|---|
| 1e-5 | BP 32 H100 | 118 updates completed | step100 loss0.913552, acc78.0624% |
| 1e-4 | BP 32 H100 | 118 updates completed and terminal verified | step100 loss0.906610, acc78.2798% |
| 5e-4 | YiCloud 32 H200 | user stopped at39 updates /156 micros per rank | step25 loss1.007512, acc76.5460% |
| 2e-4 | YiCloud 32 H200 | ongoing; all32 ranks passed gate and completed2 updates at snapshot | not yet available |

All runs use the original1,326 training and17 validation traces, globalbatch128,
accum4,118 planned updates, seed42 fresh initialization, all legal anchors,
chunk64, FP32 reduction, FULL_SHARD and CPU optimizer offload. BP and YiCloud
features have matching trajectory IDs, token/mask hashes and window supervision,
but YiCloud BF16 feature values were independently captured. Hardware differs
as well. Cross-platform results are not a strict LR-only comparison. The two
YiCloud runs use the same feature files and differ only in peak LR.

## Source provenance and scope

`bp-source-provenance.json` hashes the1,395 files used by the BP runs. YiCloud's
first AdamW snapshot copied those files byte-for-byte and added its launcher:
`9499520cbb4a8fbc1c6c70811bd565509f8268b7`. The2e-4 snapshot is
`b2e6e77f8a45ccc7d14cd3b3c1f85a2c37b41f4d` (recipe LR and assertion only).
These are experiment-local snapshot commits, not claims that they are reachable
from GitHub main. `executed-launch-hashes.json` records the original launch files.

The repository version packages this launcher with formatting cleanup and clear
path placeholders. Its library base is current main, which already includes the
Muon mixed-precision-copy fix and later capture/runtime work. Existing GPU
receipts verify the pinned experiment snapshots, not a new full training run of
this merge commit. The isolated running job's source and configuration are not
modified by this merge. The branch includes no changes to optimizer algorithms,
LR schedule implementations, library defaults or the user's concurrent WSD work.

## Completed BP1e-4

`bp-1e-4/terminal-verification.json` confirms all32 ranks completed118 updates,
472 training micros and20 validation micros, with unchanged frozen tensors;
all four node exit codes were0. The existing monitor recorded platformCompleted,
then released that exact job. Final checkpoint118 has a publication receipt,
metadata,128 nonempty shards spanning32 ranks and a matching common.pt SHA256.
Restore has not been tested. Last update was2026-11-06 11:44:08 Beijing, about
7.15h after training start including validation/checkpoint overhead. Peak GPU
allocated33.291GiB / reserved74.184GiB. All40,097,642 anchor exposures and
199,997,123 target exposures passed per-window coverage checks.

At step100,1e-4 validation accuracy is0.2174 percentage points above1e-5 and loss
is0.76% lower. No step118 evaluation was configured, and this does not establish
serving acceptance rates or inference-quality improvement.

## YiCloud replacement

5e-4 job `j-1793931458-946720` was explicitly stopped by the user; all four
replicas areStopped, not failed training. Its checkpoint25 (128 shards) remains.
The curves stop at39, and the user-stop receipt retains the exact terminal state.

2e-4 job `j-1793940402-922604`, UUID`0008929f-5488-4d37-8f2c-cc2d111fc6c7`, uses
four H200 nodes and the same input files. The startup snapshot contains actual
optimizer updates, finite values and complete per-window anchor coverage with
zero duplicates. It does not claim final convergence or checkpoint restoration.

Regenerate the three available curves with
`uv run --with matplotlib python acceptance/dspark-adamw-lr-20261106/plot_comparison.py`.
No missing steps or2e-4 validation points are synthesized. Local paths in
receipts identify retained evidence, not secrets or artifacts shipped to GitHub.

## Merge validation

On the current main library base, the2e-4 config resolves to AdamW and validates
world32. The real streamed/whole trainer differential passed on two CPU Gloo
ranks, including two optimizer updates under each policy; see
`main-cpu-gate.json`. All three rendezvous regression tests and shell syntax
checks passed. This is separate from the existing32H200 experiment acceptance.
Curves retain plotted step/accuracy/loss/gradient/LR fields from worker logs;
reported `lr` is the next LR, while per-rank `lr_used` is the actual update LR.
