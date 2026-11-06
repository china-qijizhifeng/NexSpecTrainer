# AdamW peak LR1e-4 completed

Final update: 2026-11-06T11:44:08.431854+08:00. Job job-d3b052f034c2-20261106043222 was Completed; existing monitor recorded terminal receipts and deleted it. Readback absent is expected.

All32 ranks:118 optimizer steps,472 training micros,20 validation micros,passed=true,frozen_unchanged=true. All4 node exit codes0. All step/micro values finite; every window has exhaustive legal-anchor coverage and0 duplicate anchors. Total anchor exposures40,097,642; target exposures199,997,123. Peak allocated33.291GiB; reserved74.184GiB. Start-to-final-update7.150h including validation/checkpoint overhead; median ordinary step214.53s.

Final checkpoint: training/denovo-v41-all-anchors-step118. Verified training_state step118, NFC publication receipt, nonempty .metadata,128 nonempty shards spanning32 ranks,common.pt SHA256 matching training_state. Checkpoint restore has NOT been tested. No claim of serving-quality improvement.

| Validation step | 1e-5 loss | 1e-4 loss | 1e-5 acc | 1e-4 acc |
|---|---:|---:|---:|---:|
| 25 | 0.921862 | 0.918373 | 77.898421% | 78.002025% |
| 50 | 0.916350 | 0.912357 | 78.000855% | 78.174060% |
| 75 | 0.914203 | 0.908657 | 78.048054% | 78.229986% |
| 100 | 0.913552 | 0.906610 | 78.062368% | 78.279792% |

Latest validation is step100, not step118: loss difference -0.006942; accuracy +0.2174 percentage points.

YiCloud5e-4 ongoing at28/118 updates. First validation25: loss1.0075117602,acc76.54598347%; checkpoint25 published with128 shards. Compare BP1e-4 step25 loss.9183733192,acc78.00202525%,and BP1e-5 loss.9218619599,acc77.89842136%. YiCloud uses independently captured features and H200,so this is not a strict LR-only comparison. Remaining pure training about5.24h plus validation/checkpoint overhead.

Artifacts: terminal-verification.json,heartbeat-20261106-1205.json,curves.json,plot_comparison.py,adamw-lr-comparison.png/.svg. Plot contains only actual completed steps. Combined heartbeat remains active for YiCloud5e-4.
