# YiCloud 32-GPU Muon + AdamW

Requested 2026-11-05 in the original anchor-training chat. 4 × 8 H200, independent reproducible training run, quota nex_bp, priority9. The compatible Weaver-family pool examined has no serviceable generation; pool state and unrelated workloads remain unchanged.

Source: NexSpecTrainer e480b1a28167ae3165e77b9621712427e871dddc (PR15/16) plus committed launch-only ops/yicloud-muon32. Uses the already validated YiCloud trainer image f2e360fea06554224cf3070d8c5c65bba15762ffad7cc6a7f7d7a3719297c960, runtime Core and NFC sources. No shared environment changes.

Data: original BP subset of1326 train traces and17 validation traces, selected from completed YiCloud full-trajectory capture. Full trajectory token/mask hashes, all15189 original window token hashes and split/slice/context metadata match. The recomputed BF16 features are not bitwise identical to BP and this is not an exact input-tensor controlled comparison. Reuse51 explicit padding windows to reach15104 training windows;136 unpadded validation windows. Distributed validation sampler may pad to160 samples at world32.

Recipe: all legal anchors, chunk64, fixed FP32 reduction, v41_flash_muon (Muon matrix groups plus AdamW remainder), LR1e-5, seed42, global batch128=32×1×4,118 updates/472 micros per rank, evaluation and checkpoint every25 steps. Fresh initialization, no resume. BP AdamW32 and Muon8 remain separate.

Acceptance: four real-image preflights and native feature reads;32-rank same-gradient Muon oracle; first native optimizer update plus post-update backward; finite loss/grad, exhaustive coverage, frozen tensor identity, memory/time. These are startup checks, not full training, checkpoint restoration, or quality improvement claims.

Active attempt: dspark-all-anchors-muon32-20261105-r2, JobId j-1793880042-617264, UUID 69e9394b-b815-47d2-b4b3-111784cd0d56, source 3966a11f345c8fb42b43dfe51e86ec74693deb1d. R1 startup failed before training and is terminal; see DIAGNOSIS.md.
