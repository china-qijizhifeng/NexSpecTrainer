# r1 gate failure and corrected oracle

r1 stopped before full-model training: the inherited AdamW cross-schedule master-update relative-L2 threshold (10%) failed on three GPU ranks (13.0–13.9%). This is retained as a failed equivalence experiment, not relabeled as passed.

Static inspection: whole-graph and streamed backward round BF16 gradients differently; Muon applies a nonlinear 8+2 Newton–Schulz map. The original PR explicitly uses identical reduced gradients for its optimizer differential. A small allowed input-gradient difference does not imply a fixed update-difference bound. The r1 test incorrectly reused AdamW's bound as a Muon correctness oracle.

Eight-process CPU instrumentation reproduced the behavior. First-update projection gradient relative L2 was 0.322%; complete-matrix NS direction difference was about 50%. Replaying each path's captured gradients through an unsharded MuonAdamWOptimizer reproduced every actual CPU master bitwise (diagnostic-optimizer-replay.json). The projection has small singular values; no corruption or shard mapping error is needed to produce this difference. This CPU diagnosis does not prove the full-model training trajectories equivalent or explain every custom-kernel rounding source.

r2 corrects the oracle rather than increasing the old threshold: each policy's actual reduced gradients and current masters are reconstructed, then an unsharded Muon optimizer with persistent state receives exactly those gradients. Every update must match the corresponding shard within rtol2e-6/atol2e-7, with update relative L2 <1%; reference execution must not mutate actual masters. Loss/global coverage and the original 3% gradient check remain. Cross-schedule master differences remain recorded with equivalence_accepted=false and no equivalence claim. The original failed result is preserved under r1.

Production source and train.yaml optimizer settings are unchanged. The full native training still requires a finite real update and following backward, complete anchor coverage and unchanged frozen targets.
