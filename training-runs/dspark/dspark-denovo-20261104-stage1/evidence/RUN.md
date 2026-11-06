# DeNovoSWE dSpark stage 1 — 2026-11-04

User authorized starting training after the 5.122 TB cleanup. This is a real, expanded first stage, not another three-step pilot and not a claim of full-dataset completion.

- Source: NexSpecTrainer `66f0cfd81443a6481232d8fdf7b4abaa7c1f63b7` (clean git archive).
- Target/official tokenizer/official encode_messages: DeepSeek-V4.1-Flash-1017 revision `1bae0e1088993b9b82de08f59de5be5a34e42170`; thinking, drop_thinking=False, effort75; no extra special tokens. Every token digest and supervised-token count matches the prior full audit.
- Dataset revision: `a9c33019adbcd00c93e4fdd47c9dc3e4b851d35f`; original deterministic instance split retained.
- Train: all 1,326 training trajectories <=65,536 tokens, 59,274,736 tokens; 15,053 supervised slices. Explicitly named 51 duplicate train slices pad the epoch to 15,104 slices, 118 updates at global batch128 (8 ranks × batch1 × accumulation16).
- Validation: shortest eligible complete trajectory in each of 17 disjoint validation instances; deterministic up to8 slices each, 136 fixed slices, 771,468 target-prefill tokens. Other validation instances are not covered by this stage. Test is unused.
- Remaining 2,862 train trajectories exceed the existing validated full64K prefill envelope and remain listed in deferred-train-manifest.json. No trajectory is truncated or silently skipped. Later stages require validating their larger full-prefill/memory/codec limits; they are not automatically submitted by this controller.
- Four independent TP8/EP8 servers, total32 H100, memory768Gi each; whole-trajectory prefill, chunked prefill disabled. Use already validated full64K allocator/settings and original target kernels. Sequentially persist one full record, split4096/overlap129, verify BF16 codec bit-for-bit, fsync slices/receipt, and release only that newly created temporary full record.
- Existing 8 train +1 val pilot data/checkpoints are untouched. Start stage1 from published original native MTP, not the completed three-step pilot scheduler.
- After all four workers succeed and every planned slice verifies, submit a single8 H100 trainer (640Gi host, shm288Gi). Learning rate1e-5, one epoch, warmup0.03, gamma4.0 (matching executed pilot), anchors32, objective chunks8, FULL_SHARD, CPU Adam masters/moments. Checkpoints and evaluation every25 updates and final.
- NFC native sharded pipeline only;32GiB/rank staging, complete model/FP32 masters/Adam/scheduler/RNG/consumption position. Preserve complete latest/best/final, max_checkpoints1. Initial conservative disk guard includes worst-case feature expansion+overlap and850GB for checkpoint rotation/headroom. Check during capture and before trainer.
- Draft positions retain SpecForge local slice convention; this is not a numerical equivalence claim to absolute-position full-draft training. Target features retain the full trajectory prefix.
- Single-use pipeline.py, exclusive submit-intent before each POST, exact owned identities, no automatic retry. A failed capture worker stops this controller's own four-worker job; no other resources are stopped. Completed self-owned jobs are recorded and removed.
- Checkpoint publication is distinct from restore verification. This first stage does not claim old pilot or new checkpoint restart/next-update acceptance until separately measured. Frozen embedding/head hashes and exclusion from optimizer/gradients are audited on all8 ranks around actual118 updates.

Run root: `/gpfs/users/linlizhi/dspark-denovo-specforge-stage1-20261104-r1`
Capture alias: `denovo-specforge-stage1-capture-20261104-r1`
Trainer alias: `denovo-specforge-stage1-train-20261104-r1`
Both use WeaverRSI / proj-32. This is the repository training-run skill's temporary environment; replace it with a verified pinned training image once published.
