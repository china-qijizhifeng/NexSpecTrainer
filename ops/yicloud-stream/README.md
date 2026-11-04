# Full DeNovoSWE capture/training overlap

This run consumes the existing immutable Flash-1017 full-trajectory captures.
It does not change or truncate the target prefill, loss masks, data split,
model, optimizer, 4K slices, or the 796-step / 101888-window epoch.

The publisher admits only atomic per-trajectory `sliced/*.json` receipts after
checking them against the pinned 4429-trajectory input plan. Each consumer also
checks the compressed feature SHA256 before using the native feature reader.
No `.partial` file is eligible. Model execution and NFC remain native.

Start at 16384 available training windows; retain an 8192-window reservoir
until capture succeeds. Each atomic published block is a complete global
optimizer update (8 ranks * 16 microbatches). Shuffle within the available
reservoir and persist each block exactly once. This reduces adjacency but is
NOT a uniform global shuffle: the unchanged capture workers process long
trajectories first. Arrival-dependent ordering is an explicit experiment
change. Immutable block files, not the seed alone, define replay order.

The reader exposes the full fixed slot count, so early startup does not shrink
the epoch or learning-rate schedule. DP partitions slots without a second
shuffle. The adapter preserves native loader prefetch and seek. Checkpoints
record the stream contract and its original schedule path; resumed consumers
must retain that schedule. A producer restart refuses to rewrite existing
blocks; recovery requires an explicit audit, never a new random ordering.

All original windows plus the same 16 explicit padding duplicates are consumed
once. After successful capture reconciliation and all four partition results,
the publisher flushes the reservoir and final padding block, then publishes
all 5896 validation windows. Full validation runs at step 796. NFC checkpoints
still save every 100 steps and on completion; full restore acceptance is a
separate follow-up, not claimed by this launcher.

Publisher/trainer failure aborts this training job and preserves evidence and
all source features. It never stops the independent capture job. A shortage of
ready blocks waits (up to the job's 72-hour cap), without repeating old data.
The separate capture reconciler must remain alive; terminal API reconciliation
is required before the last buffered data is released.

`test_stream.py` exercises delayed arrivals, reserve and final-tail coverage,
producer failure, uncommitted receipts, native codec/loader, eight DP partitions,
seek/replay, and rejection of corrupted feature bytes. The same tests run in
the production training image before the full training subprocess.
