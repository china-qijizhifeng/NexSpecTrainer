# AdamW peak LR 5e-4 comparison

User requested an additional run alongside 1e-4. Same baseline initialization, seed42, 1326 traces, 32H100, batch128, accumulation4, 118 steps, warmup3/cosine, all anchors/chunk64/fixedFP32 reduction, validation/save every25 steps. Only training parameter changed: learning_rate 0.0001 -> 0.0005. No resume. Independent output/cache paths and rendezvous ports29952/29953. Existing jobs preserved.
