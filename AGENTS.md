# NexSpecTrainer agent instructions

For training or evaluation operations, read `.agents/skills/use-nsp/SKILL.md`
and `docs/nsp.md` first. Use `nsp train` / `nsp eval` and the shared
`list/show/watch/logs/results/checkpoints/stop` commands for user workloads.
`python -m specforge.nsp.cli` is the equivalent source-checkout entrypoint.

Do not bypass nsp with direct `specforge train`, ad-hoc background processes,
or temporary training launch scripts. Low-level entries remain valid inside
nsp and in explicit differential acceptance tests against the original command.
If the current checkout lacks a required nsp capability, report or implement it
before launching; do not silently fall back.

Scheduler resource allocation remains platform-specific. On BP Dev, use
Northjob project `WeaverRSI` and verify its current queue before submission.
