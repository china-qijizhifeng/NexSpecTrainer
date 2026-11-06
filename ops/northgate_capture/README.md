# V4.1 CP capture numerics

[CP_NUMERICS.md](CP_NUMERICS.md) describes the opt-in Hopper TP8/CP8 numerical
compatibility mode, its runtime contract, and real-GPU regression commands.
Enable it with `NORTHGATE_CP_NUMERICAL_PARITY=1` in the serving environment.
Keep this checkout on `PYTHONPATH`: these operational modules are not included
in the published SpecForge wheel. The default capture path is unchanged.

The mode restores TP arithmetic order using current request operands and KV;
it never reads reference activations. It adds communication and workspace.
The recorded acceptance covers specific inputs through 131072 tokens, not the
full dataset or every long-context configuration.

The [evidence](evidence/cp_numerics) contains numerical metrics, tensor
fingerprints, regression results, source identity and resource completion.
The `.pt` operator fixtures and multi-GiB capture files remain in the recorded
experiment storage; they are not embedded in the repository.
