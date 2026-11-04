"""CPU native NFC -> native quantized MTP overlay; target and NFC stay read-only."""

import gc
import json
import platform
import resource
import sys
from contextlib import ExitStack

import torch
from common import (
    ROOT,
    TARGET,
    TRAIN,
    checkpoint_metadata,
    phase,
    sha,
    verify_execution,
    write,
)
from safetensors import safe_open

from specforge.export.checkpoint_io import resolve_training_state
from specforge.modeling.draft.deepseek_v41.export import export_native


def main():
    verify_execution()
    assert json.loads((TRAIN / "stage1-final-acceptance.json").read_text())["passed"]
    complete = json.loads((TARGET / ".download/complete.json").read_text())
    assert (
        complete["revision"] == "1bae0e1088993b9b82de08f59de5be5a34e42170"
        and complete["weight_sha256_verified"]
    )
    import google.protobuf
    from google.protobuf.internal.api_implementation import Type

    write(
        ROOT / "export-environment.json",
        {
            "python": sys.version,
            "torch": torch.__version__,
            "protobuf": google.protobuf.__version__,
            "protobuf_backend": Type(),
            "hostname": platform.node(),
            "cuda_available": torch.cuda.is_available(),
        },
    )
    receipts = []
    for step in (25, 118):
        checkpoint = TRAIN / "training" / f"denovo-v41-specforge-stage1-step{step}"
        phase("export_loading", step=step)
        before = checkpoint_metadata(checkpoint, step)
        state = resolve_training_state(str(checkpoint))["draft_state_dict"]
        assert state and all(
            torch.isfinite(t).all().item()
            for t in state.values()
            if t.is_floating_point()
        )
        phase("export_quantizing", step=step)
        output = ROOT / f"export-step{step}"
        export_native(state, TARGET, output)
        original = json.loads((TARGET / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        exported = json.loads((output / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        assert set(original) == set(exported)
        assert (output / "config.json").resolve() == (TARGET / "config.json").resolve()
        overlays = sorted({v for k, v in exported.items() if k.startswith("mtp.")})
        assert overlays and all(n.startswith("nexspec-mtp-") for n in overlays)
        for key, name in original.items():
            if not key.startswith("mtp."):
                assert exported[key] == name and (output / name).is_symlink()
                assert (output / name).resolve() == (TARGET / name).resolve()
        count = 0
        with ExitStack() as stack:
            src = {
                n: stack.enter_context(
                    safe_open(TARGET / n, framework="pt", device="cpu")
                )
                for n in {v for k, v in original.items() if k.startswith("mtp.")}
            }
            dst = {
                n: stack.enter_context(
                    safe_open(output / n, framework="pt", device="cpu")
                )
                for n in overlays
            }
            for key, name in exported.items():
                if not key.startswith("mtp."):
                    continue
                old = src[original[key]].get_tensor(key)
                new = dst[name].get_tensor(key)
                assert old.shape == new.shape and old.dtype == new.dtype, key
                if new.is_floating_point():
                    assert all(
                        torch.isfinite(c.float()).all().item()
                        for c in new.reshape(-1).split(4 << 20)
                    ), key
                count += 1
            for stage in range(3):
                key = f"mtp.{stage}.ffn.gate.weight"
                assert torch.equal(
                    dst[exported[key]].get_tensor(key),
                    state[key].to(dst[exported[key]].get_tensor(key).dtype),
                ), key
        assert before == checkpoint_metadata(checkpoint, step)
        receipt = {
            "passed": True,
            "step": step,
            "checkpoint": str(checkpoint),
            "checkpoint_metadata_sha256": before,
            "mtp_tensors": count,
            "target_non_mtp_shared_unchanged": True,
            "overlays": {
                n: {"bytes": (output / n).stat().st_size, "sha256": sha(output / n)}
                for n in overlays
            },
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        }
        write(ROOT / f"export-step{step}-verification.json", receipt)
        receipts.append(receipt)
        del state
        gc.collect()
    write(ROOT / "export-result.json", {"passed": True, "exports": receipts})
    phase("export_complete")


if __name__ == "__main__":
    main()
