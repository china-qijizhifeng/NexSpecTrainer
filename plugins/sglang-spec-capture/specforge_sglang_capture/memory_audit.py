"""Opt-in probe diagnostics for allocation boundaries, without tensor copies."""

import functools
import json


def install():
    import torch
    from sglang.srt.layers.engram import Engram
    from sglang.srt.models.deepseek_v2 import DeepseekV2MoE

    for cls in (Engram, DeepseekV2MoE):
        original = cls.forward

        def wrap(original, name):
            @functools.wraps(original)
            def traced(self, *args, **kwargs):
                def record(stage):
                    print("SPECFORGE_MEMORY " + json.dumps({
                        "module": name, "stage": stage,
                        "layer": getattr(self, "layer_id", getattr(self, "layer_hash_index", None)),
                        "device": torch.cuda.current_device(),
                        "allocated": torch.cuda.memory_allocated(),
                        "reserved": torch.cuda.memory_reserved(),
                        "peak": torch.cuda.max_memory_allocated(),
                    }), flush=True)

                record("before")
                out = original(self, *args, **kwargs)
                record("after")
                return out

            return traced

        cls.forward = wrap(original, cls.__name__)
