"""Pinned run policy; controls deterministic settings, not custom-kernel guarantees."""

from dataclasses import replace
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


def install():
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    original = FSDP.__init__

    def initialize(self, *args, **kwargs):
        precision = kwargs.get("mixed_precision")
        if precision is not None:
            kwargs["mixed_precision"] = replace(precision, reduce_dtype=torch.float32)
        return original(self, *args, **kwargs)

    FSDP.__init__ = initialize
