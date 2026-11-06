"""Opt-in synchronized token probes for the fixed CP/non-CP reproduction.

No collectives or model tensor mutations are added. Each rank writes its own
token probes; the CPU reader restores global token/head order afterwards.
"""
import functools
import inspect
import os
from pathlib import Path

import torch

PROBES = list(range(16)) + [31, 63, 127, 255, 511, 512, 1023, 2047, 4095]


def install():
    from sglang.srt.models import deepseek_v2, deepseek_v4
    from sglang.srt.layers.cp.utils import is_cp_active
    from sglang.srt.layers.engram import Engram
    from sglang.srt.runtime_context import get_parallel

    root = Path(os.environ['NORTHGATE_CP_DIAGNOSTICS'])
    root.mkdir(parents=True, exist_ok=True)
    parallel = get_parallel()
    rank = parallel.tp_rank

    def record(label, value, batch, layout='local', all_ranks=False):
        if not torch.is_tensor(value) or value.ndim == 0:
            return
        if not batch.forward_mode.is_extend() or len(batch.input_ids) != 4096:
            return
        cp = is_cp_active(batch)
        if not cp and rank != 0 and not all_ranks:
            return
        path = root / f'{label}-rank{rank}.pt'
        if path.exists():
            return
        tokens = [t for t in PROBES if t < len(batch.input_ids)]
        if cp and layout == 'local':
            tokens = [t for t in tokens if t % parallel.attn_cp_size == parallel.attn_cp_rank]
            rows = [t // parallel.attn_cp_size for t in tokens]
        elif cp and layout == 'rank-major':
            local = value.shape[0] // parallel.attn_cp_size
            rows = [(t % parallel.attn_cp_size) * local + t // parallel.attn_cp_size for t in tokens]
        else:
            rows = tokens
        if not rows:
            return
        assert max(rows) < value.shape[0], (label, value.shape, rows)
        selected = value[torch.tensor(rows, device=value.device)]
        if selected.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            selected = selected.float()
        selected = selected.detach().cpu().clone()
        torch.save({'tokens': tokens, 'value': selected,
                    'shape': list(value.shape), 'cp_size': parallel.attn_cp_size,
                    'attn_tp_size': parallel.attn_tp_size, 'rank': rank}, path)

    def wrap(cls, name, before=None, after=None):
        original = getattr(cls, name)
        signature = inspect.signature(original)

        @functools.wraps(original)
        def traced(self, *args, **kwargs):
            bound = signature.bind(self, *args, **kwargs)
            bound.apply_defaults()
            batch = bound.arguments.get('forward_batch')
            if batch is None:
                return original(self, *args, **kwargs)
            if before:
                before(self, bound.arguments, batch)
            result = original(self, *args, **kwargs)
            if after:
                after(self, bound.arguments, batch, result)
            return result
        setattr(cls, name, traced)

    wrap(deepseek_v4.MQALayer, 'forward',
         before=lambda s, a, b: record(f'l{s.layer_id:02d}-attn-input', a['x'], b))
    wrap(deepseek_v4.MQALayer, '_forward_prepare',
         after=lambda s, a, b, out: record(f'l{s.layer_id:02d}-q', out[0], b, all_ranks=True))
    wrap(deepseek_v2.DeepseekV2MoE, 'forward',
         before=lambda s, a, b: record(f'l{s.layer_id:02d}-moe-input', a['hidden_states'], b,
                                      layout='rank-major'))
    wrap(Engram, 'forward',
         before=lambda s, a, b: (
             record(f'engram{s.layer_hash_index}-input', a['x'], b),
             record(f'engram{s.layer_hash_index}-hash', a['hash_ids'], b)),
         after=lambda s, a, b, out: record(f'engram{s.layer_hash_index}-output', out, b))

    if rank == 0:
        import shutil
        import sglang
        source = Path(sglang.__file__).parent
        for relative in ['srt/models/deepseek_v4_boundary.py',
                         'srt/layers/layer_boundary/ops.py',
                         'srt/layers/layer_boundary/residual/batch.py',
                         'srt/layers/attention/deepseek_v4_backend.py',
                         'srt/models/deepseek_v4.py',
                         'srt/layers/moe/mega_moe.py']:
            target = root / 'source' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if (source / relative).is_file():
                shutil.copy2(source / relative, target)
    print(f'[DEBUG-cp-r5] synchronized token probes installed rank={rank}', flush=True)
