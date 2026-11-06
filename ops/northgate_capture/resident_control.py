"""Local opt-in debug commands, executed on scheduler ranks between requests.

Commands and immutable source live in the run directory. No public execution
endpoint, extra model collective, weight reload, or asynchronous model mutation.
"""
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import torch


class ResidentControl:
    def __init__(self, scheduler):
        from sglang.srt.runtime_context import get_parallel

        self.scheduler = scheduler
        self.rank = get_parallel().tp_rank
        self.root = Path(os.environ['NORTHGATE_RESIDENT_CONTROL']).resolve()
        self.last = None
        self.state = {}

    def poll(self):
        command = self.root / 'command.json'
        if not command.exists():
            return
        spec = json.loads(command.read_text())
        sequence = spec['sequence']
        if sequence == self.last:
            return
        scheduler = self.scheduler
        if not scheduler.is_fully_idle():
            return
        if not sequence or not all(c.isalnum() or c in '-_' for c in sequence):
            raise ValueError('invalid resident command sequence')
        self.last = sequence
        output = self.root / 'commands' / sequence
        output.mkdir(parents=True, exist_ok=True)
        result = {'sequence': sequence, 'rank': self.rank, 'started': time.time()}
        try:
            source = (self.root / spec['source']).resolve()
            assert source.is_relative_to(self.root / 'modules')
            data = source.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            assert digest == spec['sha256'], (digest, spec['sha256'])
            torch.cuda.synchronize()
            runner = scheduler.tp_worker.model_runner
            before = [(n, p.data_ptr(), p._version) for n, p in runner.model.named_parameters()]
            namespace = {'__name__': f'northgate_probe_{sequence}', '__file__': str(source)}
            exec(compile(data, str(source), 'exec'), namespace)
            value = namespace['run'](scheduler, output, self.state)
            torch.cuda.synchronize()
            after = [(n, p.data_ptr(), p._version) for n, p in runner.model.named_parameters()]
            assert before == after, 'debug command changed parameter pointers or versions'
            result.update(success=True, source_sha256=digest, result=value,
                          parameter_pointers_versions_unchanged=True)
        except Exception:
            result.update(success=False, error=traceback.format_exc())
        result['finished'] = time.time()
        path = output / f'rank{self.rank}.json'
        temporary = path.with_suffix('.partial')
        temporary.write_text(json.dumps(result, indent=2) + '\n')
        temporary.replace(path)
