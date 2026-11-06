import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from ops.northgate_capture.resident_control import ResidentControl


class ResidentControlTests(unittest.TestCase):
    def test_hot_commands_keep_the_same_model_and_wait_until_idle(self):
        with TemporaryDirectory() as folder, patch('torch.cuda.synchronize'):
            root = Path(folder).resolve()
            (root / 'modules').mkdir()
            model = torch.nn.Linear(2, 2)
            scheduler = SimpleNamespace(is_fully_idle=lambda: False,
                tp_worker=SimpleNamespace(model_runner=SimpleNamespace(model=model)))
            control = ResidentControl.__new__(ResidentControl)
            control.scheduler, control.rank, control.root = scheduler, 0, root
            control.last, control.state = None, {}
            original = [(p.data_ptr(), p.clone()) for p in model.parameters()]
            for sequence in ['first', 'second']:
                source = b'def run(scheduler, output, state):\n state["calls"] = state.get("calls", 0) + 1\n return state["calls"]\n'
                (root / 'modules' / f'{sequence}.py').write_bytes(source)
                (root / 'command.json').write_text(json.dumps({'sequence': sequence,
                    'source': f'modules/{sequence}.py', 'sha256': hashlib.sha256(source).hexdigest()}))
                scheduler.is_fully_idle = lambda: False
                control.poll()
                self.assertFalse((root / 'commands' / sequence).exists())
                scheduler.is_fully_idle = lambda: True
                control.poll()
                control.poll()
                result = json.loads((root / 'commands' / sequence / 'rank0.json').read_text())
                self.assertTrue(result['success'], result)
                self.assertTrue(result['parameter_pointers_versions_unchanged'])
            self.assertEqual(control.state['calls'], 2)
            for (pointer, expected), actual in zip(original, model.parameters()):
                self.assertEqual(pointer, actual.data_ptr())
                self.assertTrue(torch.equal(expected, actual))

    def test_bad_digest_never_executes_source(self):
        with TemporaryDirectory() as folder, patch('torch.cuda.synchronize'):
            root = Path(folder).resolve()
            (root / 'modules').mkdir()
            (root / 'modules' / 'bad.py').write_text('raise RuntimeError("EXECUTED")')
            (root / 'command.json').write_text(json.dumps({'sequence': 'bad',
                'source': 'modules/bad.py', 'sha256': '0' * 64}))
            control = ResidentControl.__new__(ResidentControl)
            control.scheduler = SimpleNamespace(is_fully_idle=lambda: True)
            control.rank, control.root, control.last, control.state = 0, root, None, {}
            control.poll()
            result = json.loads((root / 'commands/bad/rank0.json').read_text())
            self.assertFalse(result['success'])
            self.assertNotIn('EXECUTED', result['error'])


if __name__ == '__main__':
    unittest.main()
