"""Regression checks against the actual generated pilot files and native schema."""
import copy
import os
from pathlib import Path
import unittest

from pydantic import ValidationError
from specforge.config.schema import Config


class PilotConfigTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(os.environ['RUN_DIR'])
        self.initial = Config.from_file(str(self.root / 'initial.yaml'))
        self.resume = Config.from_file(str(self.root / 'config.yaml'))

    def test_initial_and_resume_share_schedule_and_topology(self):
        for config in (self.initial, self.resume):
            config.validate_world_size(8)
            self.assertEqual(config.training.total_steps, 3)
            self.assertEqual(config.training.eval_interval, 0)
            self.assertEqual(config.data.eval_hidden_states_path, '')
            self.assertIsNone(config.model.draft_checkpoint_path)
        self.assertIsNone(self.initial.training.resume_from)
        self.assertEqual(self.initial.training.max_steps, 1)
        self.assertEqual(self.resume.training.max_steps, 3)
        self.assertEqual(self.resume.training.resume_from,
                         str(self.root / 'training/yicloud-denovo-pilot-step1'))

    def test_eval_source_and_interval_must_be_paired(self):
        for source, interval in [('unused-validation', 0), ('', 1)]:
            with self.subTest(source=source, interval=interval):
                raw = copy.deepcopy(self.initial.model_dump())
                raw['data']['eval_hidden_states_path'] = source
                raw['training']['eval_interval'] = interval
                with self.assertRaisesRegex(ValidationError, 'configured together'):
                    Config.model_validate(raw)

    def test_resume_rejects_weights_only_warm_start(self):
        raw = self.resume.model_dump()
        raw['model']['draft_checkpoint_path'] = 'unused-weights'
        with self.assertRaisesRegex(ValidationError, 'mutually exclusive'):
            Config.model_validate(raw)


if __name__ == '__main__':
    unittest.main()
