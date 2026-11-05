"""Reject unconfigured CP and scientifically material hidden-state drift."""
import unittest

import torch

from ops.northgate_capture.cp_acceptance import compare_features
from ops.northgate_capture.wait_server import validate_cp


class TestCPAcceptance(unittest.TestCase):
    def test_reference_gate_rejects_drift_and_wrong_mask(self):
        reference = {'input_ids': torch.tensor([[1, 2]]),
                     'loss_mask': torch.tensor([[0, 1]]),
                     'hidden_states': torch.tensor([[[1., 2.], [3., 4.]]]),
                     'target_last_hidden_states': torch.tensor([[[2., 1.], [4., 3.]]])}
        self.assertTrue(compare_features(reference, reference)['passed'])
        candidate = {key: value.clone() for key, value in reference.items()}
        candidate['hidden_states'][0, 0, 0] += 1
        self.assertFalse(compare_features(candidate, reference)['passed'])
        candidate['loss_mask'][0, 0] = 1
        with self.assertRaisesRegex(ValueError, 'loss_mask mismatch'):
            compare_features(candidate, reference)

    def test_resolved_server_must_confirm_full_cp8_capture(self):
        info = {'enable_prefill_cp': True, 'attn_cp_size': 8,
                'cp_strategy': 'interleave', 'language_model_only': True,
                'chunked_prefill_size': -1, 'return_hidden_states_mode': 'full',
                'enable_decoder_swa_bounded_replay': False}
        validate_cp(info, 8)
        validate_cp({'server_args': info}, 8)
        for key, value in [('attn_cp_size', 1), ('language_model_only', False),
                           ('chunked_prefill_size', 16384),
                           ('return_hidden_states_mode', 'last')]:
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, key):
                validate_cp(dict(info, **{key: value}), 8)
