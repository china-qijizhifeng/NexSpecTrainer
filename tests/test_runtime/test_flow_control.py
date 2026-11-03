# coding=utf-8
"""Canonical producer flow-control policy (CPU-only)."""

import unittest

from specforge.runtime.control_plane.flow_control import (
    FlowControlLimits,
    ProducerFlowControl,
)


class FlowControlTest(unittest.TestCase):
    def test_reference_and_byte_hysteresis_share_one_pause_latch(self):
        policy = ProducerFlowControl(
            FlowControlLimits(
                high_watermark_refs=10,
                low_watermark_refs=4,
                high_watermark_bytes=100,
                low_watermark_bytes=40,
            )
        )
        self.assertFalse(policy.should_pause(in_flight_refs=9, resident_bytes=99))
        self.assertTrue(policy.should_pause(in_flight_refs=10, resident_bytes=20))
        self.assertTrue(policy.should_pause(in_flight_refs=5, resident_bytes=20))
        self.assertFalse(policy.should_pause(in_flight_refs=4, resident_bytes=40))
        self.assertTrue(policy.should_pause(in_flight_refs=1, resident_bytes=100))
        self.assertTrue(policy.should_pause(in_flight_refs=1, resident_bytes=50))
        self.assertFalse(policy.should_pause(in_flight_refs=1, resident_bytes=40))

        snapshot = policy.snapshot(in_flight_refs=1, resident_bytes=40)
        self.assertEqual(snapshot["pause_transitions"], 2)
        self.assertEqual(snapshot["resume_transitions"], 2)
        self.assertGreater(snapshot["wait_checks"], 0)

    def test_prompt_lease_is_capped_per_worker(self):
        policy = ProducerFlowControl(FlowControlLimits(max_prompt_lease_per_worker=3))
        self.assertEqual(policy.prompt_lease(8), 3)
        self.assertEqual(policy.prompt_lease(2), 2)

    def test_invalid_watermarks_are_rejected(self):
        with self.assertRaises(ValueError):
            FlowControlLimits(high_watermark_refs=4, low_watermark_refs=5)
        with self.assertRaises(ValueError):
            FlowControlLimits(low_watermark_bytes=10)
        with self.assertRaises(ValueError):
            FlowControlLimits(
                high_watermark_bytes=10,
                low_watermark_bytes=11,
            )

    def test_capture_reservations_share_byte_capacity_and_shrink_leases(self):
        policy = ProducerFlowControl(
            FlowControlLimits(
                high_watermark_refs=8,
                low_watermark_refs=4,
                high_watermark_bytes=50,
                low_watermark_bytes=20,
                capture_max_sample_bytes=10,
            )
        )
        policy.limits.validate_capture_window(2)
        self.assertEqual(
            policy.reserve_prompt_lease(4, in_flight_refs=1, resident_bytes=5), 4
        )
        self.assertEqual(
            policy.reserve_prompt_lease(4, in_flight_refs=1, resident_bytes=5), 0
        )
        self.assertEqual(
            policy.snapshot(in_flight_refs=1)["capture_reserved_bytes"], 40
        )
        policy.release_prompt_lease(4)
        # Publication replaced the reservation with four actual ten-byte refs.
        self.assertTrue(policy.should_pause(in_flight_refs=5, resident_bytes=45))
        self.assertEqual(
            policy.reserve_prompt_lease(8, in_flight_refs=2, resident_bytes=20), 3
        )
        policy.release_prompt_lease(3)
        with self.assertRaisesRegex(ValueError, "released more than once"):
            policy.release_prompt_lease(1)

    def test_capture_bound_requires_space_for_a_full_dispatch(self):
        with self.assertRaisesRegex(ValueError, "requires a byte high watermark"):
            FlowControlLimits(capture_max_sample_bytes=10)
        for high, low, level in ((39, 39, "high"), (50, 39, "low")):
            with (
                self.subTest(level=level),
                self.assertRaisesRegex(
                    ValueError, f"{level} byte watermark.*dispatch quantum 4"
                ),
            ):
                FlowControlLimits(
                    high_watermark_bytes=high,
                    low_watermark_bytes=low,
                    capture_max_sample_bytes=10,
                ).validate_capture_window(4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
