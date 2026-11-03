"""Bound capture calls across independent servers before their refs exist."""

import os
import tempfile
import threading
import time
import unittest

from specforge.runtime.data_plane.streaming_ref_channel import StreamingRefChannel
from tests.test_runtime.test_disagg_multiserver import (
    _adapter,
    _build,
    _FailingPublishChannel,
    _prompts,
    _TrackingMooncakeFeatureStore,
)
from tests.test_runtime.test_server_capture import (
    _FakeMooncakeStore,
    _StubCaptureServer,
)


class CaptureBufferTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="capture_buffer_")
        self.addCleanup(self.directory.cleanup)
        self.backend = _FakeMooncakeStore()
        self.stub = _StubCaptureServer(self.backend)
        self.store = _TrackingMooncakeFeatureStore(store=self.backend, store_id="run0")
        self.channel = StreamingRefChannel(
            os.path.join(self.directory.name, "refs.jsonl")
        )

    def build(self, adapters, **kwargs):
        return _build(
            adapters,
            _prompts(8),
            self.store,
            self.channel,
            lease=1,
            in_flight_high_watermark=4,
            in_flight_low_watermark=4,
            resident_high_watermark_bytes=8192,
            resident_low_watermark_bytes=8192,
            capture_max_sample_bytes=2048,
            **kwargs,
        )

    def test_four_servers_reserve_before_capture_and_resume_after_ack(self):
        lock = threading.Lock()
        gate = threading.Event()
        first_wave = threading.Event()
        calls = 0

        def post(url, json_body, timeout):
            nonlocal calls
            with lock:
                calls += 1
                if calls == 4:
                    first_wave.set()
            if not gate.wait(5):
                raise TimeoutError("test did not release capture")
            # The fake writer uses process-global torch RNG state.
            with lock:
                return self.stub(url, json_body, timeout)

        adapters = [_adapter(self.store, post, f"http://s{i}:30000") for i in range(4)]
        _workers, drive = self.build(adapters, consumer_quantum=4)
        outcome = {}
        stop = threading.Event()

        def run():
            try:
                outcome["produced"] = drive(should_stop=stop.is_set)
            except BaseException as exc:
                outcome["error"] = exc

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        try:
            self.assertTrue(first_wave.wait(5), "four captures never started")
            snapshot = drive.flow_control.snapshot(in_flight_refs=0)
            self.assertEqual(snapshot["capture_reserved_refs"], 4)
            self.assertEqual(snapshot["capture_reserved_bytes"], 8192)
            self.assertEqual(self.channel.published, 0)
            with lock:
                self.assertEqual(calls, 4)
            gate.set()
            reader = StreamingRefChannel(self.channel.path)
            for _wave in range(2):
                refs = []
                deadline = time.monotonic() + 5
                while len(refs) < 4 and time.monotonic() < deadline:
                    refs.extend(reader.poll())
                    time.sleep(0.001)
                self.assertEqual(len(refs), 4)
                self.assertLessEqual(sum(ref.estimated_bytes for ref in refs), 8192)
                for ref in refs:
                    self.store.abort(ref.sample_id, reason="test-consumer-cleanup")
                reader.mark_consumed(
                    4, num_bytes=sum(ref.estimated_bytes for ref in refs)
                )
            thread.join(5)
            self.assertFalse(thread.is_alive(), "producer did not resume after ack")
            self.assertNotIn("error", outcome)
            self.assertEqual(outcome["produced"], 8)
            self.assertEqual(
                len(
                    set(
                        ref.sample_id
                        for ref in StreamingRefChannel(self.channel.path).poll()
                    )
                ),
                8,
            )
            self.assertEqual(self.backend._d, {})
            self.assertEqual(
                drive.flow_control.snapshot(in_flight_refs=0)["capture_reserved_refs"],
                0,
            )
        finally:
            stop.set()
            gate.set()
            thread.join(5)

    def test_retryable_capture_failure_releases_reserved_capacity(self):
        calls = 0

        def post(url, json_body, timeout):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OSError("injected capture failure")
            return self.stub(url, json_body, timeout)

        # Stop after the first complete buffer; failed calls must not leak slots.
        _workers, drive = self.build([_adapter(self.store, post)])
        self.assertEqual(drive(should_stop=lambda: self.channel.published >= 4), 4)
        self.assertGreater(calls, 4)
        self.assertEqual(
            drive.flow_control.snapshot(in_flight_refs=4)["capture_reserved_refs"], 0
        )

    def test_publication_failure_drains_concurrent_reservations(self):
        self.channel = _FailingPublishChannel(self.channel.path, fail_after=0)
        _workers, drive = self.build(
            [_adapter(self.store, self.stub)], producer_concurrency=2
        )
        with self.assertRaisesRegex(OSError, "after 0 durable ref"):
            drive()
        self.assertEqual(self.backend._d, {})
        self.assertEqual(
            drive.flow_control.snapshot(in_flight_refs=0)["capture_reserved_refs"], 0
        )

    def test_partial_publication_closes_peer_admission_before_slow_drain(self):
        failed = threading.Event()
        slow_started = threading.Event()
        release = threading.Event()
        peer_entered = threading.Event()
        peer_checked = threading.Event()

        class TransientChannel(StreamingRefChannel):
            def publish(self, ref):
                if self.published == 1 and not failed.is_set():
                    failed.set()
                    raise OSError("transient failure after durable prefix")
                return super().publish(ref)

            def in_flight_remote(self):
                if threading.current_thread().name == "drive-rollout-1":
                    if not failed.wait(5):
                        raise TimeoutError("first worker did not fail publication")
                    peer_checked.set()
                return super().in_flight_remote()

        self.channel = TransientChannel(self.channel.path)
        lock = threading.Lock()
        calls = 0

        def first_post(url, json_body, timeout):
            nonlocal calls
            with lock:
                index = calls
                calls += 1
            if index == 0:
                if not slow_started.wait(5):
                    raise TimeoutError("second request never started")
            else:
                slow_started.set()
                if not release.wait(5):
                    raise TimeoutError("test did not release slow request")
            with lock:
                return self.stub(url, json_body, timeout)

        def peer_post(url, json_body, timeout):
            peer_entered.set()
            if not release.wait(5):
                raise TimeoutError("test did not release peer request")
            with lock:
                return self.stub(url, json_body, timeout)

        _workers, drive = _build(
            [_adapter(self.store, first_post), _adapter(self.store, peer_post)],
            _prompts(10),
            self.store,
            self.channel,
            lease=2,
            producer_concurrency=2,
            in_flight_high_watermark=32,
            in_flight_low_watermark=32,
            resident_high_watermark_bytes=8192,
            resident_low_watermark_bytes=8192,
            capture_max_sample_bytes=2048,
        )
        outcome = {}

        def run():
            try:
                drive()
            except BaseException as exc:
                outcome["error"] = exc

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        try:
            self.assertTrue(
                peer_checked.wait(5), "peer never observed publication failure"
            )
            self.assertFalse(
                peer_entered.wait(0.5), "peer captured while failed worker drained"
            )
            self.assertTrue(thread.is_alive(), "slow request was not still draining")
        finally:
            release.set()
            thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertIsInstance(outcome.get("error"), OSError)
        self.assertIn("after durable prefix", str(outcome["error"]))
        self.assertEqual(self.channel.published, 1)
        self.assertEqual(
            drive.flow_control.snapshot(in_flight_refs=1)["capture_reserved_refs"], 0
        )

    def test_oversized_capture_is_aborted_before_publication(self):
        _workers, drive = _build(
            [_adapter(self.store, self.stub)],
            _prompts(1),
            self.store,
            self.channel,
            lease=1,
            resident_high_watermark_bytes=1,
            resident_low_watermark_bytes=1,
            capture_max_sample_bytes=1,
        )
        with self.assertRaisesRegex(MemoryError, "exceeds capture_max_sample_bytes"):
            drive()
        self.assertEqual(self.channel.published, 0)
        self.assertEqual(self.backend._d, {})
        self.assertIn("exceeds capture_max_sample_bytes", self.channel.failure())
        self.assertEqual(
            drive.flow_control.snapshot(in_flight_refs=0)["capture_reserved_refs"], 0
        )
