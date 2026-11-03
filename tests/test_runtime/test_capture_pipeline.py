"""Bounded overlap, failure ownership, and checksum-free codec contracts."""

import io
import struct
import threading
import unittest
from contextlib import closing
from dataclasses import replace
from unittest import mock

import pytest
import torch
import zstandard

from specforge.data.feature_file import load_zstd_features
from specforge.data.wire_codec import (
    NO_HASH_WIRE_ENCODING,
    decode_wire_tensor,
    encode_wire_tensor,
)
from specforge.inference.capture_pipeline import capture_pipeline
from tests.test_scripts import test_prepare_server_hidden_states as _capture_fixture


def test_no_hash_wire_roundtrip_and_frames():
    bits = torch.arange(65536, dtype=torch.int32).to(torch.uint16)
    tensor = bits.view(torch.bfloat16).reshape(256, 256)
    with mock.patch("hashlib.sha256", side_effect=AssertionError("no hashing")):
        packet, meta = encode_wire_tensor(tensor, encoding=NO_HASH_WIRE_ENCODING)
        assert "sha256" not in meta
        restored = decode_wire_tensor(
            packet, meta, shape=tensor.shape, dtype="bfloat16"
        )
        assert torch.equal(restored.view(torch.uint16), tensor.view(torch.uint16))
        stream = io.BytesIO(bytes(packet.numpy()))
        assert stream.read(8) == b"NSACTZ02"
        header_size = struct.unpack("<I", stream.read(4))[0]
        stream.read(header_size)
        while size := stream.read(4):
            frame = stream.read(struct.unpack("<I", size)[0])
            assert not zstandard.get_frame_parameters(frame).has_checksum
        with pytest.raises(ValueError):
            decode_wire_tensor(packet[:-1], meta, shape=tensor.shape, dtype="bfloat16")
        with pytest.raises(ValueError):
            load_zstd_features(io.BytesIO(bytes(packet[:-1].numpy())))
        with pytest.raises(ValueError, match="does not match"):
            load_zstd_features(io.BytesIO(bytes(packet.numpy())), expected_version=1)
        with pytest.raises(ValueError, match="shape/dtype"):
            decode_wire_tensor(packet, meta, shape=(128, 512), dtype="bfloat16")


def test_overlap_is_bounded_and_close_discards_next():
    second_started = threading.Event()
    produced, consumed, discarded = [], [], []

    def produce(i):
        produced.append(i)
        if i == 1:
            second_started.set()
        return i

    def consume(i, value):
        assert i == value
        if i == 0:
            assert second_started.wait(5), "next producer must overlap consumer"
            assert produced == [0, 1], "must not submit a third request"
        consumed.append(i)
        return i

    with closing(
        capture_pipeline(
            range(4),
            produce=produce,
            consume=consume,
            discard=lambda i, v: discarded.append(v),
            depth=2,
        )
    ) as results:
        assert next(results) == 0
    assert consumed == [0]
    assert discarded == [1]


def test_consumer_failure_drains_inflight_and_stops_submission():
    second_started = threading.Event()
    discarded = []

    def produce(i):
        if i == 1:
            second_started.set()
        assert i < 2
        return i

    def consume(i, value):
        assert second_started.wait(5)
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        list(
            capture_pipeline(
                range(4),
                produce=produce,
                consume=consume,
                discard=lambda i, v: discarded.append(v),
                depth=2,
            )
        )
    assert discarded == [1]


def test_input_failure_discards_current_before_consume():
    def inputs():
        yield 0
        raise ValueError("bad input")

    discarded = []
    consume = mock.Mock()
    with pytest.raises(ValueError, match="bad input"):
        list(
            capture_pipeline(
                inputs(),
                produce=lambda i: i,
                consume=consume,
                discard=lambda i, v: discarded.append(v),
                depth=2,
            )
        )
    consume.assert_not_called()
    assert discarded == [0]


def test_producer_failure_is_reported_without_leaking_completed_capture():
    consumed = []

    def produce(i):
        if i:
            raise RuntimeError("server failed")
        return i

    with pytest.raises(RuntimeError, match="server failed"):
        list(
            capture_pipeline(
                range(3),
                produce=produce,
                consume=lambda i, v: consumed.append(v),
                discard=lambda i, v: pytest.fail("already consumed"),
                depth=2,
            )
        )
    assert consumed == [0]


def test_real_sender_and_mooncake_transport_never_hash():
    import tests.test_runtime.test_capture_wire_compression as transport

    case = transport.TestWireCaptureTransport()
    case.setUp()
    try:
        with (
            mock.patch.dict(
                "os.environ",
                {"SGLANG_SPEC_CAPTURE_WIRE_ENCODING": NO_HASH_WIRE_ENCODING},
            ),
            mock.patch.object(transport, "WIRE_ENCODING", NO_HASH_WIRE_ENCODING),
            mock.patch("hashlib.sha256", side_effect=AssertionError("no hashing")),
        ):
            case.test_compressed_store_smaller_than_raw_buffers_and_logical_contract_unchanged()
    finally:
        case.doCleanups()


class TestPipelinedWriter(unittest.TestCase):
    """Exercise the real writer, adapter, transport and native file reader."""

    setUp = _capture_fixture.TestServerCaptureFiles.setUp
    writer = _capture_fixture.TestServerCaptureFiles.writer

    def test_two_records_roundtrip_without_hashes(self):
        tasks = [replace(self.task, task_id=f"episode-{i}") for i in range(3)]
        with mock.patch("hashlib.sha256", side_effect=AssertionError("no hashing")):
            with self.writer(zstd=True) as writer:
                results = list(
                    writer.write_many(
                        tasks, adapter=self.adapter, store=self.store, depth=2
                    )
                )
            assert [r["sample"] for r in results] == [t.task_id for t in tasks]
            assert not self.backend._d
            for task in tasks:
                raw = load_zstd_features(self.output / (task.task_id + ".ckpt.zst"))
                assert torch.equal(
                    raw["input_ids"], torch.tensor(task.payload["input_ids"])
                )
            with self.writer(zstd=True, resume=True) as writer:
                assert all(
                    r["status"] == "resumed"
                    for r in writer.write_many(
                        tasks, adapter=self.adapter, store=self.store, depth=2
                    )
                )

    def test_disk_failure_reclaims_both_captures(self):
        tasks = [replace(self.task, task_id=f"episode-{i}") for i in range(3)]
        with self.writer(zstd=True) as writer:
            with mock.patch(
                "specforge.inference.server_capture_writer.save_zstd_features",
                side_effect=OSError("disk full"),
            ):
                with pytest.raises(OSError, match="disk full"):
                    list(
                        writer.write_many(
                            tasks, adapter=self.adapter, store=self.store, depth=2
                        )
                    )
        assert not self.backend._d
        assert not list(self.output.glob("*.partial"))

    def test_pipeline_reserves_both_source_objects_without_compression_credit(self):
        from scripts.prepare_server_hidden_states import check_capacity

        options = dict(
            capture=self.capture,
            layout=self.layout,
            client_bytes=8 << 20,
            wire_encoding=NO_HASH_WIRE_ENCODING,
            pipeline_depth=2,
        )
        with pytest.raises(ValueError, match="pipeline needs"):
            check_capacity(4, server_bytes=8 << 20, **options)
        assert check_capacity(4, server_bytes=16 << 20, **options) > 0

    def test_lost_http_response_reclaims_unadopted_server_writes(self):
        original = self.adapter.post_fn

        def lost_response(*args, **kwargs):
            original(*args, **kwargs)
            assert self.backend._d
            raise ConnectionError("response lost")

        self.adapter.post_fn = lost_response
        with self.writer(zstd=True) as writer:
            with pytest.raises(ConnectionError, match="response lost"):
                list(
                    writer.write_many(
                        [self.task], adapter=self.adapter, store=self.store, depth=2
                    )
                )
        assert not self.backend._d
        assert not list(self.output.glob("*.partial"))
