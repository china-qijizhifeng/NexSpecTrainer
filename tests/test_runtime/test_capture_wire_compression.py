"""True sender codec -> raw pointer Mooncake bytes -> logical FeatureSpec tensors."""

import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "plugins/sglang-spec-capture")
)
from specforge_sglang_capture.sink import SpecCaptureSink, gpu_put_enabled

from scripts.prepare_server_hidden_states import check_capacity
from specforge.algorithms.builtin import builtin_algorithm_registry
from specforge.data.feature_file import load_zstd_features, save_zstd_features
from specforge.data.wire_codec import (
    WIRE_ENCODING,
    decode_wire_tensor,
    encode_wire_tensor,
    validate_wire_metadata,
)
from specforge.inference.adapters.server_capture import (
    ServerCaptureSchema,
    SGLangServerCaptureAdapter,
)
from specforge.inference.capture import CaptureConfig
from specforge.inference.server_capture_writer import ServerCaptureFileWriter
from specforge.runtime.contracts import FeatureSpec, PromptTask
from specforge.runtime.data_plane.mooncake_store import MooncakeFeatureStore
from tests.test_runtime.test_mooncake_store import _FakeMooncakeStore

ENV = {
    "MOONCAKE_PROTOCOL": "tcp",
    "SGLANG_SPEC_CAPTURE_GPU_PUT": "0",
    "SGLANG_SPEC_CAPTURE_WIRE_ENCODING": WIRE_ENCODING,
    "MOONCAKE_GLOBAL_SEGMENT_SIZE": str(1 << 20),
    "MOONCAKE_LOCAL_BUFFER_SIZE": str(1 << 20),
}


class TestWireCodec(unittest.TestCase):
    def test_exact_pr2_format_and_every_bf16_bitpattern(self):
        bits = torch.arange(65536, dtype=torch.int32).to(torch.uint16)
        tensor = bits.view(torch.bfloat16).reshape(1, 256, 256)
        encoded, meta = encode_wire_tensor(tensor)
        restored = decode_wire_tensor(
            encoded, meta, shape=tensor.shape, dtype="bfloat16"
        )
        self.assertTrue(
            torch.equal(restored.view(torch.uint16), bits.reshape(1, 256, 256))
        )
        stream = io.BytesIO()
        save_zstd_features({"tensor": tensor}, stream)
        self.assertEqual(bytes(encoded.numpy()), stream.getvalue())
        self.assertTrue(
            torch.equal(
                load_zstd_features(io.BytesIO(stream.getvalue()))["tensor"].view(
                    torch.uint16
                ),
                restored.view(torch.uint16),
            )
        )

    def test_lengths_checksum_shape_and_decoded_limit_are_validated(self):
        original = torch.ones(1, 32, 24, dtype=torch.bfloat16)
        encoded, meta = encode_wire_tensor(original)
        for broken in (
            {**meta, "encoding": "unknown"},
            {**meta, "decoded_bytes": 1},
            {**meta, "encoded_bytes": 1 << 40},
            {**meta, "sha256": "bad"},
        ):
            with self.subTest(broken=broken), self.assertRaises(ValueError):
                validate_wire_metadata(broken, shape=original.shape, dtype="bfloat16")
        with self.assertRaisesRegex(ValueError, "decoded byte limit"):
            decode_wire_tensor(
                encoded,
                meta,
                shape=original.shape,
                dtype="bfloat16",
                max_decoded_bytes=1,
            )
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            decode_wire_tensor(
                encoded,
                {**meta, "sha256": "0" * 64},
                shape=original.shape,
                dtype="bfloat16",
            )
        changed = encoded.clone()
        changed[-1] ^= 1
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            decode_wire_tensor(changed, meta, shape=original.shape, dtype="bfloat16")
        # Even a matching outer SHA cannot hide frame corruption.
        bad = {
            **meta,
            "sha256": hashlib.sha256(memoryview(changed.numpy())).hexdigest(),
        }
        with self.assertRaises(ValueError):
            decode_wire_tensor(changed, bad, shape=original.shape, dtype="bfloat16")
        # Same bytes/element count but different logical shape must not pass.
        with self.assertRaisesRegex(ValueError, "shape/dtype"):
            decode_wire_tensor(encoded, meta, shape=(1, 24, 32), dtype="bfloat16")

    def test_zero_noncontiguous_and_integer_wire_payloads(self):
        for tensor in (
            torch.empty(0, dtype=torch.bfloat16),
            torch.randn(3, 5).bfloat16().t(),
            torch.arange(100),
        ):
            packet, meta = encode_wire_tensor(tensor)
            decoded = decode_wire_tensor(
                packet,
                meta,
                shape=tensor.shape,
                dtype=str(tensor.dtype).removeprefix("torch."),
            )
            self.assertTrue(torch.equal(decoded, tensor))


class TestWireCaptureTransport(unittest.TestCase):
    def setUp(self):
        self.env = mock.patch.dict(os.environ, ENV, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.backend = _FakeMooncakeStore()
        self.sink = SpecCaptureSink(aux_layer_ids=[1, 4, 7])
        self.sink._store = self.backend
        self.sink._put_config = None
        self.addCleanup(self.sink._executor.shutdown)
        self.store = MooncakeFeatureStore(store=self.backend, store_id="wire")
        self.addCleanup(self.store.close)
        self.aux = torch.ones(4096, 24, dtype=torch.bfloat16)
        self.final = torch.ones(4096, 8, dtype=torch.bfloat16)
        self.task = PromptTask(
            task_id="episode",
            run_id="wire-run",
            source_id="source",
            max_length=4096,
            payload={"input_ids": [1] * 4096, "loss_mask": [1] * 4096},
        )
        self.capture = CaptureConfig.from_strategy(
            ["hidden_states", "target_last_hidden_states", "input_ids", "loss_mask"],
            [1, 4, 7],
            target_repr="hidden_state",
            target_hidden_size=8,
        )

    def adapter(self, **kwargs):
        return SGLangServerCaptureAdapter(
            "http://server",
            self.store,
            run_id="wire-run",
            algorithm="dspark",
            schema=ServerCaptureSchema(
                aux_feature="hidden_states",
                last_hidden_feature="target_last_hidden_states",
                passthrough=(
                    ("input_ids", "input_ids", ()),
                    ("loss_mask", "loss_mask", ()),
                ),
            ),
            server_capture="plugin",
            wire_encoding=WIRE_ENCODING,
            post_fn=self.post,
            **kwargs,
        )

    def post(self, url, json_body, timeout):
        specs = [
            json.loads(p["custom_params"]["spec_capture"])
            for p in json_body["sampling_params"]
        ]
        results = self.sink.put_samples([(s, self.aux, self.final) for s in specs])
        return [{"meta_info": {"spec_capture": r}} for r in results]

    def test_compressed_store_smaller_than_raw_buffers_and_logical_contract_unchanged(
        self,
    ):
        # Source can hold compressed bytes but cannot hold the raw record.
        self.backend.batch_put_from = mock.Mock(
            side_effect=AssertionError("encoded puts must be sequential")
        )
        with mock.patch.dict(
            os.environ,
            {
                "MOONCAKE_GLOBAL_SEGMENT_SIZE": "5000",
                "MOONCAKE_LOCAL_BUFFER_SIZE": "5000",
            },
        ):
            adapter = self.adapter(
                wire_max_object_bytes=5000, wire_max_total_bytes=5000
            )
            ref = adapter.produce_refs([self.task], capture=self.capture)[0]
        self.backend.batch_put_from.assert_not_called()
        self.assertLess(ref.estimated_bytes, 5000)
        self.assertGreater(ref.metadata["decoded_bytes"], 5000)
        self.assertEqual(
            ref.feature_specs["hidden_states"],
            FeatureSpec(name="hidden_states", shape=(1, 4096, 24), dtype="bfloat16"),
        )
        self.assertEqual(ref.feature_specs["input_ids"].dtype, "int64")
        out, handle = self.store.get(ref)
        self.assertTrue(torch.equal(out["hidden_states"], self.aux.unsqueeze(0)))
        self.assertTrue(
            torch.equal(out["target_last_hidden_states"], self.final.unsqueeze(0))
        )
        self.assertEqual(out["input_ids"].shape, (1, 4096))
        self.store.release(handle)
        self.assertFalse(self.backend._d)

    def test_sender_global_local_and_requested_budgets_fail_before_any_put(self):
        for env, budgets in (
            ({"MOONCAKE_GLOBAL_SEGMENT_SIZE": "1"}, {}),
            ({"MOONCAKE_LOCAL_BUFFER_SIZE": "1"}, {}),
            ({}, {"wire_max_object_bytes": 1}),
            ({}, {"wire_max_total_bytes": 1}),
        ):
            with (
                self.subTest(env=env, budgets=budgets),
                mock.patch.dict(os.environ, env),
            ):
                with self.assertRaises(MemoryError):
                    self.adapter(**budgets).produce_refs(
                        [self.task], capture=self.capture
                    )
                self.assertFalse(self.backend._d)

    def test_receiving_checks_packet_size_checksum_and_decoded_shape(self):
        ref = self.adapter().produce_refs([self.task], capture=self.capture)[0]
        self.store._client_buffer_bytes = 1
        with self.assertRaisesRegex(MemoryError, "client local buffer"):
            self.store.get(ref)
        self.store._client_buffer_bytes = None
        key = self.store._tkey(ref.sample_id, 1, "hidden_states")
        packet = bytearray(self.backend._d[key])
        packet[-1] ^= 1
        self.backend._d[key] = bytes(packet)
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self.store.get(ref)
        self.store.abort(ref.sample_id)
        self.assertFalse(self.backend._d)

    def test_discovery_advertises_encoding_and_rejects_mismatch_before_request(self):
        from specforge_sglang_capture import _capture_internal_state

        scheduler = SimpleNamespace(
            _specforge_capture_runtime=SimpleNamespace(is_writer=False, sink=self.sink)
        )
        state = SimpleNamespace(internal_state={})
        _capture_internal_state(state, scheduler)
        self.assertEqual(
            state.internal_state["spec_capture_wire_encoding"], WIRE_ENCODING
        )
        adapter = self.adapter()
        adapter.server_capture = "auto"
        adapter.server_info_fn = lambda *_: {
            "internal_states": [{"spec_capture_backend": "plugin"}]
        }
        with mock.patch.object(adapter, "post_fn") as post:
            with self.assertRaisesRegex(ValueError, "advertises wire encoding"):
                adapter.produce_refs([self.task], capture=self.capture)
            post.assert_not_called()
        adapter.server_info_fn = lambda *_: {"internal_states": [state.internal_state]}
        ref = adapter.produce_refs([self.task], capture=self.capture)[0]
        self.assertEqual(ref.feature_specs["hidden_states"].dtype, "bfloat16")
        self.store.abort(ref.sample_id)

    def test_declared_codec_must_match_sender_and_nvlink_stays_raw(self):
        with mock.patch.dict(os.environ, {"MOONCAKE_PROTOCOL": "nvlink"}):
            with self.assertRaisesRegex(ValueError, "host publication"):
                gpu_put_enabled()
        with mock.patch.dict(os.environ, {"SGLANG_SPEC_CAPTURE_GPU_PUT": "1"}):
            with self.assertRaisesRegex(ValueError, "host publication"):
                gpu_put_enabled()
        with mock.patch.dict(os.environ, {"SGLANG_SPEC_CAPTURE_WIRE_ENCODING": "raw"}):
            with self.assertRaisesRegex(ValueError, "wire_encoding"):
                self.adapter().produce_refs([self.task], capture=self.capture)
        self.assertFalse(self.backend._d)

    def test_wire_bytes_are_recorded_separately_from_disk_and_tensor_storage(self):
        layout = (
            builtin_algorithm_registry()
            .resolve("dspark")
            .providers.offline_for("text")
            .capture_layout
        )
        with tempfile.TemporaryDirectory() as directory:
            with ServerCaptureFileWriter(
                directory,
                capture=self.capture,
                layout=layout,
                algorithm="dspark",
                target_model_version="unknown",
                zstd=True,
            ) as writer:
                result = writer.write(
                    self.task, adapter=self.adapter(), store=self.store
                )
            self.assertEqual(result["wire_decoded_bytes"], result["tensor_bytes"])
            self.assertEqual(
                result["wire_bytes"],
                sum(m["encoded_bytes"] for m in result["wire_features"].values()),
            )
            sidecar = json.loads((Path(directory) / "episode.json").read_text())
            self.assertEqual(sidecar["result"], result)
            self.assertLess(result["wire_bytes"], result["wire_decoded_bytes"])
            self.assertFalse(self.backend._d)

    def test_file_writer_drains_read_leases_after_success_and_failed_write(self):
        layout = (
            builtin_algorithm_registry()
            .resolve("dspark")
            .providers.offline_for("text")
            .capture_layout
        )

        def leased_remove(key, *, force=False):
            if key in self.backend._d and not force:
                return -706  # Mooncake read lease outlives the application lease.
            self.backend._d.pop(key, None)
            return 0

        self.backend.remove = leased_remove
        for fail_write in (False, True):
            task = replace(self.task, task_id=f"episode-{fail_write}")
            with (
                self.subTest(fail_write=fail_write),
                tempfile.TemporaryDirectory() as directory,
            ):
                with ServerCaptureFileWriter(
                    directory,
                    capture=self.capture,
                    layout=layout,
                    algorithm="dspark",
                    target_model_version="unknown",
                    zstd=True,
                ) as writer:
                    if fail_write:
                        with (
                            mock.patch(
                                "specforge.inference.server_capture_writer.save_zstd_features",
                                side_effect=OSError("disk full"),
                            ),
                            self.assertRaisesRegex(OSError, "disk full"),
                        ):
                            writer.write(task, adapter=self.adapter(), store=self.store)
                    else:
                        writer.write(task, adapter=self.adapter(), store=self.store)
                self.assertEqual(len(self.backend._d), 0)
                self.assertEqual(self.store.health()["release_pending"], 0)

    def test_wire_failed_fetch_uses_existing_quarantine_and_retry_contract(self):
        ref = self.adapter().produce_refs([self.task], capture=self.capture)[0]
        with mock.patch.object(self.backend, "get_into", return_value=-704):
            # A permanent code exercises failed-packet retention without a timed retry.
            with self.assertRaises(KeyError):
                self.store.get(ref, names=["hidden_states"])
        self.assertEqual(self.store.health()["quarantined_buffers"], 1)
        self.assertEqual(
            self.store.health()["quarantined_bytes"],
            ref.metadata["wire_features"]["hidden_states"]["encoded_bytes"],
        )
        out, handle = self.store.get(ref)
        self.assertTrue(torch.equal(out["hidden_states"], self.aux.unsqueeze(0)))
        self.store.release(handle)

    def test_256k_capacity_is_decoded_limit_not_compression_ratio_guess(self):
        layout = SimpleNamespace(
            aux_feature="hidden_states",
            last_hidden_feature="target_last_hidden_states",
            passthrough=("ids", "mask"),
        )
        capture = CaptureConfig.from_strategy(
            [], [1, 10, 19, 28, 37], target_repr="hidden_state", target_hidden_size=2048
        )
        self.assertEqual(
            check_capacity(
                262144,
                capture=capture,
                layout=layout,
                client_bytes=4 << 30,
                server_bytes=6 << 30,
                wire_encoding=WIRE_ENCODING,
            ),
            6446645248,
        )
        with self.assertRaises(ValueError):
            check_capacity(
                262144,
                capture=capture,
                layout=layout,
                client_bytes=4 << 30,
                server_bytes=6 << 30,
            )


if __name__ == "__main__":
    unittest.main()
