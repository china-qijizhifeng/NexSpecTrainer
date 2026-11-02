"""Server metadata -> actual Mooncake get -> codec -> production offline reader."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from scripts.prepare_server_hidden_states import (
    check_capacity,
    iter_tasks,
    main,
    parse_args,
)
from specforge.algorithms.builtin import builtin_algorithm_registry
from specforge.inference.adapters.server_capture import (
    ServerCaptureSchema,
    SGLangServerCaptureAdapter,
)
from specforge.inference.capture import CaptureConfig
from specforge.inference.server_capture_writer import ServerCaptureFileWriter
from specforge.runtime.contracts import PromptTask
from specforge.runtime.data_plane.feature_store import LocalFeatureStore
from specforge.runtime.data_plane.mooncake_store import MooncakeFeatureStore
from tests.test_runtime.test_server_capture import (
    AUX_LAYERS,
    HIDDEN,
    _FakeMooncakeStore,
    _StubCaptureServer,
)


class TestServerCaptureFiles(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / "features"
        self.algorithm = builtin_algorithm_registry().resolve("dspark")
        self.provider = self.algorithm.providers.offline_for("text")
        self.layout = self.provider.capture_layout
        self.capture = CaptureConfig.from_strategy(
            self.layout.output_names,
            AUX_LAYERS,
            target_repr="hidden_state",
            target_hidden_size=HIDDEN,
        )
        self.backend = _FakeMooncakeStore()
        self.server = _StubCaptureServer(self.backend)
        self.store = MooncakeFeatureStore(
            store=self.backend, store_id="files", receive_buffers="pageable"
        )
        self.addCleanup(self.store.close)
        self.adapter = SGLangServerCaptureAdapter(
            "http://server",
            self.store,
            run_id="run",
            algorithm="dspark",
            post_fn=self.server,
            target_model_version="pinned",
            schema=ServerCaptureSchema(
                aux_feature=self.layout.aux_feature,
                last_hidden_feature=self.layout.last_hidden_feature,
                passthrough=tuple((n, s, ()) for n, s in self.layout.passthrough),
            ),
        )
        self.task = PromptTask(
            task_id="episode-1",
            source_id="episode-1",
            run_id="run",
            payload={"input_ids": [1, 2, 3, 4], "loss_mask": [0, 1, 1, 1]},
            max_length=4,
            target_model_version="pinned",
            metadata={"origin": "real-row"},
        )

    def writer(self, **kwargs):
        return ServerCaptureFileWriter(
            self.output,
            capture=self.capture,
            layout=self.layout,
            algorithm="dspark",
            target_model_version="pinned",
            **kwargs,
        )

    def test_uncompressed_transport_compressed_files_feed_real_dspark_reader(self):
        for zstd in (False, True):
            with self.subTest(zstd=zstd):
                self.output = Path(self.temp.name) / str(zstd)
                task = replace(self.task, task_id=f"episode-{zstd}")
                with self.writer(zstd=zstd, interleave_layers=zstd) as writer:
                    result = writer.write(task, adapter=self.adapter, store=self.store)
                self.assertEqual(result["status"], "captured")
                self.assertFalse(
                    self.backend._d, "write must release server-owned Mooncake objects"
                )
                # Stub bytes before the client codec are raw BF16, not Zstd.
                expected = list(self.server.expected.values())[-1]
                self.assertEqual(expected["hidden_states"].dtype, torch.bfloat16)
                reader = self.provider.build_reader(
                    str(self.output), run_id="offline", ttt_length=1, max_len=4
                )
                refs = list(reader)
                self.assertEqual(len(refs), 1)
                local = LocalFeatureStore("reader")
                loaded, handle = local.get(refs[0])
                normalized = self.provider.build_normalizer(max_len=4)(loaded)
                batch = self.provider.build_collator()([normalized])
                for name, tensor in expected.items():
                    self.assertTrue(torch.equal(batch[name], tensor), name)
                local.release(handle)

    def test_resume_checks_input_and_decodes_before_skipping(self):
        with self.writer(zstd=True) as writer:
            writer.write(self.task, adapter=self.adapter, store=self.store)
        with (
            self.writer(zstd=True, resume=True) as writer,
            mock.patch.object(self.adapter, "produce_refs") as capture,
        ):
            self.assertEqual(
                writer.write(self.task, adapter=self.adapter, store=self.store)[
                    "status"
                ],
                "resumed",
            )
            capture.assert_not_called()
        self.task.payload["input_ids"][0] = 99
        with (
            self.writer(zstd=True, resume=True) as writer,
            self.assertRaisesRegex(ValueError, "resume input"),
        ):
            writer.write(self.task, adapter=self.adapter, store=self.store)
        self.task.payload["input_ids"][0] = 1
        path = self.output / "episode-1.ckpt.zst"
        path.write_bytes(path.read_bytes()[:-1])
        with (
            self.writer(zstd=True, resume=True) as writer,
            self.assertRaises(ValueError),
        ):
            writer.write(self.task, adapter=self.adapter, store=self.store)

    def test_no_publication_on_codec_failure_and_transport_is_released(self):
        with self.writer(zstd=True) as writer:
            with mock.patch(
                "specforge.inference.server_capture_writer.save_zstd_features",
                side_effect=OSError("disk full"),
            ):
                with self.assertRaisesRegex(OSError, "disk full"):
                    writer.write(self.task, adapter=self.adapter, store=self.store)
        self.assertFalse(list(self.output.glob("*.ckpt*")))
        self.assertFalse(list(self.output.glob("*.partial")))
        self.assertFalse(self.backend._d)

    def test_single_owner_and_resume_plan_guard(self):
        with self.writer() as writer:
            with self.assertRaises(FileExistsError):
                with self.writer(resume=True):
                    pass
            writer.write(self.task, adapter=self.adapter, store=self.store)
        with self.assertRaisesRegex(ValueError, "plan does not match"):
            with self.writer(zstd=True, resume=True):
                pass
        self.assertFalse((self.output / ".capture-writer.lock").exists())

    def test_token_mismatch_is_not_persisted(self):
        original_get = self.store.get

        def mismatch(*args, **kwargs):
            tensors, handle = original_get(*args, **kwargs)
            tensors["input_ids"][0, 0] = 99
            return tensors, handle

        with (
            self.writer() as writer,
            mock.patch.object(self.store, "get", side_effect=mismatch),
        ):
            with self.assertRaisesRegex(ValueError, "does not match"):
                writer.write(self.task, adapter=self.adapter, store=self.store)
        self.assertFalse(list(self.output.glob("*.ckpt*")))
        self.assertFalse(self.backend._d)

    def test_transfer_capacity_uses_raw_bf16_size(self):
        capture = CaptureConfig.from_strategy(
            self.layout.output_names,
            [1, 10, 19, 28, 37],
            target_repr="hidden_state",
            target_hidden_size=2048,
        )
        with self.assertRaisesRegex(ValueError, "disk compression does not reduce"):
            check_capacity(
                83542,
                capture=capture,
                layout=self.layout,
                client_bytes=1 << 30,
                server_bytes=4 << 30,
            )
        self.assertEqual(
            check_capacity(
                83542,
                capture=capture,
                layout=self.layout,
                client_bytes=4 << 30,
                server_bytes=4 << 30,
            ),
            83542 * (12288 * 2 + 16),
        )

    def test_input_iterator_rejects_truncation_and_revision_mismatch(self):
        path = Path(self.temp.name) / "samples.jsonl"
        cfg = SimpleNamespace(data=SimpleNamespace(max_length=3))
        row = {"id": "episode-1", **self.task.payload}
        path.write_text(json.dumps(row) + "\n")
        with self.assertRaisesRegex(ValueError, "refusing truncation"):
            list(
                iter_tasks(
                    path,
                    cfg=cfg,
                    run_id="run",
                    target_version="pinned",
                    loss_mask_filter=None,
                )
            )
        cfg.data.max_length = 4
        row["metadata"] = {"model_revision": "other"}
        path.write_text(json.dumps(row) + "\n")
        with self.assertRaisesRegex(ValueError, "model_revision"):
            list(
                iter_tasks(
                    path,
                    cfg=cfg,
                    run_id="run",
                    target_version="pinned",
                    loss_mask_filter=None,
                )
            )

    def test_cli_resolves_capture_plan_and_persists_a_complete_episode(self):
        path = Path(self.temp.name) / "samples.jsonl"
        path.write_text(
            json.dumps({"id": self.task.task_id, **self.task.payload}) + "\n"
        )
        cfg = SimpleNamespace(
            model=SimpleNamespace(input_modality="text"),
            data=SimpleNamespace(max_length=4),
        )
        plan = SimpleNamespace(
            run=SimpleNamespace(algorithm=self.algorithm),
            layout=self.layout,
            capture_layers=AUX_LAYERS,
            loss_mask_filter=None,
        )
        contract = SimpleNamespace(
            aux_layer_ids=AUX_LAYERS,
            target_hidden_size=HIDDEN,
            target_vocab_size=100,
            draft_vocab_size=100,
        )

        def adapter_factory(*args, **kwargs):
            return SGLangServerCaptureAdapter(*args, post_fn=self.server, **kwargs)

        argv = [
            "--config",
            "train.yaml",
            "--server",
            "http://server",
            "--server-capture",
            "patch",
            "--samples",
            str(path),
            "--output",
            str(self.output),
            "--target-model-version",
            "pinned",
            "--zstd",
        ]
        with (
            mock.patch.dict(
                "os.environ",
                {
                    "MOONCAKE_METADATA_SERVER": "http://metadata",
                    "MOONCAKE_MASTER_SERVER_ADDR": "master:1",
                },
            ),
            mock.patch("specforge.config.load_config", return_value=cfg),
            mock.patch(
                "specforge.application.resolve_offline_capture", return_value=plan
            ),
            mock.patch(
                "specforge.training.capture_contract.resolve_server_capture_contract",
                return_value=contract,
            ),
            mock.patch(
                "specforge.runtime.data_plane.mooncake_store.MooncakeFeatureStore",
                return_value=self.store,
            ) as store_factory,
            mock.patch(
                "specforge.inference.adapters.server_capture.SGLangServerCaptureAdapter",
                side_effect=adapter_factory,
            ),
            redirect_stdout(io.StringIO()) as stdout,
        ):
            self.assertEqual(main(argv), 0)
        self.assertEqual(json.loads(stdout.getvalue())["tokens"], 4)
        self.assertTrue((self.output / "episode-1.ckpt.zst").is_file())
        self.assertEqual(
            store_factory.call_args.kwargs["setup_kwargs"]["local_buffer_size"], 4 << 30
        )
        self.assertFalse(self.backend._d)

    def test_optional_dependency_fails_before_transport(self):
        argv = [
            "--config",
            "train.yaml",
            "--server",
            "http://server",
            "--samples",
            "samples.jsonl",
            "--output",
            "features",
            "--target-model-version",
            "pinned",
            "--zstd",
        ]
        with mock.patch(
            "specforge.data.feature_file.require_zstandard",
            side_effect=ImportError("install compression"),
        ):
            with self.assertRaisesRegex(ImportError, "install compression"):
                parse_args(argv)


if __name__ == "__main__":
    unittest.main()
