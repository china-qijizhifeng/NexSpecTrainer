"""Lossless format and capture-to-training compatibility checks."""

import gzip
import hashlib
import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from scripts.prepare_hidden_states import HiddenStatesGenerator, parse_args
from specforge.algorithms.common.hidden_states_data import (
    normalize_dspark_offline_sample,
)
from specforge.data import feature_file
from specforge.data.feature_file import load_feature_file, save_zstd_features
from specforge.data.preprocessing import OfflineEagle3Dataset, list_local_files
from specforge.runtime.data_plane.feature_dataloader import FeatureDataLoader
from specforge.runtime.data_plane.feature_store import LocalFeatureStore
from specforge.runtime.data_plane.offline_reader import OfflineManifestReader


def assert_bits_equal(left, right):
    assert left.dtype == right.dtype
    assert left.shape == right.shape
    assert torch.equal(
        left.resolve_conj().resolve_neg().contiguous().reshape(-1).view(torch.uint8),
        right.contiguous().reshape(-1).view(torch.uint8),
    )


def rewrite_header(path, mutate):
    raw = path.read_bytes()
    length = struct.unpack("<I", raw[8:12])[0]
    header = json.loads(raw[44 : 44 + length])
    mutate(header)
    encoded = json.dumps(header).encode()
    path.write_bytes(
        raw[:8]
        + struct.pack("<I", len(encoded))
        + hashlib.sha256(encoded).digest()
        + encoded
        + raw[44 + length :]
    )


def make_generator(**kwargs):
    with (
        mock.patch("scripts.prepare_hidden_states.get_tp_group", return_value=None),
        mock.patch("scripts.prepare_hidden_states.dist.get_rank", return_value=0),
    ):
        return HiddenStatesGenerator(object(), **kwargs)


@unittest.skipUnless(
    importlib.util.find_spec("zstandard") and importlib.util.find_spec("brotli"),
    "requires compression extra",
)
class TestFeatureFile(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.path = self.directory / "data.ckpt.zst"

    def test_all_bf16_patterns_across_blocks(self):
        # Includes signed zero, subnormals, infinities and every NaN payload.
        # Distinct layer values expose layer/dimension transposition errors.
        bits = torch.arange(65536, dtype=torch.int32).to(torch.int16).repeat(17)
        source = bits.view(torch.bfloat16).reshape(1, -1, 64)
        for interleave in (False, True):
            with self.subTest(interleave=interleave):
                save_zstd_features(
                    {"hidden_states": source},
                    self.path,
                    interleave_layers=(
                        {"hidden_states": (9, 2, 7, 5)} if interleave else None
                    ),
                )
                assert_bits_equal(source, load_feature_file(self.path)["hidden_states"])

    def test_dtype_shape_and_noncontiguous_values(self):
        for dtype in feature_file._DTYPES.values():
            with self.subTest(dtype=dtype):
                source = torch.arange(30).to(dtype).reshape(5, 6).t()
                if source.is_complex():
                    source = (source + 1j).conj()
                record = {
                    "noncontiguous": source,
                    "scalar": source[0, 0],
                    "empty": source[:0],
                }
                save_zstd_features(record, self.path)
                result = load_feature_file(self.path)
                for key in record:
                    assert_bits_equal(record[key], result[key])

    def test_invalid_manifest_is_rejected_before_allocation(self):
        mutations = [
            (lambda h: h.update(version=2), "version"),
            (lambda h: h["tensors"][0].update(dtype="unknown"), "dtype"),
            (lambda h: h["tensors"][0].update(shape=[-1]), "shape"),
            (lambda h: h["tensors"].append(h["tensors"][0].copy()), "duplicate"),
            (lambda h: h["tensors"][0].update(interleave_layers=[1, 1]), "layout"),
            (lambda h: h["tensors"][0].update(shape=[(1 << 30) + 1]), "byte limit"),
        ]
        for mutation, match in mutations:
            with self.subTest(match=match):
                save_zstd_features({"x": torch.ones(8, dtype=torch.float64)}, self.path)
                rewrite_header(self.path, mutation)
                with self.assertRaisesRegex(ValueError, match):
                    load_feature_file(self.path)

    def test_decode_stays_on_cpu_with_another_default_device(self):
        source = torch.ones(8, dtype=torch.bfloat16)
        with torch.device("meta"):
            save_zstd_features({"x": source}, self.path)
            actual = load_feature_file(self.path)["x"]
        self.assertEqual(actual.device.type, "cpu")
        assert_bits_equal(source, actual)

    def test_decoded_size_limit(self):
        save_zstd_features({"x": torch.ones(8, dtype=torch.float64)}, self.path)
        assert_bits_equal(
            torch.ones(8, dtype=torch.float64),
            feature_file.load_zstd_features(self.path, max_record_bytes=64)["x"],
        )
        with self.assertRaisesRegex(ValueError, "byte limit"):
            feature_file.load_zstd_features(self.path, max_record_bytes=63)

    def test_corrupt_file_fails(self):
        for damage in (
            "truncate",
            "frame_checksum",
            "header_checksum",
            "trailing",
            "frame_size",
        ):
            with self.subTest(damage=damage):
                save_zstd_features({"x": torch.randn(20, 9).bfloat16()}, self.path)
                raw = bytearray(self.path.read_bytes())
                if damage == "truncate":
                    raw = raw[:-1]
                elif damage == "frame_checksum":
                    raw[-1] ^= 1
                elif damage == "header_checksum":
                    raw[44] ^= 1
                elif damage == "frame_size":
                    offset = 44 + struct.unpack("<I", raw[8:12])[0]
                    raw[offset : offset + 4] = struct.pack("<I", 1 << 30)
                else:
                    raw += b"extra"
                self.path.write_bytes(raw)
                with self.assertRaises(ValueError):
                    load_feature_file(self.path)

    def test_failed_publish_preserves_previous_file_and_cleans_temp(self):
        save_zstd_features({"x": torch.tensor([1])}, self.path)
        previous = self.path.read_bytes()
        with mock.patch.object(
            feature_file.os, "replace", side_effect=OSError("disk error")
        ):
            with self.assertRaises(OSError):
                save_zstd_features({"x": torch.tensor([2])}, self.path)
        self.assertEqual(self.path.read_bytes(), previous)
        self.assertEqual(list(self.directory.iterdir()), [self.path])

    def test_invalid_capture_layout_does_not_publish(self):
        for layers in ({"missing": [1]}, {"x": [1, 1]}, {"x": [1, 2, 3]}):
            with self.subTest(layers=layers), self.assertRaises(ValueError):
                save_zstd_features(
                    {"x": torch.zeros(2, 8).bfloat16()},
                    self.path,
                    interleave_layers=layers,
                )
            self.assertFalse(self.path.exists())
        with self.assertRaisesRegex(ValueError, "BF16"):
            save_zstd_features(
                {"x": torch.zeros(2, 8)}, self.path, interleave_layers={"x": [1, 2]}
            )

    def test_capture_mixed_directory_to_training_loader(self):
        record = {
            "input_ids": torch.arange(8),
            "loss_mask": torch.ones(8, dtype=torch.long),
            "aux_hidden_state": torch.randn(1, 8, 12).bfloat16(),
            "hidden_state": torch.randn(1, 8, 4).bfloat16(),
        }
        generator = make_generator(
            zstd=True, interleave_layers=True, capture_layers=(1, 4, 7)
        )
        generator._save_after_transfer(record, str(self.path), [])
        hybrid_path = self.directory / "2.ckpt.hybrid"
        hybrid_generator = make_generator(
            hybrid=True,
            capture_layers=(1, 4, 7),
            hybrid_chunk_size=4096,
            compression_workers=2,
        )
        hybrid_generator._save_after_transfer(record, str(hybrid_path), [])
        torch.save(record, self.directory / "0.ckpt")
        with gzip.open(self.directory / "1.ckpt.gz", "wb") as stream:
            torch.save(record, stream)
        keys = tuple(record)
        with mock.patch("builtins.open", side_effect=AssertionError("must stay lazy")):
            refs = OfflineManifestReader(
                str(self.directory),
                run_id="compression",
                strategy="eagle3",
                feature_keys=keys,
                target_repr=None,
                max_len=8,
            ).read()
        self.assertEqual(len(refs), 4)
        self.assertEqual(len(list_local_files(self.directory)), 4)
        loader = FeatureDataLoader(
            LocalFeatureStore("compression"),
            refs=refs,
            batch_size=1,
            collate_fn=lambda samples: samples[0],
            strategy="eagle3",
        )
        batches = list(loader)
        self.assertEqual(len(batches), 4)
        for batch in batches:
            for key in record:
                assert_bits_equal(record[key], batch.tensors[key])
        eager = OfflineManifestReader(
            str(hybrid_path),
            run_id="compression",
            strategy="eagle3",
            feature_keys=keys,
            target_repr=None,
            max_len=8,
            validate_files=True,
        ).read()
        self.assertEqual(eager[0].feature_specs["aux_hidden_state"].shape, (1, 8, 12))
        # Legacy training must use the same decoder and propagate errors.
        legacy = OfflineEagle3Dataset([str(hybrid_path)], max_len=8)
        assert_bits_equal(
            legacy._open_file(0)["aux_hidden_state"], record["aux_hidden_state"]
        )
        hybrid_path.write_bytes(b"corrupt")
        with self.assertRaises(ValueError):
            legacy[0]

    def test_dspark_normalization_is_unchanged(self):
        record = {
            "input_ids": torch.arange(8),
            "loss_mask": torch.ones(8, dtype=torch.long),
            "hidden_states": torch.randn(1, 8, 12).bfloat16(),
            "target_last_hidden_states": torch.randn(1, 8, 4).bfloat16(),
        }
        save_zstd_features(
            record, self.path, interleave_layers={"hidden_states": (1, 4, 7)}
        )
        expected = normalize_dspark_offline_sample(record, max_len=5)
        actual = normalize_dspark_offline_sample(
            load_feature_file(self.path), max_len=5
        )
        for key in expected:
            assert_bits_equal(expected[key], actual[key])

    def test_resume_recognizes_compressed_file_with_raw_writer(self):
        generator = make_generator()
        path = generator._get_file_path(str(self.directory), 0, extension=".ckpt.zst")
        Path(path).parent.mkdir(parents=True)
        save_zstd_features({"x": torch.ones(1)}, path)
        with mock.patch(
            "scripts.prepare_hidden_states.is_tp_rank_0", return_value=True
        ):
            self.assertEqual(
                generator._check_existing_files_batch(str(self.directory), [0, 1]),
                [True, False],
            )

    def test_cli_compression_flags(self):
        argv = [
            "prepare_hidden_states.py",
            "--target-model-path",
            "m",
            "--data-path",
            "d",
        ]
        with mock.patch("sys.argv", argv + ["--zstd", "--interleave-layers"]):
            args = parse_args()
            self.assertTrue(args.zstd and args.interleave_layers)
            self.assertEqual(args.zstd_level, 1)
        with mock.patch("sys.argv", argv + ["--hybrid"]):
            args = parse_args()
            self.assertTrue(args.hybrid)
            self.assertEqual(args.brotli_quality, 10)
            self.assertEqual(args.hybrid_chunk_size, 256 << 10)
        for flags in (
            ["--zstd", "--compress"],
            ["--hybrid", "--zstd"],
            ["--interleave-layers"],
            ["--zstd-level", "0"],
            ["--brotli-quality", "0"],
            ["--hybrid-chunk-size", "0"],
            ["--hybrid-chunk-size", str((64 << 20) + 1)],
            ["--compression-workers", "0"],
        ):
            with mock.patch("sys.argv", argv + flags), self.assertRaises(SystemExit):
                parse_args()
        with self.assertRaisesRegex(ValueError, "2 to 64 layers"):
            make_generator(hybrid=True, capture_layers=(1,))


class TestOptionalCompressionDependency(unittest.TestCase):
    def test_missing_dependency_is_actionable_and_raw_still_loads(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.ckpt"
            torch.save({"x": torch.tensor([1])}, path)
            with mock.patch.dict("sys.modules", {"zstandard": None}):
                assert_bits_equal(load_feature_file(path)["x"], torch.tensor([1]))
                with self.assertRaisesRegex(ImportError, "specforge\\[compression\\]"):
                    feature_file.require_zstandard()
