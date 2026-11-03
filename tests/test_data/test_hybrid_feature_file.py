import hashlib
import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from specforge.data import feature_file
from specforge.data.feature_file import (
    load_feature_file,
    load_hybrid_features,
    save_hybrid_features,
)


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


@unittest.skipUnless(
    importlib.util.find_spec("zstandard") and importlib.util.find_spec("brotli"),
    "requires compression extra",
)
class TestHybridFeatureFile(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.path = self.directory / "data.ckpt.hybrid"

    def record(self, layers=3):
        bits = torch.arange(65536, dtype=torch.int32).to(torch.int16).repeat(layers)
        source = bits.view(torch.bfloat16).reshape(1, -1, layers * 16)
        return {
            "hidden_states": source,
            "input_ids": torch.arange(8),
            "final": torch.randn(6, 4).bfloat16().t(),
        }

    def save(self, record, **kwargs):
        options = {
            "interleave_layers": {
                "hidden_states": tuple(range(record["hidden_states"].shape[-1] // 16))
            },
            "chunk_size": 4096,
            "workers": 2,
        }
        options.update(kwargs)
        save_hybrid_features(record, self.path, **options)

    def test_all_bf16_patterns_dtypes_and_noncontiguous_values(self):
        record = self.record(4)
        for dtype in feature_file._DTYPES.values():
            source = torch.arange(30).to(dtype).reshape(5, 6).t()
            if source.is_complex():
                source = (source + 1j).conj()
            record[str(dtype)] = source
            record[f"{dtype}-scalar"] = source[0, 0]
            record[f"{dtype}-empty"] = source[:0]
        self.save(record)
        result = load_feature_file(self.path)
        for key in record:
            assert_bits_equal(record[key], result[key])

    def test_two_to_five_layers_and_empty_tensors(self):
        for layers in range(2, 6):
            with self.subTest(layers=layers):
                record = {
                    "hidden_states": torch.randint(
                        0, 65536, (1, 5, layers * 8), dtype=torch.int32
                    )
                    .to(torch.int16)
                    .view(torch.bfloat16),
                    "empty": torch.empty(1, 0, layers * 8, dtype=torch.bfloat16),
                }
                save_hybrid_features(
                    record,
                    self.path,
                    interleave_layers={
                        "hidden_states": tuple(range(layers)),
                        "empty": tuple(range(layers)),
                    },
                    chunk_size=4096,
                    workers=2,
                )
                result = load_feature_file(self.path)
                for key in record:
                    assert_bits_equal(record[key], result[key])

    def test_parallel_encoding_is_deterministic(self):
        record = self.record()
        other = self.directory / "other.ckpt.hybrid"
        layers = {"hidden_states": (0, 1, 2)}
        save_hybrid_features(
            record, self.path, interleave_layers=layers, chunk_size=4096, workers=1
        )
        save_hybrid_features(
            record, other, interleave_layers=layers, chunk_size=4096, workers=4
        )
        self.assertEqual(self.path.read_bytes(), other.read_bytes())
        result = load_hybrid_features(other, workers=4)
        for key in record:
            assert_bits_equal(record[key], result[key])

    def test_invalid_layout_does_not_publish(self):
        source = torch.zeros(2, 8, dtype=torch.bfloat16)
        for layers in (
            {},
            {"missing": [1, 2]},
            {"x": [1]},
            {"x": [1, 1]},
            {"x": [1, 2, 3]},
        ):
            with self.subTest(layers=layers), self.assertRaises(ValueError):
                save_hybrid_features({"x": source}, self.path, interleave_layers=layers)
            self.assertFalse(self.path.exists())
        with self.assertRaisesRegex(ValueError, "BF16"):
            save_hybrid_features(
                {"x": torch.zeros(2, 8)}, self.path, interleave_layers={"x": [1, 2]}
            )

    def test_invalid_header_and_size_limit(self):
        record = self.record()
        self.save(record)
        raw_bytes = sum(
            tensor.numel() * tensor.element_size() for tensor in record.values()
        )
        for key in record:
            assert_bits_equal(
                record[key],
                load_hybrid_features(self.path, max_record_bytes=raw_bytes)[key],
            )
        with self.assertRaisesRegex(ValueError, "byte limit"):
            load_hybrid_features(self.path, max_record_bytes=raw_bytes - 1)
        mutations = [
            (lambda h: h.update(version=2), "version"),
            (lambda h: h.update(chunk_size=1), "chunk size"),
            (lambda h: h["tensors"][0].update(codec="raw"), "codec"),
            (lambda h: h["tensors"][0].update(sha256="0"), "checksum"),
            (lambda h: h["tensors"][0].update(interleave_layers=[0]), "2 to 64 layers"),
        ]
        for mutation, match in mutations:
            with self.subTest(match=match):
                self.save(record)
                rewrite_header(self.path, mutation)
                with self.assertRaisesRegex(ValueError, match):
                    load_feature_file(self.path)

    def test_corrupt_file_fails(self):
        record = self.record()
        for damage in (
            "truncate",
            "brotli",
            "zstd",
            "header_checksum",
            "trailing",
            "frame_size",
        ):
            with self.subTest(damage=damage):
                self.save(record)
                raw = bytearray(self.path.read_bytes())
                offset = 44 + struct.unpack("<I", raw[8:12])[0]
                if damage == "truncate":
                    raw = raw[:-1]
                elif damage == "brotli":
                    compressed_size = struct.unpack(
                        "<I", raw[offset + 9 : offset + 13]
                    )[0]
                    raw[offset + 13 + compressed_size // 2] ^= 1
                elif damage == "zstd":
                    raw[-1] ^= 1
                elif damage == "header_checksum":
                    raw[44] ^= 1
                elif damage == "frame_size":
                    raw[offset + 5 : offset + 9] = struct.pack("<I", 1 << 30)
                else:
                    raw += b"extra"
                self.path.write_bytes(raw)
                with self.assertRaises(ValueError):
                    load_feature_file(self.path)

    def test_large_raw_and_small_chunk_roundtrip(self):
        hidden = (
            torch.randint(0, 65536, (1, 1024, 48), dtype=torch.int32)
            .to(torch.int16)
            .view(torch.bfloat16)
        )
        record = {
            "hidden_states": hidden,
            "raw": torch.randint(0, 256, (64 << 20,), dtype=torch.uint8),
        }
        save_hybrid_features(
            record,
            self.path,
            quality=1,
            interleave_layers={"hidden_states": (0, 1, 2)},
            chunk_size=4096,
            workers=4,
        )
        result = load_hybrid_features(self.path, workers=4)
        for key in record:
            assert_bits_equal(record[key], result[key])

    def test_brotli_output_limit_and_decoded_stream_budget(self):
        source = (
            torch.randint(0, 65536, (1, 1, 48), dtype=torch.int32)
            .to(torch.int16)
            .view(torch.bfloat16)
        )
        save_hybrid_features(
            {"hidden_states": source},
            self.path,
            interleave_layers={"hidden_states": (0, 1, 2)},
            chunk_size=4096,
            workers=1,
        )
        original = bytearray(self.path.read_bytes())
        offset = 44 + struct.unpack("<I", original[8:12])[0]
        compressed_size = struct.unpack("<I", original[offset + 9 : offset + 13])[0]
        brotli = feature_file.require_brotli()
        bomb = brotli.compress(b"\0" * (1 << 20), quality=1)
        raw = original.copy()
        raw[offset + 9 : offset + 13] = struct.pack("<I", len(bomb))
        raw = raw[: offset + 13] + bomb + raw[offset + 13 + compressed_size :]
        self.path.write_bytes(raw)
        with self.assertRaisesRegex(ValueError, "declared size"):
            load_feature_file(self.path)
        raw = original.copy()
        raw[offset + 5 : offset + 9] = struct.pack("<I", 4096)
        self.path.write_bytes(raw)
        with self.assertRaisesRegex(ValueError, "decoded stream budget"):
            load_feature_file(self.path)
        raw = original.copy()
        raw[offset + 9 : offset + 13] = struct.pack("<I", 1027)
        self.path.write_bytes(raw)
        with self.assertRaisesRegex(ValueError, "Oversized hybrid activation frame"):
            load_feature_file(self.path)

    def test_layer_limit_applies_to_empty_tensors(self):
        source = torch.empty(1, 0, 65 * 8, dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "2 to 64 layers"):
            save_hybrid_features(
                {"hidden_states": source},
                self.path,
                interleave_layers={"hidden_states": tuple(range(65))},
            )
        self.assertFalse(self.path.exists())

    def test_failed_publish_preserves_previous_file_and_cleans_temp(self):
        record = self.record()
        self.save(record)
        previous = self.path.read_bytes()
        with mock.patch.object(
            feature_file.os, "replace", side_effect=OSError("disk error")
        ):
            with self.assertRaises(OSError):
                self.save(record)
        self.assertEqual(self.path.read_bytes(), previous)
        self.assertEqual(list(self.directory.iterdir()), [self.path])


class TestHybridDependency(unittest.TestCase):
    def test_missing_brotli_is_actionable(self):
        with mock.patch.dict("sys.modules", {"brotli": None}):
            with self.assertRaisesRegex(ImportError, "specforge\\[compression\\]"):
                feature_file.require_hybrid_compression()
