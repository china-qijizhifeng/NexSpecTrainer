"""Actual Torch CPU checks for digest scope, privacy, replay and hard limits."""

import hashlib
import json
import os
import stat
import struct
import tempfile
import unittest
from pathlib import Path

import torch
from bounded_trace import (
    MAX_EVENTS,
    MAX_ROW_BYTES,
    MAX_TRACE_BYTES,
    TraceWriter,
    tensor_row_digest,
)


class DigestChecks(unittest.TestCase):
    def test_bf16_exact_bytes_and_no_mutation(self):
        value = torch.tensor([[1, -2], [3, 4]], dtype=torch.bfloat16)
        before = value.clone()
        digest = tensor_row_digest(value)
        expected = struct.pack("<HH", 0x4040, 0x4080)
        self.assertEqual(digest["sha256"], hashlib.sha256(expected).hexdigest())
        self.assertEqual(digest["selected_bytes"], 4)
        self.assertTrue(torch.equal(value, before))

    def test_noncontiguous_row_and_signed_zero(self):
        value = torch.arange(20, dtype=torch.float32).reshape(4, 5).T
        self.assertFalse(value.is_contiguous())
        result = tensor_row_digest(value, row=2)
        expected = struct.pack("<ffff", 2, 7, 12, 17)
        self.assertEqual(result["sha256"], hashlib.sha256(expected).hexdigest())
        plus = tensor_row_digest(torch.tensor([0.0]))
        minus = tensor_row_digest(torch.tensor([-0.0]))
        self.assertNotEqual(plus["sha256"], minus["sha256"])

    def test_large_tensor_small_row_and_overwide_row(self):
        height = MAX_ROW_BYTES // 8 + 1
        value = torch.ones((height, 8), dtype=torch.float32)
        result = tensor_row_digest(value)
        self.assertGreater(value.numel() * value.element_size(), MAX_ROW_BYTES)
        self.assertEqual(result["selected_bytes"], 32)
        with self.assertRaises(ValueError):
            tensor_row_digest(torch.zeros(MAX_ROW_BYTES // 4 + 1))

    def test_scope_and_nonfinite_rejection(self):
        value = torch.tensor([[float("nan")], [1.0]])
        self.assertTrue(tensor_row_digest(value)["finite"])
        with self.assertRaises(ValueError):
            tensor_row_digest(value, row=0)
        for value in (
            torch.empty(0),
            torch.ones(1, 1, 1, 1),
            torch.tensor([float("inf")]),
        ):
            with self.assertRaises(ValueError):
                tensor_row_digest(value)
        with self.assertRaises(TypeError):
            tensor_row_digest({"deferred": torch.ones(1)})

    def test_integer_and_scalar(self):
        value = torch.tensor(2**53 + 1, dtype=torch.int64)
        result = tensor_row_digest(value)
        expected = hashlib.sha256(struct.pack("<q", 2**53 + 1)).hexdigest()
        self.assertEqual(result["sha256"], expected)
        self.assertEqual(result["selected_bytes"], 8)


class WriterChecks(unittest.TestCase):
    def test_parent_umask_replay_and_symlink(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for mask in (0o077, 0o022):
                path = root / str(mask)
                previous = os.umask(mask)
                try:
                    with TraceWriter(path) as writer:
                        writer.write({"rank": 0, "digest": "a" * 64})
                    self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
                    saved = path.read_bytes()
                    with self.assertRaises(FileExistsError):
                        TraceWriter(path)
                    self.assertEqual(path.read_bytes(), saved)
                finally:
                    os.umask(previous)
            link = root / "existing-link"
            link.symlink_to(path)
            with self.assertRaises(FileExistsError):
                TraceWriter(link)

    def test_event_and_json_failures_leave_no_partial_event(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "trace"
            with TraceWriter(path) as writer:
                writer.write({"phase": "begin"})
                before = path.read_bytes()
                with self.assertRaises(ValueError):
                    writer.write({"oversize": "x" * 4096})
                with self.assertRaises(ValueError):
                    writer.write({"invalid": float("nan")})
                self.assertEqual(path.read_bytes(), before)

    def test_real_event_count_limit(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "trace"
            with TraceWriter(path) as writer:
                for index in range(MAX_EVENTS):
                    writer.write({"index": index})
                before = path.stat().st_size
                with self.assertRaises(ValueError):
                    writer.write({"index": MAX_EVENTS})
                self.assertEqual(path.stat().st_size, before)
            self.assertEqual(len(path.read_text().splitlines()), MAX_EVENTS)

    def test_real_byte_limit(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "trace"
            with TraceWriter(path) as writer:
                event = {"digest": "a" * 4000}
                size = len((json.dumps(event, sort_keys=True) + "\n").encode())
                for _ in range(MAX_TRACE_BYTES // size):
                    writer.write(event)
                before = path.stat().st_size
                with self.assertRaises(ValueError):
                    writer.write(event)
                self.assertEqual(path.stat().st_size, before)
                self.assertLessEqual(before, MAX_TRACE_BYTES)


if __name__ == "__main__":
    unittest.main()
