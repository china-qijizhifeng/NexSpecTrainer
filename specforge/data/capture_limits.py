"""Explicit process-wide bound for trusted full-trajectory capture jobs.

The default stays 8 GiB; increasing this bound does not allocate buffers or
change the model context. Readers must use the recorded bound for large files.
"""
import os

MAX_CAPTURE_RECORD_BYTES = int(os.environ.get("SPECFORGE_MAX_CAPTURE_RECORD_BYTES", str(8 << 30)))
if not 0 < MAX_CAPTURE_RECORD_BYTES <= 32 << 30:
    raise ValueError("SPECFORGE_MAX_CAPTURE_RECORD_BYTES must be in (0, 32 GiB]")
