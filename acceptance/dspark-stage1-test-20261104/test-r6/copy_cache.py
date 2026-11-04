"""Copy previous compiled artifacts into this attempt; never link writable caches."""

import os
import shutil
from pathlib import Path

from common import ROOT, verify_execution, write


def main():
    verify_execution()
    source = ROOT.parent / "dspark-test-20261104-r3/cache"
    destination = ROOT / "cache"
    assert source.is_dir() and not destination.exists()
    assert shutil.disk_usage(ROOT).free > 60_000_000_000
    destination.mkdir()
    copied = 0
    byte_count = 0
    skipped = []
    for directory, dirs, files in os.walk(source, followlinks=False):
        current = Path(directory)
        target = destination / current.relative_to(source)
        target.mkdir(exist_ok=True, parents=True)
        for name in list(dirs):
            path = current / name
            if path.is_symlink() or not os.access(path, os.R_OK | os.X_OK):
                skipped.append(str(path.relative_to(source)))
                dirs.remove(name)
        for name in files:
            path = current / name
            if path.is_symlink() or not os.access(path, os.R_OK):
                skipped.append(str(path.relative_to(source)))
                continue
            shutil.copy2(path, target / name)
            copied += 1
            byte_count += path.stat().st_size
    free_bytes = shutil.disk_usage(ROOT).free
    assert free_bytes > 50_000_000_000
    write(
        ROOT / "cache-copy-status.json",
        {
            "passed": True,
            "source": str(source),
            "destination": str(destination),
            "copied_files": copied,
            "copied_bytes": byte_count,
            "skipped": skipped,
            "free_bytes_after": free_bytes,
            "previous_cache_modified": False,
        },
    )


if __name__ == "__main__":
    main()
