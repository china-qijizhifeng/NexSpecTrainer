"""Existing identical metadata is allowed; mismatches and symlinks never overwrite."""

import json
import stat
import tempfile
from pathlib import Path

from common import sha
from prepare_remote import copy_dependency


def main():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / "source"
        destination = root / "destination"
        source.write_bytes(b"identical-receipt\n")
        digest = sha(source)
        assert copy_dependency(source, destination, digest, 0o600)
        assert stat.S_IMODE(destination.stat().st_mode) == 0o600
        before = destination.stat()
        assert copy_dependency(source, destination, digest, 0o644) is False
        after = destination.stat()
        assert (
            before.st_mtime_ns == after.st_mtime_ns and before.st_mode == after.st_mode
        )
        destination.write_bytes(b"wrong")
        for mode in (0o600, 0o644):
            try:
                copy_dependency(source, destination, digest, mode)
            except AssertionError:
                pass
            else:
                raise AssertionError("wrong existing bytes accepted")
            assert destination.read_bytes() == b"wrong"
        link = root / "link"
        link.symlink_to(source)
        try:
            copy_dependency(source, link, digest, 0o600)
        except AssertionError:
            pass
        else:
            raise AssertionError("symlink destination accepted")
    print(
        json.dumps(
            {
                "passed": True,
                "identical_existing_receipt_preserves_bytes_mode_mtime": True,
                "different_receipt_and_symlink_rejected": True,
                "new_private_copy0600": True,
            }
        )
    )


if __name__ == "__main__":
    main()
