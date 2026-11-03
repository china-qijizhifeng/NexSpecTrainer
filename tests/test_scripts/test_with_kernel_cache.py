"""Exercise the cache launcher at the subprocess environment/exec boundary."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "with_kernel_cache.py"


class KernelCacheLauncherTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve() / "cache with spaces"
        # Do not let the developer's cache paths leak into these subprocesses.
        self.env = {k: v for k, v in os.environ.items() if "CACHE" not in k}
        for key in (
            "FLASHINFER_WORKSPACE_BASE",
            "TORCH_EXTENSIONS_DIR",
            "SGLANG_JIT_DEEPGEMM_FAST_WARMUP",
        ):
            self.env.pop(key, None)

    def run_script(self, *args, env=None, namespace="image-sm90"):
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--cache-root",
                str(self.root),
                "--namespace",
                namespace,
                *args,
            ],
            env={**self.env, **(env or {})},
            text=True,
            capture_output=True,
        )

    def test_launch_sets_persistent_paths_before_imports_and_preserves_argv(self):
        code = (
            "import json, os, sys; "
            "print(json.dumps({'env': dict(os.environ), 'argv': sys.argv[1:]}))"
        )
        result = self.run_script("--", sys.executable, "-c", code, "a b", "$(literal)")
        self.assertEqual(result.returncode, 0, result.stderr)
        child = json.loads(result.stdout)
        self.assertEqual(child["argv"], ["a b", "$(literal)"])
        env = child["env"]
        base = self.root / "image-sm90"
        self.assertEqual(env["TRITON_CACHE_DIR"], str(base / "triton"))
        self.assertEqual(env["TILELANG_CACHE_DIR"], str(base / "tilelang"))
        self.assertEqual(env["DG_JIT_CACHE_DIR"], str(base / "sglang/deep_gemm"))
        self.assertEqual(env["SGLANG_DG_CACHE_DIR"], env["DG_JIT_CACHE_DIR"])
        self.assertEqual(env["SGLANG_JIT_CACHE_DIR"], str(base / "sglang/jit"))
        self.assertEqual(
            env["SGLANG_CUTE_AOT_CACHE_DIR"], str(base / "sglang/cute_aot")
        )
        self.assertEqual(env["FLASHINFER_WORKSPACE_BASE"], str(base / "flashinfer"))
        self.assertEqual(env["TRITON_CACHE_AUTOTUNING"], "1")
        self.assertEqual(env["SGLANG_FLASHINFER_AUTOTUNE_CACHE"], "1")
        self.assertEqual(env["SGLANG_JIT_DEEPGEMM_FAST_WARMUP"], "1")
        for key in ("TRITON_CACHE_DIR", "TILELANG_CACHE_DIR", "SGLANG_DG_CACHE_DIR"):
            self.assertTrue(Path(env[key]).is_dir())

    def test_preview_does_not_touch_disk_and_isolates_namespaces(self):
        first = self.run_script("--print-env")
        second = self.run_script("--print-env", namespace="trainer-sm90")
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertFalse(self.root.exists())
        a, b = json.loads(first.stdout), json.loads(second.stdout)
        for key in ("TRITON_CACHE_DIR", "TILELANG_CACHE_DIR", "SGLANG_CACHE_DIR"):
            self.assertNotEqual(a[key], b[key])

    def test_explicit_paths_and_disable_switches_win(self):
        overrides = {
            "SGLANG_CACHE_DIR": str(Path(self.temp.name) / "existing-sglang"),
            "TRITON_CACHE_DIR": str(Path(self.temp.name) / "existing-triton"),
            "SGLANG_CUTE_AOT_CACHE_DIR": "",
            "TILELANG_DISABLE_CACHE": "1",
            "TILELANG_AUTO_TUNING_DISABLE_CACHE": "1",
            "TRITON_CACHE_AUTOTUNING": "0",
            "SGLANG_FLASHINFER_AUTOTUNE_CACHE": "0",
            "SGLANG_JIT_DEEPGEMM_FAST_WARMUP": "0",
        }
        result = self.run_script("--print-env", env=overrides)
        self.assertEqual(result.returncode, 0, result.stderr)
        env = json.loads(result.stdout)
        for key, value in overrides.items():
            self.assertEqual(env[key], value)
        self.assertEqual(
            env["SGLANG_DG_CACHE_DIR"], overrides["SGLANG_CACHE_DIR"] + "/deep_gemm"
        )

    def test_deepgemm_override_is_honored_under_either_name(self):
        path = str(Path(self.temp.name) / "existing-deepgemm")
        for key in ("DG_JIT_CACHE_DIR", "SGLANG_DG_CACHE_DIR"):
            with self.subTest(key=key):
                result = self.run_script("--print-env", env={key: path})
                self.assertEqual(result.returncode, 0, result.stderr)
                env = json.loads(result.stdout)
                self.assertEqual(env["DG_JIT_CACHE_DIR"], path)
                self.assertEqual(env["SGLANG_DG_CACHE_DIR"], path)

    def test_conflicting_deepgemm_overrides_fail_before_command(self):
        result = self.run_script(
            "--",
            sys.executable,
            "-c",
            "print('must not run')",
            env={"DG_JIT_CACHE_DIR": "/one", "SGLANG_DG_CACHE_DIR": "/two"},
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("must agree", result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertFalse(self.root.exists())

    def test_exec_preserves_exit_status(self):
        result = self.run_script("--", sys.executable, "-c", "raise SystemExit(23)")
        self.assertEqual(result.returncode, 23, result.stderr)

    def test_bad_namespace_and_missing_command_do_not_create_directories(self):
        for namespace in ("../escape", "/absolute", ".", ".."):
            with self.subTest(namespace=namespace):
                result = self.run_script("--print-env", namespace=namespace)
                self.assertEqual(result.returncode, 2)
        self.assertEqual(self.run_script().returncode, 2)
        self.assertFalse(self.root.exists())

    def test_unavailable_cache_mount_fails_before_command(self):
        self.root.write_text("not a directory")
        result = self.run_script("--", sys.executable, "-c", "print('must not run')")
        self.assertEqual(result.returncode, 1)
        self.assertIn("cannot prepare cache directory", result.stderr)
        self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
