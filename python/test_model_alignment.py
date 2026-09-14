#!/usr/bin/env python3
"""End-to-end C++ versus independent Python numerical alignment test."""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from compare_logits import compare


ROOT = Path(__file__).resolve().parents[1]


def run(command):
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if result.returncode != 0:
        raise AssertionError(
            f"command failed: {' '.join(map(str, command))}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")


def run_case(binary, prompt, n_predict):
    with tempfile.TemporaryDirectory(prefix="mini_llama_alignment_") as temporary:
        base = Path(temporary)
        cpp_dir = base / "cpp"
        python_dir = base / "python"
        run([
            str(binary), "generate",
            "--model", "models/tiny/model.bin",
            "--config", "models/tiny/model.json",
            "--tokenizer", "models/tiny/vocab.json",
            "--backend", "cpu", "--threads", "1", "--temperature", "0",
            "--seed", "1", "--prompt", prompt,
            "--n-predict", str(n_predict), "--dump-logits", str(cpp_dir),
        ])
        run([
            sys.executable, str(ROOT / "python" / "reference_forward.py"),
            "--config", "models/tiny/model.json",
            "--weights", "models/tiny/model.bin",
            "--vocab", "models/tiny/vocab.json",
            "--prompt", prompt, "--n-predict", str(n_predict),
            "--output-dir", str(python_dir),
        ])
        return compare(cpp_dir, python_dir, 128, 1e-4, 1e-4)


def main():
    parser = argparse.ArgumentParser(description="Test C++ model math against Python")
    parser.add_argument("--binary", default=str(ROOT / "build-cpu" / "mini-llama"))
    args = parser.parse_args()
    binary = Path(args.binary).resolve()
    if not binary.is_file():
        raise SystemExit(f"mini-llama binary not found: {binary}")

    cases = [("hello", 8), ("a!", 5)]
    overall_abs = 0.0
    overall_rel = 0.0
    total_steps = 0
    for prompt, n_predict in cases:
        steps, max_abs, max_rel, tokens = run_case(binary, prompt, n_predict)
        total_steps += steps
        overall_abs = max(overall_abs, max_abs)
        overall_rel = max(overall_rel, max_rel)
        print(f"PASS prompt={prompt!r}: steps={steps}, tokens={tokens}, "
              f"max_abs={max_abs:.6e}, max_rel={max_rel:.6e}")

    print(f"PASS model alignment: cases={len(cases)}, steps={total_steps}, "
          f"max_abs={overall_abs:.6e}, max_rel={overall_rel:.6e}")


if __name__ == "__main__":
    main()
