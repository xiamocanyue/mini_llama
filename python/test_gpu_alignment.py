#!/usr/bin/env python3
"""Compare tiny-model CUDA logits with CPU and the independent Python path."""

import argparse
import math
import subprocess
import sys
import tempfile
from pathlib import Path

from compare_logits import collect, compare, read_floats, read_tokens


ROOT = Path(__file__).resolve().parents[1]
VOCAB_SIZE = 128


def run(command):
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if result.returncode != 0:
        raise AssertionError(
            f"command failed: {' '.join(map(str, command))}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")


def run_cpp(binary, backend, output_dir, prompt, n_predict, device, quant=None):
    command = [
        str(binary), "generate",
        "--model", "models/tiny/model.bin",
        "--config", "models/tiny/model.json",
        "--tokenizer", "models/tiny/vocab.json",
        "--backend", backend, "--threads", "1", "--temperature", "0",
        "--seed", "1", "--prompt", prompt, "--n-predict", str(n_predict),
        "--dump-logits", str(output_dir),
    ]
    if backend == "cuda":
        command.extend(["--device", str(device)])
    if quant:
        command.extend(["--quant", quant])
    run(command)


def compare_cpp_outputs(left_dir, right_dir, abs_tolerance, rel_tolerance):
    left_files = collect(left_dir, "")
    right_files = collect(right_dir, "")
    if not left_files or left_files.keys() != right_files.keys():
        raise AssertionError(
            f"logit step mismatch: left={sorted(left_files)}, "
            f"right={sorted(right_files)}")
    worst_abs = 0.0
    worst_rel = 0.0
    for step in sorted(left_files):
        left = read_floats(left_files[step], VOCAB_SIZE)
        right = read_floats(right_files[step], VOCAB_SIZE)
        if not all(math.isfinite(value) for value in left + right):
            raise AssertionError(f"step {step} contains NaN or infinity")
        errors = [abs(a - b) for a, b in zip(left, right)]
        max_abs = max(errors)
        relative = [error / abs(expected)
                    for error, expected in zip(errors, left)
                    if abs(expected) >= 1e-3]
        max_rel = max(relative, default=0.0)
        left_top1 = max(range(VOCAB_SIZE), key=left.__getitem__)
        right_top1 = max(range(VOCAB_SIZE), key=right.__getitem__)
        if (max_abs > abs_tolerance or max_rel > rel_tolerance or
                left_top1 != right_top1):
            raise AssertionError(
                f"step {step}: max_abs={max_abs:.6e}, max_rel={max_rel:.6e}, "
                f"top1 left={left_top1} right={right_top1}")
        worst_abs = max(worst_abs, max_abs)
        worst_rel = max(worst_rel, max_rel)
    left_tokens = read_tokens(Path(left_dir) / "generation_tokens.txt")
    right_tokens = read_tokens(Path(right_dir) / "generation_tokens.txt")
    if left_tokens != right_tokens:
        raise AssertionError(
            f"generation mismatch: left={left_tokens}, right={right_tokens}")
    return len(left_files), worst_abs, worst_rel, left_tokens


def main():
    parser = argparse.ArgumentParser(
        description="Test CUDA model math against CPU and independent Python")
    parser.add_argument("--binary", default=str(ROOT / "build-gpu" / "mini-llama"))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--keep-output", type=Path)
    args = parser.parse_args()
    binary = Path(args.binary).resolve()
    if not binary.is_file():
        raise SystemExit(f"mini-llama binary not found: {binary}")

    temporary = None
    if args.keep_output:
        base = args.keep_output.resolve()
        base.mkdir(parents=True, exist_ok=True)
    else:
        temporary = tempfile.TemporaryDirectory(prefix="mini_llama_gpu_alignment_")
        base = Path(temporary.name)

    try:
        cpu_dir = base / "f32_cpu"
        cuda_dir = base / "f32_cuda"
        python_dir = base / "f32_python"
        run_cpp(binary, "cpu", cpu_dir, "hello", 5, args.device)
        run_cpp(binary, "cuda", cuda_dir, "hello", 5, args.device)
        run([
            sys.executable, str(ROOT / "python" / "reference_forward.py"),
            "--config", "models/tiny/model.json",
            "--weights", "models/tiny/model.bin",
            "--vocab", "models/tiny/vocab.json",
            "--prompt", "hello", "--n-predict", "5",
            "--output-dir", str(python_dir),
        ])
        cpu_python = compare(cpu_dir, python_dir, VOCAB_SIZE, 1e-4, 1e-4)
        cuda_python = compare(cuda_dir, python_dir, VOCAB_SIZE, 2e-3, 2e-3)
        cpu_cuda = compare_cpp_outputs(cpu_dir, cuda_dir, 2e-3, 2e-3)
        print("PASS F32 CPU/Python: "
              f"steps={cpu_python[0]}, max_abs={cpu_python[1]:.6e}, "
              f"max_rel={cpu_python[2]:.6e}")
        print("PASS F32 CUDA/Python: "
              f"steps={cuda_python[0]}, max_abs={cuda_python[1]:.6e}, "
              f"max_rel={cuda_python[2]:.6e}")
        print("PASS F32 CPU/CUDA: "
              f"steps={cpu_cuda[0]}, max_abs={cpu_cuda[1]:.6e}, "
              f"max_rel={cpu_cuda[2]:.6e}, tokens={cpu_cuda[3]}")

        for quant in ("q8_0", "q4_0"):
            quant_cpu = base / f"{quant}_cpu"
            quant_cuda = base / f"{quant}_cuda"
            run_cpp(binary, "cpu", quant_cpu, "a!", 3, args.device, quant)
            run_cpp(binary, "cuda", quant_cuda, "a!", 3, args.device, quant)
            result = compare_cpp_outputs(quant_cpu, quant_cuda, 3e-3, 3e-3)
            print(f"PASS {quant} CPU/CUDA: steps={result[0]}, "
                  f"max_abs={result[1]:.6e}, max_rel={result[2]:.6e}, "
                  f"tokens={result[3]}")
        print("PASS gpu-model-alignment-test")
    finally:
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    main()
