#!/usr/bin/env python3
"""Compare tiny-model CUDA logits with CPU and the independent Python path."""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from compare_logits import collect, compare, read_floats, read_tokens


ROOT = Path(__file__).resolve().parents[1]
VOCAB_SIZE = 128


def run(command, *, optimized=False, input_text=None):
    env = {**os.environ, "MINI_LLAMA_CUDA_BATCH_PREFILL": str(int(optimized)),
           "MINI_LLAMA_CUDA_GREEDY": str(int(optimized))}
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, env=env,
                            input=input_text)
    if result.returncode != 0:
        raise AssertionError(
            f"command failed: {' '.join(map(str, command))}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    return result.stdout


def run_cpp(binary, backend, output_dir, prompt, n_predict, device, quant=None,
            temperature="0", top_k="0", optimized=False):
    command = [
        str(binary), "generate",
        "--model", "models/tiny/model.bin",
        "--config", "models/tiny/model.json",
        "--tokenizer", "models/tiny/vocab.json",
        "--backend", backend, "--threads", "1", "--temperature", temperature, "--top-k", top_k,
        "--seed", "1", "--prompt", prompt, "--n-predict", str(n_predict),
    ]
    if output_dir is not None:
        command.extend(["--dump-logits", str(output_dir)])
    if backend == "cuda":
        command.extend(["--device", str(device)])
    if quant:
        command.extend(["--quant", quant])
    return run(command, optimized=optimized)


def check_optimized_generation(binary, prompt, count, device, expected, quant=None):
    output = run_cpp(binary, "cuda", None, prompt, count, device, quant,
                     temperature="0.75", top_k="1", optimized=True)
    match = re.search(r"^generated tokens: (\[.*\])$", output, re.MULTILINE)
    if not match or json.loads(match[1]) != expected:
        raise AssertionError(f"Optimized generation differs from sequential diagnostics:\n{output}")
    if "cuda_batch" not in output or "sampling=cuda_argmax" not in output:
        raise AssertionError(f"Optimized generation did not exercise both CUDA paths:\n{output}")


def check_benchmark_paths(binary, device):
    output = run([
        str(binary), "bench", "models/tiny", "--backend", "cuda", "--device", str(device),
        "--threads", "1", "--seed", "1", "--prompt-tokens", "17", "--n-predict", "3", "--json",
    ], optimized=True)
    sample = json.loads(output.splitlines()[-1].removeprefix("benchmark_json: "))["samples"][0]
    if not sample["cuda_batch_prefill"] or not sample["cuda_greedy"]:
        raise AssertionError(f"Benchmark did not exercise both CUDA paths: {sample}")
    cuda = sample["cuda"]
    if cuda["prefill"]["linear_calls"] != 15 or cuda["prefill"]["d2h_copies"] != 0:
        raise AssertionError(f"Expected one output projection and device logits during Prefill: {cuda}")
    if (cuda["first_sample"]["d2h_copies"] != 1 or cuda["first_sample"]["d2h_bytes"] != 4 or
            cuda["decode"]["d2h_copies"] != sample["decode_steps"] or
            cuda["decode"]["d2h_bytes"] != sample["decode_steps"] * 4):
        raise AssertionError(f"Expected only token IDs downloaded during greedy sampling: {cuda}")


def check_optimized_chat(binary, device):
    command = [str(binary), "run", "models/tiny", "--backend", "cuda", "--device", str(device),
               "--seed", "1", "--n-predict", "1"]
    output = run(command, optimized=True, input_text="hello\nagain\n/clear\nhello\n/exit\n")
    prefill = re.findall(r"stage=prefill .*?prefix_reuse=(\d+), prefill=cuda_batch", output)
    if len(prefill) != 3 or int(prefill[0]) != 0 or int(prefill[1]) <= 0 or int(prefill[2]) != 0:
        raise AssertionError(f"Expected a batched prefix hit and reset in chat:\n{output}")
    if output.count("sampling=cuda_argmax") != 3 or "Chat history cleared." not in output:
        raise AssertionError(f"Expected device greedy sampling for each chat response:\n{output}")


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
        diagnostics = run_cpp(binary, "cuda", cuda_dir, "hello", 5, args.device, optimized=True)
        if "cuda_batch" in diagnostics or "sampling=cuda_argmax" in diagnostics:
            raise AssertionError("The logit dump must keep the sequential host-logits path")
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
        check_optimized_generation(binary, "hello", 5, args.device, cpu_cuda[3])
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
            check_optimized_generation(binary, "a!", 3, args.device, result[3], quant)
            print(f"PASS {quant} CPU/CUDA: steps={result[0]}, "
                  f"max_abs={result[1]:.6e}, max_rel={result[2]:.6e}, "
                  f"tokens={result[3]}")
        random_output = run_cpp(binary, "cuda", None, "hello", 2, args.device,
                                temperature="0.75", top_k="3", optimized=True)
        if "cuda_batch" not in random_output or "sampling=cuda_argmax" in random_output:
            raise AssertionError("Random sampling must use the host sampler after batched Prefill")
        check_benchmark_paths(binary, args.device)
        check_optimized_chat(binary, args.device)
        print("PASS gpu-model-alignment-test")
    finally:
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    main()
