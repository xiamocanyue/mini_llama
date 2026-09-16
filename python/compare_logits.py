#!/usr/bin/env python3
"""Compare C++ float32 logits and independent Python reference outputs."""

import argparse
import math
import re
import struct
import json
from pathlib import Path


def collect(directory, prefix):
    pattern = re.compile(rf"{prefix}logits_step(\d+)\.bin$")
    result = {}
    for path in Path(directory).glob(f"{prefix}logits_step*.bin"):
        match = pattern.fullmatch(path.name)
        if match:
            result[int(match.group(1))] = path
    return result


def read_floats(path, expected_size):
    raw = path.read_bytes()
    if len(raw) != expected_size * 4:
        raise AssertionError(
            f"{path} contains {len(raw) // 4} floats, expected {expected_size}")
    return struct.unpack(f"<{expected_size}f", raw)


def read_tokens(path):
    text = path.read_text(encoding="utf-8").strip()
    return [] if not text else [int(value) for value in text.split()]


def compare(cpp_dir, python_dir, vocab_size, abs_tolerance, rel_tolerance):
    cpp_files = collect(cpp_dir, "")
    python_files = collect(python_dir, "python_")
    if not cpp_files or cpp_files.keys() != python_files.keys():
        raise AssertionError(
            f"logit step mismatch: cpp={sorted(cpp_files)}, python={sorted(python_files)}")

    worst_abs = 0.0
    worst_rel = 0.0
    rows = []
    for step in sorted(cpp_files):
        cpp = read_floats(cpp_files[step], vocab_size)
        reference = read_floats(python_files[step], vocab_size)
        if not all(math.isfinite(value) for value in cpp + reference):
            raise AssertionError(f"step {step} contains NaN or infinity")
        differences = [abs(a - b) for a, b in zip(cpp, reference)]
        max_abs = max(differences)
        relative = [difference / abs(expected)
                    for difference, expected in zip(differences, reference)
                    if abs(expected) >= 1e-3]
        max_rel = max(relative, default=0.0)
        mae = sum(differences) / len(differences)
        rmse = math.sqrt(sum(d*d for d in differences) / len(differences))
        cpp_top1 = max(range(vocab_size), key=cpp.__getitem__)
        python_top1 = max(range(vocab_size), key=reference.__getitem__)
        if max_abs > abs_tolerance or max_rel > rel_tolerance or cpp_top1 != python_top1:
            raise AssertionError(
                f"step {step}: max_abs={max_abs:.6e}, max_rel={max_rel:.6e}, "
                f"top1 cpp={cpp_top1} python={python_top1}")
        worst_abs = max(worst_abs, max_abs)
        worst_rel = max(worst_rel, max_rel)
        rows.append({"step": step, "max_abs": max_abs, "mae": mae,
                     "rmse": rmse, "max_rel": max_rel,
                     "finite": True, "top1_match": cpp_top1 == python_top1,
                     "top1_cpp": cpp_top1, "top1_reference": python_top1,
                     "relative_count": len(relative),
                     "relative_excluded": vocab_size - len(relative)})

    cpp_tokens = read_tokens(Path(cpp_dir) / "generation_tokens.txt")
    python_tokens = read_tokens(Path(python_dir) / "python_generation_tokens.txt")
    if cpp_tokens != python_tokens:
        raise AssertionError(f"generation mismatch: cpp={cpp_tokens}, python={python_tokens}")
    return len(cpp_files), worst_abs, worst_rel, cpp_tokens, rows


def main():
    parser = argparse.ArgumentParser(description="Compare C++ and Python logits")
    parser.add_argument("--cpp-dir", required=True)
    parser.add_argument("--python-dir", required=True)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--abs-tolerance", type=float, default=1e-4)
    parser.add_argument("--rel-tolerance", type=float, default=1e-4)
    args = parser.parse_args()
    steps, max_abs, max_rel, tokens, rows = compare(
        args.cpp_dir, args.python_dir, args.vocab_size,
        args.abs_tolerance, args.rel_tolerance)
    print(f"PASS: {steps} steps")
    print(f"max_abs_error: {max_abs:.6e}")
    print(f"max_rel_error: {max_rel:.6e} (reference magnitude >= 1e-3)")
    print(f"generation tokens: {tokens}")
    output = Path(args.cpp_dir).parent / "numerical_errors.json"
    output.write_text(json.dumps({"steps": rows, "summary": {
        "step_count": steps, "max_abs": max_abs, "max_rel": max_rel,
        "tokens": tokens, "relative_threshold": 1e-3}}, indent=2), encoding="utf-8")
    print(f"numerical report: {output}")


if __name__ == "__main__":
    main()
