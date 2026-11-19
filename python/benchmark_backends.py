#!/usr/bin/env python3
"""Run mini-llama bench matrices and retain per-request timings."""

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
BENCH_PREFIX = "benchmark_json: "
TIMING_FIELDS = (
    "context_ms", "prefill_ms", "first_token_ms", "decode_ms", "total_ms",
    "prefill_tok_s", "decode_tok_s",
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path):
    return {"path": str(path), "sha256": sha256(path)}


def binary_identity(path):
    identity = file_identity(path)
    cache = path.parent / "CMakeCache.txt"
    if cache.is_file():
        keys = (
            "CMAKE_BUILD_TYPE:", "CMAKE_CXX_COMPILER:", "CMAKE_CUDA_COMPILER:",
            "CMAKE_CUDA_ARCHITECTURES:", "MINI_LLAMA_CUDA:",
        )
        identity["cmake"] = dict(
            line.split("=", 1) for line in cache.read_text().splitlines()
            if line.startswith(keys)
        )
    return identity


def device_memory():
    """Take a device-wide snapshot outside the benchmark process."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=pci.bus_id,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=5,
        )
        return {bus.strip(): int(mib) for bus, mib in csv.reader(result.stdout.splitlines())}
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def run_case(command, backend, log_path):
    before = device_memory() if backend == "cuda" else None
    start = time.perf_counter()
    process = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    process_elapsed_ms = (time.perf_counter() - start) * 1000
    after = device_memory() if backend == "cuda" else None
    log_path.write_text(process.stdout + process.stderr, encoding="utf-8")
    if process.returncode:
        return process.stderr.strip() or f"Exit code {process.returncode}", None

    lines = process.stdout.splitlines()
    if not lines or not lines[-1].startswith(BENCH_PREFIX):
        raise ValueError(f"Expected a final benchmark JSON record. See {log_path}")
    record = json.loads(lines[-1][len(BENCH_PREFIX):])
    if record["version"] != 1 or record["backend"] != backend:
        raise ValueError(f"Unexpected benchmark version or backend. See {log_path}")
    record["process_elapsed_ms"] = process_elapsed_ms
    record["cuda_device_summary"] = next(
        (line.strip() for line in lines if backend == "cuda" and line.startswith("  cuda: ")),
        None,
    )
    delta = None
    if before is not None and after is not None:
        delta = {bus: after[bus] - before[bus] for bus in before.keys() & after.keys()}
    record["idle_device_memory_mib"] = {"before": before, "after": after, "delta": delta}
    return None, record


def add_rates(sample):
    def rate(count, ms):
        return count * 1000 / ms if count > 0 and ms > 0 else None

    return {
        **sample,
        "prefill_tok_s": rate(sample["prompt_tokens"], sample["prefill_ms"]),
        "decode_tok_s": rate(sample["decode_steps"], sample["decode_ms"]),
    }


def summarize(samples):
    summary = {"samples": len(samples)}
    for field in TIMING_FIELDS:
        values = sorted(sample[field] for sample in samples if sample[field] is not None)
        summary[field + "_median"] = statistics.median(values) if values else None
        if len(values) >= 20:
            summary[field + "_p95"] = values[math.ceil(0.95 * len(values)) - 1]
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "build-cpu/mini-llama")
    parser.add_argument("--cuda-binary", type=Path)
    parser.add_argument("--cuda-only", action="store_true", help="skip CPU measurements")
    parser.add_argument("--model", type=Path, default=ROOT / "models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "models/chat/vocab.json")
    parser.add_argument("--out", type=Path, default=ROOT / "result/benchmark")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--predict", type=int, default=16)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--threads", default="1,4", help="comma-separated CPU thread counts")
    parser.add_argument("--cuda-threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--prompt", action="append", help="text prompt case, repeatable")
    parser.add_argument("--prompt-tokens", type=int, action="append", help="exact BOS count, repeatable")
    parser.add_argument("--quant", default="model-native,q8_0,q4_0",
                        help="comma-separated model-native, q8_0, q4_0 (legacy f32 means model-native)")
    args = parser.parse_args()

    if args.cuda_only and not args.cuda_binary:
        parser.error("--cuda-only requires --cuda-binary")
    if args.repeats <= 0 or min(args.warmup, args.predict, args.device, args.cuda_threads) < 0:
        parser.error("--repeats must be positive and --warmup, --predict, --device, --cuda-threads non-negative")
    if not 0 <= args.seed <= 2**32 - 1:
        parser.error("--seed must be an unsigned 32-bit integer")
    if any(not prompt for prompt in args.prompt or []):
        parser.error("--prompt must not be empty")
    if any(count <= 0 for count in args.prompt_tokens or []):
        parser.error("--prompt-tokens must be positive")
    try:
        args.cpu_threads = [int(value) for value in args.threads.split(",")]
    except ValueError:
        parser.error("--threads must be comma-separated integers")
    if any(count < 0 for count in args.cpu_threads):
        parser.error("--threads must be non-negative")

    args.quantizations = []
    for value in args.quant.lower().split(","):
        value = value.strip()
        if value == "f32":
            value = "model-native"
        if value not in {"model-native", "q8_0", "q4_0"}:
            parser.error(f"unsupported --quant value: {value!r}")
        if value not in args.quantizations:
            args.quantizations.append(value)
    return args


def main():
    args = parse_args()
    model = args.model.resolve()
    tokenizer = args.tokenizer.resolve()
    backends = []
    if not args.cuda_only:
        backends.extend(("cpu", args.binary.resolve(), threads) for threads in args.cpu_threads)
    if args.cuda_binary:
        backends.append(("cuda", args.cuda_binary.resolve(), args.cuda_threads))
    prompts = [("--prompt", text) for text in (args.prompt or [])]
    prompts += [("--prompt-tokens", str(count)) for count in (args.prompt_tokens or [])]
    if not prompts:
        prompts = [("--prompt", "hello")]

    tokenizer_paths = [tokenizer]
    for filename in ("merges.txt", "special_tokens.json"):
        path = tokenizer.parent / filename
        if path.is_file():
            tokenizer_paths.append(path)
    metadata = {
        "version": 1,
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "model": file_identity(model),
        "tokenizer_files": [file_identity(path) for path in tokenizer_paths],
        "binaries": [binary_identity(path) for path in dict.fromkeys(item[1] for item in backends)],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
        "cuda_batch_prefill": os.environ.get("MINI_LLAMA_CUDA_BATCH_PREFILL"),
        "cuda_device_greedy": os.environ.get("MINI_LLAMA_CUDA_GREEDY"),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=args.out))
    cases, rows, summaries, skipped = [], [], [], []
    case_number = 0
    for backend, binary, threads in backends:
        context_window = None
        for quant in args.quantizations:
            for prompt_flag, prompt_value in prompts:
                case_number += 1
                case = {
                    "case": case_number, "backend": backend, "threads_requested": threads,
                    "quant_requested": quant, "prompt_flag": prompt_flag, "prompt": prompt_value,
                }
                synthetic = prompt_flag == "--prompt-tokens"
                if synthetic and context_window is not None and int(prompt_value) > context_window:
                    skipped.append({**case, "reason": f"Context window is {context_window} tokens"})
                    continue

                command = [
                    str(binary), "bench", str(model), "--tokenizer", str(tokenizer),
                    "--backend", backend, "--threads", str(threads), "--seed", str(args.seed),
                    "--n-predict", str(args.predict), "--warmup", str(args.warmup),
                    "--repeats", str(args.repeats), "--json", prompt_flag, prompt_value,
                ]
                if backend == "cuda":
                    command.extend(("--device", str(args.device)))
                if quant != "model-native":
                    command.extend(("--quant", quant))
                log_path = run_dir / f"case-{case_number:03d}.log"
                error, record = run_case(command, backend, log_path)
                if synthetic and error == "Prompt too long.":
                    skipped.append({**case, "reason": error, "command": command, "log": log_path.name})
                    continue
                if error:
                    raise SystemExit(f"Benchmark case {case_number} failed. See {log_path}\n{error}")
                context_window = record["context_window"]
                samples = [add_rates(sample) for sample in record["samples"]]
                if len(samples) != args.repeats:
                    raise ValueError(f"Unexpected sample count. See {log_path}")
                cold = record["cold_sample"]
                cases.append({**case, **record, "command": command, "log": log_path.name,
                              "cold_sample": add_rates(cold) if cold is not None else None,
                              "samples": samples})
                summaries.append({**case, "threads": record["threads"],
                                  "linear_weight_types": record["linear_weight_types"],
                                  **summarize(samples)})
                for index, sample in enumerate(samples):
                    rows.append({**case, "threads": record["threads"], "sample": index,
                                 "linear_weight_types": json.dumps(record["linear_weight_types"]),
                                 **sample, "generated_ids": json.dumps(sample["generated_ids"]),
                                 "cuda": json.dumps(sample["cuda"])})
    (run_dir / "raw.json").write_text(
        json.dumps({**metadata, "cases": cases, "skipped": skipped}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps({"cases": summaries, "skipped": skipped}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if not rows:
        raise SystemExit(f"No cases fit the model context window. See {run_dir}")
    with (run_dir / "raw_timings.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    print(f"PASS benchmark cases={len(cases)} skipped={len(skipped)} output={run_dir}")


if __name__ == "__main__":
    main()
