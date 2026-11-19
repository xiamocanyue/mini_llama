#!/usr/bin/env python3
"""Exercise the benchmark CLI and the runner with the tiny CPU model."""

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

from benchmark_backends import BENCH_PREFIX, ROOT


TINY = ROOT / "models/tiny"


def run(command, error=None):
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if error is None and result.returncode == 0:
        return result.stdout
    if error is not None and result.returncode != 0 and error in result.stderr:
        return None
    raise AssertionError(f"Unexpected result from {command}:\n{result.stdout}\n{result.stderr}")


def bench(binary, *options):
    output = run([binary, "bench", TINY, "--threads", "1", "--seed", "1", "--json", *options])
    line = output.splitlines()[-1]
    assert line.startswith(BENCH_PREFIX), line
    return json.loads(line[len(BENCH_PREFIX):])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--check-no-cuda", action="store_true")
    args = parser.parse_args()
    binary = args.binary.resolve()

    for limit in (0, 1, 2):
        record = bench(binary, "--n-predict", str(limit), "--warmup", "1", "--repeats", "2")
        assert record["prompt_ids"] == [1, 104, 101, 108, 108, 111]
        assert record["linear_weight_types"]["f32"] > 0
        assert len(record["samples"]) == 2
        for sample in [record["cold_sample"], *record["samples"]]:
            assert sample["prompt_tokens"] == 6
            assert sample["generated_tokens"] == limit
            assert sample["generated_ids"] == [61, 92][:limit]
            assert sample["decode_steps"] == max(0, limit - 1)
            assert (sample["first_token_ms"] is None) == (limit == 0)
            assert (sample["decode_ms"] == 0) == (limit <= 1)
            assert sample["cuda"] is None
            assert math.isclose(sample["total_ms"],
                                (sample["first_token_ms"] or sample["prefill_ms"]) + sample["decode_ms"])

    exact = bench(binary, "--prompt-tokens", "16", "-n", "0")
    assert exact["prompt_ids"] == [1] * 16 and exact["cold_sample"] is None
    for length, effective in ((127, 1), (128, 0)):
        full = bench(binary, "--prompt-tokens", str(length), "-n", "2")
        assert full["effective_predict"] == effective
        assert full["samples"][0]["generated_tokens"] == effective
        assert full["samples"][0]["decode_steps"] == 0

    base = [binary, "bench", TINY, "--threads", "1"]
    for options, error in (
        (["--prompt-tokens", "0"], "positive integer"),
        (["--prompt-tokens", "129"], "Prompt too long"),
        (["--prompt-tokens", "1", "-p", "hello"], "mutually exclusive"),
        (["-p", ""], "must not be empty"),
        (["-n", "-1"], "Invalid --n-predict"),
        (["--repeats", "0"], "positive integer"),
    ):
        run([*base, *options], error)
    if args.check_no_cuda:
        run([*base, "--backend", "cuda"], "CUDA backend was not built")

    with tempfile.TemporaryDirectory(prefix="mini-llama-benchmark-test-") as directory:
        temp = Path(directory)
        vocab = temp / "vocab.json"
        vocab.write_text('[{"id": 1, "content": "a"}]', encoding="utf-8")
        run([*base, "--tokenizer", vocab, "--prompt-tokens", "1"], "valid tokenizer BOS")
        vocab.write_text('{"a": 97}', encoding="utf-8")
        (temp / "merges.txt").write_text("", encoding="utf-8")
        (temp / "special_tokens.json").write_text("{}", encoding="utf-8")
        run([binary, "bench", TINY / "model.gguf", "--tokenizer", vocab, "--prompt-tokens", "1"],
            "valid tokenizer BOS")

        out = temp / "results"
        run([sys.executable, ROOT / "python/benchmark_backends.py", "--binary", binary,
             "--model", TINY / "model.bin", "--tokenizer", TINY / "vocab.json", "--out", out,
             "--threads", "1", "--quant", "model-native", "--predict", "1", "--repeats", "2",
             "--prompt", "a!", "--prompt-tokens", "16", "--prompt-tokens", "129"])
        result_dir, = out.iterdir()
        raw = json.loads((result_dir / "raw.json").read_text(encoding="utf-8"))
        assert len(raw["cases"]) == 2 and len(raw["skipped"]) == 1
        for case, count in zip(raw["cases"], (3, 16)):
            for sample in case["samples"]:
                assert sample["prompt_tokens"] == count
                assert math.isclose(sample["prefill_tok_s"], count * 1000 / sample["prefill_ms"])
                assert sample["decode_tok_s"] is None
        with (result_dir / "raw_timings.csv").open(encoding="utf-8", newline="") as file:
            assert len(list(csv.DictReader(file))) == 4
        summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
        assert all(case["decode_tok_s_median"] is None and "prefill_ms_p95" not in case
                   for case in summary["cases"])

    print("PASS benchmark CLI and runner")


if __name__ == "__main__":
    main()
