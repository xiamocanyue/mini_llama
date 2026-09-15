#!/usr/bin/env python3
"""Build, test, and benchmark the CPU backend, saving a timestamped report."""

import argparse
import datetime as dt
import platform
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_stage(name, command, cwd=ROOT):
    result = subprocess.run(
        command, cwd=cwd, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    return {
        "name": name,
        "command": " ".join(str(item) for item in command),
        "returncode": result.returncode,
        "output": result.stdout or "",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run CPU tests and benchmarks and save a Markdown report")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build-cpu")
    parser.add_argument("--result-dir", type=Path, default=ROOT / "result")
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()

    build_dir = args.build_dir.resolve()
    result_dir = args.result_dir.resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    binary = build_dir / "mini-llama"
    stages = []

    stages.append(run_stage("configure", [
        "cmake", "-S", str(ROOT), "-B", str(build_dir),
        "-DMINI_LLAMA_CUDA=OFF", "-DMINI_LLAMA_BUILD_TESTS=ON",
        "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
    ]))
    if stages[-1]["returncode"] == 0:
        stages.append(run_stage("build", [
            "cmake", "--build", str(build_dir), "--parallel", str(args.jobs)
        ]))
    if binary.is_file():
        stages.append(run_stage("ctest", [
            "ctest", "--test-dir", str(build_dir), "--output-on-failure"
        ]))
        stages.append(run_stage("python-model-alignment", [
            sys.executable, str(ROOT / "python" / "test_model_alignment.py"),
            "--binary", str(binary)
        ]))
        for label, extra in (
                ("benchmark-native", []),
                ("benchmark-q8_0", ["--quant", "q8_0"]),
                ("benchmark-q4_0", ["--quant", "q4_0"])):
            stages.append(run_stage(label, [
                str(binary), "bench", "models/tiny", "--backend", "cpu",
                "--threads", "1", "--seed", "1", "--n-predict", "4",
                *extra
            ]))

    timestamp = dt.datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    report_path = result_dir / f"cpu_test_{timestamp}.md"
    passed = bool(stages) and all(stage["returncode"] == 0 for stage in stages)
    lines = [
        "# Mini Llama CPU Test Report", "",
        f"- Status: **{'PASS' if passed else 'FAIL'}**",
        f"- Time: `{dt.datetime.now().astimezone().isoformat(timespec='seconds')}`",
        f"- Platform: `{platform.platform()}`",
        f"- Build directory: `{build_dir}`", "",
        "## Summary", "",
    ]
    for stage in stages:
        status = "PASS" if stage["returncode"] == 0 else "FAIL"
        lines.append(f"- **{status}** `{stage['name']}` (exit {stage['returncode']})")
    if not stages:
        lines.append("- **FAIL** no stages were executed")
    lines.extend(["", "## Stage Output", ""])
    for stage in stages:
        lines.extend([
            f"### {stage['name']}", "", f"Command: `{stage['command']}`", "",
            "```text", stage["output"].rstrip(), "```", ""
        ])
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"{'PASS' if passed else 'FAIL'}: report written to {report_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
