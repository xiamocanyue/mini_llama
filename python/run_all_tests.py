#!/usr/bin/env python3
"""One-command CPU + CUDA test runner and final report generator."""

import argparse
import datetime
import subprocess
import sys
import hashlib
import json
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(name, command, log_path):
    print(f"[{name}] {' '.join(command)}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        with subprocess.Popen(command, cwd=ROOT, text=True,
                              stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, bufsize=1) as process:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            return process.wait()


def main():
    parser = argparse.ArgumentParser(description="Run CPU and 4090 CUDA tests")
    parser.add_argument("--arch", default="89", help="CUDA arch; 4090D uses 89")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=None)
    parser.add_argument("--skip-sanitizer", action="store_true")
    parser.add_argument("--with-asan", action="store_true")
    parser.add_argument("--result-dir", type=Path, default=ROOT / "result")
    args = parser.parse_args()
    result_dir = args.result_dir.resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"format_version": 1, "run_id": datetime.datetime.now().astimezone().strftime("%Y%m%d_%H%M%S"),
                "root": str(ROOT), "python": sys.version, "platform": platform.platform(),
                "attention_sha256": hashlib.sha256((ROOT / "src/cuda_attention.cu").read_bytes()).hexdigest(),
                "command": " ".join(sys.argv)}
    (result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    jobs = str(args.jobs) if args.jobs else ""
    common = [sys.executable]
    cpu_command = common + [str(ROOT / "python" / "run_cpu_tests.py"),
                            "--report-dir", str(result_dir)]
    if jobs:
        cpu_command += ["--jobs", jobs]
    gpu_command = common + [str(ROOT / "python" / "run_gpu_tests.py"),
                            "--arch", args.arch, "--device", str(args.device),
                            "--report-dir", str(result_dir)]
    if jobs:
        gpu_command += ["--jobs", jobs]
    if args.skip_sanitizer:
        gpu_command.append("--skip-sanitizer")
    if args.with_asan:
        gpu_command.append("--with-asan")

    cpu_code = run("CPU", cpu_command, result_dir / "Cpu_runner.log")
    gpu_code = run("GPU", gpu_command, result_dir / "Gpu_runner.log")
    status = "PASS" if cpu_code == 0 and gpu_code == 0 else "FAIL"
    final = [
        "# Final test result", "", f"- Status: **{status}**",
        f"- Time: `{datetime.datetime.now().astimezone().isoformat(timespec='seconds')}`",
        f"- CPU report: [`Cpu_result.md`](Cpu_result.md)",
        f"- GPU report: [`Gpu_result.md`](Gpu_result.md)", "", "## Results", "",
        f"- CPU test exit code: `{cpu_code}`",
        f"- GPU test exit code: `{gpu_code}`", "",
        "The final status is PASS only when both reports pass. "
        "Inspect the two detailed reports and runner logs for numerical errors.", "",
    ]
    (result_dir / "Final_result.md").write_text("\n".join(final), encoding="utf-8")
    print(f"{status}: final report written to {result_dir / 'Final_result.md'}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
