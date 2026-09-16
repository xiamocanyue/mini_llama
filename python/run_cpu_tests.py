#!/usr/bin/env python3
"""Build and run the CPU suite, writing a self-contained Markdown report."""

import argparse
import datetime
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def select_build_dir(requested):
    cache = requested / "CMakeCache.txt"
    if cache.is_file():
        entries = dict(line.split("=", 1) for line in
                       cache.read_text(encoding="utf-8").splitlines()
                       if "=" in line and not line.startswith(("#", "//")))
        expected = {"CMAKE_HOME_DIRECTORY:INTERNAL": ROOT,
                    "CMAKE_CACHEFILE_DIR:INTERNAL": requested}
        if any(key in entries and Path(entries[key]).resolve() != path.resolve()
               for key, path in expected.items()):
            replacement = Path(tempfile.mkdtemp(prefix="build-cpu-relocated-", dir=ROOT))
            return replacement, f"Relocated CMake cache in {requested}; preserved it and selected {replacement}."
    return requested, "Using requested build directory."


def main():
    parser = argparse.ArgumentParser(description="Run all mini-llama CPU tests")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build-cpu")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "result")
    parser.add_argument("--jobs", type=int, default=max(1, os.cpu_count() or 1))
    args = parser.parse_args()
    report_dir = args.report_dir.resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    build_dir, build_note = select_build_dir(args.build_dir.resolve())
    print(build_note, flush=True)
    steps = []
    outputs = []
    failure = None

    commands = [
        ("CMake CPU configure", ["cmake", "-S", str(ROOT), "-B", str(build_dir),
                                 "-DMINI_LLAMA_CUDA=OFF", "-DMINI_LLAMA_BUILD_TESTS=ON",
                                 "-DCMAKE_BUILD_TYPE=RelWithDebInfo"]),
        ("CPU compile and link", ["cmake", "--build", str(build_dir), "--parallel",
                                  str(args.jobs)]),
        ("CPU CTest", ["ctest", "--test-dir", str(build_dir), "-LE", "gpu",
                       "--output-on-failure", "--timeout", "600"]),
    ]
    for name, command in commands:
        try:
            print(f"[{name}] {' '.join(command)}", flush=True)
            chunks = []
            with subprocess.Popen(command, cwd=ROOT, text=True,
                                  stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, bufsize=1) as process:
                for line in process.stdout:
                    chunks.append(line)
                    print(line, end="", flush=True)
                returncode = process.wait()
            output = "".join(chunks)
            outputs.append((name, command, output))
            if returncode != 0:
                failure = f"{name} failed with exit code {returncode}"
                break
            steps.append(name)
        except OSError as error:
            failure = f"{name} could not start: {error}"
            break

    status = "PASS" if failure is None else "FAIL"
    lines = [
        "# CPU test result", "", f"- Status: **{status}**",
        f"- Time: `{datetime.datetime.now().astimezone().isoformat(timespec='seconds')}`",
        f"- Build directory: `{build_dir}`", f"- Build selection: {build_note}",
        "", "## Completed stages", "",
    ]
    if steps:
        lines.extend(f"- {step}" for step in steps)
    else:
        lines.append("- None")
    if failure:
        lines.extend(["", "## Failure", "", "```text", failure, "```"])
    lines.extend(["", "## Command output", ""])
    for name, command, output in outputs:
        lines.extend([f"### {name}", "", "```text", "$ " + " ".join(command),
                      output.rstrip(), "```", ""])
    (report_dir / "Cpu_result.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"{status}: report written to {report_dir / 'Cpu_result.md'}")
    return 0 if failure is None else 1


if __name__ == "__main__":
    sys.exit(main())
