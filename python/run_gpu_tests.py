#!/usr/bin/env python3
"""Configure, build, run, and report the complete CUDA test suite."""

import argparse
import datetime
import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TestFailure(RuntimeError):
    pass


def tool(name):
    path = shutil.which(name)
    if not path:
        raise TestFailure(f"required program not found: {name}")
    return path


def capture(command, env=None):
    result = subprocess.run(
        command, cwd=ROOT, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise TestFailure(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"{result.stdout}")
    return result.stdout.strip()


def parse_cuda_version(nvcc_output):
    match = re.search(r"release\s+(\d+)\.(\d+)", nvcc_output)
    if not match:
        raise TestFailure("could not determine CUDA Toolkit version from nvcc")
    return int(match.group(1)), int(match.group(2))


def normalize_arch(value):
    value = value.strip().lower().replace("sm_", "").replace("compute_", "")
    if "." in value:
        major, minor = value.split(".", 1)
        value = major + minor
    if not value.isdigit():
        raise TestFailure(f"invalid CUDA architecture: {value}")
    return value


def run_logged(name, command, log_dir, env):
    print(f"[{name}] {' '.join(command)}", flush=True)
    with (log_dir / f"{name}.log").open("w", encoding="utf-8") as log:
        with subprocess.Popen(command, cwd=ROOT, env=env, text=True,
                              stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, bufsize=1) as process:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            returncode = process.wait()
    if returncode != 0:
        raise TestFailure(f"{name} failed with exit code {returncode}")


def write_report(path, status, details, completed, failure=None):
    now = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    lines = [
        "# GPU test run",
        "",
        f"- Status: **{status}**",
        f"- Time: `{now}`",
        f"- GPU: `{details.get('gpu', 'unknown')}`",
        f"- Compute capability: `{details.get('compute_capability', 'unknown')}`",
        f"- CMake CUDA architecture: `{details.get('arch', 'unknown')}`",
        f"- CUDA Toolkit: `{details.get('cuda', 'unknown')}`",
        f"- CUDA visible device: `{details.get('device', 'unknown')}`",
        f"- Attention source SHA-256: `{details.get('attention_sha256', 'unknown')}`",
        f"- Real GGUF model: `{details.get('real_model', 'not reached')}`",
        f"- Real GGUF SHA-256: `{details.get('real_model_sha256', 'not reached')}`",
        f"- Real tokenizer SHA-256: `{details.get('real_tokenizer_sha256', 'not reached')}`",
        "- Build policy: clean-first (recompile sources, do not reuse object files)",
        "",
        "## Completed stages",
        "",
    ]
    lines.extend(f"- {stage}" for stage in completed)
    if not completed:
        lines.append("- None")
    if failure:
        lines.extend(["", "## Failure", "", "```text", str(failure), "```"])
    lines.extend([
        "", "## Logs", "",
        "All command output is stored next to this report as `*.log` files.", "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Build and run all mini-llama GPU tests; failures are never skipped")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build-gpu")
    parser.add_argument("--report-dir", type=Path,
                        default=ROOT / "result")
    parser.add_argument("--arch", default="auto",
                        help="CUDA architecture, e.g. 89 for 4090D or 120 for RTX 5060")
    parser.add_argument("--device", type=int, default=0,
                        help="physical GPU index made visible to the tests")
    parser.add_argument("--jobs", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--skip-sanitizer", action="store_true")
    parser.add_argument("--with-asan", action="store_true",
                        help="also enable host AddressSanitizer (leak checks disabled)")
    args = parser.parse_args()

    report_dir = args.report_dir.resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    details = {"device": str(args.device)}
    completed = []
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.device)
    if args.with_asan:
        env["ASAN_OPTIONS"] = (
            "detect_leaks=0:protect_shadow_gap=0:abort_on_error=1:halt_on_error=1")

    try:
        attention_source = ROOT / "src" / "cuda_attention.cu"
        details["attention_sha256"] = hashlib.sha256(attention_source.read_bytes()).hexdigest()
        provenance = (f"Source root: {ROOT}\nAttention source: {attention_source}\n"
                      f"SHA-256: {details['attention_sha256']}\n")
        (report_dir / "00-gpu-source.log").write_text(provenance, encoding="utf-8")
        print(provenance, flush=True)
        nvidia_smi = tool("nvidia-smi")
        nvcc = tool("nvcc")
        cmake = tool("cmake")
        ctest = tool("ctest")
        tool(sys.executable)
        gpu_rows = capture([
            nvidia_smi, "--query-gpu=index,name,compute_cap",
            "--format=csv,noheader"], env=os.environ.copy()).splitlines()
        if args.device < 0 or args.device >= len(gpu_rows):
            raise TestFailure(
                f"GPU index {args.device} is unavailable; detected {len(gpu_rows)} GPU(s)")
        fields = [field.strip() for field in gpu_rows[args.device].split(",")]
        if len(fields) < 3:
            raise TestFailure(f"unexpected nvidia-smi output: {gpu_rows[args.device]}")
        details["gpu"] = fields[1]
        details["compute_capability"] = fields[2]
        detected_arch = normalize_arch(fields[2])
        arch = detected_arch if args.arch == "auto" else normalize_arch(args.arch)
        if arch != detected_arch:
            raise TestFailure(
                f"requested architecture {arch} does not match selected GPU "
                f"architecture {detected_arch}")
        details["arch"] = arch
        nvcc_output = capture([nvcc, "--version"], env=env)
        cuda_version = parse_cuda_version(nvcc_output)
        details["cuda"] = f"{cuda_version[0]}.{cuda_version[1]}"
        if int(arch) >= 120 and cuda_version < (12, 8):
            raise TestFailure(
                "RTX 50-series/sm_120 requires CUDA Toolkit 12.8 or newer; "
                f"detected {details['cuda']}")
        completed.append("Environment and GPU capability checks")

        configure = [
            cmake, "-S", str(ROOT), "-B", str(args.build_dir.resolve()),
            "-DMINI_LLAMA_CUDA=ON", "-DMINI_LLAMA_BUILD_TESTS=ON",
            "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
            f"-DCMAKE_CUDA_ARCHITECTURES={arch}",
        ]
        if args.with_asan:
            configure.extend([
                "-DCMAKE_CXX_FLAGS=-fsanitize=address -fno-omit-frame-pointer -g -fno-pie",
                "-DCMAKE_EXE_LINKER_FLAGS=-fsanitize=address -no-pie",
            ])
        run_logged("01-configure", configure, report_dir, env)
        completed.append("CMake CUDA configuration")
        run_logged("02-build", [cmake, "--build", str(args.build_dir.resolve()),
                                 "--clean-first", "--parallel", str(args.jobs)], report_dir, env)
        completed.append("CUDA compile and link")
        run_logged("03-gpu-ctest", [
            ctest, "--test-dir", str(args.build_dir.resolve()), "-L", "gpu",
            "--output-on-failure", "--timeout", "600"], report_dir, env)
        completed.append("GPU core and model-alignment CTest")

        real_model = ROOT / "models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf"
        real_tokenizer = ROOT / "models/chat/vocab.json"
        if not real_model.is_file() or not real_tokenizer.is_file():
            raise TestFailure("real GGUF chat model or tokenizer is missing: "
                              f"{real_model}, {real_tokenizer}")
        details["real_model"] = str(real_model)
        details["real_model_sha256"] = hashlib.sha256(real_model.read_bytes()).hexdigest()
        details["real_tokenizer_sha256"] = hashlib.sha256(real_tokenizer.read_bytes()).hexdigest()
        run_logged("04-real-gguf-generate", [
            str(args.build_dir.resolve() / "mini-llama"), "generate",
            "--model", str(real_model), "--tokenizer", str(real_tokenizer),
            "--backend", "cuda", "--device", "0", "--threads", "1",
            "--prompt", "你好，请用一句话介绍自己。", "--temperature", "0",
            "--seed", "1", "--n-predict", "8"], report_dir, env)
        completed.append("Real Qwen2 GGUF CUDA generation")

        if not args.skip_sanitizer:
            sanitizer = shutil.which("compute-sanitizer")
            if not sanitizer:
                raise TestFailure(
                    "compute-sanitizer not found; install it or explicitly use "
                    "--skip-sanitizer")
            run_logged("05-compute-sanitizer-core", [
                sanitizer, "--tool", "memcheck", "--error-exitcode", "99",
                str(args.build_dir.resolve() / "gpu-core-test")], report_dir, env)
            completed.append("compute-sanitizer memcheck on GPU core test")
            run_logged("06-compute-sanitizer-generate", [
                sanitizer, "--tool", "memcheck", "--error-exitcode", "99",
                str(args.build_dir.resolve() / "mini-llama"), "generate",
                "--model", str(real_model), "--tokenizer", str(real_tokenizer),
                "--backend", "cuda", "--device", "0", "--prompt",
                "你好", "--temperature", "0", "--seed", "1", "--n-predict", "2"], report_dir, env)
            completed.append("compute-sanitizer memcheck on real GGUF CUDA generation")
            for number, tool_name in (("07", "racecheck"), ("08", "synccheck")):
                run_logged(f"{number}-compute-sanitizer-{tool_name}", [
                    sanitizer, "--tool", tool_name, "--error-exitcode", "99",
                    str(args.build_dir.resolve() / "gpu-core-test")], report_dir, env)
                completed.append(f"compute-sanitizer {tool_name} on GPU core test")

        write_report(report_dir / "Gpu_result.md", "PASS", details, completed)
        print(f"PASS: report written to {report_dir / 'Gpu_result.md'}")
        return 0
    except (OSError, subprocess.SubprocessError, TestFailure) as error:
        write_report(report_dir / "Gpu_result.md", "FAIL", details,
                     completed, error)
        print(f"FAIL: {error}", file=sys.stderr)
        print(f"report written to {report_dir / 'Gpu_result.md'}",
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
