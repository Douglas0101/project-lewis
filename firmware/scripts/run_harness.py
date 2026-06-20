#!/usr/bin/env python3
"""Project-Lewis Firmware Test Harness Runner.

Orquestra a compilacao e execucao do harness em dois ambientes:
  - native: executavel host x86_64 (make harness-native)
  - renode: binario ARM executado no Renode (make harness-renode)

A saida UART e parseada e convertida em `firmware/test_harness_report.json`.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIRMWARE_ROOT = PROJECT_ROOT
BUILD_DIR = FIRMWARE_ROOT / "build"
HARNESS_ELF_NATIVE = BUILD_DIR / "native" / "lewis_harness"
HARNESS_BIN_STM = BUILD_DIR / "stm32f4" / "lewis_harness.bin"
HARNESS_REPORT = FIRMWARE_ROOT / "test_harness_report.json"
RENODE_SCRIPT = FIRMWARE_ROOT / "renode" / "harness.resc"
RENODE_UART_LOG = Path("/tmp/renode_harness_uart.log")
RENODE_TIMEOUT_S = 30

HARNESS_LINE_RE = re.compile(
    r"^HARNESS\s+(?P<suite>\S+)\s+(?P<name>\S+)\s+(?P<status>PASS|FAIL)(?:\s+(?P<detail>.+))?"
)
SUMMARY_RE = re.compile(
    r"^HARNESS\s+SUMMARY\s+PASS\s+(?P<pass>\d+)\s+FAIL\s+(?P<fail>\d+)\s+TOTAL\s+(?P<total>\d+)"
)


def generate_fixtures(num_beats: int = 5) -> bool:
    """Gera headers C com fixtures a partir das referencias Python."""
    script = FIRMWARE_ROOT / "scripts" / "generate_harness_fixtures.py"
    if not script.exists():
        print(f"WARNING: fixture generator not found: {script}", file=sys.stderr)
        return False
    print("=== Generating harness fixtures ===")
    r = run(
        [sys.executable, str(script), "--num-beats", str(num_beats)],
        cwd=FIRMWARE_ROOT,
        timeout=120,
    )
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr, file=sys.stderr)
        return False
    return True


def run(cmd: list[str], cwd: Path, timeout: int = 120) -> subprocess.CompletedProcess:
    print("$ " + " ".join(cmd))
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, timeout=timeout, stdin=subprocess.DEVNULL)


def parse_output(text: str) -> dict:
    results = []
    summary = {"pass": 0, "fail": 0, "total": 0}
    for line in text.splitlines():
        line = line.strip()
        m = HARNESS_LINE_RE.match(line)
        if m:
            results.append(
                {
                    "suite": m.group("suite"),
                    "name": m.group("name"),
                    "status": m.group("status"),
                    "detail": m.group("detail") or "",
                }
            )
            continue
        s = SUMMARY_RE.match(line)
        if s:
            summary = {
                "pass": int(s.group("pass")),
                "fail": int(s.group("fail")),
                "total": int(s.group("total")),
            }
    return {"summary": summary, "tests": results}


def build_native() -> int:
    r = run(["make", str(HARNESS_ELF_NATIVE)], cwd=FIRMWARE_ROOT, timeout=300)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr, file=sys.stderr)
    return r.returncode


def build_renode() -> int:
    r = run(["make", str(HARNESS_BIN_STM)], cwd=FIRMWARE_ROOT, timeout=300)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr, file=sys.stderr)
    return r.returncode


def run_native() -> dict:
    print("Running native harness executable: %s" % HARNESS_ELF_NATIVE, flush=True)
    r = run([str(HARNESS_ELF_NATIVE)], cwd=FIRMWARE_ROOT, timeout=60)
    print("Native harness finished with return code %d" % r.returncode, flush=True)
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
    return parse_output(r.stdout)


def run_renode(renode: Path) -> dict:
    # Limpa log anterior para evitar leitura de execucoes antigas.
    RENODE_UART_LOG.unlink(missing_ok=True)

    # Usa o utilitario `timeout` do sistema para limitar a execucao do Renode.
    cmd = [
        "timeout",
        "--signal=KILL",
        str(RENODE_TIMEOUT_S),
        str(renode),
        "--disable-xwt",
        "--console",
        str(RENODE_SCRIPT),
    ]
    print("Running Renode harness (timeout %ds)..." % RENODE_TIMEOUT_S, flush=True)
    r = run(cmd, cwd=FIRMWARE_ROOT, timeout=RENODE_TIMEOUT_S + 10)
    print("Renode finished with return code %d" % r.returncode, flush=True)
    print(r.stdout)
    if r.returncode not in (0, 124, 137):
        # 124 = timeout encerrou, 137 = SIGKILL (comportamento esperado)
        print(r.stderr, file=sys.stderr)

    text = ""
    if RENODE_UART_LOG.exists():
        text = RENODE_UART_LOG.read_text()
        print("UART log size: %d bytes" % len(text), flush=True)
    else:
        print("UART log not found at %s" % RENODE_UART_LOG, flush=True)
    return parse_output(text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Project-Lewis firmware harness")
    parser.add_argument(
        "--mode",
        choices=["native", "renode", "both"],
        default="both",
        help="Execution mode",
    )
    parser.add_argument(
        "--renode",
        type=Path,
        default=FIRMWARE_ROOT / "tools" / "renode-1.15.3" / "renode",
        help="Path to Renode executable",
    )
    args = parser.parse_args()

    report = {"native": None, "renode": None}

    if not generate_fixtures():
        print("ERROR: fixture generation failed", file=sys.stderr)
        return 1

    if args.mode in ("native", "both"):
        print("=== Building harness (native) ===")
        rc = build_native()
        if rc != 0:
            print("ERROR: native build failed", file=sys.stderr)
            return rc
        print("=== Running harness (native) ===")
        report["native"] = run_native()

    if args.mode in ("renode", "both"):
        print("=== Building harness (Renode) ===")
        rc = build_renode()
        if rc != 0:
            print("ERROR: renode build failed", file=sys.stderr)
            return rc
        print("=== Running harness (Renode) ===")
        report["renode"] = run_renode(args.renode)

    # Mescla com relatorio existente para preservar resultados de outro ambiente.
    existing = {"native": None, "renode": None}
    if HARNESS_REPORT.exists():
        try:
            existing = json.loads(HARNESS_REPORT.read_text())
        except json.JSONDecodeError:
            existing = {"native": None, "renode": None}
    if args.mode == "native":
        existing["native"] = report["native"]
    elif args.mode == "renode":
        existing["renode"] = report["renode"]
    elif args.mode == "both":
        existing = report

    HARNESS_REPORT.write_text(json.dumps(existing, indent=2))
    print(f"Report written to {HARNESS_REPORT}")

    failures = 0
    if existing.get("native"):
        failures += existing["native"]["summary"]["fail"]
    if existing.get("renode"):
        failures += existing["renode"]["summary"]["fail"]

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
