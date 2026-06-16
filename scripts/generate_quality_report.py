#!/usr/bin/env python3
"""Gera relatorio consolidado dos Quality Gates (QG0–QG9).

Executa os testes por marker usando ``--junitxml`` e coleta estatisticas
pass/fail/skip/error/xfail/xpass de forma robusta.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"

GATES = [
    ("QG0", "Download e integridade", "qg0"),
    ("QG1", "Resample, loader e pré-processamento", "qg1"),
    ("QG2", "AMPT detector @ 500 Hz", "qg2"),
    ("QG3", "Feature engineering e segmentação", "qg3"),
    ("QG4", "Pré-treino (Chapman)", "qg4"),
    ("QG5", "Fine-tuning (MIT-BIH+)", "qg5"),
    ("QG6", "Quantização e exportação TFLM", "qg6"),
    ("QG7", "Build do firmware", "qg7"),
    ("QG8", "Bit-exatidão TFLM", "qg8"),
    ("QG9", "Latência e memória do firmware", "qg9"),
]

DLQ_PATHS = [
    PROJECT_ROOT / "data" / ".dlq" / "failed_downloads.jsonl",
    PROJECT_ROOT / "data" / ".dlq" / "preprocess_failures.jsonl",
    PROJECT_ROOT / "data" / ".dlq" / "ci_failures.jsonl",
]


def _parse_junit_stats(xml_text: str) -> dict:
    """Extrai estatisticas de um XML JUnit gerado pelo pytest."""
    stats = {
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "error": 0,
        "xfail": 0,
        "xpass": 0,
        "total_collected": 0,
    }
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return stats

    testsuites = root if root.tag == "testsuites" else None
    if testsuites is None and root.tag == "testsuite":
        testsuites = root

    if testsuites is not None:
        attr_map = {
            "tests": "total_collected",
            "failures": "failed",
            "errors": "error",
            "skipped": "skipped",
        }
        for attr, key in attr_map.items():
            try:
                stats[key] += int(testsuites.get(attr, 0))
            except (TypeError, ValueError):
                pass

    # Contagem refinada por tag de testcase para xfail/xpass.
    for testcase in root.iter("testcase"):
        status = "passed"
        for child in testcase:
            if child.tag in ("failure", "error", "skipped"):
                status = child.tag
                message = (child.get("message") or "").lower()
                if "xfail" in message:
                    status = "xfail" if child.tag == "skipped" else status
                break
        # xpass: teste esperado para falhar mas passou -> nenhuma tag de falha.
        if status == "passed":
            for prop in testcase.iter("property"):
                if prop.get("name") == "pytest_result" and "xpass" in (
                    prop.get("value") or ""
                ).lower():
                    status = "xpass"
                    break

        if status == "failure":
            stats["failed"] += 1
        elif status == "error":
            stats["error"] += 1
        elif status == "skipped":
            stats["skipped"] += 1
        elif status == "xfail":
            stats["xfail"] += 1
        elif status == "xpass":
            stats["xpass"] += 1
        else:
            stats["passed"] += 1

    if stats["total_collected"] == 0:
        stats["total_collected"] = sum(
            stats[k] for k in ("passed", "failed", "skipped", "error", "xfail", "xpass")
        )
    return stats


def _run_pytest(marker: str, timeout: int = 300) -> dict:
    """Roda pytest para um marker e retorna estatisticas via JUnit XML."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as tmp:
        junit_path = tmp.name

    try:
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "tests/",
            "-m",
            marker,
            "-q",
            "--tb=no",
            "--no-header",
            f"--junitxml={junit_path}",
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT / "src")

        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )

        summary = result.stdout.strip().splitlines()[-1] if result.stdout else ""
        xml_text = Path(junit_path).read_text(encoding="utf-8") if Path(junit_path).exists() else ""
        stats = _parse_junit_stats(xml_text)
    finally:
        try:
            Path(junit_path).unlink(missing_ok=True)
        except OSError:
            pass

    stats.update(
        {
            "summary": summary,
            "returncode": result.returncode,
        }
    )

    if result.returncode == 0:
        stats["status"] = "✅ PASS"
    elif stats["failed"] == 0 and stats["error"] == 0 and stats["total_collected"] > 0:
        stats["status"] = "✅ PASS"
    elif stats["total_collected"] == 0:
        stats["status"] = "⚠️ SKIP"
    else:
        stats["status"] = "❌ FAIL"
    return stats


def _dlq_summary() -> tuple[int, list[str]]:
    """Conta entradas nao-vazias nos arquivos DLQ."""
    total = 0
    files_with_data: list[str] = []
    for path in DLQ_PATHS:
        if not path.exists():
            continue
        lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if lines:
            total += len(lines)
            files_with_data.append(str(path.relative_to(PROJECT_ROOT)))
    return total, files_with_data


def _git_info() -> dict:
    """Obtem commit e branch atuais, se disponiveis."""
    info = {"commit": "unknown", "branch": "unknown"}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        info["commit"] = commit.stdout.strip()
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        info["branch"] = branch.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return info


def generate_report(output_dir: Path) -> Path:
    """Executa os gates e gera ``quality_report.md``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "quality_report.md"

    git = _git_info()
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    rows = []
    for gate, name, marker in GATES:
        stats = _run_pytest(marker)
        rows.append(
            {
                "gate": gate,
                "name": name,
                "marker": marker,
                **stats,
            }
        )

    dlq_count, dlq_files = _dlq_summary()

    lines = [
        "# Quality Report — Project-Lewis",
        "",
        f"**Data:** {now} | **Commit:** {git['commit']} | **Branch:** {git['branch']}",
        "",
        "## Resumo por Quality Gate",
        "",
        "| Gate | Nome | Status | Pass | Fail | Skip | Error | Total | Sumario |",
        "| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- |",
    ]

    for r in rows:
        lines.append(
            f"| {r['gate']} | {r['name']} | {r['status']} | "
            f"{r['passed']} | {r['failed']} | {r['skipped']} | "
            f"{r['error']} | {r['total_collected']} | {r['summary']} |"
        )

    all_passed = all(r["status"] == "✅ PASS" for r in rows)
    overall = "✅ TODOS OS GATES PASSARAM" if all_passed else "❌ EXISTEM GATES COM FALHA"

    lines.extend(
        [
            "",
            f"**Status geral:** {overall}",
            "",
            "## DLQ (Dead Letter Queue)",
            "",
            f"**Falhas pendentes:** {dlq_count}",
        ]
    )

    if dlq_files:
        lines.append("")
        lines.append("Arquivos com entradas:")
        for f in dlq_files:
            lines.append(f"- `{f}`")
    else:
        lines.append("")
        lines.append("Nenhuma falha pendente nos arquivos DLQ monitorados.")

    lines.extend(
        [
            "",
            "## Detalhes",
            "",
            "```json",
            json.dumps(rows, indent=2, ensure_ascii=False),
            "```",
            "",
            "---",
            "_Relatorio gerado automaticamente por ``scripts/generate_quality_report.py``._",
        ]
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Relatorio gerado: {report_path}")
    return report_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Gera relatorio de quality gates para o Project-Lewis."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPORTS_DIR,
        help="Diretorio onde o relatorio sera salvo (padrao: reports/).",
    )
    args = parser.parse_args(argv)

    try:
        generate_report(args.output_dir)
    except Exception as exc:
        print(f"Erro ao gerar relatorio: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
