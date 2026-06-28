#!/usr/bin/env python3
"""
Runner de testes de simulacao Renode para Project-Lewis.

Executa o cenario de shutdown controlado via UART usando o teste Robot
(`test_shutdown.robot`), captura a saida UART e gera relatorio JSON com
checks estruturais e metricas.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path


UART_LOG = Path("/tmp/renode_lewis_uart.log")


def run_command(cmd, cwd=None, timeout_sec=120, env=None):
    """Executa comando retornando (returncode, stdout)."""
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_sec,
        env=env,
    )
    return proc.returncode, proc.stdout


def build_firmware(project_root, make_extra=None):
    """Invoca make stm32f4 a partir de firmware/ com RENODE_SIMULATION=1."""
    cmd = ["make", "RENODE_SIMULATION=1", "stm32f4"]
    if make_extra:
        cmd.extend(make_extra)
    print(f"[build] {' '.join(cmd)} em {project_root}")
    rc, out = run_command(cmd, cwd=project_root, timeout_sec=300)
    if rc != 0:
        print(out)
        raise RuntimeError(f"Build falhou com rc={rc}")
    return (
        Path(project_root) / "build" / "stm32f4" / "lewis.bin",
        Path(project_root) / "build" / "stm32f4" / "lewis.elf",
    )


def generate_resc(bin_path, uart_log):
    """Gera conteudo de script .resc temporario com path absoluto do binario.

    O renode-test executa a partir do diretorio do arquivo .robot, portanto
    o script nao pode depender de CWD para resolver o binario.
    """
    return f"""using sysbus
mach create "stm32f4discovery-lewis"
include @scripts/single-node/stm32f4_discovery.resc
sysbus LoadBinary @{bin_path} 0x08000000
cpu VectorTableOffset 0x08000000
$uart_log?="{uart_log}"
sysbus.uart4 CreateFileBackend $uart_log true
start
"""


def _compile_beat_re() -> re.Pattern:
    """Regex tolerante para linhas de beat da UART (two-stage v2.0).

    Aceita tanto ``output=[...]`` (formato atual) quanto ``output [...]``
    (formato legado usado em logs de teste), com espacamento flexivel.
    """
    return re.compile(
        r"Beat\s+(?P<idx>\d+)\s*:\s*(?P<time>\d+)\s*ms"
        r"(?:\s*\(\s*(?P<us>\d+)\s*us\s*\))?"
        r"\s*,\s*class=\w+\s*,\s*output\s*=?\s*\[\s*(?P<values>[-\d,\s]+)\s*\]",
        re.IGNORECASE,
    )


def parse_uart_log(log_text):
    """Extrai metricas do log de texto plano da UART."""
    lines = [ln.strip() for ln in log_text.splitlines() if ln.strip()]
    result = {
        "header": False,
        "model_size_bytes": None,
        "inference_init": False,
        "arena_used_bytes": None,
        "beats": [],
        "end": False,
        "raw_lines": lines,
    }

    header_re = re.compile(r"===\s*Project-Lewis\s+Firmware\s+v(?P<version>[\d.]+)\s*===")
    model_re = re.compile(r"Model\s+size\s*(?:\([^)]*\))?\s*:\s*(?P<size>\d+)\s*bytes")
    init_ok_re = re.compile(r"Stage1\s+inference\s+init\s+OK", re.IGNORECASE)
    arena_re = re.compile(r"Arena\s+used\s*:\s*(?P<arena>\d+)\s*bytes")
    beat_re = _compile_beat_re()
    end_re = re.compile(r"===\s*Fim\s*===")

    for line in lines:
        m = header_re.search(line)
        if m:
            result["header"] = True
            result["firmware_version"] = m.group("version")
            continue
        m = model_re.search(line)
        if m:
            result["model_size_bytes"] = int(m.group("size"))
            continue
        if init_ok_re.search(line):
            result["inference_init"] = True
            continue
        m = arena_re.search(line)
        if m:
            result["arena_used_bytes"] = int(m.group("arena"))
            continue
        m = beat_re.search(line)
        if m:
            values = [int(v.strip()) for v in m.group("values").split(",") if v.strip()]
            result["beats"].append(
                {
                    "index": int(m.group("idx")),
                    "time_ms": int(m.group("time")),
                    "output": values,
                }
            )
            continue
        if end_re.search(line):
            result["end"] = True

    return result


def check_report(parsed, expected_beats=3):
    """Avalia checks estruturais."""
    model_size = parsed["model_size_bytes"]
    checks = {
        "header": parsed["header"],
        # Two-stage v2.0: soma dos FlatBuffers (stage1+stage2) deve caber na Flash.
        "model_size": model_size is not None and model_size < 512 * 1024,
        "inference_init": parsed["inference_init"],
        "beats": len(parsed["beats"]) >= expected_beats,
        "end": parsed["end"],
    }
    checks["all_passed"] = all(checks.values())
    return checks


def _find_venv_python(project_root: Path) -> Path | None:
    """Retorna o Python do venv do projeto se existir.

    O venv pode estar na raiz do projeto (../.. em relacao a este script)
    ou, futuramente, dentro de firmware/.
    """
    roots = [project_root, project_root.parent]
    for root in roots:
        for name in ("python3", "python"):
            cand = root / ".venv" / "bin" / name
            if cand.exists():
                return cand
    return None


def _python_has_robot(python: Path) -> bool:
    """Verifica se o interpretador possui robotframework."""
    try:
        rc, _ = run_command([str(python), "-c", "import robot; print(robot.__version__)"], timeout_sec=10)
        return rc == 0
    except Exception:
        return False


def _choose_python_runner(project_root: Path) -> str:
    """Escolhe um interpretador Python com robotframework disponivel."""
    venv_python = _find_venv_python(project_root)
    if venv_python and _python_has_robot(venv_python):
        return str(venv_python)
    if _python_has_robot(Path(sys.executable)):
        return sys.executable
    # Fallback: confia no python3 do PATH
    return "python3"


def run_renode_test(renode_bin: Path, robot_path: Path, resc_path: Path, project_root: Path, timeout_sec=120):
    """Executa Renode via renode-test usando o cenario Robot fornecido."""
    renode_dir = renode_bin.parent
    renode_test = renode_dir / "renode-test"
    if not renode_test.exists():
        raise FileNotFoundError(f"renode-test nao encontrado em {renode_test}")

    python_runner = _choose_python_runner(project_root)
    print(f"[renode-test] usando python: {python_runner}")

    # Garante que python3 no PATH aponte para o interpretador escolhido,
    # pois renode-test invoca 'python3' via common.sh.
    env = os.environ.copy()
    python_runner_path = Path(python_runner).parent
    env["PATH"] = str(python_runner_path) + os.pathsep + env.get("PATH", "")

    cmd = [
        str(renode_test),
        "--show-log",
        "--variable",
        f"RESC:{resc_path}",
        str(robot_path),
    ]
    print(f"[renode-test] {' '.join(cmd)}")
    return run_command(cmd, timeout_sec=timeout_sec, env=env)


def _copy_report(report, report_path):
    """Serializa e salva o relatorio JSON."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[report] salvo em {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Run Project-Lewis Renode tests")
    parser.add_argument("--renode", required=True, help="caminho para executavel renode")
    parser.add_argument("--resc", help="script .resc base (legado, nao usado)")
    parser.add_argument("--robot", required=True, help="arquivo .robot do cenario de shutdown")
    parser.add_argument("--bin", help="caminho para lewis.bin")
    parser.add_argument("--elf", help="caminho para lewis.elf")
    parser.add_argument("--run-time", type=int, default=5, help="tempo de emulacao em segundos")
    parser.add_argument("--build", action="store_true", help="forca rebuild do firmware")
    parser.add_argument("--project-root", default=None, help="raiz do firmware/")
    parser.add_argument("--report", default=None, help="caminho do report JSON")
    args = parser.parse_args()

    if args.project_root:
        project_root = Path(args.project_root).resolve()
    else:
        project_root = Path(__file__).resolve().parents[1]

    if args.build or not args.bin or not args.elf:
        bin_path, elf_path = build_firmware(project_root)
    else:
        bin_path = Path(args.bin).resolve()
        elf_path = Path(args.elf).resolve()

    if not bin_path.exists():
        raise FileNotFoundError(f"Binario nao encontrado: {bin_path}")
    if not elf_path.exists():
        raise FileNotFoundError(f"ELF nao encontrado: {elf_path}")

    robot_path = Path(args.robot).resolve()
    if not robot_path.exists():
        raise FileNotFoundError(f"Arquivo robot nao encontrado: {robot_path}")

    # Gera script .resc temporario com path absoluto do binario.
    with tempfile.TemporaryDirectory(prefix="lewis_renode_") as tmpdir:
        resc_path = Path(tmpdir) / "run.resc"
        resc_path.write_text(generate_resc(bin_path, UART_LOG))

        # Remove log anterior para garantir que o teste comecou do zero.
        if UART_LOG.exists():
            UART_LOG.unlink()

        timeout = max(args.run_time + 30, 60)
        start = time.time()
        rc, renode_out = run_renode_test(
            Path(args.renode), robot_path, resc_path, project_root, timeout_sec=timeout
        )
    elapsed = time.time() - start
    print(f"[renode-test] finalizado em {elapsed:.1f}s (rc={rc})")

    if UART_LOG.exists():
        uart_text = UART_LOG.read_text(errors="replace")
    else:
        print("[aviso] arquivo de log UART nao foi criado")
        uart_text = ""

    parsed = parse_uart_log(uart_text)
    checks = check_report(parsed)

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "firmware_bin": str(bin_path),
        "firmware_elf": str(elf_path),
        "model_size_bytes": parsed["model_size_bytes"],
        "arena_used_bytes": parsed["arena_used_bytes"],
        "beat_count": len(parsed["beats"]),
        "beat_times_ms": [b["time_ms"] for b in parsed["beats"]],
        "checks": checks,
        "uart_log": str(UART_LOG),
        "uart_log_text": uart_text,
        "renode_output_tail": renode_out[-4000:] if renode_out else "",
    }

    if args.report:
        report_path = Path(args.report)
    else:
        report_path = project_root / "build" / "stm32f4" / "firmware_simulation_report.json"
    _copy_report(report, report_path)

    if not checks["all_passed"]:
        print("[falha] Algum check estrutural nao passou:")
        print(json.dumps(checks, indent=2))
        sys.exit(1)

    if rc != 0:
        print(f"[falha] Renode retornou rc={rc}")
        sys.exit(rc)

    print("[ok] Todos os checks estruturais passaram.")


if __name__ == "__main__":
    main()
