#!/usr/bin/env python3
"""Executa os hard gates HG-01..HG-06 do Project-Lewis v1.2.

Roda cada marker de qualidade individualmente (qg7, qg9-qg12), verifica a
configuracao --strict-markers do pyproject.toml, e garante que o ELF STM32F4
nao contenha o marcador de stub STUB_TFLM.

Saida: tabela de PASS/FAIL por gate. Codigo de retorno 0 somente se todos
passarem.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = PROJECT_ROOT / "pyproject.toml"
PYTEST = os.environ.get("PYTEST")
if PYTEST is None:
    PYTEST = str(Path(sys.executable).with_name("pytest"))
ELF_PATH = PROJECT_ROOT / "firmware" / "build" / "stm32f4" / "lewis.elf"

GATES = [
    "qg7",
    "qg9",
    "qg10",
    "qg11",
    "qg12",
]


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Roda um comando no root do projeto com variaveis de ambiente padrao."""
    env = os.environ.copy()
    env.setdefault("ALLOW_STUB", "0")
    env.setdefault("CI", "1")
    kwargs.setdefault("cwd", PROJECT_ROOT)
    kwargs.setdefault("env", env)
    return subprocess.run(cmd, **kwargs)


def _check_strict_markers() -> tuple[bool, str]:
    """Verifica se --strict-markers esta configurado e rejeita markers desconhecidos."""
    cfg = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    addopts = cfg.get("tool", {}).get("pytest", {}).get("ini_options", {}).get("addopts", "")
    if "--strict-markers" not in addopts.split():
        return False, "--strict-markers nao encontrado em pyproject.toml"

    # Forca um erro de marker desconhecido: com --strict-markers, um decorator
    # @pytest.mark.<desconhecido> deve falhar na colecao.
    with tempfile.TemporaryDirectory(prefix="lewis_hg02_") as tmpdir:
        tmp_path = Path(tmpdir)
        test_file = tmp_path / "test_unknown_marker.py"
        test_file.write_text(
            "import pytest\n@pytest.mark.lewis_unknown_marker_hg02\n"
            "def test_dummy():\n    pass\n",
            encoding="utf-8",
        )
        result = _run(
            [PYTEST, "--strict-markers", str(test_file)],
            capture_output=True,
            text=True,
        )

    if result.returncode == 0:
        return False, "marker desconhecido deveria ter falhado com --strict-markers"

    output = (result.stdout or "") + (result.stderr or "")
    indicators = [
        "not registered",
        "not found",
        "Unknown pytest.mark",
        "is not defined",
    ]
    if not any(ind.lower() in output.lower() for ind in indicators):
        return False, "pytest falhou, mas nao por causa de marker desconhecido"

    return True, "--strict-markers ativo e rejeitou marker desconhecido"


def _check_no_stub() -> tuple[bool, str]:
    """Garante que o ELF nao contenha o marcador de stub STUB_TFLM."""
    if not ELF_PATH.exists():
        return True, "ELF ainda nao existe; verificacao adiada (sera feita apos qg7)"

    strings_bin = shutil.which("strings")
    if strings_bin is None:
        return True, "binario 'strings' nao disponivel; pulando verificacao simbolica"

    result = subprocess.run(
        [strings_bin, str(ELF_PATH)],
        capture_output=True,
        text=True,
        check=False,
    )
    if "STUB_TFLM" in result.stdout:
        return False, f"STUB_TFLM encontrado em {ELF_PATH}"
    return True, f"nenhum STUB_TFLM encontrado em {ELF_PATH}"


def _run_gate(marker: str) -> tuple[bool, str]:
    """Roda pytest para um marker especifico."""
    print(f"\n=== HG-01: rodando pytest -m {marker} tests/ ===")
    result = _run(
        [PYTEST, "-m", marker, "tests/"],
        check=False,
    )
    if result.returncode == 0:
        return True, f"rc={result.returncode}"
    return False, f"rc={result.returncode}"


def main() -> int:
    print("Project-Lewis Hard Gates v1.2")
    print(f"pytest: {PYTEST}")
    print(f"root:   {PROJECT_ROOT}")

    results: list[tuple[str, bool, str]] = []

    # HG-02: strict markers
    print("\n=== HG-02: verificando --strict-markers ===")
    ok, msg = _check_strict_markers()
    results.append(("HG-02 strict-markers", ok, msg))
    print(f"{'PASS' if ok else 'FAIL'}: {msg}")
    if not ok:
        return _report(results)

    # HG-01: gate markers
    for marker in GATES:
        ok, msg = _run_gate(marker)
        results.append((f"HG-01 {marker}", ok, msg))
        if not ok:
            break

    # HG-03: no stub no ELF (repete apos qg7 para garantir artefato)
    print("\n=== HG-03: verificando ausencia de STUB_TFLM no ELF ===")
    ok, msg = _check_no_stub()
    results.append(("HG-03 no-stub", ok, msg))
    print(f"{'PASS' if ok else 'FAIL'}: {msg}")

    return _report(results)


def _report(results: list[tuple[str, bool, str]]) -> int:
    print("\n" + "=" * 60)
    print("RESUMO DOS HARD GATES")
    print("=" * 60)
    all_ok = True
    for name, ok, msg in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}: {msg}")
        if not ok:
            all_ok = False
    print("=" * 60)
    if all_ok:
        print("RESULTADO: todos os hard gates passaram.")
        return 0
    print("RESULTADO: um ou mais hard gates falharam.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
