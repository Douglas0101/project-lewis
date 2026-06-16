"""Valida artefatos de firmware gerados pela Fase 1.

Executa verificações sintáticas nos headers C em ``firmware/src/`` e, quando
``arm-none-eabi-gcc`` está disponível, compila um stub de teste com
``-c -Werror`` para garantir que os headers são compiláveis no target ARM
Cortex-M.

Uso:
    python scripts/validate_firmware_deliverables.py
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

LOGGER = logging.getLogger("lewis.validate_firmware")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIRMWARE_DIR = PROJECT_ROOT / "firmware"
ML_DIR = FIRMWARE_DIR / "src" / "ml"
FEATURES_DIR = FIRMWARE_DIR / "src" / "features"
DSP_DIR = FIRMWARE_DIR / "src" / "dsp"

# Headers que devem existir após ``make export``
EXPECTED_ML_HEADERS = [
    "model_data.h",
    "quantization_params.h",
]


def _find_headers(directory: Path) -> list[Path]:
    """Lista todos os arquivos ``.h`` em um diretório."""
    if not directory.exists():
        return []
    return sorted(directory.glob("*.h"))


def _check_syntax(header: Path) -> list[str]:
    """Verifica regras básicas de sintaxe/estilo dos headers C."""
    errors: list[str] = []
    content = header.read_text(encoding="utf-8")

    # Regra: arrays de modelo devem ser const unsigned char[]
    if "model_data" in header.name:
        if "const unsigned char" not in content:
            errors.append("model_data.h deve declarar 'const unsigned char[]'")
        if "alignas" not in content and "__attribute__((aligned" not in content:
            errors.append("model_data.h deve usar alinhamento (alignas/aligned)")
        if "unsigned int" not in content or "_len" not in content:
            errors.append("model_data.h deve exportar tamanho do array (_len)")

    # Regra: quantization_params.h deve ter struct com scale e zero_point
    if "quantization" in header.name:
        if "scale" not in content.lower():
            errors.append("quantization_params.h deve conter campo 'scale'")
        if "zero_point" not in content.lower():
            errors.append("quantization_params.h deve conter campo 'zero_point'")

    return errors


def _compile_headers(headers: list[Path]) -> tuple[bool, str]:
    """Tenta compilar um stub C incluindo todos os headers encontrados."""
    compiler = shutil.which("arm-none-eabi-gcc")
    if compiler is None:
        return False, "arm-none-eabi-gcc não encontrado no PATH"

    includes = sorted({str(h.parent) for h in headers})
    stub = "\n".join(f'#include "{h.name}"' for h in headers)
    stub += "\nint main(void) { return 0; }\n"

    with tempfile.TemporaryDirectory() as tmp:
        stub_path = Path(tmp) / "validate_stub.c"
        stub_path.write_text(stub, encoding="utf-8")
        cmd = [
            compiler,
            "-c",
            "-Werror",
            "-Wall",
            "-Wextra",
            "-mcpu=cortex-m4",
            "-mthumb",
            "-mfloat-abi=hard",
            "-mfpu=fpv4-sp-d16",
            "-std=c11",
            *[f"-I{inc}" for inc in includes],
            str(stub_path),
            "-o",
            str(Path(tmp) / "validate_stub.o"),
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
            return True, result.stdout or "compilação OK"
        except subprocess.CalledProcessError as exc:
            return False, exc.stderr or exc.stdout or "erro de compilação"


def validate(
    *,
    require_compiler: bool = False,
) -> bool:
    """Executa todas as validações de firmware.

    Parameters
    ----------
    require_compiler
        Se ``True``, falha quando ``arm-none-eabi-gcc`` não está disponível.

    Returns
    -------
    bool
        ``True`` se todos os checks passaram.
    """
    ok = True

    # 1. Verificar presença de headers esperados
    for name in EXPECTED_ML_HEADERS:
        path = ML_DIR / name
        if not path.exists():
            LOGGER.warning("Header esperado não encontrado: %s", path)
            ok = False

    # 2. Coletar todos os headers existentes
    headers = _find_headers(ML_DIR) + _find_headers(FEATURES_DIR) + _find_headers(DSP_DIR)
    if not headers:
        LOGGER.warning("Nenhum header .h encontrado em firmware/src/")
        return False

    LOGGER.info("Headers encontrados: %d", len(headers))
    for h in headers:
        LOGGER.info("  - %s", h.relative_to(PROJECT_ROOT))

    # 3. Validação sintática
    for h in headers:
        errors = _check_syntax(h)
        if errors:
            ok = False
            for err in errors:
                LOGGER.error("[%s] %s", h.name, err)

    # 4. Compilação com ARM GCC (quando disponível)
    compile_ok, compile_msg = _compile_headers(headers)
    if compile_ok:
        LOGGER.info("Compilação ARM: %s", compile_msg)
    else:
        if require_compiler:
            LOGGER.error("Compilação ARM falhou: %s", compile_msg)
            ok = False
        else:
            LOGGER.warning("Compilação ARM ignorada: %s", compile_msg)

    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Valida headers C de firmware gerados pela Fase 1."
    )
    parser.add_argument(
        "--require-compiler",
        action="store_true",
        help="Falha se arm-none-eabi-gcc não estiver disponível.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Loga informações detalhadas.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    ok = validate(require_compiler=args.require_compiler)
    if ok:
        LOGGER.info("Validação de firmware concluída com sucesso.")
        return 0
    LOGGER.error("Validação de firmware encontrou problemas.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
