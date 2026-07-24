"""Write an immutable, ZIP-only inspection report for the Stage 1 Keras artifact."""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.keras_artifact_inspector import inspect_keras_archive  # noqa: E402

MODEL_PATH = PROJECT_ROOT / "models" / "stage1_float32_v2.0.keras"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "stage1_recall_investigation" / "R02"
REPORT_PATH = OUTPUT_DIR / "artifact_inspection.json"
CONFIG_PATH = OUTPUT_DIR / "config.json"
METADATA_PATH = OUTPUT_DIR / "metadata.json"
MEMBERS_PATH = OUTPUT_DIR / "archive_members.json"


def _write_text_atomic(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    """Inspect without deserializing or rewriting the original model."""
    inspection = inspect_keras_archive(MODEL_PATH)
    if inspection.model_sha256_before != inspection.model_sha256_after:
        raise RuntimeError("Stage 1 model changed during ZIP inspection")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(MODEL_PATH, "r") as archive:
            config = json.loads(archive.read("config.json"))
            metadata = json.loads(archive.read("metadata.json"))
    except (KeyError, json.JSONDecodeError, zipfile.BadZipFile) as error:
        raise ValueError("Unable to extract required Stage 1 JSON members") from error
    _write_text_atomic(CONFIG_PATH, json.dumps(config, indent=2) + "\n")
    _write_text_atomic(METADATA_PATH, json.dumps(metadata, indent=2) + "\n")
    _write_text_atomic(
        MEMBERS_PATH,
        json.dumps([member.model_dump() for member in inspection.archive_members], indent=2) + "\n",
    )
    _write_text_atomic(REPORT_PATH, inspection.model_dump_json(indent=2) + "\n")
    print(f"Wrote {REPORT_PATH}")
    print(
        f"Keras family={inspection.keras_family} input={inspection.input_shape} "
        f"output={inspection.output_shape} activation={inspection.output_activation}"
    )
    print(f"Model SHA-256 preserved: {inspection.model_sha256_after}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
