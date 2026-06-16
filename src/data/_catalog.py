"""WFDB metadata extraction and catalog builder for Project-Lewis Camada 1.

Walks ``data/raw_*/`` and emits one normalized JSON line per record into
``data/catalog/dataset_catalog.jsonl``. Each line is validated against the
canonical schema in :mod:`src.data._schemas`.

The output contract is what Camada 2 (``resampler.py`` / ``loader.py``) reads.
See ``docs/Camada-01-Ingestao-v1.1.md`` §1.8 and §10.2.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

try:
    import wfdb as _wfdb  # type: ignore
except ImportError:  # pragma: no cover - presence validated by callers
    _wfdb = None  # type: ignore[assignment]

from ._downloader import project_root

LOGGER = logging.getLogger("lewis.camada01")

RAW_DATASETS: dict[str, Path] = {
    "chapman": Path("data/raw_chapman"),
    "mitdb": Path("data/raw_mitbih"),
    "svdb": Path("data/raw_svdb"),
    "afdb": Path("data/raw_afdb"),
    "incart": Path("data/raw_incart"),
    "ptbxl": Path("data/raw_ptbxl"),
}

CATALOG_PATH = Path("data/catalog/dataset_catalog.jsonl")

_AGE_RE = re.compile(r"\bage[:\s]+(\d+)\b", re.IGNORECASE)
_SEX_RE = re.compile(r"\bsex[:\s]+([mfMF])\b", re.IGNORECASE)
_DX_RE = re.compile(r"\b(?:dx|diagnosis)[:\s]+(.+?)(?:\n|$)", re.IGNORECASE)


@dataclass
class RecordMetadata:
    record_name: str
    dataset: str
    fs: int
    n_sig: int
    sig_len: int
    duration_sec: float
    units: list[str] = field(default_factory=list)
    gains: list[int] = field(default_factory=list)
    baselines: list[int] = field(default_factory=list)
    adc_res: list[int] = field(default_factory=list)
    adc_zero: list[int] = field(default_factory=list)
    initial_value: list[int] = field(default_factory=list)
    checksum: list[int] = field(default_factory=list)
    sig_name: list[str] = field(default_factory=list)
    comments: list[str] = field(default_factory=list)
    age: Optional[int] = None
    sex: Optional[str] = None
    diagnosis: Optional[str] = None
    source_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_comments(comments: Iterable[str]) -> dict[str, Any]:
    """Heuristically extract age/sex/diagnosis from WFDB comment lines."""
    text = "\n".join(comments or [])
    age_m = _AGE_RE.search(text)
    sex_m = _SEX_RE.search(text)
    dx_m = _DX_RE.search(text)
    return {
        "age": int(age_m.group(1)) if age_m else None,
        "sex": sex_m.group(1).upper() if sex_m else None,
        "diagnosis": dx_m.group(1).strip() if dx_m else None,
    }


def extract_metadata(record_path: Path, dataset: str) -> RecordMetadata:
    """Read a single WFDB record header and return normalized metadata.

    ``record_path`` is the full path to the record (with or without ``.hea``).
    Uses :func:`wfdb.rdheader` so gain/baseline are read live, never hardcoded.
    """
    if _wfdb is None:
        raise RuntimeError("wfdb is not installed; run `make env` to install requirements")
    rec = _wfdb.rdheader(str(Path(record_path).with_suffix("")))
    demo = parse_comments(getattr(rec, "comments", []) or [])

    fs = int(rec.fs) if rec.fs else 0
    sig_len = int(rec.sig_len) if rec.sig_len else 0
    duration = sig_len / fs if fs else 0.0

    return RecordMetadata(
        record_name=rec.record_name,
        dataset=dataset,
        fs=fs,
        n_sig=int(rec.n_sig) if rec.n_sig else 0,
        sig_len=sig_len,
        duration_sec=duration,
        units=list(rec.units or []),
        gains=list(getattr(rec, "adc_gain", []) or []),
        baselines=list(getattr(rec, "baseline", []) or []),
        adc_res=list(getattr(rec, "adc_res", []) or []),
        adc_zero=list(getattr(rec, "adc_zero", []) or []),
        initial_value=list(getattr(rec, "init_value", []) or []),
        checksum=list(getattr(rec, "checksum", []) or []),
        sig_name=list(rec.sig_name or []),
        comments=list(getattr(rec, "comments", []) or []),
        age=demo["age"],
        sex=demo["sex"],
        diagnosis=demo["diagnosis"],
        source_path=str(record_path),
    )


def iter_record_headers(root: Path) -> Iterator[tuple[str, Path]]:
    """Yield ``(dataset_name, hea_path)`` for every ``.hea`` under ``root``."""
    if not root.exists():
        return
    for hea in sorted(root.rglob("*.hea")):
        yield ("__raw__", hea)


def _dataset_for(root: Path) -> Optional[str]:
    resolved = root.resolve()
    for ds, ds_root in RAW_DATASETS.items():
        try:
            ds_resolved = (project_root() / ds_root).resolve()
        except (OSError, RuntimeError):
            continue
        if resolved == ds_resolved or ds_resolved in resolved.parents:
            return ds
    return None


def _iter_dataset(dataset: str, root: Path) -> Iterator[Path]:
    if not root.exists():
        return
    if dataset in ("chapman", "ptbxl"):
        yield from sorted(root.rglob("*.hea"))
    else:
        yield from sorted(root.glob("*.hea"))


def build_catalog(
    catalog_path: Path = CATALOG_PATH,
    *,
    overwrite: bool = False,
) -> dict[str, int]:
    """Build the JSONL catalog from ``data/raw_*/``.

    Writes atomically (``.tmp`` + rename). Returns counts ``{ok, fail, skipped}``.
    """
    from ._schemas import validate_catalog_line

    catalog_path = project_root() / catalog_path
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    n_ok = n_fail = n_skip = 0
    tmp = catalog_path.with_suffix(".tmp")

    with tmp.open("w", encoding="utf-8") as out:
        for ds, root in RAW_DATASETS.items():
            ds_root = project_root() / root
            if not ds_root.exists():
                LOGGER.warning("dataset %s missing under %s", ds, ds_root)
                n_skip += 1
                continue
            for hea in _iter_dataset(ds, ds_root):
                try:
                    meta = extract_metadata(hea, dataset=ds)
                    if meta.sig_len == 0 or meta.n_sig == 0:
                        LOGGER.warning(
                            "[catalog skip] %s/%s: no signal data (sig_len=%d, n_sig=%d)",
                            ds,
                            meta.record_name,
                            meta.sig_len,
                            meta.n_sig,
                        )
                        n_skip += 1
                        continue
                    line = json.dumps(meta.to_dict(), ensure_ascii=False)
                    if not validate_catalog_line(line):
                        LOGGER.error("invalid catalog line (schema): %s", hea)
                        n_fail += 1
                        continue
                    out.write(line + "\n")
                    n_ok += 1
                except Exception as exc:
                    LOGGER.error("[catalog skip] %s: %s", hea, exc)
                    n_fail += 1
    if not overwrite and catalog_path.exists() and n_ok == 0:
        tmp.unlink(missing_ok=True)
        LOGGER.warning("catalog would be empty; keeping previous file at %s", catalog_path)
        return {"ok": 0, "fail": n_fail, "skipped": n_skip}
    tmp.replace(catalog_path)
    LOGGER.info("catalog: %d ok, %d failed, %d datasets skipped", n_ok, n_fail, n_skip)
    return {"ok": n_ok, "fail": n_fail, "skipped": n_skip}


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    counts = build_catalog()
    return 0 if counts["fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
