"""Integrity and provenance helpers for Project-Lewis Camada 1.

Provides:

* :func:`sha256_of` — streaming SHA256 over a local file.
* :func:`verify_all` — check every entry of ``src/data/checksums.json`` against
  the corresponding ZIP in ``data/.cache/zips/``.
* :func:`update_entry` — write a new manifest entry under TOFU policy.
* :func:`write_provenance` — regenerate ``PROVENANCE.md`` from the manifest.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ._downloader import project_root

LOGGER = logging.getLogger("lewis.camada01")

CHECKSUMS_PATH = Path("src/data/checksums.json")
CACHE_ZIPS_DIR = Path("data/.cache/zips")
MIRROR_DIR = Path("data/mirrors")
DEFAULT_ZIP_PATTERN = re.compile(r".*\.zip$", re.IGNORECASE)
DEFAULT_ARCHIVE_PATTERN = re.compile(r".*\.(zip|tar\.gz|tgz)$", re.IGNORECASE)

_SIZE_TOLERANCE = 0.05


def sha256_of(path: Path, *, chunk: int = 1 << 20) -> str:
    """Compute the SHA256 hex digest of ``path`` by streaming in ``chunk`` bytes."""
    path = project_root() / path
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def _archive_for_dataset(dataset: str, cache: Path, mirror: Path) -> Optional[Path]:
    """Return the first existing archive for ``dataset``.

    Preference order:
    1. ZIP under ``cache`` (legacy MIT-BIH family downloads).
    2. Tarball (``.tar.gz`` / ``.tgz``) under ``mirror`` (Chapman, PTB-XL).
    """
    if cache.exists():
        for cand in cache.glob(f"{dataset}*.zip"):
            if cand.exists():
                return cand
    if mirror.exists():
        for ext in ("*.tar.gz", "*.tgz"):
            for cand in mirror.glob(f"{dataset}{ext}"):
                if cand.exists():
                    return cand
    return None


def verify_all(
    checksums_path: Path = CHECKSUMS_PATH,
    cache_dir: Path = CACHE_ZIPS_DIR,
    mirror_dir: Path = MIRROR_DIR,
) -> dict[str, bool]:
    """Verify every entry of the manifest against local archives.

    Returns ``{dataset: ok_bool}``. An empty dict means the manifest is missing
    or unreadable. A ``False`` value means either the archive is absent, the size
    is off by more than :data:`_SIZE_TOLERANCE`, or the SHA256 mismatches.
    """
    checksums_path = project_root() / checksums_path
    if not checksums_path.exists():
        LOGGER.error("checksums.json missing at %s", checksums_path)
        return {}
    manifest = json.loads(checksums_path.read_text(encoding="utf-8"))
    cache = project_root() / cache_dir
    mirror = project_root() / mirror_dir
    results: dict[str, bool] = {}
    for ds, entry in manifest.get("datasets", {}).items():
        expected_hash = entry.get("sha256")
        expected_size = entry.get("size_bytes")
        # Entries marked with null sha256/size are documentation-only fallbacks
        # (e.g., PTB-XL distributed as many files) and do not require a local archive.
        if expected_hash is None or expected_size is None:
            LOGGER.info("skipping archive checksum for %s (null manifest entry)", ds)
            results[ds] = True
            continue
        archive = _archive_for_dataset(ds, cache, mirror)
        if archive is None or not archive.exists():
            LOGGER.warning("no cached archive for %s under %s or %s", ds, cache, mirror)
            results[ds] = False
            continue
        actual_size = archive.stat().st_size
        actual_hash = sha256_of(archive)
        expected_size_int = int(expected_size)
        expected_hash_str = str(expected_hash)
        size_ok = (
            expected_size_int == 0
            or abs(actual_size - expected_size_int) / max(expected_size_int, 1) < _SIZE_TOLERANCE
        )
        hash_ok = bool(expected_hash_str) and actual_hash == expected_hash_str
        results[ds] = size_ok and hash_ok
        if not results[ds]:
            LOGGER.warning(
                "checksum fail %s: size_ok=%s hash_ok=%s (actual %d bytes)",
                ds,
                size_ok,
                hash_ok,
                actual_size,
            )
    return results


def update_entry(
    checksums_path: Path,
    dataset: str,
    *,
    zip_path: Path,
    source: str,
    url: str,
    verified_by: str,
) -> None:
    """Add or update the manifest entry for ``dataset`` under TOFU policy.

    Writes the manifest atomically (``tmp`` + ``replace``). Caller is
    responsible for ensuring TOFU invariants (size within tolerance, header
    parseable, etc.) per Camada-01 spec §7.2.
    """
    checksums_path = project_root() / checksums_path
    if checksums_path.exists():
        manifest = json.loads(checksums_path.read_text(encoding="utf-8"))
    else:
        manifest = {"version": "1.0", "generated_at": "", "datasets": {}}
    manifest.setdefault("version", "1.0")
    manifest.setdefault("datasets", {})
    manifest["datasets"][dataset] = {
        "sha256": sha256_of(zip_path),
        "size_bytes": zip_path.stat().st_size,
        "source": source,
        "url": url,
        "verified_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "verified_by": verified_by,
    }
    manifest["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    tmp = checksums_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(checksums_path)


def write_provenance(
    manifest: dict[str, Any],
    output_path: Path = Path("PROVENANCE.md"),
) -> None:
    """Render ``PROVENANCE.md`` from the checksums manifest.

    The output is deterministic markdown with a per-dataset table sourced from
    the live manifest entries (sha256, size, source, url, verified_at).
    """
    output_path = project_root() / output_path
    datasets = manifest.get("datasets", {})
    lines: list[str] = []
    lines.append("# Project-Lewis — Proveniência de Dados")
    lines.append("")
    lines.append(
        "> Documento exigido pela LGPD (Lei 13.709/2018) e GDPR para "
        "rastreabilidade de finalidade, base legal e sub-processadores de "
        "dados pessoais (mesmo de-identificados)."
    )
    lines.append("")
    lines.append("## 1. Inventário de Datasets")
    lines.append("")
    lines.append("| Dataset | URL canônica | Source | Tamanho (bytes) | SHA256 | Verificado em |")
    lines.append("| :--- | :--- | :--- | ---: | :--- | :--- |")
    for ds, entry in datasets.items():
        url = entry.get("url", "—")
        size = entry.get("size_bytes", 0)
        sha = entry.get("sha256", "—")
        verified = entry.get("verified_at", "—")
        source = entry.get("source", "—")
        lines.append(f"| {ds} | {url} | {source} | {size} | `{sha}` | {verified} |")
    lines.append("")
    lines.append("## 2. Base Legal (LGPD Art. 7º)")
    lines.append("")
    lines.append(
        "- **Chapman-Shaoxing:** uso de pesquisa pública, dado de-identificado pelo "
        "publicador. IRB de Shaoxing People's Hospital e Ningbo First Hospital."
    )
    lines.append(
        "- **MIT-BIH Family e INCART:** pesquisa pública (PhysioNet), dados históricos "
        "de-identificados pelo publicador. Sem IRB formal (dados > 40 anos)."
    )
    lines.append("")
    lines.append("## 3. Finalidade")
    lines.append("")
    lines.append(
        "Treinamento, validação e exportação de modelo de classificação de "
        "arritmias cardíacas em ECG, embarcado em microcontrolador STM32F4. "
        "Não inclui re-identificação, não inclui comercialização de dados "
        "pessoais, não inclui cruzamento com outras bases."
    )
    lines.append("")
    lines.append("## 4. Sub-processadores")
    lines.append("")
    lines.append("| Nome | Serviço | Dados compartilhados | Localização |")
    lines.append("| :--- | :--- | :--- | :--- |")
    lines.append(
        "| PhysioNet (MIT Lab) | Distribuição de datasets | Hash do dataset (auditoria) | EUA |"
    )
    lines.append(
        "| Kaggle (Google) | Distribuição de Chapman | Hash do dataset (auditoria) | EUA |"
    )
    lines.append("")
    lines.append("## 5. Retenção e Descarte")
    lines.append("")
    lines.append(
        "- **Dados brutos (`data/raw_*/`):** retidos pelo tempo de vida do projeto. "
        "Ao final, delete via `rm -rf data/raw_*/` e arquive o SHA256 final em "
        "`data/audit/final_checksums.jsonl`."
    )
    lines.append("- **Catálogo (`data/catalog/`):** retido indefinidamente (curado, pequeno).")
    lines.append("- **DLQ e audit logs:** retidos por 7 anos (LGPD Art. 37) em cold storage.")
    lines.append("")
    lines.append("## 6. Direitos do Titular")
    lines.append("")
    lines.append(
        "Por se tratar de dados públicos de-identificados pelo publicador, "
        "não há canal direto de exercício de direitos do titular (Art. 18 LGPD) "
        "pelo Project-Lewis. Encaminhar solicitações às instituições de origem."
    )
    lines.append("")
    lines.append("## 7. Histórico de Mudanças")
    lines.append("")
    lines.append("| Data | Versão | Mudança | Responsável |")
    lines.append("| :--- | :--- | :--- | :--- |")
    lines.append("| 2026-06-09 | 1.0 | Criação | Douglas Souza |")
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")
