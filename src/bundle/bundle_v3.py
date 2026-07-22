"""Bundle indivisível de artefatos v3 (docs/rebuild_spec/08).

Regras:

- 15 componentes obrigatórios, todos ligados por SHA-256;
- ``H_bundle`` com serialização canônica (prefixo de domínio + pares
  nome/hash com prefixo de comprimento) — proibida concatenação textual simples;
- modelo, scaler, calibrador e threshold DEVEM apontar para a mesma geração
  (``training_run_id``); divergência → ``GENERATION_MISMATCH``;
- bundle parcial não é promovível nem carregável como candidato de produção;
- qualquer hash divergente → ``HASH_MISMATCH`` (hard reject);
- ``passes_qg5=false`` em métricas → ``GATE_FAILED``.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable

LOGGER = logging.getLogger("lewis.bundle.v3")

BUNDLE_DOMAIN = "project-lewis/bundle/v1"

BUNDLE_COMPONENTS: tuple[str, ...] = (
    "raw_data_manifest",
    "processed_data_manifest",
    "ontology",
    "preprocessing_contract",
    "feature_schema",
    "patient_split_manifest",
    "training_configuration",
    "model",
    "scaler",
    "calibrator",
    "threshold_policy",
    "metrics",
    "environment",
    "source_revision",
    "risk_report",
)

# Componentes que precisam compartilhar a mesma geração (training_run_id)
CO_GENERATED = ("model", "scaler", "calibrator", "threshold_policy")


class BundleState(StrEnum):
    BUNDLE_VALID = "BUNDLE_VALID"
    BUNDLE_INCOMPLETE = "BUNDLE_INCOMPLETE"
    ORPHAN_ARTIFACT = "ORPHAN_ARTIFACT"
    GENERATION_MISMATCH = "GENERATION_MISMATCH"
    HASH_MISMATCH = "HASH_MISMATCH"
    GATE_FAILED = "GATE_FAILED"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_bundle_digest(components: dict[str, str]) -> str:
    """Digest composto do bundle com serialização canônica e separação de domínio.

    Formato: SHA256( "project-lewis/bundle/v1" || para cada componente, em
    ordem alfabética de nome: len(nome):nome || len(hash):hash ), com
    comprimentos em decimal ASCII — separação inequívoca de campos.
    """
    h = hashlib.sha256()
    h.update(BUNDLE_DOMAIN.encode("ascii"))
    for name in sorted(components):
        digest = components[name]
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"hash inválido para componente {name!r}")
        h.update(f"{len(name)}:".encode("ascii"))
        h.update(name.encode("ascii"))
        h.update(f"{len(digest)}:".encode("ascii"))
        h.update(digest.encode("ascii"))
    return h.hexdigest()


@dataclass(frozen=True)
class BundleManifest:
    """Manifest imutável do bundle v3."""

    bundle_version: str
    training_run_id: str
    created_utc: str
    components: dict[str, str]  # nome → sha256 do conteúdo
    component_paths: dict[str, str]  # nome → path relativo (informativo)
    bundle_digest: str
    shadow: bool = True
    operational: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {
                "bundle_version": self.bundle_version,
                "training_run_id": self.training_run_id,
                "created_utc": self.created_utc,
                "components": self.components,
                "component_paths": self.component_paths,
                "bundle_digest": self.bundle_digest,
                "shadow": self.shadow,
                "operational": self.operational,
                "extra": self.extra,
            },
            indent=2,
            sort_keys=True,
        )


def build_bundle_manifest(
    training_run_id: str,
    created_utc: str,
    component_files: dict[str, Path],
    component_paths: dict[str, str] | None = None,
    extra: dict[str, Any] | None = None,
) -> BundleManifest:
    """Constrói o manifest calculando os hashes dos arquivos informados."""
    missing = [c for c in BUNDLE_COMPONENTS if c not in component_files]
    if missing:
        raise ValueError(f"componentes ausentes: {missing}")
    components = {name: sha256_file(path) for name, path in component_files.items()}
    digest = canonical_bundle_digest(components)
    return BundleManifest(
        bundle_version="3.0.0",
        training_run_id=training_run_id,
        created_utc=created_utc,
        components=components,
        component_paths=component_paths or {k: str(v) for k, v in component_files.items()},
        bundle_digest=digest,
        extra=extra or {},
    )


def verify_bundle(
    manifest: BundleManifest,
    resolver: Callable[[str], Path],
    metrics_gate: Callable[[], bool] | None = None,
) -> tuple[BundleState, list[str]]:
    """Verifica o bundle contra o filesystem e as regras estruturais.

    Parameters
    ----------
    manifest : BundleManifest
        Manifest a verificar.
    resolver : Callable[[str], Path]
        Resolve nome do componente → path confiável (root-constrained; nunca
        abrir paths vindos de payload não confiável).
    metrics_gate : Callable[[], bool], optional
        Retorna True se os gates de desempenho passaram (ex.: passes_qg5).
    """
    problems: list[str] = []

    missing = [c for c in BUNDLE_COMPONENTS if c not in manifest.components]
    if missing:
        return BundleState.BUNDLE_INCOMPLETE, [f"componentes ausentes: {missing}"]

    # hashes no filesystem
    for name in BUNDLE_COMPONENTS:
        expected = manifest.components[name]
        try:
            actual = sha256_file(resolver(name))
        except (OSError, ValueError) as exc:
            problems.append(f"{name}: não resolvível ({exc})")
            continue
        if actual != expected:
            problems.append(f"{name}: hash divergente")

    if problems:
        return BundleState.HASH_MISMATCH, problems

    # digest composto
    recomputed = canonical_bundle_digest(manifest.components)
    if recomputed != manifest.bundle_digest:
        return BundleState.HASH_MISMATCH, ["bundle_digest não recomputa"]

    # geração única para modelo/scaler/calibrador/threshold
    gen_ids = manifest.extra.get("generation_ids", {})
    gens = {gen_ids.get(c, manifest.training_run_id) for c in CO_GENERATED}
    if len(gens) != 1 or manifest.training_run_id not in gens:
        return BundleState.GENERATION_MISMATCH, [
            f"gerações divergentes em {CO_GENERATED}: {sorted(gens)}"
        ]

    # gates de desempenho
    if metrics_gate is not None and not metrics_gate():
        return BundleState.GATE_FAILED, ["passes_qg5=false ou gate ratificado falho"]

    return BundleState.BUNDLE_VALID, []
