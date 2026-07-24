"""Hashing, atomic artifacts, runtime identity, and resumable run state."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.stage2_research.contracts import (
    REQUIRED_RUN_ARTIFACTS,
    DoneMarker,
    EnvironmentManifest,
    ExitCode,
    ResearchError,
    RunManifest,
)


def utc_now() -> str:
    """Return a stable UTC timestamp."""
    return datetime.now(UTC).isoformat()


def canonical_json_bytes(value: Any) -> bytes:
    """Encode JSON deterministically for content addressing."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    """Hash bytes."""
    return hashlib.sha256(value).hexdigest()


def hash_canonical(value: Any) -> str:
    """Hash a JSON-compatible object canonically."""
    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: Path) -> str:
    """Stream a file into SHA-256."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ResearchError(
            f"cannot hash file: {path}",
            ExitCode.DATA_INTEGRITY,
        ) from error
    return digest.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    """Hash dtype, shape, and C-order bytes of an ndarray."""
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(canonical_json_bytes(list(array.shape)))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def atomic_write_bytes(path: Path, content: bytes) -> None:
    """Atomically publish a file in its destination directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_text(path: Path, content: str) -> None:
    """Atomically publish UTF-8 text."""
    atomic_write_bytes(path, content.encode("utf-8"))


def atomic_write_json(path: Path, value: Any) -> None:
    """Atomically publish readable deterministic JSON."""
    content = json.dumps(
        value,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        default=str,
    )
    atomic_write_text(path, content + "\n")


def load_json(path: Path) -> Any:
    """Load JSON with a stable integrity error."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ResearchError(
            f"invalid JSON artifact: {path}",
            ExitCode.INCOMPATIBLE_ARTIFACT,
        ) from error


def git_identity(project_root: Path) -> tuple[str, bool]:
    """Return HEAD and dirty state."""
    git_executable = shutil.which("git")
    if git_executable is None:
        raise ResearchError("git executable not found", ExitCode.BLOCKED_PRECONDITION)
    try:
        head = subprocess.run(
            [git_executable, "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                [git_executable, "status", "--porcelain"],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ResearchError("cannot resolve Git identity", ExitCode.BLOCKED_PRECONDITION) from error
    return head, dirty


def source_fingerprint(project_root: Path) -> dict[str, Any]:
    """Content-address every source file that can change research semantics."""
    root = project_root.resolve()
    patterns = (
        "src/stage2_research/*.py",
        "src/cli/stage2_research.py",
        "src/features/e06_*.py",
        "src/models/e06_protocol.py",
        "config/stage2_research.yaml",
    )
    files: dict[str, str] = {}
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            if path.is_file():
                files[str(path.relative_to(root))] = sha256_file(path)
    if not files:
        raise ResearchError(
            "research source fingerprint is empty",
            ExitCode.BLOCKED_PRECONDITION,
        )
    head, dirty = git_identity(root)
    payload = {"git_head": head, "git_dirty": dirty, "files": files}
    payload["source_manifest_hash"] = hash_canonical(payload)
    return payload


def runtime_identity_hash(runtime: Mapping[str, Any]) -> str:
    """Hash runtime versions/device without ephemeral timestamps."""
    stable_keys = (
        "python_version",
        "tensorflow_version",
        "keras_version",
        "numpy_version",
        "platform",
        "device_requested",
        "physical_devices",
        "deterministic_requested",
        "split_random_state",
    )
    return hash_canonical({key: runtime.get(key) for key in stable_keys})


def validate_project_output_root(project_root: Path, output_root: Path) -> Path:
    """Reject output paths that could overwrite source, datasets, or production models."""
    root = project_root.resolve()
    resolved = output_root.resolve()
    protected = tuple((root / name).resolve() for name in ("src", "data", "models", "firmware"))
    if resolved == root or any(
        resolved == item or resolved.is_relative_to(item) for item in protected
    ):
        raise ResearchError(
            f"unsafe output root: {resolved}",
            ExitCode.BLOCKED_PRECONDITION,
        )
    if not resolved.is_relative_to(root / "experiments"):
        raise ResearchError(
            "Stage 2 output root must remain under experiments/",
            ExitCode.BLOCKED_PRECONDITION,
        )
    return resolved


def validate_path_segment(value: str, *, label: str) -> str:
    """Reject absolute, empty, or multi-segment run identifiers."""
    if (
        not value
        or value in {".", ".."}
        or "\x00" in value
        or "/" in value
        or "\\" in value
        or Path(value).is_absolute()
    ):
        raise ResearchError(
            f"unsafe {label}: {value!r}",
            ExitCode.BLOCKED_PRECONDITION,
        )
    return value


def validate_descendant_path(root: Path, candidate: Path, *, label: str) -> Path:
    """Resolve and require a strict descendant of an allowed filesystem root."""
    resolved_root = root.resolve(strict=False)
    resolved_candidate = candidate.resolve(strict=False)
    if resolved_candidate == resolved_root or not resolved_candidate.is_relative_to(resolved_root):
        raise ResearchError(
            f"{label} escapes allowed root: {resolved_candidate}",
            ExitCode.BLOCKED_PRECONDITION,
        )
    return resolved_candidate


def configure_determinism(
    seed: int,
    *,
    deterministic: bool,
    device: str,
    split_random_state: int,
    sampler_random_state: int,
) -> EnvironmentManifest:
    """Configure and record Python/NumPy/Keras/TensorFlow random state."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    if device.lower() == "cpu":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    random.seed(seed)
    np.random.seed(seed)

    import keras
    import tensorflow as tf

    keras.utils.set_random_seed(seed)
    deterministic_enabled = False
    if deterministic:
        try:
            tf.config.experimental.enable_op_determinism()
            deterministic_enabled = True
        except (RuntimeError, ValueError) as error:
            raise ResearchError(
                f"TensorFlow deterministic mode unavailable: {error}",
                ExitCode.BLOCKED_PRECONDITION,
            ) from error
    devices = tf.config.list_physical_devices()
    device_names = ",".join(item.device_type for item in devices) or "none"
    return EnvironmentManifest(
        python_version=platform.python_version(),
        tensorflow_version=tf.__version__,
        keras_version=keras.__version__,
        numpy_version=np.__version__,
        platform=platform.platform(),
        device=f"{device}:{device_names}",
        deterministic_requested=deterministic,
        deterministic_enabled=deterministic_enabled,
        pythonhashseed=os.environ.get("PYTHONHASHSEED", ""),
        numpy_seed=seed,
        tensorflow_seed=seed,
        keras_seed=seed,
        split_random_state=split_random_state,
        sampler_random_state=sampler_random_state,
        started_monotonic_seconds=time.monotonic(),
    )


def collect_environment_without_reseeding(
    *,
    deterministic: bool,
    device: str,
    split_random_state: int,
) -> dict[str, Any]:
    """Collect preflight runtime identity without changing model state."""
    import keras
    import tensorflow as tf

    return {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "tensorflow_version": tf.__version__,
        "keras_version": keras.__version__,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "device_requested": device,
        "physical_devices": [item.name for item in tf.config.list_physical_devices()],
        "deterministic_requested": deterministic,
        "split_random_state": split_random_state,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED", ""),
    }


def artifact_hashes(run_dir: Path, names: tuple[str, ...]) -> dict[str, str]:
    """Hash required run artifacts."""
    hashes: dict[str, str] = {}
    for name in names:
        path = run_dir / name
        if not path.is_file():
            raise ResearchError(
                f"required artifact missing: {path}",
                ExitCode.INCOMPATIBLE_ARTIFACT,
            )
        hashes[name] = sha256_file(path)
    return hashes


def write_done_marker(
    run_dir: Path,
    manifest: RunManifest,
    required_artifacts: tuple[str, ...],
) -> DoneMarker:
    """Verify artifacts and publish DONE last."""
    if tuple(required_artifacts) != tuple(REQUIRED_RUN_ARTIFACTS):
        raise ResearchError(
            "DONE requires the canonical run artifact set",
            ExitCode.INCOMPATIBLE_ARTIFACT,
        )
    if manifest.outer_test_used_for_selection:
        raise ResearchError("outer test used for selection", ExitCode.LEAKAGE)
    hashes = artifact_hashes(run_dir, required_artifacts)
    if manifest.artifact_hashes != hashes:
        raise ResearchError(
            "run manifest artifact hashes do not match post-check",
            ExitCode.INCOMPATIBLE_ARTIFACT,
        )
    marker = DoneMarker(
        run_manifest_hash=sha256_file(run_dir / "run_manifest.json"),
        config_hash=manifest.config_hash,
        completed_at=utc_now(),
        artifact_hashes=hashes,
    )
    atomic_write_json(run_dir / "DONE", marker.model_dump(mode="json"))
    return marker


def validate_done_marker(
    run_dir: Path,
    *,
    expected_config_hash: str | None = None,
) -> DoneMarker | None:
    """Return a valid DONE marker or None for an incomplete run."""
    done_path = run_dir / "DONE"
    if not done_path.exists():
        return None
    marker = DoneMarker.model_validate(load_json(done_path))
    if expected_config_hash is not None and marker.config_hash != expected_config_hash:
        raise ResearchError(
            f"completed run config mismatch: {run_dir}",
            ExitCode.INCOMPATIBLE_ARTIFACT,
        )
    required = set(REQUIRED_RUN_ARTIFACTS)
    recorded = set(marker.artifact_hashes)
    if recorded != required:
        raise ResearchError(
            f"DONE artifact set mismatch: {run_dir}",
            ExitCode.INCOMPATIBLE_ARTIFACT,
            details={
                "missing": sorted(required - recorded),
                "unexpected": sorted(recorded - required),
            },
        )
    manifest_path = run_dir / "run_manifest.json"
    try:
        manifest_hash = sha256_file(manifest_path)
        manifest = RunManifest.model_validate(load_json(manifest_path))
    except (OSError, ValueError) as error:
        raise ResearchError(
            f"invalid completed run manifest: {run_dir}",
            ExitCode.INCOMPATIBLE_ARTIFACT,
        ) from error
    if manifest_hash != marker.run_manifest_hash:
        raise ResearchError(
            f"DONE manifest hash mismatch: {run_dir}",
            ExitCode.INCOMPATIBLE_ARTIFACT,
        )
    if manifest.status != "PASS" or manifest.config_hash != marker.config_hash:
        raise ResearchError(
            f"DONE is not bound to a passing run manifest: {run_dir}",
            ExitCode.INCOMPATIBLE_ARTIFACT,
        )
    if manifest.artifact_hashes != marker.artifact_hashes:
        raise ResearchError(
            f"DONE artifact map differs from run manifest: {run_dir}",
            ExitCode.INCOMPATIBLE_ARTIFACT,
        )
    for name, expected_hash in marker.artifact_hashes.items():
        try:
            actual_hash = sha256_file(run_dir / name)
        except OSError as error:
            raise ResearchError(
                f"completed artifact is missing: {run_dir / name}",
                ExitCode.INCOMPATIBLE_ARTIFACT,
            ) from error
        if actual_hash != expected_hash:
            raise ResearchError(
                f"completed artifact hash mismatch: {run_dir / name}",
                ExitCode.INCOMPATIBLE_ARTIFACT,
            )
    return marker


@contextmanager
def run_lock(run_dir: Path, *, output_root: Path) -> Iterator[None]:
    """Prevent concurrent writers for one root-bound run cell."""
    resolved_run_dir = validate_descendant_path(
        output_root,
        run_dir,
        label="run directory",
    )
    resolved_run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = resolved_run_dir / ".RUNNING.lock"
    try:
        handle = lock_path.open("x", encoding="ascii")
    except FileExistsError as error:
        raise ResearchError(
            f"run is already active: {resolved_run_dir}",
            ExitCode.BLOCKED_PRECONDITION,
        ) from error
    try:
        os.fchmod(handle.fileno(), 0o600)
        handle.write(f"pid={os.getpid()}\n")
        handle.flush()
        os.fsync(handle.fileno())
        yield
    finally:
        handle.close()
        if lock_path.exists():
            lock_path.unlink()


def reset_incomplete_run(
    run_dir: Path,
    *,
    output_root: Path,
    keep_manifest: bool = False,
) -> None:
    """Remove root-bound partial files while preserving a completed run."""
    resolved_run_dir = validate_descendant_path(
        output_root,
        run_dir,
        label="incomplete run directory",
    )
    if (resolved_run_dir / "DONE").exists():
        raise ResearchError(
            f"refusing to overwrite completed run: {resolved_run_dir}",
            ExitCode.INCOMPATIBLE_ARTIFACT,
        )
    try:
        for path in resolved_run_dir.iterdir() if resolved_run_dir.exists() else ():
            if path.name == ".RUNNING.lock":
                continue
            if keep_manifest and path.name in {"run_manifest.json", "config_resolved.json"}:
                continue
            if path.is_symlink() or not path.is_dir():
                path.unlink()
            else:
                shutil.rmtree(path)
    except OSError as error:
        raise ResearchError(
            f"cannot reset incomplete run: {resolved_run_dir}",
            ExitCode.INTERRUPTED_RESUMABLE,
        ) from error


def verify_hash_mapping(root: Path, expected: Mapping[str, str]) -> list[dict[str, Any]]:
    """Validate a set of relative paths and hashes."""
    results: list[dict[str, Any]] = []
    for relative, expected_hash in expected.items():
        path = root / relative
        actual = sha256_file(path) if path.is_file() else ""
        results.append(
            {
                "path": relative,
                "exists": path.is_file(),
                "expected_sha256": expected_hash,
                "actual_sha256": actual,
                "match": actual == expected_hash,
                "size_bytes": path.stat().st_size if path.is_file() else 0,
            }
        )
    return results
