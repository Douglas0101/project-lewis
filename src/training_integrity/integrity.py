"""Content-addressed hashing and immutable publication primitives."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
from pydantic import BaseModel

from .contracts import FileManifest, HashedFile


def _json_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return value


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a JSON value deterministically without non-finite extensions."""
    return json.dumps(
        _json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def hash_canonical(domain: str, value: Any) -> str:
    """Hash a canonical payload with an unambiguous domain prefix."""
    domain_bytes = domain.encode("utf-8")
    payload = canonical_json_bytes(value)
    digest = hashlib.sha256()
    digest.update(len(domain_bytes).to_bytes(4, "big"))
    digest.update(domain_bytes)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)
    return digest.hexdigest()


def beat_sample_id(
    dataset_id: str,
    record_id: str,
    beat_index: int,
    annotation_index_target: int,
) -> str:
    """Construct the canonical source-bound identity for one beat window."""
    return f"{dataset_id}:{record_id}:beat:{beat_index}:" f"target:{annotation_index_target}"


def afdb_episode_sample_id(
    record_id: str,
    episode_index: int,
    start_sample_target: int,
    end_sample_target: int,
) -> str:
    """Construct the canonical source-bound identity for one AFDB episode."""
    return (
        f"afdb:{record_id}:episode:{episode_index}:"
        f"target:{start_sample_target}:{end_sample_target}"
    )


def waveform_row_sha256(waveform: np.ndarray) -> str:
    """Hash one canonical float32 waveform including its shape."""
    row = np.ascontiguousarray(waveform, dtype=np.float32)
    digest = hashlib.sha256()
    digest.update(b"project-lewis/waveform-row/v1\x00")
    digest.update(np.asarray(row.shape, dtype="<i8").tobytes())
    digest.update(row.tobytes(order="C"))
    return digest.hexdigest()


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Stream SHA-256 and reject files that change while being read."""
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    after = path.stat()
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after:
        raise RuntimeError(f"file changed while hashing: {path}")
    return digest.hexdigest()


def resolve_project_path(project_root: Path, relative_path: str) -> Path:
    """Resolve a configured project-relative path and reject escapes."""
    root = project_root.resolve()
    candidate = (root / relative_path).resolve()
    if candidate == root or not candidate.is_relative_to(root):
        raise ValueError(f"path escapes project root: {relative_path}")
    return candidate


def build_file_manifest(
    project_root: Path,
    paths: Iterable[Path],
    *,
    category: str,
) -> FileManifest:
    """Build a sorted exact-byte manifest for existing project files."""
    root = project_root.resolve()
    unique: dict[str, Path] = {}
    for source in paths:
        path = source.resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"manifest path escapes project root: {path}")
        if not path.is_file():
            raise FileNotFoundError(path)
        relative = path.relative_to(root).as_posix()
        unique[relative] = path
    files = tuple(
        HashedFile(
            project_relative_path=relative,
            size_bytes=path.stat().st_size,
            sha256=sha256_file(path),
        )
        for relative, path in sorted(unique.items())
    )
    payload = {
        "schema_version": "file-manifest-v3.1.0",
        "category": category,
        "files": [file.model_dump(mode="json") for file in files],
    }
    return FileManifest(
        schema_version="file-manifest-v3.1.0",
        category=category,
        files=files,
        payload_hash=hash_canonical(f"file-manifest:{category}", payload),
    )


def _json_document_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            _json_value(value),
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


@contextmanager
def temporary_staging_path(target_path: Path) -> Iterator[Path]:
    """Yield a same-filesystem temporary path suitable for a target writer."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{target_path.stem}.",
        suffix=target_path.suffix,
        dir=target_path.parent,
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        yield temporary
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def exclusive_publication(
    lock_path: Path,
    targets: Sequence[Path],
) -> Iterator[None]:
    """Serialize canonical producers and reject every pre-existing target."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fd: int | None = None
    lock_owned = False
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        lock_owned = True
        os.close(lock_fd)
        lock_fd = None
        existing = [str(path) for path in targets if path.exists()]
        if existing:
            raise FileExistsError(f"immutable publication targets already exist: {existing}")
        yield
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
        if lock_owned:
            lock_path.unlink(missing_ok=True)


def publish_staged_file_exclusive(staged_path: Path, target_path: Path) -> None:
    """Hard-link one staged file into a write-once path without replacement."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(staged_path, target_path)
        target_path.chmod(0o444)
    finally:
        staged_path.unlink(missing_ok=True)


def write_bytes_exclusive(path: Path, content: bytes) -> None:
    """Publish bytes atomically and fail when the destination already exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise
        path.chmod(0o444)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_exclusive(path: Path, value: Any) -> None:
    """Publish deterministic JSON once; never reinterpret an existing path."""
    write_bytes_exclusive(path, _json_document_bytes(value))


def write_detached_sha256(path: Path) -> Path:
    """Create a detached SHA-256 file next to an immutable artifact."""
    digest_path = path.with_name(f"{path.name}.sha256")
    line = f"{sha256_file(path)}  {path.name}\n".encode("ascii")
    write_bytes_exclusive(digest_path, line)
    return digest_path


def verify_detached_sha256(path: Path) -> str:
    """Verify the exact detached digest and filename for an artifact."""
    digest_path = path.with_name(f"{path.name}.sha256")
    try:
        line = digest_path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError) as error:
        raise ValueError(f"cannot read detached digest: {digest_path}") from error
    parts = line.split()
    if len(parts) != 2 or parts[1] != path.name:
        raise ValueError(f"invalid detached digest format: {digest_path}")
    expected = parts[0]
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise ValueError(f"invalid detached SHA-256: {digest_path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"detached SHA-256 mismatch: {path}")
    return actual
