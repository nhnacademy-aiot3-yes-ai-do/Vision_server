#!/usr/bin/env python3
"""Verify and atomically stage approved model files for local/Docker runtime."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import predict_mushroom_health as predictor


DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "models" / "model-manifest.json"
EXPECTED_ROLES = ("detector", "health")
EXPECTED_RUNTIME_PATHS = {
    "detector": Path("runtime/models/detector/best.pt"),
    "health": Path("runtime/models/health/best.pt"),
}
EXPECTED_DOCKER_PATHS = {
    "detector": "/models/detector/best.pt",
    "health": "/models/health/best.pt",
}


class ModelPreparationError(RuntimeError):
    """Safe preparation error that does not expose a local absolute path."""


@dataclass(frozen=True)
class ModelSpec:
    """Validated manifest fields needed to prepare one approved model."""

    role: str
    name: str
    size_bytes: int
    sha256: str
    source_relative_path: Path
    runtime_relative_path: Path
    docker_runtime_path: str


def _safe_relative_path(value: object, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ModelPreparationError(f"manifest {field} is invalid")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ModelPreparationError(f"manifest {field} must be repository-relative")
    return path


def _model_spec(raw: Mapping[str, Any]) -> ModelSpec:
    role = raw.get("role")
    name = raw.get("name")
    size_bytes = raw.get("sizeBytes")
    sha256 = raw.get("sha256")
    docker_runtime_path = raw.get("dockerRuntimePath")
    if role not in EXPECTED_ROLES:
        raise ModelPreparationError("manifest model role is invalid")
    if not isinstance(name, str) or not name.strip():
        raise ModelPreparationError(f"{role} model name is invalid")
    if not isinstance(size_bytes, int) or size_bytes <= 0:
        raise ModelPreparationError(f"{role} model size is invalid")
    if (
        not isinstance(sha256, str)
        or len(sha256) != 64
        or any(character not in "0123456789abcdef" for character in sha256)
    ):
        raise ModelPreparationError(f"{role} SHA-256 is invalid")
    if docker_runtime_path != EXPECTED_DOCKER_PATHS[role]:
        raise ModelPreparationError(f"{role} Docker runtime path is invalid")
    source = _safe_relative_path(
        raw.get("sourceLocalPath"),
        "sourceLocalPath",
    )
    runtime = _safe_relative_path(
        raw.get("runtimeRelativePath"),
        "runtimeRelativePath",
    )
    if runtime != EXPECTED_RUNTIME_PATHS[role]:
        raise ModelPreparationError(f"{role} runtime path is invalid")
    return ModelSpec(
        role=role,
        name=name,
        size_bytes=size_bytes,
        sha256=sha256,
        source_relative_path=source,
        runtime_relative_path=runtime,
        docker_runtime_path=docker_runtime_path,
    )


def load_manifest(path: Path = DEFAULT_MANIFEST_PATH) -> tuple[ModelSpec, ...]:
    """Load and validate the two role-specific model preparation records."""

    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ModelPreparationError("model manifest cannot be read") from exc
    if not isinstance(payload, Mapping) or payload.get("schemaVersion") != 1:
        raise ModelPreparationError("model manifest schemaVersion is invalid")
    raw_models = payload.get("models")
    if not isinstance(raw_models, list):
        raise ModelPreparationError("model manifest models must be a list")
    specs = tuple(
        _model_spec(raw)
        for raw in raw_models
        if isinstance(raw, Mapping)
    )
    if len(specs) != len(raw_models):
        raise ModelPreparationError("model manifest entry is invalid")
    by_role = {spec.role: spec for spec in specs}
    if set(by_role) != set(EXPECTED_ROLES) or len(by_role) != len(specs):
        raise ModelPreparationError("model manifest must contain each role once")
    return tuple(by_role[role] for role in EXPECTED_ROLES)


def _fingerprint(path: Path, role: str) -> tuple[int, int, str]:
    try:
        return predictor.file_fingerprint(path)
    except OSError as exc:
        raise ModelPreparationError(f"{role} model cannot be read") from exc


def _validated_sources(
    specs: Sequence[ModelSpec],
    project_root: Path,
) -> dict[str, tuple[Path, tuple[int, int, str]]]:
    validated: dict[str, tuple[Path, tuple[int, int, str]]] = {}
    resolved_root = project_root.resolve()
    for spec in specs:
        source = project_root / spec.source_relative_path
        if not source.resolve(strict=False).is_relative_to(resolved_root):
            raise ModelPreparationError(
                f"{spec.role} source model escapes project root"
            )
        if not source.is_file():
            raise ModelPreparationError(f"{spec.role} source model is missing")
        fingerprint = _fingerprint(source, spec.role)
        if fingerprint[0] != spec.size_bytes:
            raise ModelPreparationError(
                f"{spec.role} source model size mismatch"
            )
        if fingerprint[2] != spec.sha256:
            raise ModelPreparationError(
                f"{spec.role} source model SHA-256 mismatch"
            )
        validated[spec.role] = (source, fingerprint)
    return validated


def _atomic_copy(
    source: Path,
    destination: Path,
    *,
    role: str,
    expected_size: int,
    expected_sha256: str,
    source_fingerprint: tuple[int, int, str],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    activated = False
    try:
        with os.fdopen(descriptor, "wb") as target, source.open("rb") as source_file:
            shutil.copyfileobj(source_file, target, length=1024 * 1024)
            target.flush()
            os.fsync(target.fileno())
        temporary_fingerprint = _fingerprint(temporary, role)
        if temporary_fingerprint[0] != expected_size:
            raise ModelPreparationError(f"{role} copied model size mismatch")
        if temporary_fingerprint[2] != expected_sha256:
            raise ModelPreparationError(f"{role} copied model SHA-256 mismatch")
        if _fingerprint(source, role) != source_fingerprint:
            raise ModelPreparationError(f"{role} source model changed during copy")
        os.chmod(temporary, 0o644)
        os.replace(temporary, destination)
        activated = True
        destination_fingerprint = _fingerprint(destination, role)
        if destination_fingerprint[0] != expected_size:
            raise ModelPreparationError(f"{role} runtime model size mismatch")
        if destination_fingerprint[2] != expected_sha256:
            raise ModelPreparationError(f"{role} runtime model SHA-256 mismatch")
    except OSError as exc:
        raise ModelPreparationError(f"{role} runtime model cannot be prepared") from exc
    finally:
        temporary.unlink(missing_ok=True)
        if activated and destination.is_file():
            try:
                destination_fingerprint = _fingerprint(destination, role)
                if (
                    destination_fingerprint[0] != expected_size
                    or destination_fingerprint[2] != expected_sha256
                ):
                    destination.unlink()
            except ModelPreparationError:
                destination.unlink(missing_ok=True)


def prepare_runtime_models(
    *,
    project_root: Path = PROJECT_ROOT,
    manifest_path: Path | None = None,
    check_only: bool = False,
    overwrite: bool = False,
) -> tuple[ModelSpec, ...]:
    """Validate sources and atomically prepare or check both runtime models."""

    resolved_manifest = (
        manifest_path
        if manifest_path is not None
        else project_root / "models" / "model-manifest.json"
    )
    specs = load_manifest(resolved_manifest)
    sources = _validated_sources(specs, project_root)
    for spec in specs:
        destination = project_root / spec.runtime_relative_path
        if not destination.resolve(strict=False).is_relative_to(
            project_root.resolve()
        ):
            raise ModelPreparationError(
                f"{spec.role} runtime model escapes project root"
            )
        if check_only:
            if not destination.is_file():
                raise ModelPreparationError(
                    f"{spec.role} runtime model is missing"
                )
            destination_fingerprint = _fingerprint(destination, spec.role)
            if (
                destination_fingerprint[0] != spec.size_bytes
                or destination_fingerprint[2] != spec.sha256
            ):
                raise ModelPreparationError(
                    f"{spec.role} runtime model SHA-256 mismatch"
                )
            print(f"{spec.role}: runtime model verified")
            continue
        source, source_fingerprint = sources[spec.role]
        destination_fingerprint = (
            _fingerprint(destination, spec.role)
            if destination.is_file()
            else None
        )
        if (
            destination_fingerprint is not None
            and destination_fingerprint[0] == spec.size_bytes
            and destination_fingerprint[2] == spec.sha256
        ):
            print(f"{spec.role}: runtime model already verified")
            continue
        if destination_fingerprint is not None and not overwrite:
            raise ModelPreparationError(
                f"{spec.role} runtime model differs; use --overwrite"
            )
        _atomic_copy(
            source,
            destination,
            role=spec.role,
            expected_size=spec.size_bytes,
            expected_sha256=spec.sha256,
            source_fingerprint=source_fingerprint,
        )
        print(f"{spec.role}: runtime model prepared and verified")
    for spec in specs:
        source, before = sources[spec.role]
        if _fingerprint(source, spec.role) != before:
            raise ModelPreparationError(
                f"{spec.role} source model changed during preparation"
            )
    return specs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="승인된 두 모델을 검증하고 runtime/models에 원자적으로 준비합니다."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check-only",
        action="store_true",
        help="복사하지 않고 원본과 runtime 모델의 존재·크기·SHA-256 검증",
    )
    mode.add_argument(
        "--overwrite",
        action="store_true",
        help="기존 runtime 모델이 승인본과 다를 때 원자적으로 교체",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        prepare_runtime_models(
            check_only=args.check_only,
            overwrite=args.overwrite,
        )
    except ModelPreparationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
