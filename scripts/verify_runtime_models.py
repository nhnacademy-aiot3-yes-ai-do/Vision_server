#!/usr/bin/env python3
"""Verify the two Git-managed runtime models against their manifest."""

# 목적: private Git에서 코드와 함께 관리하는 detector/health 모델이
#       승인된 manifest의 경로·크기·SHA-256과 정확히 일치하는지 확인한다.
# 입력: models/model-manifest.json과 runtime/models 아래의 두 best.pt 파일.
# 출력: 역할별 검증 성공 메시지 또는 로컬 경로를 노출하지 않는 오류.
# 처리 흐름: manifest 스키마 검증 -> 고정 repository 경로 확인 ->
#            일반 파일·크기·SHA-256 확인 -> 검증 중 파일 불변성 확인.

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "models" / "model-manifest.json"
EXPECTED_ROLES = ("detector", "health")
EXPECTED_REPOSITORY_PATHS = {
    "detector": Path("runtime/models/detector/best.pt"),
    "health": Path("runtime/models/health/best.pt"),
}
EXPECTED_TASKS = {
    "detector": "object-detection",
    "health": "image-classification",
}
REQUIRED_HEALTH_STATUSES = frozenset(
    {"HEALTHY", "DISEASE_SUSPECTED"}
)
HASH_CHUNK_BYTES = 1024 * 1024


class ModelVerificationError(RuntimeError):
    """검증 실패 원인만 나타내고 실제 로컬 절대경로는 노출하지 않는 오류다."""


@dataclass(frozen=True)
class ModelSpec:
    """manifest에서 검증 완료된 한 모델의 실행·해석 계약이다."""

    role: str
    name: str
    version: str
    task: str
    size_bytes: int
    sha256: str
    repository_relative_path: Path
    image_size: int
    class_mapping: tuple[tuple[int, str], ...]
    raw_class_mapping: tuple[tuple[int, str], ...]


def _safe_relative_path(value: object, field: str) -> Path:
    # 절대경로와 상위 디렉터리 이동을 막아 저장소 밖 파일을 읽지 않게 한다.
    if not isinstance(value, str) or not value.strip():
        raise ModelVerificationError(f"manifest {field} is invalid")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ModelVerificationError(
            f"manifest {field} must be repository-relative"
        )
    return path


def _class_mapping(
    value: object,
    *,
    role: str,
    field: str,
) -> tuple[tuple[int, str], ...]:
    # JSON object key를 연속된 0 기반 class id로 제한해 확률·label 해석 오류를 막는다.
    if not isinstance(value, Mapping) or not value:
        raise ModelVerificationError(
            f"{role} {field} must be a non-empty object"
        )
    parsed: dict[int, str] = {}
    for raw_class_id, raw_name in value.items():
        if (
            not isinstance(raw_class_id, str)
            or not raw_class_id.isascii()
            or not raw_class_id.isdecimal()
            or len(raw_class_id) > 10
            or str(int(raw_class_id)) != raw_class_id
        ):
            raise ModelVerificationError(
                f"{role} {field} class id is invalid"
            )
        if (
            not isinstance(raw_name, str)
            or not raw_name.strip()
            or raw_name != raw_name.strip()
        ):
            raise ModelVerificationError(
                f"{role} {field} class name is invalid"
            )
        class_id = int(raw_class_id)
        if class_id in parsed:
            raise ModelVerificationError(
                f"{role} {field} class id is duplicated"
            )
        parsed[class_id] = raw_name
    if sorted(parsed) != list(range(len(parsed))):
        raise ModelVerificationError(
            f"{role} {field} class ids must be contiguous from zero"
        )
    if len(set(parsed.values())) != len(parsed):
        raise ModelVerificationError(
            f"{role} {field} class names must be unique"
        )
    return tuple(sorted(parsed.items()))


def _model_spec(raw: Mapping[str, Any]) -> ModelSpec:
    # manifest가 임의 역할이나 모델 경로를 추가할 수 없도록 역할별 경로를 고정한다.
    role = raw.get("role")
    name = raw.get("name")
    version = raw.get("version")
    task = raw.get("task")
    size_bytes = raw.get("sizeBytes")
    expected_sha256 = raw.get("sha256")
    if role not in EXPECTED_ROLES:
        raise ModelVerificationError("manifest model role is invalid")
    if not isinstance(name, str) or not name.strip():
        raise ModelVerificationError(f"{role} model name is invalid")
    if (
        not isinstance(version, str)
        or not version.strip()
        or version != version.strip()
    ):
        raise ModelVerificationError(f"{role} model version is invalid")
    if task != EXPECTED_TASKS[role]:
        raise ModelVerificationError(f"{role} model task is invalid")
    if not isinstance(size_bytes, int) or isinstance(size_bytes, bool):
        raise ModelVerificationError(f"{role} model size is invalid")
    if size_bytes <= 0:
        raise ModelVerificationError(f"{role} model size is invalid")
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in expected_sha256
        )
    ):
        raise ModelVerificationError(f"{role} SHA-256 is invalid")
    repository_path = _safe_relative_path(
        raw.get("repositoryPath"),
        "repositoryPath",
    )
    if repository_path != EXPECTED_REPOSITORY_PATHS[role]:
        raise ModelVerificationError(f"{role} repository path is invalid")
    image_size = raw.get("imageSize")
    if (
        not isinstance(image_size, int)
        or isinstance(image_size, bool)
        or image_size <= 0
        or image_size > 16_384
    ):
        raise ModelVerificationError(f"{role} imageSize is invalid")
    class_mapping = _class_mapping(
        raw.get("classMapping"),
        role=role,
        field="classMapping",
    )
    raw_class_mapping = _class_mapping(
        raw.get("rawClassMapping"),
        role=role,
        field="rawClassMapping",
    )
    if (
        tuple(class_id for class_id, _name in class_mapping)
        != tuple(class_id for class_id, _name in raw_class_mapping)
    ):
        raise ModelVerificationError(
            f"{role} class mapping ids do not match"
        )
    if role == "health" and {
        name for _class_id, name in class_mapping
    } != REQUIRED_HEALTH_STATUSES:
        raise ModelVerificationError(
            "health classMapping must define HEALTHY and DISEASE_SUSPECTED"
        )
    return ModelSpec(
        role=role,
        name=name,
        version=version,
        task=task,
        size_bytes=size_bytes,
        sha256=expected_sha256,
        repository_relative_path=repository_path,
        image_size=image_size,
        class_mapping=class_mapping,
        raw_class_mapping=raw_class_mapping,
    )


def load_manifest(path: Path = DEFAULT_MANIFEST_PATH) -> tuple[ModelSpec, ...]:
    """manifest를 읽고 detector와 health가 정확히 한 번씩 있는지 검증한다."""

    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ModelVerificationError("model manifest cannot be read") from exc
    if not isinstance(payload, Mapping) or payload.get("schemaVersion") != 1:
        raise ModelVerificationError(
            "model manifest schemaVersion is invalid"
        )
    raw_models = payload.get("models")
    if not isinstance(raw_models, list):
        raise ModelVerificationError(
            "model manifest models must be a list"
        )
    specs = tuple(
        _model_spec(raw)
        for raw in raw_models
        if isinstance(raw, Mapping)
    )
    if len(specs) != len(raw_models):
        raise ModelVerificationError("model manifest entry is invalid")
    by_role = {spec.role: spec for spec in specs}
    if set(by_role) != set(EXPECTED_ROLES) or len(by_role) != len(specs):
        raise ModelVerificationError(
            "model manifest must contain each role once"
        )
    return tuple(by_role[role] for role in EXPECTED_ROLES)


def _fingerprint(path: Path, role: str) -> tuple[int, int, str]:
    # 검증 전후 stat을 비교해 해시 계산 중 파일이 바뀐 경우도 승인하지 않는다.
    try:
        before = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(HASH_CHUNK_BYTES):
                digest.update(chunk)
        after = path.stat()
    except OSError as exc:
        raise ModelVerificationError(f"{role} model cannot be read") from exc
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_identity != after_identity:
        raise ModelVerificationError(
            f"{role} model changed during verification"
        )
    return after.st_size, after.st_mtime_ns, digest.hexdigest()


def _verify_model(spec: ModelSpec, project_root: Path) -> None:
    resolved_root = project_root.resolve()
    model_path = project_root / spec.repository_relative_path
    # Git에는 실제 모델 파일만 허용하며 symlink로 저장소 밖을 우회하지 못하게 한다.
    if model_path.is_symlink():
        raise ModelVerificationError(
            f"{spec.role} model must not be a symbolic link"
        )
    if not model_path.resolve(strict=False).is_relative_to(resolved_root):
        raise ModelVerificationError(
            f"{spec.role} model escapes project root"
        )
    if not model_path.is_file():
        raise ModelVerificationError(f"{spec.role} model is missing")
    actual_size, _mtime_ns, actual_sha256 = _fingerprint(
        model_path,
        spec.role,
    )
    if actual_size != spec.size_bytes:
        raise ModelVerificationError(
            f"{spec.role} model size mismatch"
        )
    if actual_sha256 != spec.sha256:
        raise ModelVerificationError(
            f"{spec.role} model SHA-256 mismatch"
        )


def verify_runtime_models(
    *,
    project_root: Path = PROJECT_ROOT,
    manifest_path: Path | None = None,
) -> tuple[ModelSpec, ...]:
    """Git에서 관리하는 두 runtime 모델을 변경하지 않고 검증한다."""

    resolved_manifest = (
        manifest_path
        if manifest_path is not None
        else project_root / "models" / "model-manifest.json"
    )
    specs = load_manifest(resolved_manifest)
    # 두 모델을 모두 검증한 뒤에만 성공 메시지를 출력해 부분 성공으로 오해하지 않게 한다.
    for spec in specs:
        _verify_model(spec, project_root)
    for spec in specs:
        print(f"{spec.role}: Git-managed runtime model verified")
    return specs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "private Git의 두 runtime 모델을 manifest 크기·SHA-256으로 "
            "검증합니다."
        )
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    parse_args(argv)
    try:
        verify_runtime_models()
    except ModelVerificationError as exc:
        # CI/CD가 예측 가능한 종료 코드로 실패를 감지할 수 있게 한다.
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
