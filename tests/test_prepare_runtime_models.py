from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import predict_mushroom_health as predictor
from scripts import prepare_runtime_models as preparation


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_manifest(
    root: Path,
    *,
    detector_payload: bytes,
    health_payload: bytes,
    detector_hash: str | None = None,
    health_hash: str | None = None,
) -> Path:
    manifest = {
        "schemaVersion": 1,
        "service": "synthetic",
        "models": [
            {
                "role": "detector",
                "name": "synthetic-detector",
                "sizeBytes": len(detector_payload),
                "sha256": detector_hash or sha256(detector_payload),
                "sourceLocalPath": "sources/detector.pt",
                "runtimeRelativePath": "runtime/models/detector/best.pt",
                "dockerRuntimePath": "/models/detector/best.pt",
            },
            {
                "role": "health",
                "name": "synthetic-health",
                "sizeBytes": len(health_payload),
                "sha256": health_hash or sha256(health_payload),
                "sourceLocalPath": "sources/health.pt",
                "runtimeRelativePath": "runtime/models/health/best.pt",
                "dockerRuntimePath": "/models/health/best.pt",
            },
        ],
    }
    path = root / "models" / "model-manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def synthetic_project(
    tmp_path: Path,
) -> tuple[Path, Path, bytes, bytes]:
    detector = b"small synthetic detector"
    health = b"small synthetic health classifier"
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    (source_dir / "detector.pt").write_bytes(detector)
    (source_dir / "health.pt").write_bytes(health)
    manifest = write_manifest(
        tmp_path,
        detector_payload=detector,
        health_payload=health,
    )
    return tmp_path, manifest, detector, health


def test_repository_manifest_matches_approved_model_contract() -> None:
    specs = preparation.load_manifest()
    by_role = {spec.role: spec for spec in specs}
    payload = json.loads(
        preparation.DEFAULT_MANIFEST_PATH.read_text(encoding="utf-8")
    )
    raw_by_role = {
        model["role"]: model for model in payload["models"]
    }

    assert set(by_role) == {"detector", "health"}
    assert by_role["detector"].sha256 == predictor.DETECTOR_MODEL_SHA256
    assert by_role["health"].sha256 == predictor.HEALTH_MODEL_SHA256
    assert (
        by_role["detector"].size_bytes
        == predictor.DETECTOR_MODEL_SIZE_BYTES
    )
    assert (
        by_role["health"].size_bytes
        == predictor.HEALTH_MODEL_SIZE_BYTES
    )
    assert by_role["detector"].docker_runtime_path == (
        "/models/detector/best.pt"
    )
    assert by_role["health"].docker_runtime_path == "/models/health/best.pt"
    assert raw_by_role["detector"]["imageSize"] == predictor.DETECTOR_IMAGE_SIZE
    assert raw_by_role["health"]["imageSize"] == predictor.HEALTH_IMAGE_SIZE
    assert raw_by_role["detector"]["classMapping"] == {
        str(key): value for key, value in predictor.DETECTOR_MODEL_NAMES.items()
    }
    assert raw_by_role["health"]["rawClassMapping"] == {
        str(key): value for key, value in predictor.HEALTH_MODEL_NAMES.items()
    }
    assert raw_by_role["health"]["classMapping"] == {
        "0": "HEALTHY",
        "1": "DISEASE_SUSPECTED",
    }
    assert raw_by_role["detector"]["name"] == predictor.DETECTOR_MODEL_NAME
    assert raw_by_role["health"]["name"] == predictor.HEALTH_MODEL_NAME
    assert raw_by_role["detector"]["version"] == "v1"
    assert raw_by_role["health"]["version"] == "v1"
    assert raw_by_role["detector"]["validationScope"]
    assert raw_by_role["health"]["validationScope"]
    assert raw_by_role["detector"]["knownLimitations"]
    assert raw_by_role["health"]["knownLimitations"]
    serialized = json.dumps(payload, ensure_ascii=False)
    assert "/mnt/" not in serialized
    assert "/home/" not in serialized


def test_prepare_and_check_only_use_atomic_verified_copies(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, manifest, detector_payload, health_payload = synthetic_project(
        tmp_path
    )
    source_stats = {
        path.name: predictor.file_fingerprint(path)
        for path in (root / "sources").iterdir()
    }

    preparation.prepare_runtime_models(
        project_root=root,
        manifest_path=manifest,
    )
    runtime_before = {
        str(path.relative_to(root)): predictor.file_fingerprint(path)
        for path in (root / "runtime/models").rglob("best.pt")
    }
    preparation.prepare_runtime_models(
        project_root=root,
        manifest_path=manifest,
    )
    preparation.prepare_runtime_models(
        project_root=root,
        manifest_path=manifest,
        check_only=True,
    )

    assert (
        root / "runtime/models/detector/best.pt"
    ).read_bytes() == detector_payload
    assert (
        root / "runtime/models/health/best.pt"
    ).read_bytes() == health_payload
    assert not list((root / "runtime/models").rglob("*.tmp"))
    assert {
        str(path.relative_to(root)): predictor.file_fingerprint(path)
        for path in (root / "runtime/models").rglob("best.pt")
    } == runtime_before
    assert {
        path.name: predictor.file_fingerprint(path)
        for path in (root / "sources").iterdir()
    } == source_stats
    output = capsys.readouterr().out
    assert str(root) not in output
    assert "/home/" not in output
    assert "/mnt/" not in output


def test_source_sha_mismatch_is_rejected_without_runtime_file(
    tmp_path: Path,
) -> None:
    root, _manifest, detector_payload, health_payload = synthetic_project(
        tmp_path
    )
    manifest = write_manifest(
        root,
        detector_payload=detector_payload,
        health_payload=health_payload,
        detector_hash="0" * 64,
    )

    with pytest.raises(
        preparation.ModelPreparationError,
        match="detector source model SHA-256 mismatch",
    ):
        preparation.prepare_runtime_models(
            project_root=root,
            manifest_path=manifest,
        )

    assert not (root / "runtime/models/detector/best.pt").exists()
    assert not (root / "runtime/models/health/best.pt").exists()


def test_missing_source_is_rejected_without_partial_copy(
    tmp_path: Path,
) -> None:
    root, manifest, _detector_payload, _health_payload = synthetic_project(
        tmp_path
    )
    (root / "sources/health.pt").unlink()

    with pytest.raises(
        preparation.ModelPreparationError,
        match="health source model is missing",
    ):
        preparation.prepare_runtime_models(
            project_root=root,
            manifest_path=manifest,
        )

    assert not (root / "runtime").exists()


def test_source_size_mismatch_is_rejected(
    tmp_path: Path,
) -> None:
    root, manifest, _detector_payload, _health_payload = synthetic_project(
        tmp_path
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["models"][0]["sizeBytes"] += 1
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        preparation.ModelPreparationError,
        match="detector source model size mismatch",
    ):
        preparation.prepare_runtime_models(
            project_root=root,
            manifest_path=manifest,
        )

    assert not (root / "runtime").exists()


def test_check_only_never_creates_missing_runtime_models(
    tmp_path: Path,
) -> None:
    root, manifest, _detector_payload, _health_payload = synthetic_project(
        tmp_path
    )

    with pytest.raises(
        preparation.ModelPreparationError,
        match="detector runtime model is missing",
    ):
        preparation.prepare_runtime_models(
            project_root=root,
            manifest_path=manifest,
            check_only=True,
        )

    assert not (root / "runtime").exists()


def test_existing_different_runtime_requires_explicit_overwrite(
    tmp_path: Path,
) -> None:
    root, manifest, detector_payload, _health_payload = synthetic_project(
        tmp_path
    )
    destination = root / "runtime/models/detector/best.pt"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"unapproved existing model")

    with pytest.raises(
        preparation.ModelPreparationError,
        match="detector runtime model differs",
    ):
        preparation.prepare_runtime_models(
            project_root=root,
            manifest_path=manifest,
        )
    assert destination.read_bytes() == b"unapproved existing model"

    preparation.prepare_runtime_models(
        project_root=root,
        manifest_path=manifest,
        overwrite=True,
    )
    assert destination.read_bytes() == detector_payload


def test_source_symlink_escape_is_rejected(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    detector_payload = b"outside detector"
    health_payload = b"inside health"
    (outside / "detector.pt").write_bytes(detector_payload)
    sources = project / "sources"
    sources.mkdir()
    (sources / "detector.pt").symlink_to(outside / "detector.pt")
    (sources / "health.pt").write_bytes(health_payload)
    manifest = write_manifest(
        project,
        detector_payload=detector_payload,
        health_payload=health_payload,
    )

    with pytest.raises(
        preparation.ModelPreparationError,
        match="detector source model escapes project root",
    ):
        preparation.prepare_runtime_models(
            project_root=project,
            manifest_path=manifest,
        )

    assert not (project / "runtime").exists()


def test_manifest_path_traversal_is_rejected(
    tmp_path: Path,
) -> None:
    root, manifest, _detector_payload, _health_payload = synthetic_project(
        tmp_path
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["models"][0]["sourceLocalPath"] = "../outside.pt"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        preparation.ModelPreparationError,
        match="sourceLocalPath must be repository-relative",
    ):
        preparation.load_manifest(manifest)
