# private Git에서 직접 관리하는 두 runtime 모델의 공급망 계약을 검증한다.
# manifest의 고정 경로·크기·SHA-256, 무변경 검증, 누락·변조·symlink와
# 저장소 밖 경로 차단을 작은 합성 모델로 확인한다.
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
from scripts import verify_runtime_models as verification


# 합성 모델 payload와 manifest의 기대 해시를 일관되게 만드는 SHA-256 helper이다.
def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


# 임시 프로젝트의 고정 runtime 경로를 가리키는 두 모델 manifest를 작성한다.
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
                "version": "test-v1",
                "task": "object-detection",
                "sizeBytes": len(detector_payload),
                "sha256": detector_hash or sha256(detector_payload),
                "repositoryPath": "runtime/models/detector/best.pt",
                "imageSize": 640,
                "classMapping": {
                    "0": "느타리",
                    "1": "표고",
                },
                "rawClassMapping": {
                    "0": "oyster",
                    "1": "shiitake",
                },
            },
            {
                "role": "health",
                "name": "synthetic-health",
                "version": "test-v1",
                "task": "image-classification",
                "sizeBytes": len(health_payload),
                "sha256": health_hash or sha256(health_payload),
                "repositoryPath": "runtime/models/health/best.pt",
                "imageSize": 320,
                "classMapping": {
                    "0": "HEALTHY",
                    "1": "DISEASE_SUSPECTED",
                },
                "rawClassMapping": {
                    "0": "healthy",
                    "1": "disease",
                },
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


# 두 합성 모델을 실제 서비스와 같은 runtime 경로에 직접 만드는 공통 helper이다.
def synthetic_project(
    tmp_path: Path,
) -> tuple[Path, Path, bytes, bytes]:
    detector = b"small synthetic detector"
    health = b"small synthetic health classifier"
    detector_path = tmp_path / "runtime/models/detector/best.pt"
    health_path = tmp_path / "runtime/models/health/best.pt"
    detector_path.parent.mkdir(parents=True)
    health_path.parent.mkdir(parents=True)
    detector_path.write_bytes(detector)
    health_path.write_bytes(health)
    manifest = write_manifest(
        tmp_path,
        detector_payload=detector,
        health_payload=health,
    )
    return tmp_path, manifest, detector, health


# 저장소 manifest와 실제 모델이 predictor의 승인된 계약과 모두 일치하는지 검증한다.
def test_repository_manifest_and_models_match_approved_contract() -> None:
    specs = verification.verify_runtime_models()
    by_role = {spec.role: spec for spec in specs}
    payload = json.loads(
        verification.DEFAULT_MANIFEST_PATH.read_text(encoding="utf-8")
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
    assert by_role["detector"].repository_relative_path == (
        Path("runtime/models/detector/best.pt")
    )
    assert by_role["health"].repository_relative_path == (
        Path("runtime/models/health/best.pt")
    )
    assert by_role["detector"].image_size == predictor.DETECTOR_IMAGE_SIZE
    assert by_role["health"].image_size == predictor.HEALTH_IMAGE_SIZE
    assert dict(by_role["detector"].class_mapping) == (
        predictor.SPECIES_BY_CLASS_ID
    )
    assert dict(by_role["detector"].raw_class_mapping) == (
        predictor.DETECTOR_MODEL_NAMES
    )
    assert dict(by_role["health"].class_mapping) == (
        predictor.HEALTH_STATUS_BY_CLASS_ID
    )
    assert dict(by_role["health"].raw_class_mapping) == (
        predictor.HEALTH_MODEL_NAMES
    )
    assert raw_by_role["detector"]["imageSize"] == predictor.DETECTOR_IMAGE_SIZE
    assert raw_by_role["health"]["imageSize"] == predictor.HEALTH_IMAGE_SIZE
    assert raw_by_role["detector"]["classMapping"] == {
        str(key): value
        for key, value in predictor.SPECIES_BY_CLASS_ID.items()
    }
    assert raw_by_role["detector"]["rawClassMapping"] == {
        str(key): value
        for key, value in predictor.DETECTOR_MODEL_NAMES.items()
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
    assert "sourceLocalPath" not in serialized
    assert "runtimeRelativePath" not in serialized
    assert "dockerRuntimePath" not in serialized
    assert "artifacts/models" not in serialized
    assert "/mnt/" not in serialized
    assert "/home/" not in serialized


# predictor의 호환용 public constants가 별도 hard-code가 아니라 manifest spec에서 파생되는지 확인한다.
def test_predictor_public_constants_are_manifest_derived() -> None:
    specs = predictor._load_approved_model_specs(
        predictor.MODEL_MANIFEST_PATH
    )
    detector = specs["detector"]
    health = specs["health"]

    assert predictor.DETECTOR_MODEL_PATH == (
        PROJECT_ROOT / detector.repository_relative_path
    )
    assert predictor.HEALTH_MODEL_PATH == (
        PROJECT_ROOT / health.repository_relative_path
    )
    assert predictor.DETECTOR_MODEL_NAME == detector.name
    assert predictor.HEALTH_MODEL_NAME == health.name
    assert predictor.DETECTOR_MODEL_SHA256 == detector.sha256
    assert predictor.HEALTH_MODEL_SHA256 == health.sha256
    assert predictor.DETECTOR_MODEL_SIZE_BYTES == detector.size_bytes
    assert predictor.HEALTH_MODEL_SIZE_BYTES == health.size_bytes
    assert predictor.DETECTOR_IMAGE_SIZE == detector.image_size
    assert predictor.HEALTH_IMAGE_SIZE == health.image_size
    assert predictor.SPECIES_BY_CLASS_ID == dict(detector.class_mapping)
    assert predictor.DETECTOR_MODEL_NAMES == dict(
        detector.raw_class_mapping
    )
    assert predictor.HEALTH_STATUS_BY_CLASS_ID == dict(
        health.class_mapping
    )
    assert predictor.HEALTH_MODEL_NAMES == dict(health.raw_class_mapping)


# 검증을 반복해도 모델의 내용·크기·mtime이 전혀 바뀌지 않는지 확인한다.
def test_verification_is_read_only(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, manifest, _detector, _health = synthetic_project(tmp_path)
    model_paths = tuple((root / "runtime/models").rglob("best.pt"))
    before = {
        path.relative_to(root): predictor.file_fingerprint(path)
        for path in model_paths
    }

    verification.verify_runtime_models(
        project_root=root,
        manifest_path=manifest,
    )
    verification.verify_runtime_models(
        project_root=root,
        manifest_path=manifest,
    )

    after = {
        path.relative_to(root): predictor.file_fingerprint(path)
        for path in model_paths
    }
    assert after == before
    output = capsys.readouterr().out
    assert output.count("detector: Git-managed runtime model verified") == 2
    assert output.count("health: Git-managed runtime model verified") == 2
    assert str(root) not in output


# 실제 모델 바이트의 SHA-256이 manifest와 다르면 변조로 거부한다.
def test_sha_mismatch_is_rejected(tmp_path: Path) -> None:
    root, _manifest, detector, health = synthetic_project(tmp_path)
    manifest = write_manifest(
        root,
        detector_payload=detector,
        health_payload=health,
        detector_hash="0" * 64,
    )

    with pytest.raises(
        verification.ModelVerificationError,
        match="detector model SHA-256 mismatch",
    ):
        verification.verify_runtime_models(
            project_root=root,
            manifest_path=manifest,
        )


# 두 필수 모델 중 하나라도 없으면 전체 검증을 실패시킨다.
def test_missing_model_is_rejected(tmp_path: Path) -> None:
    root, manifest, _detector, _health = synthetic_project(tmp_path)
    (root / "runtime/models/health/best.pt").unlink()

    with pytest.raises(
        verification.ModelVerificationError,
        match="health model is missing",
    ):
        verification.verify_runtime_models(
            project_root=root,
            manifest_path=manifest,
        )


# 빠른 손상 탐지를 위해 manifest와 다른 파일 크기도 별도 오류로 거부한다.
def test_size_mismatch_is_rejected(tmp_path: Path) -> None:
    root, manifest, _detector, _health = synthetic_project(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["models"][0]["sizeBytes"] += 1
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        verification.ModelVerificationError,
        match="detector model size mismatch",
    ):
        verification.verify_runtime_models(
            project_root=root,
            manifest_path=manifest,
        )


# runtime 모델 자리에 symlink를 두어 외부 파일을 우회 참조하지 못하게 한다.
def test_model_symlink_is_rejected(tmp_path: Path) -> None:
    root, manifest, detector, _health = synthetic_project(tmp_path)
    outside = tmp_path / "outside-detector.pt"
    outside.write_bytes(detector)
    model_path = root / "runtime/models/detector/best.pt"
    model_path.unlink()
    model_path.symlink_to(outside)

    with pytest.raises(
        verification.ModelVerificationError,
        match="detector model must not be a symbolic link",
    ):
        verification.verify_runtime_models(
            project_root=root,
            manifest_path=manifest,
        )


# manifest가 상위 디렉터리 이동 경로를 지정하면 파싱 단계에서 거부한다.
def test_manifest_path_traversal_is_rejected(tmp_path: Path) -> None:
    root, manifest, _detector, _health = synthetic_project(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["models"][0]["repositoryPath"] = "../outside.pt"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        verification.ModelVerificationError,
        match="repositoryPath must be repository-relative",
    ):
        verification.load_manifest(manifest)


# detector나 health가 중복되면 한 모델을 다른 역할 대신 사용하는 오류를 막는다.
def test_manifest_requires_each_role_once(tmp_path: Path) -> None:
    root, manifest, _detector, _health = synthetic_project(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["models"][1] = dict(payload["models"][0])
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        verification.ModelVerificationError,
        match="model manifest must contain each role once",
    ):
        verification.load_manifest(manifest)


# 입력 크기와 task의 잘못된 타입·값은 predictor import 전에 계약 오류로 잡는다.
@pytest.mark.parametrize(
    ("role_index", "field", "value", "message"),
    [
        (0, "imageSize", True, "detector imageSize is invalid"),
        (0, "imageSize", "640", "detector imageSize is invalid"),
        (0, "task", "classify", "detector model task is invalid"),
        (1, "task", None, "health model task is invalid"),
    ],
)
def test_manifest_rejects_invalid_runtime_contract_types(
    tmp_path: Path,
    role_index: int,
    field: str,
    value: object,
    message: str,
) -> None:
    root, manifest, _detector, _health = synthetic_project(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["models"][role_index][field] = value
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        verification.ModelVerificationError,
        match=message,
    ):
        verification.load_manifest(manifest)


# class id는 0부터 연속이어야 하고 display/raw mapping이 같은 id 집합을 가져야 한다.
@pytest.mark.parametrize(
    ("field", "mapping", "message"),
    [
        (
            "classMapping",
            {"0": "느타리", "2": "표고"},
            "detector classMapping class ids must be contiguous from zero",
        ),
        (
            "rawClassMapping",
            {"0": "oyster"},
            "detector class mapping ids do not match",
        ),
        (
            "classMapping",
            {"0": "HEALTHY", "1": "UNKNOWN"},
            "health classMapping must define HEALTHY and DISEASE_SUSPECTED",
        ),
    ],
)
def test_manifest_rejects_invalid_class_mapping(
    tmp_path: Path,
    field: str,
    mapping: dict[str, str],
    message: str,
) -> None:
    root, manifest, _detector, _health = synthetic_project(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    role_index = 1 if "health " in message else 0
    payload["models"][role_index][field] = mapping
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        verification.ModelVerificationError,
        match=message,
    ):
        verification.load_manifest(manifest)
