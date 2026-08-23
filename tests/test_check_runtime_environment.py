# Mac/Linux 런타임 진단 보고서가 네트워크 없이 안전하게 만들어지는지 검증한다.
# fake MPS/CUDA 객체로 버전·가속기·모델 유무는 정확히 보고하면서
# 로컬 경로나 위험한 환경변수 값은 숨기는지 확인한다.
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import check_runtime_environment as doctor


# MPS 지원이 빌드되어 있고 현재 장치에서도 사용 가능하다고 응답하는 torch backend 대역이다.
class FakeMPS:
    @staticmethod
    def is_built() -> bool:
        return True

    @staticmethod
    def is_available() -> bool:
        return True


# CUDA를 사용할 수 없다고 응답하여 macOS 계열 런타임 상황을 재현하는 대역이다.
class FakeCUDA:
    @staticmethod
    def is_available() -> bool:
        return False


# 설치 메타데이터 조회를 실제 환경과 분리하고 기대 버전을 고정하는 helper이다.
def fake_version(package: str) -> str:
    return {
        "torch": "2.11.0",
        "torchvision": "0.26.0",
        "ultralytics": "8.4.106",
    }[package]


# 오프라인 진단이 버전·가속기·모델 존재 여부를 완전하게 보고하되 절대 경로는 숨기는지 검증한다.
def test_environment_report_is_offline_path_safe_and_complete(
    tmp_path: Path,
) -> None:
    for relative_path in doctor.MODEL_RELATIVE_PATHS.values():
        target = tmp_path / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"synthetic")
    fake_torch = SimpleNamespace(
        backends=SimpleNamespace(mps=FakeMPS()),
        cuda=FakeCUDA(),
    )

    report = doctor.collect_environment_report(
        project_root=tmp_path,
        environment={"HEALTH_DEVICE": "mps"},
        version_reader=fake_version,
        torch_module=fake_torch,
    )

    assert report["torchVersion"] == "2.11.0"
    assert report["torchvisionVersion"] == "0.26.0"
    assert report["ultralyticsVersion"] == "8.4.106"
    assert report["mpsBuilt"] is True
    assert report["mpsAvailable"] is True
    assert report["cudaAvailable"] is False
    assert report["healthDevice"] == "mps"
    assert all(report["modelFiles"].values())
    serialized = json.dumps(report)
    assert str(tmp_path) not in serialized
    assert "/home/" not in serialized
    assert "/mnt/" not in serialized


# torch 패키지가 없어도 import 예외 세부정보 없이 NOT_INSTALLED와 미확인 상태를 반환하는지 검증한다.
def test_missing_torch_is_reported_without_import_or_exception_details() -> None:
    # 요청한 모든 패키지가 설치되지 않은 상황을 재현하는 메타데이터 reader이다.
    def missing_version(package: str) -> str:
        raise doctor.metadata.PackageNotFoundError(package)

    report = doctor.collect_environment_report(
        environment={"HEALTH_DEVICE": "   "},
        version_reader=missing_version,
        import_torch=False,
    )

    assert report["torchVersion"] == "NOT_INSTALLED"
    assert report["torchvisionVersion"] == "NOT_INSTALLED"
    assert report["ultralyticsVersion"] == "NOT_INSTALLED"
    assert report["mpsBuilt"] is None
    assert report["mpsAvailable"] is None
    assert report["cudaAvailable"] is None
    assert report["healthDevice"] == "auto"


# 경로처럼 위험한 장치 설정값을 그대로 출력하지 않고 REDACTED로 치환하는지 검증한다.
def test_unsafe_device_value_is_not_printed() -> None:
    secret_like_value = "/home/example/private-device"

    report = doctor.collect_environment_report(
        environment={"HEALTH_DEVICE": secret_like_value},
        version_reader=fake_version,
        import_torch=False,
    )

    assert report["healthDevice"] == "REDACTED"
    assert secret_like_value not in json.dumps(report)
