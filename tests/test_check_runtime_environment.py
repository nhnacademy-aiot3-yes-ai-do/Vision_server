from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import check_runtime_environment as doctor


class FakeMPS:
    @staticmethod
    def is_built() -> bool:
        return True

    @staticmethod
    def is_available() -> bool:
        return True


class FakeCUDA:
    @staticmethod
    def is_available() -> bool:
        return False


def fake_version(package: str) -> str:
    return {
        "torch": "2.11.0",
        "torchvision": "0.26.0",
        "ultralytics": "8.4.106",
    }[package]


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


def test_missing_torch_is_reported_without_import_or_exception_details() -> None:
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


def test_unsafe_device_value_is_not_printed() -> None:
    secret_like_value = "/home/example/private-device"

    report = doctor.collect_environment_report(
        environment={"HEALTH_DEVICE": secret_like_value},
        version_reader=fake_version,
        import_torch=False,
    )

    assert report["healthDevice"] == "REDACTED"
    assert secret_like_value not in json.dumps(report)
