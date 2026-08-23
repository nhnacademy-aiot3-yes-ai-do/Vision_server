#!/usr/bin/env python3
"""Print an offline, path-safe summary of the local inference environment."""

# 목적: 모델 추론을 실행하기 전에 Python/딥러닝 패키지/가속기/모델 파일의
#       준비 상태를 네트워크 접속 없이 한 번에 확인한다.
# 입력: 현재 프로세스의 환경 변수와 로컬 패키지 메타데이터, 고정 모델 경로.
# 출력: 민감한 로컬 경로를 포함하지 않는 JSON 환경 점검 보고서(stdout).
# 처리 흐름: 패키지 버전 조회 -> torch 가속기 기능 확인 -> 장치 설정값 정제
#            -> 원본·runtime 모델 존재 여부 확인 -> JSON 직렬화.

from __future__ import annotations

import importlib
import json
import os
import platform
import re
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import predict_mushroom_health as predictor


MODEL_RELATIVE_PATHS = {
    "sourceDetector": predictor.DETECTOR_MODEL_PATH.relative_to(PROJECT_ROOT),
    "sourceHealth": predictor.HEALTH_MODEL_PATH.relative_to(PROJECT_ROOT),
    "runtimeDetector": Path("runtime/models/detector/best.pt"),
    "runtimeHealth": Path("runtime/models/health/best.pt"),
}
SAFE_DEVICE_PATTERN = re.compile(r"^[A-Za-z0-9_,:.-]{1,64}$")
VersionReader = Callable[[str], str]


def _distribution_version(
    package: str,
    *,
    version_reader: VersionReader = metadata.version,
) -> str:
    """Return installed distribution version without contacting a package index."""

    # import 여부가 아니라 설치 배포판 메타데이터를 읽으므로 패키지를 실행하지 않고
    # 버전만 확인할 수 있으며, 미설치 상태도 예외 대신 보고서 값으로 표현한다.
    try:
        return version_reader(package)
    except metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def _import_torch() -> Any | None:
    """Import torch for capability flags, suppressing unsafe local error details."""

    # torch는 설치되어 있어도 드라이버/동적 라이브러리 문제로 import가 실패할 수 있다.
    # 환경 점검 자체는 계속 수행해야 하므로 예상 가능한 로딩 오류를 None으로 축약한다.
    try:
        return importlib.import_module("torch")
    except (ImportError, OSError, RuntimeError):
        return None


def _backend_flag(torch_module: Any, backend: str, method: str) -> bool:
    # torch 버전이나 빌드에 따라 backend API가 없을 수 있어 단계별 getattr로 탐색한다.
    backends = getattr(torch_module, "backends", None)
    selected = getattr(backends, backend, None)
    probe = getattr(selected, method, None)
    if not callable(probe):
        return False
    # 기능 조회도 런타임 드라이버 오류를 낼 수 있으므로 단순 미사용 상태로 처리한다.
    try:
        return bool(probe())
    except (OSError, RuntimeError):
        return False


def _cuda_available(torch_module: Any) -> bool:
    # CUDA 모듈/메서드가 없는 CPU 전용 torch에서도 점검 스크립트가 실패하지 않게 한다.
    cuda = getattr(torch_module, "cuda", None)
    probe = getattr(cuda, "is_available", None)
    if not callable(probe):
        return False
    try:
        return bool(probe())
    except (OSError, RuntimeError):
        return False


def _safe_device(environment: Mapping[str, str]) -> str:
    # 환경 변수 전체를 노출하지 않고, 장치 지정에 필요한 제한된 문자만 통과시킨다.
    value = environment.get("HEALTH_DEVICE", "auto").strip() or "auto"
    if SAFE_DEVICE_PATTERN.fullmatch(value):
        return value
    return "REDACTED"


def collect_environment_report(
    *,
    project_root: Path = PROJECT_ROOT,
    environment: Mapping[str, str] | None = None,
    version_reader: VersionReader = metadata.version,
    torch_module: Any | None = None,
    import_torch: bool = True,
) -> dict[str, Any]:
    """Collect versions, accelerator flags, and model-presence booleans."""

    # 테스트에서는 주입된 환경/torch를 사용하고, 실제 실행에서는 현재 프로세스 상태를 읽는다.
    source = os.environ if environment is None else environment
    loaded_torch = (
        _import_torch()
        if torch_module is None and import_torch
        else torch_module
    )
    if loaded_torch is None:
        # False는 "조회했지만 사용할 수 없음", None은 "torch를 불러오지 못해 미확인"을 뜻한다.
        mps_built: bool | None = None
        mps_available: bool | None = None
        cuda_available: bool | None = None
    else:
        mps_built = _backend_flag(loaded_torch, "mps", "is_built")
        mps_available = _backend_flag(loaded_torch, "mps", "is_available")
        cuda_available = _cuda_available(loaded_torch)

    return {
        "pythonVersion": platform.python_version(),
        "operatingSystem": platform.system(),
        "architecture": platform.machine(),
        "torchVersion": _distribution_version(
            "torch",
            version_reader=version_reader,
        ),
        "torchvisionVersion": _distribution_version(
            "torchvision",
            version_reader=version_reader,
        ),
        "ultralyticsVersion": _distribution_version(
            "ultralytics",
            version_reader=version_reader,
        ),
        "mpsBuilt": mps_built,
        "mpsAvailable": mps_available,
        "cudaAvailable": cuda_available,
        "healthDevice": _safe_device(source),
        "modelFiles": {
            # 경로 문자열 대신 존재 여부만 내보내 로컬 디렉터리 구조를 숨긴다.
            key: (project_root / relative_path).is_file()
            for key, relative_path in MODEL_RELATIVE_PATHS.items()
        },
    }


def main() -> int:
    # CLI의 유일한 부작용은 점검 결과를 표준 출력으로 내보내는 것이다.
    report = collect_environment_report()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
