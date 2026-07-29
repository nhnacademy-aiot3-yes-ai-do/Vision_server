#!/usr/bin/env python3
"""Print an offline, path-safe summary of the local inference environment."""

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

    try:
        return version_reader(package)
    except metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def _import_torch() -> Any | None:
    """Import torch for capability flags, suppressing unsafe local error details."""

    try:
        return importlib.import_module("torch")
    except (ImportError, OSError, RuntimeError):
        return None


def _backend_flag(torch_module: Any, backend: str, method: str) -> bool:
    backends = getattr(torch_module, "backends", None)
    selected = getattr(backends, backend, None)
    probe = getattr(selected, method, None)
    if not callable(probe):
        return False
    try:
        return bool(probe())
    except (OSError, RuntimeError):
        return False


def _cuda_available(torch_module: Any) -> bool:
    cuda = getattr(torch_module, "cuda", None)
    probe = getattr(cuda, "is_available", None)
    if not callable(probe):
        return False
    try:
        return bool(probe())
    except (OSError, RuntimeError):
        return False


def _safe_device(environment: Mapping[str, str]) -> str:
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

    source = os.environ if environment is None else environment
    loaded_torch = (
        _import_torch()
        if torch_module is None and import_torch
        else torch_module
    )
    if loaded_torch is None:
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
            key: (project_root / relative_path).is_file()
            for key, relative_path in MODEL_RELATIVE_PATHS.items()
        },
    }


def main() -> int:
    report = collect_environment_report()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
