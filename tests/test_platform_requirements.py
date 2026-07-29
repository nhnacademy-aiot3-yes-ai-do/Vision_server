from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _active_requirements(name: str) -> list[str]:
    return [
        line
        for raw_line in (PROJECT_ROOT / name).read_text(encoding="utf-8").splitlines()
        if (line := raw_line.strip()) and not line.startswith("#")
    ]


def test_cuda_build_is_not_pinned_in_common_or_macos_requirements() -> None:
    common = _active_requirements("requirements-common.txt")
    macos = _active_requirements("requirements-macos.txt")
    runtime = _active_requirements("requirements-runtime.txt")

    assert not any(line.startswith(("torch==", "torchvision==")) for line in common)
    assert "torch==2.11.0" in macos
    assert "torchvision==0.26.0" in macos
    assert runtime == ["-r requirements-common.txt"]
    for line in common + macos + runtime:
        assert "+cu" not in line
        assert "--index-url" not in line
        assert "--extra-index-url" not in line


def test_linux_docker_requires_base_torch_and_omits_macos_profile() -> None:
    dockerfile = (PROJECT_ROOT / "Dockerfile.template").read_text(
        encoding="utf-8"
    )

    assert 'RUN python -c "import torch, torchvision"' in dockerfile
    assert "COPY requirements-common.txt" in dockerfile
    assert "COPY requirements-runtime.txt" in dockerfile
    assert "requirements-macos.txt" not in dockerfile


def test_mac_make_targets_use_platform_profile_and_explicit_devices() -> None:
    makefile = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")

    assert "install-mac:" in makefile
    assert "-r requirements-macos.txt" in makefile
    assert "-r requirements-dev.txt" in makefile
    assert "doctor-mac:" in makefile
    assert "HEALTH_DEVICE=mps $(MAKE) run" in makefile
    assert "HEALTH_DEVICE=cpu $(MAKE) run" in makefile
