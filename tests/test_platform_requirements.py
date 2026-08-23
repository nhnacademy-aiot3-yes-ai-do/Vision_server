# 플랫폼별 의존성 파일과 Docker/Make 진입점의 정적 계약을 검증한다.
# macOS에 CUDA wheel이 섞이지 않고 Linux는 base image의 torch를 사용하며,
# MPS/CPU 실행 장치를 Make target에서 명시한다는 전제를 보호한다.
from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# requirements 파일에서 빈 줄과 설명 주석을 제외하고 실제 설치 항목만 추출하는 helper이다.
def _active_requirements(name: str) -> list[str]:
    return [
        line
        for raw_line in (PROJECT_ROOT / name).read_text(encoding="utf-8").splitlines()
        if (line := raw_line.strip()) and not line.startswith("#")
    ]


# 공통 CI는 torch가 없고 macOS/Linux runtime만 Ultralytics를 설치하는지 검증한다.
def test_cuda_build_is_not_pinned_in_common_or_macos_requirements() -> None:
    common = _active_requirements("requirements-common.txt")
    macos = _active_requirements("requirements-macos.txt")
    runtime = _active_requirements("requirements-runtime.txt")

    assert not any(line.startswith(("torch==", "torchvision==")) for line in common)
    assert "ultralytics==8.4.106" not in common
    assert "torch==2.11.0" in macos
    assert "torchvision==0.26.0" in macos
    assert "ultralytics==8.4.106" in macos
    assert runtime == [
        "-r requirements-common.txt",
        "ultralytics==8.4.106",
    ]
    for line in common + macos + runtime:
        assert "+cu" not in line
        assert "--index-url" not in line
        assert "--extra-index-url" not in line


# Linux Docker 이미지가 base image의 torch를 확인하고 macOS 전용 의존성을 복사하지 않는지 검증한다.
def test_linux_docker_requires_base_torch_and_omits_macos_profile() -> None:
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert 'RUN python -c "import torch, torchvision"' in dockerfile
    assert "COPY requirements-common.txt" in dockerfile
    assert "COPY requirements-runtime.txt" in dockerfile
    assert "COPY scripts/verify_runtime_models.py" in dockerfile
    assert "RUN python scripts/verify_runtime_models.py" in dockerfile
    assert "runtime/models/detector/best.pt" in dockerfile
    assert "runtime/models/health/best.pt" in dockerfile
    assert "requirements-macos.txt" not in dockerfile


# macOS Make target이 전용 requirements와 MPS·CPU 장치를 명시적으로 선택하는지 검증한다.
def test_mac_make_targets_use_platform_profile_and_explicit_devices() -> None:
    makefile = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")

    assert "install-mac:" in makefile
    assert "-r requirements-macos.txt" in makefile
    assert "-r requirements-dev.txt" in makefile
    assert "doctor-mac:" in makefile
    assert "HEALTH_DEVICE=mps $(MAKE) run" in makefile
    assert "HEALTH_DEVICE=cpu $(MAKE) run" in makefile


# 서버 실행과 Docker 빌드가 private Git 모델의 manifest 검증을 먼저 수행하는지 확인한다.
def test_make_run_and_docker_build_verify_models_first() -> None:
    makefile = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")

    assert "verify-models:" in makefile
    assert "$(PYTHON) scripts/verify_runtime_models.py" in makefile
    assert "run: verify-models" in makefile
    assert "docker-build: check-base-image verify-models" in makefile
    assert "--file Dockerfile" in makefile


# GHCR workflow가 최소 권한 GITHUB_TOKEN과 지정된 공식 action major를 사용하는지 검증한다.
def test_ghcr_workflow_uses_supported_actions_and_base_image_input() -> None:
    workflow = (
        PROJECT_ROOT / ".github/workflows/publish-image.yml"
    ).read_text(encoding="utf-8")

    for action in (
        "actions/checkout@v6",
        "actions/setup-python@v6",
        "docker/login-action@v4",
        "docker/metadata-action@v6",
        "docker/setup-buildx-action@v4",
        "docker/build-push-action@v7",
    ):
        assert action in workflow
    assert "contents: read" in workflow
    assert "packages: write" in workflow
    assert "pull_request:" in workflow
    assert 'python-version: "3.12"' in workflow
    assert (
        "python -m pip install -r requirements-common.txt "
        "-r requirements-dev.txt"
    ) in workflow
    assert "python -m pytest -q" in workflow
    assert "needs: test" in workflow
    assert "github.event_name != 'pull_request'" in workflow
    assert "password: ${{ secrets.GITHUB_TOKEN }}" in workflow
    assert "${{ vars.VISION_BASE_IMAGE }}" in workflow
    assert "${{ github.event.inputs.base_image }}" in workflow
    assert "python3 scripts/verify_runtime_models.py" in workflow
    assert "file: Dockerfile" in workflow
