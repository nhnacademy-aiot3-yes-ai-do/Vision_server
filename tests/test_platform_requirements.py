# 플랫폼별 의존성 파일과 Docker/Make 진입점의 정적 계약을 검증한다.
# macOS에 CUDA wheel이 섞이지 않고 Linux는 공식 CPU wheel을 사용하며,
# MPS/CPU 실행 장치를 명시한다는 전제를 보호한다.
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


# Linux Docker 이미지가 고정 CPU runtime을 설치하고 macOS 전용 의존성을 제외하는지 검증한다.
def test_linux_docker_installs_pinned_cpu_runtime_and_omits_macos_profile() -> None:
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "python:3.12.14-slim-bookworm@sha256:" in dockerfile
    assert "torch==2.11.0" in dockerfile
    assert "torchvision==0.26.0" in dockerfile
    assert "https://download.pytorch.org/whl/cpu" in dockerfile
    assert "libgl1" in dockerfile
    assert "libglib2.0-0" in dockerfile
    assert "libxcb1" in dockerfile
    assert "groupadd" in dockerfile
    assert "useradd" in dockerfile
    assert "USER ${APP_UID}:${APP_GID}" in dockerfile
    assert "assert torch.version.cuda is None" in dockerfile
    assert "assert not torch.cuda.is_available()" in dockerfile
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


# 서버 실행과 Docker 빌드가 Git 모델의 manifest 검증을 먼저 수행하는지 확인한다.
def test_make_run_and_docker_build_verify_models_first() -> None:
    makefile = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")

    assert "verify-models:" in makefile
    assert "$(PYTHON) scripts/verify_runtime_models.py" in makefile
    assert "run: verify-models" in makefile
    assert "docker-build: verify-models" in makefile
    assert "--build-arg BASE_IMAGE" not in makefile
    assert "--platform linux/amd64" in makefile
    assert "--file Dockerfile" in makefile


# GitHub workflow가 경량 테스트와 중앙 배포 계약을 분리하는지 검증한다.
def test_github_workflows_match_ci_and_central_deploy_contract() -> None:
    reusable = (
        PROJECT_ROOT / ".github/workflows/_reusable-test.yml"
    ).read_text(encoding="utf-8")
    develop = (
        PROJECT_ROOT / ".github/workflows/develop-ci.yml"
    ).read_text(encoding="utf-8")
    pr_check = (
        PROJECT_ROOT / ".github/workflows/pr-check.yml"
    ).read_text(encoding="utf-8")
    deploy = (
        PROJECT_ROOT / ".github/workflows/deploy.yml"
    ).read_text(encoding="utf-8")

    for action in (
        "actions/checkout@v6",
        "actions/setup-python@v6",
    ):
        assert action in reusable
    assert 'python-version: "3.12"' in reusable
    assert (
        "python -m pip install -r requirements-common.txt "
        "-r requirements-dev.txt"
    ) in reusable
    assert "python scripts/verify_runtime_models.py" in reusable
    assert "python -m pytest -q" in reusable
    assert "branches: [develop]" in develop
    assert "branches: [develop, main]" in pr_check

    for action in (
        "actions/checkout@v6",
        "docker/login-action@v4",
        "docker/setup-buildx-action@v4",
        "docker/build-push-action@v7",
        "actions/create-github-app-token@v3",
    ):
        assert action in deploy
    assert "packages: write" in deploy
    assert "password: ${{ secrets.GITHUB_TOKEN }}" in deploy
    assert "platforms: linux/amd64" in deploy
    assert "${{ needs.image-metadata.outputs.image_name }}:${{ github.sha }}" in deploy
    assert "vars.VISION_CENTRAL_DEPLOY_ENABLED == 'true'" in deploy
    assert "docker buildx imagetools inspect" in deploy
    assert "GHCR image를 인증 없이 조회할 수 없습니다" in deploy
    assert "QUALITY_RESULT: ${{ needs.test-and-model-verify.result }}" in deploy
    assert "COVERAGE_RESULT: not_configured" in deploy
    assert "bash .github/scripts/config-deployment.sh" in deploy
    assert "SONAR" not in reusable
