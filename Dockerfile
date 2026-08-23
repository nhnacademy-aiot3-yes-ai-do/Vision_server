# syntax=docker/dockerfile:1.7
#
# Linux amd64 CPU 배포 이미지다. macOS/MPS 및 CUDA용 이미지가 아니다.
# Docker Official Python 이미지의 변경 불가능한 multi-arch digest를 고정한다.
ARG BASE_IMAGE=docker.io/library/python:3.12.14-slim-bookworm@sha256:a116514e19457bcb7af7efe9c3dd0b9b71e85b317694e7882a1c52aa15a78134
FROM ${BASE_IMAGE}

# 배포판별 사용자 생성 명령에 의존하지 않도록 숫자 UID/GID를 사용한다.
ARG APP_UID=10001
ARG APP_GID=10001

USER 0

# Ultralytics가 설치하는 OpenCV wheel의 headless 추론 import에 필요한
# 최소 Linux shared library만 설치하고 apt metadata는 image에 남기지 않는다.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# 캐시와 설정 파일은 읽기 전용 root filesystem에서도 쓸 수 있는 /tmp로 모은다.
# 모델 경로와 SHA 검증 기본값은 컨테이너의 고정 배치 계약이다.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/tmp/mushroom-home \
    YOLO_CONFIG_DIR=/tmp/mushroom-yolo \
    MPLCONFIGDIR=/tmp/mushroom-matplotlib \
    TORCH_HOME=/tmp/mushroom-torch \
    XDG_CACHE_HOME=/tmp/mushroom-cache \
    DETECTOR_MODEL_PATH=/opt/mushroom-vision/runtime/models/detector/best.pt \
    HEALTH_MODEL_PATH=/opt/mushroom-vision/runtime/models/health/best.pt \
    HEALTH_VERIFY_MODEL_SHA256=true

WORKDIR /opt/mushroom-vision

# PyTorch 공식 CPU wheel을 먼저 고정 설치한다. 이후 runtime 의존성이
# macOS wheel이나 CUDA build로 교체하지 않도록 설치 후 버전과 CPU 상태를 검증한다.
COPY requirements-common.txt ./requirements-common.txt
COPY requirements-runtime.txt ./requirements-runtime.txt
RUN python -m pip install --no-cache-dir \
        torch==2.11.0 \
        torchvision==0.26.0 \
        --index-url https://download.pytorch.org/whl/cpu
RUN python -m pip install --no-cache-dir -r requirements-runtime.txt
RUN python -m pip check \
    && python -c "import cv2, torch, torchvision, ultralytics; assert torch.__version__.split('+')[0] == '2.11.0'; assert torchvision.__version__.split('+')[0] == '0.26.0'; assert torch.version.cuda is None; assert not torch.cuda.is_available()"

# 운영에 필요한 표면만 복사한다. `.dockerignore`의 허용 목록이 데이터셋,
# 보고서, 테스트, 로컬 환경과 임의 산출물이 build context에 들어오는 것을 막는다.
COPY app ./app
COPY scripts/predict_mushroom_health.py ./scripts/predict_mushroom_health.py
COPY scripts/verify_runtime_models.py ./scripts/verify_runtime_models.py
COPY models/model-manifest.json ./models/model-manifest.json

# Git의 runtime 경로를 그대로 복사해 모델 위치에 두 번째 기준을 만들지 않는다.
# 빌드 안에서도 manifest의 크기와 SHA-256을 다시 검증한 뒤 읽기 전용으로 고정한다.
COPY runtime/models/detector/best.pt ./runtime/models/detector/best.pt
COPY runtime/models/health/best.pt ./runtime/models/health/best.pt
RUN python scripts/verify_runtime_models.py \
    && chmod 0444 \
        runtime/models/detector/best.pt \
        runtime/models/health/best.pt \
        models/model-manifest.json \
    && chmod -R a-w app scripts models runtime/models \
    && chmod 0555 runtime runtime/models runtime/models/detector runtime/models/health

# PyTorch가 runtime cache 경로를 계산할 때 UID의 사용자명을 조회하므로
# 로그인 권한과 home directory가 없는 전용 계정을 등록한다.
RUN groupadd --gid "${APP_GID}" vision \
    && useradd \
        --uid "${APP_UID}" \
        --gid "${APP_GID}" \
        --home-dir /tmp/mushroom-home \
        --no-create-home \
        --shell /usr/sbin/nologin \
        vision

# 전용 계정으로 전환하여 이후 서버가 root 권한으로 동작하지 않게 한다.
USER ${APP_UID}:${APP_GID}

EXPOSE 8000

# readiness는 FastAPI lifespan에서 두 모델 로드가 끝난 뒤에만 성공한다.
# worker마다 모델 두 개가 다시 올라가므로 단일 worker 계약을 유지한다.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=3).read(1)"]

# 컨테이너의 최종 프로세스도 Makefile과 동일하게 worker 1개로 서버를 띄운다.
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
