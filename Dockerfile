# syntax=docker/dockerfile:1.7
#
# Linux 배포 구조를 검증하기 위한 템플릿이며 macOS/MPS용 이미지가 아니다.
# 팀이 승인한 BASE_IMAGE를 빌드 시 직접 전달해야 한다.
# TODO(BASE_IMAGE): Python 3.12/PyTorch 이미지의 변경 불가능한 digest를 확정한다.
# TODO(GPU): 배포 대상이 정해진 뒤 CPU 또는 CUDA 런타임을 확정한다.
# base image에는 서로 호환되는 torch/torchvision 조합이 이미 있어야 한다.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# 배포판별 사용자 생성 명령에 의존하지 않도록 숫자 UID/GID를 사용한다.
ARG APP_UID=10001
ARG APP_GID=10001

USER 0

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

# requirements-runtime에는 플랫폼 중립적인 서비스 패키지만 들어 있다.
# torch/torchvision은 승인된 Linux base가 제공하므로 pip가 Mac wheel이나
# 승인되지 않은 CUDA build로 교체해서는 안 된다. 패키지 저장소 인증정보도
# requirements 또는 build argument에 넣지 않는다.
COPY requirements-common.txt ./requirements-common.txt
COPY requirements-runtime.txt ./requirements-runtime.txt
# 설치 전에 base image가 두 핵심 런타임을 실제로 제공하는지 빠르게 실패시킨다.
RUN python -c "import torch, torchvision"
RUN python -m pip install --no-cache-dir -r requirements-runtime.txt

# 운영에 필요한 표면만 복사한다. `.dockerignore`의 허용 목록이 데이터셋,
# 보고서, 테스트, 로컬 환경과 임의 산출물이 build context에 들어오는 것을 막는다.
COPY app ./app
COPY scripts/predict_mushroom_health.py ./scripts/predict_mushroom_health.py
COPY scripts/verify_runtime_models.py ./scripts/verify_runtime_models.py
COPY models/model-manifest.json ./models/model-manifest.json

# private Git의 runtime 경로를 그대로 복사해 모델 위치에 두 번째 기준을 만들지 않는다.
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

# 숫자 계정으로 전환하여 이후 서버가 root 권한으로 동작하지 않게 한다.
USER ${APP_UID}:${APP_GID}

EXPOSE 8000

# readiness는 FastAPI lifespan에서 두 모델 로드가 끝난 뒤에만 성공한다.
# worker마다 모델 두 개가 다시 올라가므로 단일 worker 계약을 유지한다.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=3).read(1)"]

# 컨테이너의 최종 프로세스도 Makefile과 동일하게 worker 1개로 서버를 띄운다.
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
