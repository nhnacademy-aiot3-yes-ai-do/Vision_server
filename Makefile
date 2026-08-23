# 로컬 실행과 Docker 프로토타입이 공통으로 사용하는 기본값이다.
# `?=`를 사용하므로 `make run PORT=9000`처럼 호출 시 안전하게 덮어쓸 수 있다.
PYTHON ?= python
HOST ?= 0.0.0.0
PORT ?= 8000
IMAGE_NAME ?= mushroom-vision-service:prototype
BASE_IMAGE ?=

# 실제 파일을 만드는 규칙이 아니라 명령 진입점임을 Make에 알린다.
.PHONY: \
	test run run-mps run-cpu install-mac doctor-mac \
	verify-models check-models check-base-image docker-build docker-run

# Apple Silicon Mac인지 먼저 확인한 뒤 로컬 추론·테스트 의존성을 설치한다.
install-mac:
	@$(PYTHON) -c 'import platform, sys; ok = platform.system() == "Darwin" and platform.machine() == "arm64"; print("Apple Silicon macOS required" if not ok else "Apple Silicon macOS detected"); sys.exit(0 if ok else 2)'
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements-macos.txt
	$(PYTHON) -m pip install -r requirements-dev.txt

# 실제 모델을 매번 읽지 않는 기본 pytest 회귀 테스트를 실행한다.
test:
	$(PYTHON) -m pytest -q

# private Git의 두 모델이 manifest와 일치할 때만 서버를 시작한다.
# GPU 모델이 프로세스마다 복제되므로 Uvicorn worker는 반드시 1개만 사용한다.
run: verify-models
	$(PYTHON) -m uvicorn app.main:app \
		--host $(HOST) \
		--port $(PORT) \
		--workers 1

# 같은 서버를 실행하되 장치 선택만 명시적으로 MPS 또는 CPU로 고정한다.
run-mps:
	HEALTH_DEVICE=mps $(MAKE) run

run-cpu:
	HEALTH_DEVICE=cpu $(MAKE) run

# 패키지 버전, 가속기 가용성, 모델 파일 유무를 경로 노출 없이 점검한다.
doctor-mac:
	$(PYTHON) scripts/check_runtime_environment.py

# private Git에서 직접 관리하는 두 모델의 존재·크기·SHA-256을 검증한다.
verify-models:
	$(PYTHON) scripts/verify_runtime_models.py

# 기존 자동화가 사용하던 이름은 복사 없이 같은 검증을 수행하는 별칭으로 유지한다.
check-models: verify-models

# 팀이 승인한 Python/PyTorch base image 없이 우연히 이미지를 만들지 못하게 막는다.
check-base-image:
	@if [ -z "$(strip $(BASE_IMAGE))" ]; then \
		echo "BASE_IMAGE is required; use a team-approved Python/PyTorch image."; \
		echo "Example: make docker-build BASE_IMAGE=<approved-image>"; \
		exit 2; \
	fi

# 모델 검증이 성공한 경우에만 Docker 이미지를 빌드한다.
docker-build: check-base-image verify-models
	docker build \
		--build-arg BASE_IMAGE="$(BASE_IMAGE)" \
		--file Dockerfile \
		--tag "$(IMAGE_NAME)" \
		.

# 운영 환경의 읽기 전용 파일 시스템 조건을 로컬에서도 가깝게 재현한다.
docker-run: docker-build
	docker run --rm \
		--read-only \
		--tmpfs /tmp:rw,nosuid,nodev,size=128m \
		--publish "$(PORT):8000" \
		--env DETECTOR_MODEL_PATH=/opt/mushroom-vision/runtime/models/detector/best.pt \
		--env HEALTH_MODEL_PATH=/opt/mushroom-vision/runtime/models/health/best.pt \
		"$(IMAGE_NAME)"
