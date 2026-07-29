PYTHON ?= python
HOST ?= 0.0.0.0
PORT ?= 8000
IMAGE_NAME ?= mushroom-vision-service:prototype
BASE_IMAGE ?=

.PHONY: test run prepare-models check-models check-base-image docker-build docker-run

test:
	$(PYTHON) -m pytest -q

run:
	$(PYTHON) -m uvicorn app.main:app \
		--host $(HOST) \
		--port $(PORT) \
		--workers 1

prepare-models:
	$(PYTHON) scripts/prepare_runtime_models.py

check-models:
	$(PYTHON) scripts/prepare_runtime_models.py --check-only

check-base-image:
	@if [ -z "$(strip $(BASE_IMAGE))" ]; then \
		echo "BASE_IMAGE is required; use a team-approved Python/PyTorch image."; \
		echo "Example: make docker-build BASE_IMAGE=<approved-image>"; \
		exit 2; \
	fi

docker-build: check-base-image check-models
	docker build \
		--build-arg BASE_IMAGE="$(BASE_IMAGE)" \
		--file Dockerfile.template \
		--tag "$(IMAGE_NAME)" \
		.

docker-run: docker-build
	docker run --rm \
		--read-only \
		--tmpfs /tmp:rw,nosuid,nodev,size=128m \
		--publish "$(PORT):8000" \
		--env DETECTOR_MODEL_PATH=/models/detector/best.pt \
		--env HEALTH_MODEL_PATH=/models/health/best.pt \
		"$(IMAGE_NAME)"
