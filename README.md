# Mushroom Vision Service

버섯 사진에서 품종을 탐지하고 품종별 건강 상태 후보를 반환하는
Python 3.12 / FastAPI 기반 내부 AI 서비스 프로토타입입니다.

이 서비스의 결과는 **AI 참고 정보**이며 병명을 확정하는 진단이 아닙니다.
외부 스마트폰 사진에 대한 독립 일반화 검증 전에는 최종 서비스 성능으로
간주하지 않습니다.

## MSA에서의 역할

```text
Cultivation Service
        ↓
Spring AI-Service
        ↓ HTTP multipart
Mushroom Vision Service
        ↓ 구조화된 JSON
Spring AI-Service가 센서·제어·RAG 정보와 종합
        ↓
사용자 응답
```

Spring AI-Service는 Vision 결과를 다른 도메인 정보와 종합하고 사용자용
설명을 생성합니다. Mushroom Vision Service는 이미지 분석 경계만 담당합니다.

### 담당하는 일

- 업로드 이미지의 형식·크기·픽셀 수 검증
- 5개 버섯 품종 탐지와 bbox·개수·confidence 계산
- 품종별 detection grouping 및 union bbox 생성
- 원본 크기 기준 15% padding crop 생성
- `HEALTHY` / `DISEASE_SUSPECTED` 후보 분류
- confidence 안전 규칙 적용과 camelCase JSON 반환

### 담당하지 않는 일

- 병명·원인 확정 진단 또는 병반 위치 탐지
- 센서 데이터 조회와 재배 제어
- 사용자용 최종 LLM 설명 생성
- 사진 영구 저장
- 수확 적기, 생육 단계, 예상 수확일 판단
- 실제 갓·대 크기 또는 해충 판단
- 모델 재학습·변환

## 요청 처리 흐름

```text
입력 이미지
→ 품종 탐지
→ 품종별 detection grouping
→ union bbox
→ 이미지 폭·높이 기준 15% padding crop
→ 건강 이진 분류
→ 구조화된 JSON
```

같은 품종 객체가 여러 개면 해당 품종 bbox 전체의 union crop을 한 번
분류합니다. 서로 다른 품종은 각각 별도 결과로 반환합니다.

`detectionConfidenceMin < HEALTH_MIN_DETECTION_CONFIDENCE`이면 해당 품종의
건강 classifier를 호출하지 않습니다. 이 경우 `healthStatus=UNCERTAIN`이고
건강 confidence와 두 확률은 `null`입니다. 다른 품종은 계속 분석합니다.

## 지원 범위

### 품종

| class ID | 품종 |
| ---: | --- |
| 0 | 느타리 |
| 1 | 양송이 |
| 2 | 큰느타리 |
| 3 | 팽이 |
| 4 | 표고 |

### 건강 상태

| 값 | 의미 |
| --- | --- |
| `HEALTHY` | 정상 후보 |
| `DISEASE_SUSPECTED` | 병해 의심 후보 |
| `UNCERTAIN` | 탐지 또는 건강 분류 confidence가 안전 기준 미달 |

## 빠른 시작

### 1. Python 환경

검증 기준은 Python 3.12.3입니다.

의존성 파일의 역할은 플랫폼별로 다릅니다.

| 파일 | 역할 |
| --- | --- |
| `requirements-common.txt` | FastAPI, Ultralytics, Pillow 등 플랫폼 공통 패키지 |
| `requirements-macos.txt` | Apple Silicon 호스트용 공통 패키지와 비-CUDA PyTorch/torchvision |
| `requirements-runtime.txt` | Linux container용 공통 패키지; PyTorch/torchvision은 승인된 base image가 제공 |
| `requirements-dev.txt` | pytest 등 개발·테스트 전용 패키지 |

Apple Silicon Mac에서는 다음처럼 macOS runtime과 개발 의존성을 함께
설치합니다.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-macos.txt
python -m pip install -r requirements-dev.txt
```

같은 설치 절차를 한 명령으로 실행하려면 다음 target을 사용합니다.

```bash
make install-mac
make doctor-mac
```

`make install-mac`은 Apple Silicon host인지 확인한 뒤
`requirements-macos.txt`와 `requirements-dev.txt`를 설치합니다.
`make doctor-mac`은 패키지 버전, MPS/CUDA 가용성, 모델 파일 존재 여부를
로컬 절대경로 없이 출력합니다.

`requirements-macos.txt`의 `torch==2.11.0`과
`torchvision==0.26.0`은 Apple Silicon용 wheel을 사용하며 CUDA local
version suffix를 사용하지 않습니다. Mac에서 CUDA wheel이나 Linux용
`requirements-runtime.txt`를 대신 설치하지 마세요.

현재 설치된 `ultralytics==8.4.106` metadata의 하한은
`torch>=1.8.0`, `torchvision>=0.9.0`으로 넓습니다. 이 하한만으로 모델
호환성이 증명되지는 않으므로 Mac은 현재 검증 환경과 같은 release pair인
PyTorch 2.11 / torchvision 0.26을 명시적으로 사용합니다. 실제 MPS 동작은
MacBook에서 별도로 확인해야 합니다.

Linux 배포는 bare host 설치가 아니라 팀이 승인한 CPU 또는 CUDA base
image를 기준으로 합니다. base image가 호환되는 PyTorch와 torchvision을
제공하고 `requirements-runtime.txt`는 플랫폼 공통 패키지만 추가합니다.
CPU와 CUDA base의 digest·아키텍처·PyTorch 조합은 각각 별도로 승인해야
합니다.

Apple Silicon의 자세한 설치, MPS/CPU 실행과 검증 절차는
[docs/MAC_VALIDATION.md](docs/MAC_VALIDATION.md)를 참고하세요.

### 2. 모델 준비

원본 모델은 Git에 포함하지 않습니다. 로컬 기본 경로에 승인된 두 `best.pt`가
있어야 합니다.

```text
artifacts/models/yolo11n_camera_holdout_v1/best.pt
artifacts/models/yolo11n_health_date_camera_holdout_v1/best.pt
```

manifest의 크기와 SHA-256을 검증한 뒤 runtime 경로에 원자적으로 준비합니다.

```bash
make prepare-models
make check-models
```

기존 runtime 파일이 승인본과 다르면 묵시적으로 덮어쓰지 않습니다. 검토 후
명시적으로 교체할 때만 다음 명령을 사용합니다.

```bash
python scripts/prepare_runtime_models.py --overwrite
```

### 3. 설정

```bash
cp .env.example .env
```

애플리케이션은 process 환경변수를 읽으며 `.env`를 자동 로딩하지 않습니다.
로컬 shell에서 필요하면 `.env`를 export한 뒤 실행합니다.

```bash
set -a
source .env
set +a
```

빈 모델 경로는 기존 `artifacts/models/...` 기본값을 사용합니다. runtime
준비본을 사용할 때는 다음처럼 지정합니다.

```text
DETECTOR_MODEL_PATH=runtime/models/detector/best.pt
HEALTH_MODEL_PATH=runtime/models/health/best.pt
```

### 4. 테스트와 실행

```bash
make test
make run
```

Apple Silicon에서 MPS를 명시적으로 선택하려면 다음 명령을 사용합니다.

```bash
make run-mps
```

CPU 기준선은 다음처럼 실행합니다.

```bash
make run-cpu
```

두 target은 각각 `HEALTH_DEVICE=mps make run`과
`HEALTH_DEVICE=cpu make run`의 짧은 진입점입니다.

현재 고정 Ultralytics 버전에서 `HEALTH_DEVICE=auto`는 CUDA가 있으면 CUDA,
없으면 CPU를 선택하며 MPS를 자동 선택하지 않습니다. MPS 사용 가능 여부를
먼저 확인하고 실제 선택 장치를 시작 로그에서도 확인하세요.

직접 실행 명령은 다음과 같습니다.

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

GPU 모델은 process-local singleton이므로 worker마다 모델이 중복 로딩됩니다.
이 프로토타입은 반드시 worker 1개를 사용합니다.

## API

### 건강 체크

`POST /api/v1/mushroom/health-check`

- 요청: `multipart/form-data`
- 파일 필드: `image`
- 지원 형식: JPG, JPEG, PNG, WEBP
- 최대 업로드: 기본 10 MiB
- 업로드 원본은 영구 저장하지 않음

```bash
curl -X POST \
  http://localhost:8000/api/v1/mushroom/health-check \
  -H "accept: application/json" \
  -F "image=@sample.jpg"
```

성공 응답 예:

```json
{
  "analysisType": "MUSHROOM_HEALTH_CHECK_V1",
  "status": "SUCCESS",
  "detectorModel": "mushroom-yolo11n-camera-holdout-v1",
  "healthModel": "mushroom-health-yolo11n-date-camera-holdout-v1",
  "thresholds": {
    "detection": 0.25,
    "minDetectionConfidence": 0.5,
    "healthUncertain": 0.7
  },
  "results": [
    {
      "species": "느타리",
      "speciesClassId": 0,
      "detectedCount": 1,
      "detectionConfidence": 0.97,
      "detectionConfidenceMin": 0.97,
      "healthStatus": "HEALTHY",
      "healthConfidence": 0.99,
      "healthyProbability": 0.99,
      "diseaseSuspectedProbability": 0.01,
      "bbox": [187, 769, 902, 1372],
      "cropBbox": [25, 481, 1064, 1660]
    }
  ],
  "warnings": [
    "AI 분석 참고 결과이며 확정 진단이 아닙니다."
  ]
}
```

주요 응답 상태:

| HTTP | `status` | 설명 |
| ---: | --- | --- |
| 200 | `SUCCESS` | 품종별 분석 결과 반환 |
| 200 | `NO_MUSHROOM_DETECTED` | 탐지 결과 없음, `results=[]` |
| 400 | `INVALID_IMAGE` | 빈 파일 또는 손상 이미지 |
| 413 | `INVALID_IMAGE` | 업로드 크기 초과 |
| 415 | `INVALID_IMAGE` | 형식·MIME 불일치 또는 미지원 |
| 422 | FastAPI validation error | `image` 필드 누락 |
| 500 | `INFERENCE_FAILED` | 내부 상세를 숨긴 추론 실패 |

### 상태 확인

```bash
curl http://localhost:8000/health/live
curl http://localhost:8000/health/ready
```

- `GET /health/live`: 모델을 호출하지 않고 `200 {"status":"UP"}`
- `GET /health/ready`: registry가 `READY`면 200, 아니면 503
- 두 endpoint 모두 모델 추론과 사용자 이미지 처리를 실행하지 않음

OpenAPI UI는 `/docs`, schema는 `/openapi.json`에서 확인할 수 있습니다.
상세 계약은 [docs/API.md](docs/API.md)를 참고하세요.

## 환경변수

| 환경변수 | 기본값 | 설명 |
| --- | --- | --- |
| `DETECTOR_MODEL_PATH` | 로컬 artifacts 경로 | 품종 detector 파일 |
| `HEALTH_MODEL_PATH` | 로컬 artifacts 경로 | 건강 classifier 파일 |
| `HEALTH_DETECTION_CONFIDENCE` | `0.25` | detector가 객체로 인정하는 최소 threshold |
| `HEALTH_MIN_DETECTION_CONFIDENCE` | `0.50` | 건강 분류 실행을 허용하는 품종별 최소 탐지 confidence |
| `HEALTH_UNCERTAIN_THRESHOLD` | `0.70` | 건강 결과를 확정 상태로 반환하는 최소 confidence |
| `HEALTH_PADDING_RATIO` | `0.15` | 원본 폭·높이 기준 crop padding, 허용 0~0.5 |
| `HEALTH_MAX_UPLOAD_BYTES` | `10485760` | 최대 업로드, 허용 1 byte~100 MiB |
| `HEALTH_VERIFY_MODEL_SHA256` | `true` | 시작 시 승인 SHA-256 검증 |
| `HEALTH_DEVICE` | `auto` | Ultralytics device 값. Linux 예: `cpu`, `cuda:0`; Apple Silicon MPS: `mps`. `auto`는 MPS를 자동 선택하지 않음 |

모든 숫자와 boolean 설정은 애플리케이션 시작 시 검증됩니다. 모델 파일
존재 여부, SHA-256과 class mapping도 요청을 받기 전에 확인합니다.

세 confidence 설정은 서로 다른 단계에 적용됩니다.

1. `HEALTH_DETECTION_CONFIDENCE`: bbox 후보를 detector 결과로 채택할지 결정
2. `HEALTH_MIN_DETECTION_CONFIDENCE`: 채택된 품종 그룹의 건강 분류 실행 여부
3. `HEALTH_UNCERTAIN_THRESHOLD`: 실행된 건강 classifier 결과를 확정할지 결정

## 고정 모델

| 역할 | 모델 | 입력 | SHA-256 |
| --- | --- | ---: | --- |
| 품종 탐지 | `mushroom-yolo11n-camera-holdout-v1` | 640 | `8d17eb493f2eeccccd832c56da2f346dfc730e5605c460016386c6bd0be10d32` |
| 건강 분류 | `mushroom-health-yolo11n-date-camera-holdout-v1` | 320 | `720efb30093c2fbaf8866243a60870c7816f3e141c81e0fac015a358edce1f92` |

세부 클래스, 검증 범위와 제한은
[models/model-manifest.json](models/model-manifest.json)에 기록합니다.

## 주요 의존성

| 패키지 | 역할 |
| --- | --- |
| FastAPI | HTTP REST API와 OpenAPI 제공 |
| Uvicorn | ASGI 서버 실행 |
| python-multipart | `multipart/form-data` 업로드 파싱 |
| Pillow | 이미지 형식 검증, EXIF 방향 처리와 crop |
| PyTorch | 모델 실행 runtime |
| torchvision | 검증된 PyTorch/Ultralytics 영상 runtime 구성 |
| Ultralytics | YOLO11n 모델 로딩과 추론 |
| Pydantic | 설정·외부 응답 schema 검증 |
| pytest | 개발·회귀 테스트 |

플랫폼 공통 버전은 `requirements-common.txt`가 기준입니다. Apple Silicon은
`requirements-macos.txt`, Linux container는 `requirements-runtime.txt`를
사용하고 테스트 추가분은 `requirements-dev.txt`를 함께 설치합니다. 전체
`pip freeze`를 사용하지 않습니다.

## Docker 프로토타입

팀의 Spring Dockerfile은 Java 21 서비스용이며 Vision 서비스의
Python/PyTorch/CUDA base image와는 별도입니다. GPU 노드와 내부 base image가
확정되지 않아 현재는 `Dockerfile.template`을 제공합니다.

Docker Desktop for Mac도 Linux VM 안에서 Linux container를 실행하므로
호스트의 Metal/MPS를 Vision container에 제공하지 않습니다. Mac에서 MPS를
검증할 때는 위의 host virtual environment를 사용하고, Mac Docker는 승인된
Linux CPU image의 구조 검증 범위로만 취급합니다. CUDA image의 실제 build와
GPU 검증은 대상 아키텍처의 Linux/NVIDIA runner에서 수행합니다.

먼저 runtime 모델을 준비한 뒤 팀 승인 base image를 명시합니다.

```bash
make prepare-models
make docker-build BASE_IMAGE=<team-approved-python-pytorch-image>
make docker-run BASE_IMAGE=<team-approved-python-pytorch-image>
```

Docker image에는 `app`, 실행에 필요한 predictor, manifest, runtime
requirements와 두 runtime 모델만 포함됩니다. datasets, artifacts, reports,
tests, `.git`, `.venv`, 외부 이미지와 사용자 업로드는 `.dockerignore`가
차단합니다.

컨테이너는 numeric non-root user, worker 1개, 읽기 전용 모델 파일을 전제로
합니다. Cluster의 GPU/NVIDIA 설정은
[docs/CI_CD_HANDOFF.md](docs/CI_CD_HANDOFF.md)의 TODO를 확인하세요.

## 디렉터리 구조

```text
.
├── app/
│   ├── api/                 # versioned API와 service health probes
│   ├── core/                # 환경설정과 process-local model registry
│   ├── schemas/             # 외부 Pydantic DTO
│   ├── services/            # 업로드 검증과 추론 orchestration
│   └── main.py              # FastAPI factory와 lifespan
├── scripts/
│   ├── check_runtime_environment.py
│   ├── predict_mushroom_health.py
│   └── prepare_runtime_models.py
├── tests/                   # fake 기본 테스트와 opt-in 모델 통합 테스트
├── docs/                    # 아키텍처, API, 모델, CI/CD 인계
├── models/model-manifest.json
├── runtime/models/          # Git 제외 runtime best.pt 준비 위치
├── Dockerfile.template
├── requirements-common.txt
├── requirements-macos.txt
├── requirements-runtime.txt
├── requirements-dev.txt
├── Makefile
└── README.md
```

기존 데이터 감사·파일럿 스크립트는 연구 재현을 위해 `scripts/`에 남겨 두되
production Docker image에는 포함하지 않습니다.

## 테스트

기본 테스트는 fake detector/classifier와 작은 합성 파일을 사용하며 실제
모델을 반복 로딩하지 않습니다.

```bash
pytest -q
```

실제 모델 registry 통합 테스트는 명시적으로 활성화할 때만 실행합니다.

```bash
RUN_MODEL_INTEGRATION_TESTS=true \
pytest -q tests/test_model_registry.py -k integration
```

## 검증 범위와 제한

- 품종 모델 camera holdout mAP50-95: 0.899
- 건강 date+camera holdout Validation 1,000장 Top-1: 0.999
- 내부 end-to-end 100장: Accuracy/Macro F1 1.000
- AIHub 미사용 느타리 10장 smoke: 탐지 10/10, 건강 상태 10/10
- 외부 사진에서 낮은 confidence 품종 오분류 사례가 있어 0.50 안전 gate 적용

높은 내부 성능은 외부 일반화 성능을 보장하지 않습니다. 서비스 응답에는
확정 진단 표현을 사용하지 않으며, 낮은 confidence는 `UNCERTAIN`으로
보수적으로 처리합니다.

## 모델 관리와 MinIO 전환

현재 프로토타입은 Git 밖의 승인 모델을 SHA-256으로 검증해 Docker image에
포함합니다. 다음 단계에서는 image에서 모델 binary를 제거하고 다음 흐름으로
전환할 수 있습니다.

1. manifest에 MinIO object key·version·SHA-256 기록
2. initContainer가 임시 경로로 모델 다운로드
3. 크기와 SHA-256 검증 후 공유 read-only volume에 atomic rename
4. Vision container는 동일한 `DETECTOR_MODEL_PATH` /
   `HEALTH_MODEL_PATH`로 mount 경로 사용
5. 이전 immutable object version으로 배포 설정을 되돌려 rollback

애플리케이션의 경로 주입과 registry 검증은 그대로 재사용하므로 추론 로직을
다시 구현하지 않습니다. 자세한 절차는
[docs/MODEL_MANAGEMENT.md](docs/MODEL_MANAGEMENT.md)를 참고하세요.

## 개발 시 주의사항

- `best.pt`, artifacts, 데이터셋, ZIP, 외부 이미지를 Git에 추가하지 않음
- `git add .` 대신 검토한 source/config/docs 파일만 명시적으로 stage
- 모델 재학습·변환·양자화는 이 서비스 정리 작업의 범위가 아님
- route에 YOLO 로직을 중복하지 않고 service와 predictor를 재사용
- 모델 경로는 `HealthAPISettings`에서 결정하고 registry loader까지 전달
- 업로드 파일명으로 저장 경로를 만들지 않으며 이미지를 영구 저장하지 않음
- 외부 응답에 stack trace, 로컬 경로 또는 모델 경로를 노출하지 않음
- worker 수를 늘리기 전에 GPU 메모리와 model singleton 전략을 재검토

추가 설계 문서:

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/API.md](docs/API.md)
- [docs/MODEL_MANAGEMENT.md](docs/MODEL_MANAGEMENT.md)
- [docs/CI_CD_HANDOFF.md](docs/CI_CD_HANDOFF.md)
- [docs/MAC_VALIDATION.md](docs/MAC_VALIDATION.md)
