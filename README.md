# Mushroom Vision Service

버섯 사진에서 품종을 탐지하고 품종별 건강 상태 후보를 반환하는
Python 3.12 / FastAPI 내부 서비스입니다.

이 서비스가 반환하는 결과는 AI 참고 정보이며 확정 진단이 아닙니다.
모델의 검증 범위와 한계는 [모델 카드](docs/MODEL_CARD.md)를 확인하세요.

## 한눈에 보는 역할

```text
MinIO의 사용자 이미지
        │ Spring AI-Server가 읽음
        ▼
Spring AI-Server
        │ OpenFeign multipart(image)
        ▼
Vision_server
        │ 품종 탐지 → 품종별 crop → 건강 분류
        ▼
구조화된 JSON
        │ 같은 HTTP 요청의 응답
        ▼
Spring AI-Server가 센서·RAG 정보와 결합해 요약·가공
```

Vision_server는 MinIO에 접속하지 않습니다. Spring AI-Server가 MinIO에서
이미지를 읽어 실제 이미지 바이트를 `multipart/form-data`의 `image` 필드로
보냅니다. MinIO는 사용자 이미지 저장소일 뿐, 모델 저장·배포 경로가
아닙니다.

OpenFeign 호출은 동기 요청·응답입니다. `POST` 요청이 FastAPI route를
실행시키고, Vision_server는 두 모델의 분석과 JSON 생성을 모두 마친 뒤에만
응답합니다. OpenFeign 메서드가 DTO를 반환한 시점이 곧 분석 완료 시점이며,
그다음 AI-Server 코드에서 요약과 가공을 시작합니다.

## 코드 읽는 순서

1. `app/main.py`: FastAPI 시작·종료와 모델 최초 로드
2. `app/api/health.py`: OpenFeign 요청이 도착하는 REST endpoint
3. `app/services/inference_gateway.py`: 동시 분석 요청 admission 제한
4. `app/services/mushroom_health_service.py`: 업로드 검증과 추론 실행 조정
5. `scripts/predict_mushroom_health.py`: 품종 탐지·crop·건강 분류
6. `app/schemas/health.py`: AI-Server로 보내는 camelCase 응답 계약
7. `app/core/model_registry.py`: 두 모델의 SHA-256·클래스 검증과 준비 상태
8. `app/api/status.py`: liveness/readiness probe

요청 처리 순서는 다음과 같습니다.

```text
multipart 이미지 수신
→ 확장자·MIME·크기·손상 여부 검증
→ 동시 분석 permit 획득, 용량 초과 시 HTTP 429
→ 품종 탐지
→ 같은 품종 bbox를 union하고 설정된 비율로 padding crop 생성
→ 건강 분류
→ confidence 안전 규칙 적용
→ camelCase JSON 응답
```

모델은 요청마다 다시 로드하지 않습니다. 애플리케이션 시작 시 두 모델을 한
번 검증·로드하고, worker 1개 안에서 재사용합니다.

## 최소 저장소 구조

```text
.
├── app/
│   ├── api/                         # 분석 API와 live/ready probe
│   ├── core/                        # 설정과 모델 registry
│   ├── schemas/                     # 외부 요청·응답 계약
│   ├── services/                    # 업로드 검증과 추론 orchestration
│   └── main.py                      # FastAPI 애플리케이션
├── scripts/
│   ├── predict_mushroom_health.py   # 실제 추론
│   ├── verify_runtime_models.py     # manifest 기반 모델 검증
│   └── check_runtime_environment.py # Mac 개발 환경 확인
├── runtime/models/
│   ├── detector/best.pt             # 품종 탐지 모델
│   └── health/best.pt               # 건강 분류 모델
├── models/model-manifest.json       # 모델 크기·SHA-256·클래스 계약
├── tests/                            # 서비스·API·모델 계약 회귀 테스트
├── docs/                             # 운영 문서와 모델 카드
├── Dockerfile
├── Makefile
└── requirements-*.txt
```

학습 데이터 준비, 데이터 감사, 모델 평가 스크립트와 생성 보고서는 이
서비스 저장소의 실행 범위에 포함하지 않습니다.

## 로컬 실행

### 1. 의존성 설치

Apple Silicon Mac:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
make install-mac
make doctor-mac
```

플랫폼별 의존성의 역할은 다음과 같습니다.

| 파일 | 용도 |
| --- | --- |
| `requirements-common.txt` | FastAPI, Pillow 등 torch 없는 API·CI 공통 패키지 |
| `requirements-macos.txt` | Ultralytics와 Apple Silicon용 PyTorch/torchvision |
| `requirements-runtime.txt` | Linux CPU image의 API·Ultralytics 의존성 |
| `requirements-dev.txt` | pytest 등 개발·테스트 도구 |

Mac의 CPU/MPS 검증은 [Mac 검증 가이드](docs/MAC_VALIDATION.md)를
참고하세요.

### 2. 모델 검증

두 `best.pt`는 공개 팀 Vision_server Git 저장소의 `runtime/models`에서
코드와 함께 버전 관리합니다.

```bash
make verify-models
```

이 명령은 두 파일의 크기와 SHA-256을
`models/model-manifest.json`과 비교합니다. 불일치하면 실행·Docker build를
계속하지 않습니다.

### 3. 환경변수와 실행

```bash
cp .env.example .env
set -a
source .env
set +a

make run
```

애플리케이션은 `.env`를 자동으로 읽지 않으므로 필요한 경우 위처럼 shell에
export합니다. 직접 실행할 때도 worker는 반드시 1개입니다.

```bash
python -m uvicorn app.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 1
```

모델이 프로세스마다 별도로 로드되고 하나의 worker 내부 추론도 안전하게
직렬화되기 때문에, worker 수를 임의로 늘리지 않습니다.

### 4. 준비 상태와 분석 확인

```bash
curl http://localhost:8000/health/live
curl http://localhost:8000/health/ready

curl --fail-with-body \
  --request POST \
  --header "accept: application/json" \
  --form "image=@sample.jpg" \
  http://localhost:8000/api/v1/internal/mushrooms/health-check
```

- `/health/live`: FastAPI process가 살아 있으면 200
- `/health/ready`: 두 모델이 검증·로드돼 요청 가능하면 200
- `/api/v1/internal/mushrooms/health-check`: JPG/JPEG, PNG, WEBP 이미지 분석
- multipart 필드명: `image`
- 기본 최대 업로드: 10 MiB

상세 요청·응답은 [API 계약](docs/API.md)을 확인하세요.

## 테스트

```bash
make test
```

기본 테스트는 fake model을 사용하므로 실제 모델을 반복 로드하지 않습니다.
승인된 두 모델을 직접 로드하는 통합 테스트는 명시적으로 실행합니다.

```bash
RUN_MODEL_INTEGRATION_TESTS=true \
python -m pytest -q tests/test_model_registry.py -k integration
```

## 모델과 Docker 배포

모델 배포 흐름은 다음 하나로 고정합니다.

```text
public team Vision_server Git
  └─ runtime/models/*/best.pt
        ↓ make verify-models
models/model-manifest.json과 무결성 확인
        ↓
Vision 코드와 모델을 하나의 Docker image로 build
        ↓
team GitHub Container Registry(GHCR)에 push
        ↓
Kubernetes가 전체 Git commit SHA tag의 image를 pull
```

실행 중 MinIO나 외부 저장소에서 모델을 내려받지 않습니다. 모델을 바꾸려면
manifest와 모델 파일을 함께 검토하고 새 이미지를 발행합니다. 배포와
rollback은 중앙 배포가 기록한 이전 Git commit SHA image 기준으로 수행합니다.

로컬 image build 예:

```bash
make verify-models
make docker-build IMAGE_NAME=vision-server:cpu-smoke
```

저장소가 public이므로 소스와 두 모델 가중치는 누구나 내려받을 수 있습니다.
AIHub 데이터 출처 표기와 모델·런타임 라이선스 고지는 배포 전 유지합니다.
GHCR 공개 범위는 Kubernetes pull 정책과 함께 별도로 확정합니다. 자세한 정책은
[모델 관리](docs/MODEL_MANAGEMENT.md)와
[CI/CD 인계](docs/CI_CD_HANDOFF.md)를 참고하세요.

## 주요 설정

| 환경변수 | 기본값 | 설명 |
| --- | --- | --- |
| `DETECTOR_MODEL_PATH` | `runtime/models/detector/best.pt` | 품종 탐지 모델 |
| `HEALTH_MODEL_PATH` | `runtime/models/health/best.pt` | 건강 분류 모델 |
| `HEALTH_DETECTION_CONFIDENCE` | `0.25` | detector bbox 채택 기준 |
| `HEALTH_MIN_DETECTION_CONFIDENCE` | `0.50` | 건강 분류 실행 허용 기준 |
| `HEALTH_UNCERTAIN_THRESHOLD` | `0.70` | 건강 상태 확정 기준 |
| `HEALTH_PADDING_RATIO` | `0.15` | 품종별 crop padding |
| `HEALTH_MAX_UPLOAD_BYTES` | `10485760` | 최대 업로드 크기 |
| `HEALTH_MAX_INFLIGHT_REQUESTS` | `1` | 프로세스별 동시 분석 요청 상한 |
| `HEALTH_VERIFY_MODEL_SHA256` | `true` | 시작 시 모델 SHA 검증 |
| `HEALTH_DEVICE` | `auto` | `cpu`, `mps`, `cuda:0` 등 실행 장치 |

운영에서는 `HEALTH_VERIFY_MODEL_SHA256=true`를 유지합니다.

## 더 읽기

- [API 계약](docs/API.md)
- [서비스 구조](docs/ARCHITECTURE.md)
- [모델 관리](docs/MODEL_MANAGEMENT.md)
- [모델 카드](docs/MODEL_CARD.md)
- [CI/CD 인계](docs/CI_CD_HANDOFF.md)
- [Apple Silicon Mac 검증](docs/MAC_VALIDATION.md)
