# Vision Server 코드 흐름 및 리팩터링 기준

이 문서는 리팩터링 전 기준 커밋 `1c704b6`의 실행 흐름을 설명합니다.
Python에 익숙하지 않은 개발자가 Java/Spring 코드와 비교하며 읽을 수 있도록 작성했습니다.

## 1. 서비스의 책임

Vision Server는 이미지를 저장하거나 사용자 권한을 관리하지 않습니다.

입력:

- `multipart/form-data` 형식의 이미지 한 장
- 필드명 `image`
- 지원 형식 JPG/JPEG, PNG, WEBP

출력:

- 탐지된 버섯 품종과 개수
- 탐지 영역과 신뢰도
- `HEALTHY`, `DISEASE_SUSPECTED`, `UNCERTAIN` 중 하나인 건강 관찰 상태
- 분석에 사용된 모델과 임계값

Vision Server가 하지 않는 일:

- 업로드 이미지 영구 저장
- MinIO 직접 접근
- 사용자 및 재배 권한 검증
- 병명 확정 또는 치료법 결정
- 성장 단계나 수확 시기 판정
- 분석 결과 DB 저장

이미지 조회와 결과 저장은 AI Server의 책임입니다.

## 2. Java/Spring 코드와 비교

| Python 코드 | Java/Spring에서 비슷한 역할 |
|---|---|
| `app/main.py` | 애플리케이션 시작 클래스와 Bean 구성 |
| `app/api/health.py` | Controller |
| `app/schemas/health.py` | Request/Response DTO와 검증 |
| `app/services/mushroom_health_service.py` | Application Service |
| `app/core/config.py` | `@ConfigurationProperties` |
| `app/core/model_registry.py` | 모델 Lifecycle Manager와 process-local cache |
| `scripts/predict_mushroom_health.py` | Domain Service, 모델 Adapter, CLI가 섞여 있는 현재 리팩터링 대상 |
| `scripts/verify_runtime_models.py` | 모델 artifact 검증기 |

Python의 `dataclass(frozen=True)`는 변경 불가능한 Java record와 비슷합니다.
`Protocol`은 Java interface와 비슷하고, Pydantic `BaseModel`은 검증 기능이 있는 DTO와 비슷합니다.

## 3. 전체 실행 흐름

```text
서버 시작
  → 설정 읽기
  → manifest 검증
  → detector와 health classifier를 한 번 로드
  → 운영 기본 설정에서는 모델 SHA 확인
  → class mapping·fingerprint 확인
  → HTTP 요청 수신 가능 상태

이미지 요청
  → 확장자와 MIME type 검증
  → 1 MiB 단위로 읽으며 크기 제한 확인
  → 실제 이미지 형식·손상·픽셀 수 검증
  → EXIF 방향 보정 및 RGB 변환
  → 버섯 품종 탐지
  → 같은 품종의 bbox를 하나로 묶기
  → 그룹의 최저 탐지 신뢰도 확인
    → 기준 미만이면 UNCERTAIN
    → 기준 이상이면 품종별 union crop 생성·건강 상태 분류
  → 공개 DTO 재검증
  → camelCase JSON 응답
```

## 4. 서버 시작과 모델 Lifecycle

시작점은 `app/main.py`의 `create_app()`입니다.

1. `HealthAPISettings`가 환경 변수를 타입이 있는 값으로 변환합니다.
2. `ModelRegistry`를 생성합니다.
3. FastAPI lifespan 시작 시 전용 thread pool을 생성합니다.
4. HTTP 요청을 받기 전에 `ModelRegistry.load()`를 실행합니다.
5. 두 모델이 모두 검증된 경우에만 `MushroomHealthService`를 공개합니다.

`ModelRegistry.load()`는 다음 순서를 지킵니다.

```text
모델 파일 fingerprint 기록
  → detector와 classifier 로드
  → 모델 class mapping 검증
  → 로드 후 fingerprint 재검사
  → 모두 성공하면 READY
```

모델은 요청마다 다시 로드하지 않습니다. 한 프로세스에서 한 번 로드하고 모든 요청이 같은 객체를 사용합니다.

manifest, class mapping 또는 모델 파일 fingerprint가 잘못되면 부분적으로 시작하지 않고 애플리케이션 시작을 실패시킵니다. SHA 비교는 `HEALTH_VERIFY_MODEL_SHA256=true`인 운영 기본 설정에서 적용되며, false이면 승인 SHA 비교만 생략합니다.

관련 파일:

- `app/main.py`
- `app/core/config.py`
- `app/core/model_registry.py`
- `models/model-manifest.json`
- `scripts/verify_runtime_models.py`

## 5. HTTP 요청 접수

계약:

```http
POST /api/v1/internal/mushrooms/health-check
Content-Type: multipart/form-data

image=<이미지 파일>
```

`app/api/health.py`의 `health_check()`는 Controller 역할만 수행합니다.

1. lifespan에서 만든 `MushroomHealthService`를 가져옵니다.
2. `analyze_upload(image)`를 호출합니다.
3. 내부 분석 결과를 `HealthCheckResponse`로 변환합니다.
4. 예상 가능한 서비스 오류를 안전한 공개 응답으로 변환합니다.

Controller는 YOLO 모델이나 MinIO를 직접 알지 않습니다.

## 6. 이미지 검증

`app/services/mushroom_health_service.py`가 다음 검증을 순서대로 수행합니다.

### 6.1 확장자와 MIME type

`_upload_rule()`은 확장자와 MIME type이 주장하는 기대 형식을 다음 조합으로 제한합니다.

| 기대 형식 | 허용 확장자 | 허용 MIME type |
|---|---|---|
| JPEG | `.jpg`, `.jpeg` | `image/jpeg` |
| PNG | `.png` | `image/png` |
| WEBP | `.webp` | `image/webp` |

PNG 확장자를 JPEG MIME type과 함께 보내는 것처럼 메타데이터가 서로 다른 형식을 주장하면 거부합니다. 실제 이미지 bytes의 형식은 이후 `decode_image_bytes()`에서 별도로 확인합니다.

### 6.2 파일 크기

`_read_upload_limited()`는 1 MiB씩 읽으며 최대 `설정 상한 + 1 byte`까지만 확인합니다. 정상 파일은 추론을 위해 전체 bytes를 메모리에 보관합니다.

- 기본 최대 크기: 10 MiB
- 설정 가능한 절대 상한: 100 MiB
- 빈 파일: HTTP 400
- 크기 초과: HTTP 413

정상과 실패 경로 모두 업로드 파일 핸들을 닫습니다.

### 6.3 실제 이미지 검증

`decode_image_bytes()`는 Pillow로 다음 항목을 확인합니다.

- 선언된 형식과 실제 이미지 형식 일치
- 애니메이션 이미지가 아닌 단일 프레임
- 폭과 높이가 양수
- 픽셀 수 안전 상한
- 손상된 이미지 여부
- 압축 폭탄 이미지 여부

검증 후 EXIF 방향을 반영하고 RGB `PIL.Image`로 변환합니다.

## 7. 품종 탐지

`MushroomHealthService._predict_sync()`가 준비된 모델 쌍을 가져와 `predict_health()`를 호출합니다.

`UltralyticsDetector`는 YOLO detector 결과를 다음 값 객체로 변환합니다.

```python
Detection(
    class_id=0,
    species="느타리",
    bbox=(x_min, y_min, x_max, y_max),
    confidence=0.91,
)
```

탐지 품종은 다섯 종류입니다.

| model class id | 표시명 | 서비스 간 코드 |
|---:|---|---|
| 0 | 느타리 | `OYSTER` |
| 1 | 양송이 | `BUTTON` |
| 2 | 큰느타리 | `KING_OYSTER` |
| 3 | 팽이 | `ENOKI` |
| 4 | 표고 | `SHIITAKE` |

`speciesClassId`는 모델 내부 번호이고 `speciesCode`는 서비스 간 업무 식별자입니다. AI Server와 DB 연동에서는 `speciesCode`를 사용해야 합니다.

버섯이 하나도 탐지되지 않으면 장애가 아니라 HTTP 200과 `NO_MUSHROOM_DETECTED`를 반환합니다.

## 8. 품종 grouping과 crop

같은 품종의 버섯이 여러 개 탐지되더라도 건강 classifier를 각 객체마다 호출하지 않습니다.

```text
느타리 bbox 3개
  → 느타리 그룹 1개
  → bbox 3개를 포함하는 union bbox
  → padding을 적용한 crop 1개
  → classifier 1회
```

다른 품종은 각각 별도의 그룹과 결과를 만듭니다.

좌표 형식:

```text
[xMin, yMin, xMax, yMax]
```

현재 padding은 bbox 크기가 아니라 원본 이미지 폭과 높이에 `HEALTH_PADDING_RATIO`를 곱해 계산합니다. 기본값은 15%이며 설정 가능한 범위는 0~50%입니다. 리팩터링 중 이 산식을 임의로 바꾸면 안 됩니다.

## 9. 건강 상태 판정

서로 다른 역할의 임계값 세 개를 사용합니다.

| 설정 | 기본값 | 역할 |
|---|---:|---|
| detection | 0.25 | detector 결과를 채택할 최소 신뢰도 |
| min detection | 0.50 | 건강 classifier를 실행할 최소 탐지 신뢰도 |
| health uncertain | 0.70 | 건강 상태를 확정할 최소 분류 신뢰도 |

### 낮은 탐지 신뢰도

기본 설정에서는 같은 품종 그룹에서 가장 낮은 detection confidence가 0.50 미만이면 classifier를 실행하지 않습니다. 실제 경계는 설정된 `HEALTH_MIN_DETECTION_CONFIDENCE`입니다.

```json
{
  "healthStatus": "UNCERTAIN",
  "healthConfidence": null,
  "healthyProbability": null,
  "diseaseSuspectedProbability": null
}
```

### 분류 결과의 신뢰도가 낮은 경우

classifier는 실행했지만 가장 높은 확률이 설정된 `HEALTH_UNCERTAIN_THRESHOLD` 미만이면 `UNCERTAIN`입니다. 기본값은 0.70이며, 이 경우 두 확률은 숫자로 반환됩니다.

```text
HEALTHY 0.60 / DISEASE_SUSPECTED 0.40 → UNCERTAIN
HEALTHY 0.70 / DISEASE_SUSPECTED 0.30 → HEALTHY
HEALTHY 0.20 / DISEASE_SUSPECTED 0.80 → DISEASE_SUSPECTED
```

기본 설정에서는 정확히 0.70인 경계값부터 확정 상태입니다.

`UNCERTAIN`을 `HEALTHY`로 변환하거나 병해 없음으로 집계하면 안 됩니다.

## 10. 공개 응답 변환

추론 파이프라인은 내부에서 snake_case `dict`를 사용합니다.

`app/schemas/health.py`의 `HealthCheckResponse.from_internal()`이 다음 작업을 수행합니다.

1. 내부 경로나 민감한 값이 없는지 검사합니다.
2. 내부 필드를 공개 필드로 명시적으로 매핑합니다.
3. Pydantic으로 값 범위와 필수 필드를 재검증합니다.
4. camelCase JSON으로 직렬화합니다.

대표적인 유효 응답:

```json
{
  "analysisType": "MUSHROOM_HEALTH_CHECK_V1",
  "status": "SUCCESS",
  "detectorModel": "mushroom-species-detector",
  "healthModel": "mushroom-health-classifier",
  "thresholds": {
    "detection": 0.25,
    "minDetectionConfidence": 0.50,
    "healthUncertain": 0.70
  },
  "results": [
    {
      "species": "느타리",
      "speciesCode": "OYSTER",
      "speciesClassId": 0,
      "detectedCount": 1,
      "detectionConfidence": 0.93,
      "detectionConfidenceMin": 0.93,
      "healthStatus": "HEALTHY",
      "healthConfidence": 0.91,
      "healthyProbability": 0.91,
      "diseaseSuspectedProbability": 0.09,
      "bbox": [10, 12, 90, 76],
      "cropBbox": [0, 0, 100, 88]
    }
  ],
  "warnings": []
}
```

## 11. 오류와 업무 상태

| 상황 | HTTP 또는 결과 |
|---|---|
| 모델 manifest 오류 | 애플리케이션 시작 실패 |
| 모델 누락, class mapping 또는 fingerprint 불일치 | 애플리케이션 시작 실패 |
| SHA 불일치 | SHA 검증을 켠 운영 기본 설정에서 애플리케이션 시작 실패 |
| multipart `image` 누락 | HTTP 422 |
| 서비스 미준비 | HTTP 503 |
| 확장자·MIME·실제 형식 불일치 | HTTP 415, `INVALID_IMAGE` |
| 파일 크기 초과 | HTTP 413, `INVALID_IMAGE` |
| 빈 파일·손상 이미지·픽셀 제한 | HTTP 400, `INVALID_IMAGE` |
| 동시 분석 요청 상한 초과 | HTTP 429, `SERVICE_BUSY` |
| 버섯 미탐지 | HTTP 200, `NO_MUSHROOM_DETECTED` |
| 낮은 신뢰도 | HTTP 200, 결과의 `UNCERTAIN` |
| 예상하지 못한 추론 오류 | HTTP 500, `INFERENCE_FAILED` |

## 12. 동시성 구조와 첫 개선 대상

YOLO wrapper가 동시에 안전하게 호출된다고 가정하지 않기 때문에 `inference_lock`으로 추론을 한 번에 하나만 실행합니다.

기준 커밋 `1c704b6`의 요청은 다음 순서였습니다.

```text
확장자·MIME type 검증
  → 파일 읽기
  → 이미지 디코딩
  → inference_lock 대기
  → 모델 추론
```

동시 요청이 많으면 여러 요청의 파일 bytes와 PIL 이미지가 메모리에서 lock을 기다릴 수 있습니다. thread pool의 작업 대기열도 별도 상한이 없습니다.

첫 기능 리팩터링에서는 파일을 읽기 전에 제한된 admission permit을 획득하도록 변경했습니다.

- 허용된 요청만 읽기와 디코딩 수행
- 초과 요청은 빠르게 HTTP 429 `SERVICE_BUSY`로 거부
- 현재 기본 상한은 프로세스별 1이며 초과 요청을 기다리게 하지 않음
- 추론 직렬화 계약은 유지

## 13. 리팩터링에서 보존할 계약

다음 동작은 테스트와 팀 합의 없이 바꾸지 않습니다.

1. `POST /api/v1/internal/mushrooms/health-check`
2. multipart 필드명 `image`
3. 한 요청이 완료된 분석 JSON 하나를 받는 동기 계약
4. startup 시 모델 한 번 로드
5. 운영 기본 설정의 모델 SHA 검증과 class mapping·fingerprint 검증
6. worker 1개와 process 내부 추론 직렬화
7. 같은 품종 여러 bbox를 union crop 하나로 분류
8. 다른 품종은 별도 결과 생성
9. 원본 이미지 폭·높이에 설정된 비율을 적용하는 padding 산식과 기본값 15%
10. 0.25, 0.50, 0.70 세 기본 임계값의 서로 다른 의미
11. 두 종류 `UNCERTAIN`의 확률 null 여부
12. `speciesCode`와 `speciesClassId` 구분
13. camelCase 공개 응답
14. 이미지 미저장
15. 로컬 경로와 내부 예외 비노출

## 14. Data Generator 방식의 리팩터링 순서

각 단계는 한 가지 책임만 이동하고 전체 회귀 테스트를 통과해야 완료입니다.

1. 과부하 거부 동작을 테스트로 먼저 고정 — 완료
2. `app/services/inference_gateway.py` 추가 — 완료
3. 기존 서비스가 gateway를 사용하도록 변경 — 완료
4. `app/domain/health.py`에 상태와 immutable 결과 모델 추가
5. 공개 schema가 typed domain 결과만 변환하도록 변경
6. inference `Protocol` 분리
7. 순수 추론 pipeline 분리
8. Ultralytics adapter 분리
9. 업로드 decoder 분리
10. 기존 1,000줄 script를 얇은 CLI로 축소
11. Ruff와 Pyright 정적 검사 추가
12. 실제 모델 smoke test와 Docker 검증

구조를 옮기는 동안 threshold나 응답 의미를 함께 바꾸지 않습니다.

## 15. 로컬 확인 명령

가상환경이 준비된 경우:

```bash
make test PYTHON=.venv/bin/python
make verify-models PYTHON=.venv/bin/python
make doctor-mac PYTHON=.venv/bin/python
make run-cpu PYTHON=.venv/bin/python
```

기본 테스트는 실제 YOLO 모델을 매번 로드하지 않습니다. 모델 artifact 검증과 실제 모델 로드 smoke test는 별도 단계로 실행합니다.

```bash
RUN_MODEL_INTEGRATION_TESTS=true \
HEALTH_DEVICE=cpu \
.venv/bin/python -m pytest -q tests/test_model_registry.py -k integration
```

## 16. Python 용어 정리

| 용어 | 의미 |
|---|---|
| `async def` | coroutine을 정의하며, 실행 중 `await`한 대상이 기다려야 할 때 event loop에 제어권을 양보할 수 있는 함수 |
| event loop | 여러 비동기 요청의 실행을 조정하는 루프 |
| executor | 동기 파일 처리와 모델 추론을 별도 thread에서 실행하는 도구 |
| `asyncio.Lock` | 한 프로세스에서 추론을 한 번에 하나만 실행하는 잠금 |
| Pydantic | DTO의 필수값과 타입·범위를 검사하는 라이브러리 |
| Pillow/PIL | 이미지 파일을 읽고 검증·변환하는 라이브러리 |
| Ultralytics | YOLO 모델을 로드하고 추론하는 라이브러리 |
| `Protocol` | 구현체가 따라야 할 메서드 모양을 표현하는 Python interface |
| fingerprint | 파일 크기·수정정보·SHA를 조합한 모델 파일 식별 정보 |
