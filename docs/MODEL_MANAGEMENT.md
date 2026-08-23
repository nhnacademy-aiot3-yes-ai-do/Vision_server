# Model Management

## 결정된 방식

Vision_server가 두 모델의 소유권과 실행 책임을 가집니다. 현재 모델은
private Vision_server Git 저장소에서 코드와 함께 관리하고, Docker image에
포함해 private GitHub Container Registry(GHCR)로 배포합니다.

```text
private Git repository
├─ runtime/models/detector/best.pt
├─ runtime/models/health/best.pt
└─ models/model-manifest.json
        ↓ make verify-models
size + SHA-256 검증
        ↓
Vision 코드와 두 모델을 하나의 Docker image로 build
        ↓
private GHCR package
        ↓
배포 환경이 승인된 image digest로 pull
```

MinIO는 사용자 이미지 저장에만 사용합니다. 모델용 bucket이나 credential을
두지 않으며, Vision_server는 image에 포함된 모델만 사용합니다.

## 저장 위치

| 역할 | private Git | Container |
| --- | --- | --- |
| 품종 탐지 | `runtime/models/detector/best.pt` | `/opt/mushroom-vision/runtime/models/detector/best.pt` |
| 건강 분류 | `runtime/models/health/best.pt` | `/opt/mushroom-vision/runtime/models/health/best.pt` |
| 모델 계약 | `models/model-manifest.json` | 애플리케이션 내부 manifest 경로 |

`best.pt`는 Git에서 직접 검토하는 배포 입력입니다.
로컬 `artifacts`에서 복사하거나 동일 경로에 덮어쓰는 준비 단계가 없습니다.
private 저장소를 clone한 개발자와 CI는 같은 모델 파일을 받습니다.

모델 또는 Docker image를 public 저장소·public package·public release에
게시하지 않습니다. AIHub 데이터 이용조건, 모델 재배포 조건과
Ultralytics/PyTorch 관련 라이선스는 공개 범위를 바꾸기 전에 별도로
확인해야 합니다.

## Manifest 계약

`models/model-manifest.json`은 두 binary의 승인된 정체성과 predictor
실행 상수를 기록하는 단일 기준입니다. predictor의 모델 경로·이름·크기·
SHA-256·입력 크기·class mapping은 import 시 이 파일을 검증한 결과에서
파생됩니다. 각 모델에는 최소한 다음 정보가 있어야 합니다.

- 역할과 안정적인 model name/version
- repository 상대 경로
- byte 크기와 전체 SHA-256
- framework, task와 입력 크기
- 외부 class mapping과 실제 weight의 raw class mapping
- 검증 범위와 알려진 제한

Container의 절대 모델 경로는 manifest에 중복 저장하지 않고 `Dockerfile`의
복사 위치와 `DETECTOR_MODEL_PATH`·`HEALTH_MODEL_PATH` 환경변수로
고정합니다.

host 절대경로, 사용자명, credential, presigned URL과 임시 다운로드 주소는
manifest에 넣지 않습니다. 모델 성능과 적용 한계는
[MODEL_CARD.md](MODEL_CARD.md)에 요약합니다.

## 검증 지점

### 개발자와 CI

```bash
make verify-models
```

이 target은 `scripts/verify_runtime_models.py`를 통해 두 파일의 존재 여부,
크기와 SHA-256을 manifest와 비교합니다. 검증은 파일을 복사·수정하지 않는
read-only 작업입니다.

검증 실패 시 다음 행동을 하지 않습니다.

- SHA 검증 비활성화
- manifest의 hash만 임의 수정
- 다른 `best.pt`를 같은 이름으로 덮어쓰기
- Docker build 또는 배포 계속

### Docker build

`make docker-build`는 모델 검증이 통과한 경우에만 실행되어야 합니다.
Dockerfile은 두 model file의 repository 상대 경로를 container 안에서도
유지하고, non-root 애플리케이션이 수정할 수 없게 read-only로 둡니다.

### 애플리케이션 시작

predictor import 시 manifest schema·역할·고정 repository 경로·필드 타입과
class mapping을 먼저 검증합니다. FastAPI lifespan에서는 `ModelRegistry`가
다시 SHA-256과 실제 weight의 class mapping을 검증하고 두 모델을 한 번
로드합니다. 모든 검증이 끝나야
`GET /health/ready`가 200을 반환합니다.

즉, Git/CI의 사전 검증과 애플리케이션의 시작 검증은 서로 대체하지 않는 두
안전장치입니다.

## 모델 변경 절차

모델 교체는 일반 코드 변경과 같은 PR 검토를 받습니다.

1. 새 모델의 provenance, 학습·검증 split, class 순서, 성능과 제한을
   확인합니다.
2. 해당 역할의 `runtime/models/.../best.pt`를 새 파일로 교체합니다.
3. `models/model-manifest.json`의 model version, size, SHA-256과 관련
   정보를 함께 갱신합니다.
4. predictor가 새 manifest 계약에서 상수를 정상 파생하는지 확인합니다.
5. `make verify-models`와 `make test`를 실행합니다.
6. 승인된 환경에서 실제 model registry/API smoke test를 실행합니다.
7. PR 승인 후 새 Docker image를 build해 private GHCR에 push합니다.
8. staging 검증을 통과한 image digest만 production에 배포합니다.

모델 파일과 manifest는 하나의 commit/PR에서 함께 변경해야 합니다. 둘 중
하나만 바뀐 commit은 배포하지 않습니다.

## image version과 배포

사람이 읽는 tag에는 코드와 model bundle 버전을 함께 표현할 수 있습니다.

```text
ghcr.io/<organization>/vision-server:0.1.0-model-v1
```

`latest`만으로 배포 대상을 지정하지 않습니다. 실제 Kubernetes 배포는
변경 불가능한 digest를 사용합니다.

```text
ghcr.io/<organization>/vision-server@sha256:<image-digest>
```

release마다 다음을 함께 기록합니다.

- Git commit SHA
- detector와 health model version
- detector와 health model SHA-256
- 사람이 읽는 image tag
- 실제 배포한 image digest
- 테스트 결과와 승인자

## rollback

실행 중인 Pod의 모델 파일을 바꾸지 않습니다. 문제가 생기면 이전에 검증된
Docker image digest로 Deployment를 되돌립니다.

```text
현재 digest에서 문제 확인
→ 이전 승인 digest로 Deployment 변경
→ 새 Pod readiness 확인
→ 트래픽 전환
```

image가 코드와 두 모델을 모두 포함하므로 이전 digest 하나만 지정하면 같은
코드·의존성·모델 조합을 재현할 수 있습니다.

## 접근 권한과 Secret

- Vision_server Git repository를 private으로 유지합니다.
- GHCR package도 private으로 유지합니다.
- CI push 권한과 Kubernetes pull 권한은 최소 범위로 분리합니다.
- GHCR token이나 registry credential을 Git과 `.env.example`에 넣지
  않습니다.
- CI 로그와 artifact에 token, 로컬 경로와 모델 binary를 별도로 노출하지
  않습니다.
- 배포 환경에는 image pull secret 또는 승인된 workload identity만
  제공합니다.

Git history에는 과거 모델 binary도 남습니다. 모델 배포 권한이 없는
사용자에게 저장소 read 권한을 주지 않습니다.

## 향후 재검토 기준

다음 상황이 오면 모델 전용 OCI artifact나 승인된 모델 registry 분리를
재검토할 수 있습니다.

- 모델 수·용량이 크게 증가함
- 코드와 모델의 release 주기가 완전히 달라짐
- 저장소 clone 비용이 팀 개발을 방해함
- 모델별 접근 권한 또는 retention 정책이 필요함

그전까지는 모델 두 개와 Vision 코드를 하나의 검증된 Docker image로
배포하는 현재 방식이 기준입니다. 구조를 바꾸더라도 image digest, manifest
무결성, private 접근 권한과 rollback 가능성은 유지해야 합니다.
