# CI/CD Handoff

## 현재 배포 계약

| 항목 | 값 |
| --- | --- |
| 서비스명 | `vision-server` |
| 플랫폼 | Linux `amd64`, CPU |
| container port | `8000` |
| Uvicorn workers | `1` |
| 분석 API | `POST /api/v1/internal/mushrooms/health-check` |
| liveness | `GET /health/live` |
| readiness | `GET /health/ready` |
| image | `ghcr.io/nhnacademy-aiot3-yes-ai-do/vision_server` |
| 배포 tag | 전체 40자리 Git commit SHA |
| 모델 공급 | Git에서 관리하는 두 `best.pt`를 image에 포함 |

Vision은 외부 Ingress를 만들지 않습니다. AI Server가 Kubernetes 내부 DNS인
`http://vision-server`를 통해 multipart `image`를 전송합니다. MinIO는 사용자
사진 저장소이며 모델 배포 경로가 아닙니다.

## 브랜치별 자동화

```text
feature → develop PR
  ├─ PR 리뷰 Discord 알림
  └─ 모델 SHA 검증 + pytest

develop push
  └─ 모델 SHA 검증 + pytest

main push
  └─ 모델 SHA 검증 + pytest
     → Linux amd64 CPU image build
     → GHCR latest + 전체 commit SHA tag push
     → Config repository_dispatch
     → Kubernetes rollout 및 중앙 Discord 결과 알림
```

SonarQube와 커버리지 검사는 Vision 배포 관문에 포함하지 않습니다. 대신
모델 manifest/SHA 검증과 전체 핵심 pytest가 성공해야 image를 게시합니다.

## CPU image

Dockerfile은 다음 runtime을 고정합니다.

- base: Docker Official `python:3.12.14-slim-bookworm`의 immutable digest
- `torch==2.11.0`
- `torchvision==0.26.0`
- `ultralytics==8.4.106`
- PyTorch official CPU wheel index

빌드 중 `pip check`, OpenCV import, PyTorch/torchvision 버전, CUDA 비활성화를
검증합니다. slim Linux에서 OpenCV import에 필요한 최소 GL/XCB runtime도
image에 포함합니다. 모델 파일은 manifest의 size와 SHA-256이 맞아야 합니다.

```bash
make verify-models
make docker-build IMAGE_NAME=vision-server:cpu-smoke
```

로컬 Mac의 Docker가 `amd64` emulation을 사용하면 빌드가 오래 걸릴 수 있습니다.
최종 기준은 GitHub의 Linux amd64 runner에서 생성한 image입니다.

## GitHub Actions 파일

| 파일 | 역할 |
| --- | --- |
| `_reusable-test.yml` | Python 3.12, 모델 검증, pytest 공통 절차 |
| `develop-ci.yml` | develop push 검증 |
| `pr-check.yml` | develop/main 대상 PR 검증 |
| `pr-review-notify.yml` | PR Discord 알림 |
| `deploy.yml` | main image 게시와 Config 배포 요청 |

배포 요청 JSON은 `.github/scripts/config-deployment.sh`와
`dispatch-payload.jq`가 만듭니다. Sonar 관련 파일과 설정은 필요하지 않습니다.

## 필요한 GitHub 설정

Vision Server repository 또는 접근이 허용된 Organization 설정:

- Variable `GH_APP_CLIENT_ID`
- Variable `VISION_CENTRAL_DEPLOY_ENABLED`
- Secret `GH_APP_PRIVATE_KEY`
- Secret `DISCORD_PR_NOTIFICATION_HOOK_URL`

`GITHUB_TOKEN`은 Actions가 자동 제공하므로 따로 만들지 않습니다. GHCR push용
PAT, Sonar secret, workspace_log 설정도 필요하지 않습니다.

GitHub App은 `Config` repository에 접근할 수 있어야 합니다. GHCR package가
private이면 Kubernetes에 image pull credential이 필요합니다. 현재 MVP에서는
package를 public으로 두는 구성이 가장 단순합니다.

첫 게시 전에는 `VISION_CENTRAL_DEPLOY_ENABLED`를 만들지 않거나 `false`로 둡니다.
Config의 Vision manifest를 먼저 병합한 뒤 Vision `main` workflow로 image만 게시하고,
GitHub Packages의 `vision_server` package를 Public으로 전환합니다. 그다음 변수를
`true`로 바꾸고 `deploy` workflow를 `main`에서 수동 실행합니다. 이후 `main` push는
자동으로 중앙 배포를 요청합니다.

중앙 배포 요청 직전에는 새 GitHub runner가 해당 SHA image를 인증 없이 조회합니다.
이 검증이 실패하면 Config 요청을 보내지 않으므로 Kubernetes에 `ImagePullBackOff`
상태를 만들지 않습니다.

## Kubernetes 계약

Config repository가 다음 resource를 관리합니다.

- `Deployment/vision-server`: replica 1, worker 1, CPU
- `Service/vision-server`: ClusterIP 80 → container 8000
- `ConfigMap/vision-config`: threshold, upload limit, CPU device
- AI ConfigMap의 `VISION_SERVER_URL=http://vision-server`

Pod는 `/etc/passwd`에 등록된 non-login UID 10001, read-only root filesystem,
capability drop, RuntimeDefault seccomp를 사용합니다. 쓰기가 필요한 cache는
크기 제한된 `/tmp`에만 둡니다.

Kubernetes probe:

- startup: `/health/ready`
- readiness: `/health/ready`
- liveness: `/health/live`

Config 중앙 배포는 `ghcr.io/.../vision_server:<40자리 SHA>`를 rollout합니다.
`latest`는 사람이 확인할 수 있도록 함께 게시하지만 실제 rollout 기준이 아닙니다.

## 배포 smoke

Pod가 READY가 된 뒤 AI Pod 또는 동일 cluster 내부에서 확인합니다.

```bash
curl --fail http://vision-server/health/live
curl --fail http://vision-server/health/ready

curl --fail-with-body \
  --form "image=@approved-smoke-image.jpg" \
  http://vision-server/api/v1/internal/mushrooms/health-check
```

확인 항목:

- 두 모델 load가 끝난 뒤에만 readiness가 200인지
- 응답이 camelCase이며 경로·credential·stack trace를 노출하지 않는지
- 로그의 device가 `cpu`인지
- rollout 실패 시 Config가 직전 revision으로 rollback하는지

## 후속 개선

- Linux 전이 의존성 lock/hash 파일
- 실제 스마트폰 이미지 smoke fixture와 성능 기준
- CPU memory/latency 실측 후 resource 및 replica 조정
- AI Server의 Vision timeout·오류 변환·결과 저장
- 필요 시 NetworkPolicy 적용을 위한 중앙 deploy script 확장
