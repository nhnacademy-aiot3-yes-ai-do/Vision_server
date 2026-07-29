# Apple Silicon Mac 검증 가이드

## 목적과 범위

이 문서는 Apple Silicon Mac에서 Mushroom Vision Service의 동일한 승인
모델을 CPU 또는 MPS로 검증하는 절차를 정의합니다. Mac 전용 모델을
재학습하거나 Core ML로 변환하는 절차가 아닙니다.

Mac 검증에서도 Linux 배포와 동일한 두 `best.pt`와 동일한
`models/model-manifest.json`을 사용합니다.

| 역할 | 승인 모델 |
| --- | --- |
| 품종 탐지 | `mushroom-yolo11n-camera-holdout-v1` |
| 건강 분류 | `mushroom-health-yolo11n-date-camera-holdout-v1` |

운영 배포는 별도의 승인된 Linux CPU/CUDA container profile을 사용합니다.
Mac 결과는 API 호환성과 host 실행 가능성을 확인하는 증거이며 Linux
container 또는 CUDA 성능을 대신하지 않습니다.

## 빠른 검증 순서

아래 순서를 그대로 따르면 설치, 모델 검증, CPU/MPS 통합 테스트와 API
확인을 빠뜨리지 않을 수 있습니다. 승인 모델 전달 방법은 팀 내부 저장소
정책을 따르며 이 문서는 source URL이나 credential을 가정하지 않습니다.

1. Python 3.12 arm64 virtual environment를 만들고 Mac·dev requirements를
   설치합니다.
2. 승인된 두 `best.pt`를 아래 source 위치에 배치합니다.
3. `shasum -a 256`으로 원본 SHA-256을 직접 확인합니다.
4. `make prepare-models`와 `make check-models`를 실행합니다.
5. `make doctor-mac`으로 패키지·MPS·모델 존재 여부를 기록합니다.
6. `HEALTH_DEVICE=cpu`로 실제 모델 통합 테스트를 실행합니다.
7. MPS preflight 후 `HEALTH_DEVICE=mps`로 같은 통합 테스트를 실행합니다.
8. `make run-cpu` 또는 `make run-mps`로 FastAPI를 실행합니다.
9. `/health/live`, `/health/ready`와 실제 이미지 `curl` 응답을 확인합니다.

## 의존성 파일 계약

| 파일 | 포함 범위 |
| --- | --- |
| `requirements-common.txt` | FastAPI, Uvicorn, Pillow, Ultralytics, Pydantic 등 플랫폼 공통 항목 |
| `requirements-macos.txt` | common include와 `torch==2.11.0`, `torchvision==0.26.0` |
| `requirements-dev.txt` | pytest 등 개발·테스트 전용 항목 |
| `requirements-runtime.txt` | Linux container에서 common만 include; PyTorch/torchvision은 승인 base가 제공 |

macOS wheel에는 `+cu128` 같은 CUDA local version suffix를 사용하지
않습니다. `requirements-runtime.txt`는 Linux container 계약이므로 Mac
virtual environment 설치 파일로 사용하지 않습니다.

## 1. 호스트와 Python 확인

검증 기준은 Apple Silicon과 Python 3.12입니다.

```bash
uname -m
python3.12 --version
```

예상 아키텍처는 `arm64`입니다. Rosetta의 x86_64 Python과 arm64 wheel을
섞지 않습니다.

```bash
python3.12 -c "import platform; print(platform.machine())"
```

## 2. virtual environment 설치

```bash
python3.12 -m venv .venv
source .venv/bin/activate
make install-mac
```

이 target은 Apple Silicon macOS인지 먼저 확인하고
`requirements-macos.txt`, `requirements-dev.txt` 순서로 설치합니다. 이를
우회해 수동 설치해야 한다면 동일한 두 requirements 파일을 함께 사용합니다.

설치 후 플랫폼과 accelerator 정보를 확인합니다.

```bash
make doctor-mac
```

이 진단은 Python·패키지 버전, 아키텍처, MPS/CUDA 가용성, 승인 모델 파일
존재 여부를 host 절대경로 없이 출력합니다. 정상적인 Mac wheel이면
`cudaAvailable`은 false입니다. 패키지 자체의 CUDA build 여부도 확인해야
하면 `python -c "import torch; print(torch.version.cuda)"`가 `None`인지
확인합니다.
`mps_built=True`이더라도 OS·하드웨어·현재 실행 환경에 따라
`mps_available=False`일 수 있습니다.

## 3. 동일 승인 모델 준비

모델 binary는 Git과 dependency package에 포함하지 않습니다. 기존 승인
source의 두 파일을 다음 기본 위치에 둡니다.

```text
artifacts/models/yolo11n_camera_holdout_v1/best.pt
artifacts/models/yolo11n_health_date_camera_holdout_v1/best.pt
```

Linux에서 검증한 파일과 byte 단위로 동일한 `best.pt`를 사용합니다. Mac
전용 checkpoint를 만들거나 파일 이름만 같은 다른 모델로 교체하지 않습니다.

```bash
make prepare-models
make check-models
```

이 명령은 manifest의 크기와 SHA-256을 확인해 다음 runtime 위치를
준비합니다.

```text
runtime/models/detector/best.pt
runtime/models/health/best.pt
```

검증 실패 시 모델을 묵시적으로 덮어쓰거나 SHA 검증을 끄지 않습니다.

준비 전에 source 파일을 직접 확인할 때는 다음 명령과 승인값을 비교합니다.

```bash
shasum -a 256 artifacts/models/yolo11n_camera_holdout_v1/best.pt
shasum -a 256 artifacts/models/yolo11n_health_date_camera_holdout_v1/best.pt
```

```text
detector  8d17eb493f2eeccccd832c56da2f346dfc730e5605c460016386c6bd0be10d32
health    720efb30093c2fbaf8866243a60870c7816f3e141c81e0fac015a358edce1f92
```

## 4. fake-model 기본 테스트

기본 테스트는 실제 `best.pt`를 로드하지 않습니다.

```bash
make test
```

실제 모델 registry 통합 테스트는 승인 모델이 준비된 로컬 환경에서만
명시적으로 활성화합니다.

```bash
HEALTH_DEVICE=cpu RUN_MODEL_INTEGRATION_TESTS=true \
python -m pytest -q tests/test_model_registry.py -k integration
```

MPS integration을 증거로 남길 때는 다음 절의 preflight가 성공한 뒤
`HEALTH_DEVICE=mps`로 실행하고 실제 선택 device 로그도 함께 보존합니다.

```bash
HEALTH_DEVICE=mps RUN_MODEL_INTEGRATION_TESTS=true \
python -m pytest -q tests/test_model_registry.py -k integration
```

## 5. MPS 실행

먼저 MPS backend가 실제로 사용 가능한지 확인합니다.

```bash
python -c "import torch; assert torch.backends.mps.is_built(), 'PyTorch wheel has no MPS support'; assert torch.backends.mps.is_available(), 'MPS is not available in this process'; print('MPS available')"
```

MPS를 명시적으로 선택해 worker 1개로 실행합니다.

```bash
make run-mps
```

이는 `HEALTH_DEVICE=mps make run`과 같습니다.

다른 terminal에서 probe를 확인합니다.

```bash
curl http://localhost:8000/health/live
curl http://localhost:8000/health/ready
```

승인된 비민감 sample 이미지로 실제 multipart API도 확인합니다.

```bash
curl -X POST \
  http://localhost:8000/api/v1/mushroom/health-check \
  -H "accept: application/json" \
  -F "image=@sample.jpg"
```

현재 고정 Ultralytics 버전에서 `HEALTH_DEVICE=auto`는 CUDA가 있으면 CUDA,
없으면 CPU를 선택합니다. MPS를 자동 선택하지 않으므로 Mac MPS 검증에는
반드시 `HEALTH_DEVICE=mps`를 지정합니다.

또한 요청한 문자열이 `mps`라는 사실만으로 실제 MPS 실행이 입증되지는
않습니다. MPS backend를 사용할 수 없는 환경에서는 현재 Ultralytics device
선택 과정에서 CPU가 선택될 수 있습니다. preflight 결과와 애플리케이션 시작
로그의 실제 device를 함께 확인합니다.

## 6. CPU 기준선

재현 가능한 CPU 기준선은 device를 명시합니다.

```bash
make run-cpu
```

이는 `HEALTH_DEVICE=cpu make run`과 같습니다.

MPS 오류 후 결과 정확성만 확인해야 할 때도 `auto` 대신 이 명령을
사용합니다. CPU 결과는 MPS 성능 수치로 보고하지 않습니다.

## 7. MPS fallback의 정확한 의미

일부 PyTorch 연산이 MPS에 구현되지 않아 실패할 때만 다음을 제한적으로
검토합니다.

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 HEALTH_DEVICE=mps make run
```

`PYTORCH_ENABLE_MPS_FALLBACK=1`은 MPS가 지원하지 않는 연산을 CPU에서
실행하도록 허용합니다.

- MPS를 선택하는 환경변수가 아닙니다.
- 전체 추론을 CPU 모드로 전환한다는 뜻이 아닙니다.
- 모든 연산이 MPS에서 실행됐다는 증거가 아닙니다.
- CPU 왕복 때문에 latency가 달라질 수 있으므로 성능 기준으로 사용하지
  않습니다.
- unsupported operation을 숨길 수 있으므로 기본 CI 성공 조건으로 켜지
  않습니다.

fallback 없이 MPS가 실패한다면 오류와 사용한 버전을 기록하고, 기능
기준선이 필요하면 `HEALTH_DEVICE=cpu`로 별도 검증합니다.

## 8. 기능 검증 체크리스트

### 빠른 결과 기록

- [ ] Apple Silicon architecture 확인
- [ ] torch 설치 성공
- [ ] MPS available 확인
- [ ] 모델 SHA 일치
- [ ] CPU 통합 테스트 성공
- [ ] MPS 통합 테스트 성공 또는 CPU fallback 결정
- [ ] `/health/live` 성공
- [ ] `/health/ready` 성공
- [ ] 실제 이미지 API 응답 성공

### 설치와 모델

- [ ] `uname -m`과 Python process가 모두 `arm64`이다.
- [ ] Python은 3.12 계열이다.
- [ ] `torch.version.cuda is None`이다.
- [ ] CUDA suffix wheel 또는 CUDA package index를 사용하지 않았다.
- [ ] `make check-models`가 동일한 두 모델의 크기와 SHA-256을 통과했다.
- [ ] 모델 변환·재학습·manifest 우회가 없었다.

### CPU/MPS

- [ ] MPS 검증이면 `is_built()`와 `is_available()`가 모두 true이다.
- [ ] `HEALTH_DEVICE=mps` 또는 `HEALTH_DEVICE=cpu`를 명시했다.
- [ ] 시작 로그에서 실제 선택 device를 확인했다.
- [ ] fallback 사용 여부를 결과에 기록했다.
- [ ] worker 수가 1이다.

### API

- [ ] fake-model 기본 테스트가 통과했다.
- [ ] `/health/live`가 200을 반환한다.
- [ ] 모델 준비 후 `/health/ready`가 200을 반환한다.
- [ ] 승인된 비민감 sample로 health-check 응답 schema를 확인했다.
- [ ] 응답과 로그에 host 절대경로, credential, stack trace가 없다.
- [ ] 외부 사진 결과를 확정 진단이나 최종 일반화 성능으로 해석하지 않았다.

## 9. 진단 순서

| 증상 | 우선 확인 | 조치 |
| --- | --- | --- |
| macOS에서 torch wheel을 찾지 못함 | CUDA suffix 또는 Linux requirements 사용 여부 | Mac venv를 새로 만들고 `requirements-macos.txt` 사용 |
| `mps_available=False` | arm64 Python, macOS, PyTorch wheel, 실행 환경 | CPU 기준선 사용; MPS 성공으로 보고하지 않음 |
| `HEALTH_DEVICE=mps`인데 CPU로 보임 | preflight와 시작 device 로그 | MPS 가용성을 복구하거나 명시적 CPU로 재검증 |
| MPS unsupported operation | fallback 없는 오류 원문과 패키지 버전 | 제한적으로 fallback 비교; 운영 결정 전 CPU/MPS 결과 분리 |
| readiness 503 | 두 모델 경로, 크기, SHA-256, class mapping | `make check-models`; 검증을 끄거나 모델을 덮어쓰지 않음 |
| MPS와 CPU 예측 차이 | 동일 입력·동일 두 `best.pt`·동일 threshold 확인 | 차이와 confidence를 기록하고 허용 기준을 별도 승인 |

진단 결과에는 다음 정보를 함께 남깁니다.

```bash
python --version
python -c "import platform, torch, torchvision, ultralytics; print(platform.platform()); print(platform.machine()); print(torch.__version__); print(torchvision.__version__); print(ultralytics.__version__); print(torch.version.cuda); print(torch.backends.mps.is_built()); print(torch.backends.mps.is_available())"
```

로컬 사용자명, 절대경로, 모델 source URL과 credential은 결과에 포함하지
않습니다.

## 10. Mac Docker와 Linux 배포의 경계

Docker Desktop for Mac은 Linux VM에서 Linux container를 실행합니다.
Apple Metal/MPS device는 현재 Vision Docker 경로에 전달되지 않습니다.

- Mac에서 MPS를 검증하려면 host virtual environment를 사용합니다.
- Mac Docker에서 `HEALTH_DEVICE=mps`를 사용하지 않습니다.
- Mac Docker CPU smoke는 Linux CPU image의 제한된 호환성 점검일 뿐입니다.
- Linux CUDA image는 대상 아키텍처의 NVIDIA runner에서 build·실행합니다.
- `--gpus` 옵션 또는 cross-build 성공만으로 CUDA 동작을 주장하지 않습니다.

`Dockerfile.template`은 Apple용 image로 바꾸지 않습니다. 팀이 승인한 Linux
CPU 또는 CUDA `BASE_IMAGE`를 입력받고, 그 base가 호환되는
PyTorch/torchvision을 제공합니다. Java Spring 서비스의 Java 21 base는
Vision Python/PyTorch base와 계속 분리합니다.

## 11. CI 체크리스트

- [ ] macOS arm64 job은 `requirements-macos.txt`와
  `requirements-dev.txt`를 사용한다.
- [ ] macOS job은 CUDA wheel, CUDA index 또는 Linux CUDA base를 설치하지
  않는다.
- [ ] 일반 PR job은 fake-model 테스트만 실행하고 모델을 다운로드하지 않는다.
- [ ] 전용 Mac MPS integration job은 MPS preflight와 실제 device를
  증거로 남긴다.
- [ ] Linux CPU와 Linux CUDA는 별도 base digest·job·release 증거를 갖는다.
- [ ] Linux CUDA integration은 승인된 NVIDIA runner에서만 실행한다.
- [ ] 모델 binary, `.env`, credential과 host 절대경로를 artifact에 남기지
  않는다.

runner, wheel source, Linux base digest와 platform별 lock 정책은
[CI/CD handoff](CI_CD_HANDOFF.md)의 TODO가 승인되기 전까지 확정된 운영
설정으로 간주하지 않습니다.
