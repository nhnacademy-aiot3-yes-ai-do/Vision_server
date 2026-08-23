# Apple Silicon Mac 검증 가이드

## 목적

Apple Silicon Mac에서 팀 Git 저장소에 포함된 동일한 두 `best.pt`를
CPU 또는 MPS로 실행해 서비스와 API 계약을 검증합니다. Mac 전용 모델을
학습하거나 Core ML로 변환하는 절차가 아닙니다.

운영은 승인된 Linux CPU/CUDA container를 사용합니다. Mac 결과는 로컬
개발과 기능 호환성 증거이며 Linux container 또는 CUDA 성능을 대신하지
않습니다.

## 전체 순서

```text
private repository clone
→ Python 3.12 arm64 가상환경
→ Mac 의존성 설치
→ runtime/models의 두 모델 manifest 검증
→ fake-model 테스트
→ 실제 모델 CPU 테스트
→ MPS 가능 시 MPS 테스트
→ FastAPI와 multipart API 확인
```

모델은 Git clone에 포함됩니다. `artifacts`나 외부 저장소에서 모델을
복사하거나 다운로드하는 준비 단계는 없습니다.

## 1. 호스트 확인

```bash
uname -m
python3.12 --version
python3.12 -c "import platform; print(platform.machine())"
```

`uname -m`과 Python process 모두 `arm64`, Python은 3.12 계열이어야
합니다. Rosetta의 x86_64 Python과 arm64 wheel을 섞지 않습니다.

## 2. 환경 설치

```bash
python3.12 -m venv .venv
source .venv/bin/activate
make install-mac
```

`make install-mac`은 Apple Silicon인지 확인한 뒤 다음 파일을 설치합니다.

| 파일 | 용도 |
| --- | --- |
| `requirements-macos.txt` | 공통 패키지, Ultralytics와 Mac용 PyTorch/torchvision |
| `requirements-dev.txt` | pytest 등 개발·테스트 패키지 |

macOS에서는 `+cu...` suffix가 붙은 CUDA wheel이나 Linux용
`requirements-runtime.txt`를 설치하지 않습니다.

설치 결과와 accelerator 상태를 확인합니다.

```bash
make doctor-mac
```

이 명령은 Python·패키지 버전, 아키텍처, MPS/CUDA 가용성과 모델 파일
존재 여부를 host 절대경로 없이 출력합니다. 정상적인 Mac 환경에서
`cudaAvailable`은 false입니다.

## 3. Git 모델 검증

팀 저장소에는 다음 두 파일이 있어야 합니다.

```text
runtime/models/detector/best.pt
runtime/models/health/best.pt
```

manifest와 일치하는지 확인합니다.

```bash
make verify-models
```

이 명령은 파일을 복사하거나 덮어쓰지 않고 크기와 SHA-256을
`models/model-manifest.json`과 비교합니다. 실패하면 다음을 확인합니다.

- Git checkout이 완전한지
- 모델 파일과 manifest가 같은 commit의 것인지
- 파일이 전송·압축 해제 과정에서 변경되지 않았는지
- 현재 branch가 팀이 승인한 branch인지

검증을 끄거나 manifest의 hash만 임의로 바꾸지 않습니다.

## 4. 기본 테스트

```bash
make test
```

기본 회귀 테스트는 fake model을 사용하므로 실제 `best.pt`를 반복해서
로드하지 않습니다.

실제 모델 registry 통합 테스트는 명시적으로 활성화합니다.

```bash
HEALTH_DEVICE=cpu RUN_MODEL_INTEGRATION_TESTS=true \
python -m pytest -q tests/test_model_registry.py -k integration
```

CPU 통합 테스트를 먼저 통과시키면 모델 파일·class mapping과 API 코드의
문제인지 MPS backend 문제인지 분리하기 쉽습니다.

## 5. CPU 실행

```bash
make run-cpu
```

다른 terminal에서 확인합니다.

```bash
curl --fail http://localhost:8000/health/live
curl --fail http://localhost:8000/health/ready
```

승인된 비민감 sample 이미지로 실제 API를 호출합니다.

```bash
curl --fail-with-body \
  --request POST \
  --header "accept: application/json" \
  --form "image=@sample.jpg" \
  http://localhost:8000/api/v1/internal/mushrooms/health-check
```

응답이 camelCase이고 `status`, `results`, `warnings` 계약을 따르는지
확인합니다. 상세 계약은 [API.md](API.md)를 참고하세요.

## 6. MPS 실행

먼저 현재 process에서 MPS를 실제로 사용할 수 있는지 확인합니다.

```bash
python -c "import torch; assert torch.backends.mps.is_built(), 'PyTorch wheel has no MPS support'; assert torch.backends.mps.is_available(), 'MPS is not available in this process'; print('MPS available')"
```

preflight가 성공한 경우에만 실제 모델 통합 테스트를 실행합니다.

```bash
HEALTH_DEVICE=mps RUN_MODEL_INTEGRATION_TESTS=true \
python -m pytest -q tests/test_model_registry.py -k integration
```

서버도 MPS를 명시해서 실행합니다.

```bash
make run-mps
```

그다음 CPU와 동일한 live, ready와 multipart API를 확인합니다. 시작 로그에
실제 선택 device가 기대한 값인지도 확인합니다.

현재 `HEALTH_DEVICE=auto`는 MPS 검증 방법으로 사용하지 않습니다. MPS를
검증할 때는 반드시 `HEALTH_DEVICE=mps`를 명시하고 preflight와 시작 로그를
함께 증거로 남깁니다.

## 7. MPS fallback

일부 연산이 MPS에 구현되지 않아 실패할 때만 다음 설정을 제한적으로
비교합니다.

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 HEALTH_DEVICE=mps make run
```

이 환경변수는 지원하지 않는 일부 연산을 CPU로 실행하도록 허용할 뿐입니다.

- MPS를 선택하는 설정이 아닙니다.
- 모든 연산이 MPS에서 실행됐다는 증거가 아닙니다.
- CPU 왕복으로 latency가 달라질 수 있습니다.
- 기본 CI 성공 조건으로 사용하지 않습니다.

fallback 없이 실패하면 오류와 패키지 버전을 기록하고, 기능 확인은
`HEALTH_DEVICE=cpu`로 별도 수행합니다.

## 8. 기능 체크리스트

### 설치와 모델

- [ ] macOS와 Python process가 모두 arm64이다.
- [ ] Python 3.12 가상환경을 사용한다.
- [ ] Mac용 PyTorch/torchvision을 설치했다.
- [ ] CUDA wheel/index를 사용하지 않았다.
- [ ] `make verify-models`가 통과했다.
- [ ] 모델과 manifest를 임의로 변경하지 않았다.

### 테스트와 device

- [ ] `make test`가 통과했다.
- [ ] CPU 실제 모델 통합 테스트가 통과했다.
- [ ] MPS 검증이면 `is_built()`와 `is_available()`이 모두 true이다.
- [ ] `HEALTH_DEVICE=cpu` 또는 `mps`를 명시했다.
- [ ] 시작 로그에서 실제 device를 확인했다.
- [ ] fallback 사용 여부를 기록했다.
- [ ] Uvicorn worker가 1개이다.

### API

- [ ] `/health/live`가 200이다.
- [ ] 두 모델 로드 후 `/health/ready`가 200이다.
- [ ] 실제 이미지 multipart 분석 응답이 계약과 일치한다.
- [ ] 응답과 로그에 host 경로, credential과 stack trace가 없다.
- [ ] 결과를 확정 진단이나 외부 일반화 성능으로 해석하지 않았다.

## 9. 자주 발생하는 문제

| 증상 | 확인 | 조치 |
| --- | --- | --- |
| torch wheel 설치 실패 | x86_64 Python 또는 CUDA/Linux requirements 사용 | arm64 Python 3.12 venv를 새로 만들고 `make install-mac` |
| `make verify-models` 실패 | 모델/manifest가 같은 commit인지 | 승인 branch를 다시 checkout하고 파일 임의 수정 금지 |
| readiness 503 | 시작 로그, 모델 SHA와 class mapping | `make verify-models`; 검증 비활성화 금지 |
| `mpsAvailable=false` | arm64, macOS, wheel과 실행 환경 | CPU 기준선 사용; MPS 성공으로 기록하지 않음 |
| MPS unsupported operation | fallback 없는 오류와 버전 | 제한적 fallback 비교 후 CPU 결과와 분리 |
| CPU/MPS 예측 차이 | 같은 image, 모델, threshold인지 | 입력과 confidence 차이를 기록하고 별도 검토 |

진단 시 다음 정보를 함께 기록할 수 있습니다.

```bash
python --version
python -c "import platform, torch, torchvision, ultralytics; print(platform.platform()); print(platform.machine()); print(torch.__version__); print(torchvision.__version__); print(ultralytics.__version__); print(torch.version.cuda); print(torch.backends.mps.is_built()); print(torch.backends.mps.is_available())"
```

로컬 사용자명, 절대경로, repository token과 모델 binary는 결과 artifact에
포함하지 않습니다.

## 10. Mac Docker의 범위

Docker Desktop for Mac은 Linux VM에서 Linux container를 실행하므로 Mac의
Metal/MPS를 현재 Vision container에 제공하지 않습니다.

- MPS 검증은 Mac host virtual environment에서 수행합니다.
- Mac Docker에서 `HEALTH_DEVICE=mps`를 사용하지 않습니다.
- Mac Docker CPU smoke는 Linux CPU image의 제한된 구조 확인입니다.
- CUDA image는 대상 Linux/NVIDIA runner에서 build·실행합니다.

production profile, base digest와 runner 결정은
[CI/CD handoff](CI_CD_HANDOFF.md)를 참고하세요.
