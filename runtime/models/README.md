# Runtime model directory

이 디렉터리는 Git에 모델 가중치를 저장하지 않고, 로컬 실행 또는 Docker
build 직전에 승인된 두 모델을 준비하는 위치입니다.

```text
runtime/models/
├── detector/best.pt
└── health/best.pt
```

다음 명령은 원본 모델과 manifest의 크기·SHA-256을 검증한 뒤 위 경로에
원자적으로 복사합니다.

```bash
python scripts/prepare_runtime_models.py
python scripts/prepare_runtime_models.py --check-only
```

기존 runtime 파일이 승인본과 다르면 자동으로 덮어쓰지 않습니다. 검토 후
교체가 필요할 때만 `--overwrite`를 명시합니다.

`*.pt`는 `.gitignore`로 제외됩니다. 모델을 Git에 강제로 추가하지 마세요.
