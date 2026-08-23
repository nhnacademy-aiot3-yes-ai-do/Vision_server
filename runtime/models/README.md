# Runtime model directory

이 디렉터리의 두 모델은 Vision_server 비공개 Git 저장소에서 코드와 함께
버전 관리하는 서비스 실행 모델입니다. 다른 `artifacts/` 원본을 복사하지 않고
아래 두 파일만 로컬 실행과 Docker 이미지 빌드의 단일 기준으로 사용합니다.

```text
runtime/models/
├── detector/best.pt
└── health/best.pt
```

모델을 교체할 때는 해당 `best.pt`와 `models/model-manifest.json`의
`sizeBytes`, `sha256`, 버전을 같은 변경으로 갱신합니다. 다음 명령은 두 파일을
변경하지 않고 manifest와 일치하는지만 검증합니다.

```bash
python scripts/verify_runtime_models.py
make verify-models
```

두 `best.pt`만 `.gitignore`의 명시적 예외이며 저장소와 GHCR Package는
비공개로 유지해야 합니다. 임의 모델, 학습 checkpoint와 사용자 이미지는
여기에 추가하지 않습니다.
