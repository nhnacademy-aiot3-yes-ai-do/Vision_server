# Vision 근거 목록과 인용 규칙

조사일: 2026-09-14 · 읽기 전용 원본 조사 · 연결 문서: [Vision 경험](DEVELOPMENT_HISTORY.md)

## 1. 출처 기준

- 저장소 표기 `Vision_server/`는 NHN 작업 폴더의 팀 Vision checkout을 의미한다. 문서에는 사용자 홈 절대경로·이메일·토큰·운영 내부 접속 정보를 복사하지 않았다.
- **S ref:** `b5a9a8e17f9d3cbef3926bf1463794403aacc4df` — 현재 checkout HEAD. 코드·서비스 문서의 기준.
- **R ref:** `8420e23628f628adad4ebca983c4b59a0db63e79` — 로컬 보존 브랜치 `archive/research-reports-2026-07-30`. 연구 보고서 기준.
- 기존 미커밋 파일: `Makefile`, `app/main.py`, `app/services/mushroom_health_service.py`. 현재 코드 확인은 이 상태를 포함한다. 변경 규모는 작더라도 운영 반영은 확정하지 않는다.
- `Vision_server/AGENTS.md`는 확인 시 0바이트였다. 추가 지시 없음.
- 이번 조사에서는 원본 수정·모델 로드·학습·테스트·전체 데이터 압축 해제·fetch·네트워크·배포를 하지 않았다.
- 9/6에 작성한 과거 조사 메모는 탐색 시작점으로만 사용했다. 아래 주요 사실은 이 저장소의 코드·문서와 보존 Git ref로 재확인했으므로 별도 포트폴리오 작업 폴더 없이 검토할 수 있다.

원본을 다시 볼 때는 Vision checkout에서 다음처럼 고정 ref를 사용한다. 아래 명령은 설명용이며 새 작업을 실행한 기록이 아니다.

```bash
git show 8420e23628f628adad4ebca983c4b59a0db63e79:reports/full_dataset_analysis.md
git show b5a9a8e17f9d3cbef3926bf1463794403aacc4df:docs/MODEL_CARD.md
```

줄 번호는 조사 시 출력의 기준이다. 파일 변경 후에는 고정 ref로 다시 확인한다. `app/services/mushroom_health_service.py` 관련 줄은 기존 미커밋 변경을 포함한 작업 트리 기준임을 별도로 표시했다.

## 2. 사실별 근거

### VE01 — 전체 JSON 스캔과 기능 제약

- `Vision_server/reports/full_dataset_analysis.md` **R**, 3행: 보고서 생성일 2026-07-26.
- 같은 파일 **R**, 7~16행: 전체 JSON 332,100, 성공 332,100, 실패 0. 이미지 ZIP 파일명만 조사, 이미지 bytes 디코딩·전체 압축 해제·학습 없음.
- 같은 파일 **R**, 22~28행: 품종·작업·정상 여부·bbox 분포.
- 같은 파일 **R**, 38~42행: 갓/대/무게 5항목 결측 100%.
- 같은 파일 **R**, 44~54행: 생육 단계 유사 필드 미발견, 농가/재배사 식별 후보 유효값 없음.
- 해석 한계: 라벨 분석 범위의 결과이지 이미지 전체 무결성·라벨 정답성·전국 농가 일반화 검증이 아님.

### VE02 — 원본 split 중복 신호

- `Vision_server/reports/data_leakage_analysis.md` **R**, 3~12행: JSON 스캔, 이미지 bytes 미조사, SQLite 집계, ‘중복 가능성 신호’ 명시.
- 같은 파일 **R**, 14~19행: 이미지명/동일 stem 중복 0, 세션 중복 994키, 카메라+날짜 중복 560키.
- 해석 한계: 중복 키 수를 이미지 중복 장수로 바꾸지 않으며 누수 원인을 확정하지 않음.

### VE03 — 제공 수확 모델의 정적 검토

- `Vision_server/reports/aihub_harvest_logic_review.md` **R**, 5~16행: 품종+번호단계 객체 탐지 설계, 최종 weight 없음.
- 같은 파일 **R**, 20~45행: 단계 의미·READY 기준 미확인.
- 같은 파일 **R**, 119~147행: 수확 날짜·남은 일수·크기 기준 없음, 설정 그대로 실행 불가의 이유.
- 현재 결과: `Vision_server/docs/MODEL_CARD.md` **S**, 170~180행은 수확 적기·생육 단계·물리량 등을 비지원으로 명시.
- 해석 한계: 실제 수확 모델을 실행해 성능 불량을 확인한 결과가 아니라 제공 자료의 정적 감사.

### VE04 — 라벨 의미 정제와 탐지 후보 manifest

- `Vision_server/reports/model_scope_decision.md` **R**, 5~23행: 탐지/분류/segmentation/생육/회귀 후보 비교, 배양 bbox 의미, 초기 탐지 선택.
- `Vision_server/reports/detection_manifest_summary.md` **R**, 5~17행: 생육·병해, 5품종, 그룹 키, 303,388 JSON, 575,702 bbox.
- 같은 파일 **R**, 19~42행: Train/Validation/Test 후보 규모, 제외 이유, 그룹·image_member 중복 0.
- 같은 파일 **R**, 68~75행: 이 작업은 이미지 디코딩·YOLO 변환·학습을 수행하지 않음.
- 해석 한계: 후보 manifest 규모와 최종 detector 실제 학습 이미지 수는 다르다. 초기 group split과 뒤의 camera holdout release 평가도 구분한다.

### VE05 — 날짜·마스킹 편향과 분할의 한계

- `Vision_server/reports/health_date_holdout_feasibility.md` **R**, 12~23행: 날짜 lookup 93.79%, 품종+날짜 97.59%, 날짜+카메라 95.30%; 같은 데이터 다수결 진단치임을 명시.
- 같은 파일 **R**, 27~45행: chronological/연속 날짜의 균형 분리 제약, 고정 Test 포함 3-way date isolation의 가용량 부족, mask-only 진단치와 한계.
- 같은 파일 **R**, 48~53행: raw/masked 6,000장 대응 점검 결과.
- 해석 한계: metadata lookup은 모델 성능이 아니고, mask-only lookup은 완전한 timestamp ablation도 아니다. 날짜 편향이 높은 정확도의 확정 원인이라고 표현하지 않는다.

### VE06 — 건강 분류 데이터 구성

- `Vision_server/reports/health_date_holdout_split_plan.md` **R**, 5~12행: 고정 seed, 날짜·global camera 분리, 교차 조합 제외.
- 같은 파일 **R**, 16~29행: 5품종×2상태, 각 조합 Train 500/Validation 100, 합계 5,000/1,000.
- 같은 파일 **R**, 36~42행: 이 문서 자체는 계획 단계이며 일부 병해 조합 미지원.
- 후속 `3ddbbe0` (2026-07-28, `feat: create health date camera holdout dataset`)와 `1f69736` (2026-07-29, 분류 평가), `Vision_server/docs/MODEL_CARD.md` **S**, 62~81행을 함께 확인해 후속 구성·평가 기록과 연결.
- 주의: 계획 문서만 보고 실제 학습/평가 완료로 주장하지 않았으며 후속 평가 자료로 교차 확인했다.

### VE07 — 현재 모델, 내부 점수와 일반화 한계

- `Vision_server/docs/MODEL_CARD.md` **S**, 35~58행: YOLO11n detector / YOLO11n-cls binary classifier, UNCERTAIN은 서비스 상태.
- 같은 파일 **S**, 45~50행: detector mAP50-95 0.899는 release manifest 인용이며 재계산 상세표 미보존.
- 같은 파일 **S**, 64~92행: 건강 Val 1,000, 999정답, Test 사용 없음.
- 같은 파일 **S**, 109~140행: crop 평균 IoU 0.982953, 제한된 e2e 100장 결과, 15% 최적 비교 없음.
- 같은 파일 **S**, 146~161행: 건강 날짜-상태 결합, e2e 100장이 detector Train의 날짜/camera와 겹침, 외부 스마트폰 독립평가 미완료.
- `Vision_server/models/model-manifest.json` **S**, 7~38행 / 43~69행: 두 모델명, 입력 640/320, 역할과 제한.
- 해석 한계: 모델 카드가 사용하는 ‘배포 모델’ 용어는 승인 모델 artifact의 의미로 인용하며 현재 live 클러스터 버전 확인으로 취급하지 않는다.

### VE08 — union crop 선택과 구현

- `Vision_server/reports/health_input_strategy.md` **R**, 5~9행: 건강 라벨은 이미지 단위, bbox는 객체/군집, 촬영 프로토콜과 라벨 결합.
- 같은 파일 **R**, 13~32행: 전체 이미지/union/padding/최대 bbox 후보의 trade-off, 15% 정책, 병반 출력 제외.
- `Vision_server/scripts/predict_mushroom_health.py` **S**, 333~355행: 이미지 width/height의 15% padding, 이미지 경계 clipping.
- 같은 파일 **S**, 489~546행: 품종별 grouping, union bbox, 품종별 한 번의 classifier 호출과 응답 구성.
- 해석 한계: 분석 대상 영역을 병변 위치로 표시하지 않는다. 15% 최적성이나 개체별 진단 정확도 근거가 아니다.

### VE09 — 건강 분류 평가 원자료

- `Vision_server/reports/health_classifier_overall_metrics.json` **R**, 3~12행: 모델 경로·validation 1,000장·train 평가 0·test 접근 0·입력 320.
- 같은 파일 **R**, 24~47행: 999정답, accuracy 0.999, macro F1 0.9989999989999989, confusion matrix `[[499,1],[0,500]]`.
- 같은 파일 **R**, 140~179행: threshold 후보 검토; 0.70에서 자동 992/보류 8, 해당 자동판정 subset 992정답. **전체/외부 100% 정확도로 사용 금지.**
- `Vision_server/reports/health_classifier_evaluation.md` **R**, 74~82행: threshold 분석은 Validation 후보 분석이며 그 자체가 최종 threshold 확정은 아님.
- 주의: 실제 서비스 기본값 0.70은 VE10의 API/config 계약과 별도로 확인한다.

### VE10 — 불확실·미탐지·품종별 실패 격리

- `Vision_server/docs/API.md` **S**, 52~67행: 탐지 gate와 classifier 결과 보류의 차이; gate 시 확률 null, classifier 실행 후에는 숫자.
- 같은 파일 **S**, 149~164행: 미탐지 200, 오류 400/413/415/422/429/500 구분.
- 같은 파일 **S**, 196~199행: detection 0.25, min detection 0.50, health 0.70, padding 0.15 기본값.
- `Vision_server/scripts/predict_mushroom_health.py` **S**, 359~379행 / 482~531행: 두 확률 검증, strict `<` 보류 조건, 낮은 탐지 시 분류 생략, 미탐지 반환.
- `Vision_server/tests/test_health_api.py` **S**, 397·427·451·485행: 저신뢰, 임계값 동일, 품종별 격리, 미탐지 테스트.
- 조치 이력: `a1e16ee`, `6148372` (2026-07-29). 동일 목적 변경을 독립 성과로 중복 계산하지 않는다.

### VE11 — Java 연동/API 입력 계약

- `Vision_server/docs/API.md` **S**, 5~17행: multipart image, JPEG/PNG/WEBP, 10MiB, Vision이 저장소를 직접 읽거나 영구 저장하지 않음.
- 같은 파일 **S**, 21~48행: 동기 요청/응답, OpenFeign 응답과 분석 완료, timeout/오류 구분.
- 같은 파일 **S**, 137~168행: 응답 필드 의미·건강 상태·HTTP 계약.
- `Vision_server/tests/test_health_api.py` **S**, 352·543·564·583·595·650·716·736행: 정상·오류·로드·OpenAPI 계약 테스트 존재.
- 구현 상세 `app/services/mushroom_health_service.py`는 작업 트리 기준 99~215행. 해당 파일에 기존 미커밋 변경이 있으므로 완료 배포와 분리.

### VE12 — 동시성/과부하/취소 처리

- `Vision_server/app/services/inference_gateway.py` **S**, 14~35행: process-local admission, 초과 거부, finally permit 반납.
- `Vision_server/app/services/mushroom_health_service.py` **작업 트리**, 71~96행: 취소 후에도 동기 작업 종료를 기다림.
- 같은 파일 **작업 트리**, 269~340행: executor 디코딩, inference lock, admission·429 변환.
- `Vision_server/tests/test_mushroom_health_service.py` **S**, 199·226·304·363행: event loop 외 작업, 취소 대기, 직렬화, admission 테스트.
- `Vision_server/docs/API.md` **S**, 171~174행: multipart 네트워크 수신 단계가 이 제한보다 앞섬.
- 해석 한계: Vision 프로세스의 메모리 보호 설계이며 Cultivation Java OOM의 직접 해결책으로 연결하지 않는다. 부하 테스트 개선률은 확인하지 않았다.

### VE13 — 모델 재현성과 준비 상태

- `Vision_server/models/model-manifest.json` **S**, 5~69행: role/version/hash/size/input/class mapping/한계.
- `Vision_server/scripts/verify_runtime_models.py` **S**, 4~9행: schema→경로→크기/SHA→검증 중 불변성.
- `Vision_server/app/core/model_registry.py` **S**, 93~143행: class mapping, 단일 로드, 전후 fingerprint, partial failure와 READY 상태.
- `Vision_server/tests/test_model_registry.py` **S**, 97·133·170·302·375·476행: singleton/동시성/실패/비노출/실모델 opt-in 테스트.
- `Vision_server/docs/MODEL_MANAGEMENT.md` **S**, 3~25행 / 35~54행 / 89~98행: Git+image 방식, manifest 중심, 시작 검증.
- 과거 `Vision_server/scripts/prepare_runtime_models.py` **R**, 1~28행: 승인 모델 staging 방식. 현재 tree에서는 verify 방식으로 변경된 것을 구분.

### VE14 — 개인 담당·날짜 교차 확인

조회 방법: 로컬 Git log의 날짜·author·변경 파일만 읽었고 커밋 수·줄 수를 기여율 산식으로 쓰지 않았다. 연락 이메일은 이 문서에 넣지 않았다.

| ref | 기록일 | author 표기 | 확인한 주제 |
| --- | --- | --- | --- |
| `754ece5` | 2026-07-26 | kim75503 | 전체 JSON 분석·누수 가능성 보고서 |
| `b3f6b19` | 2026-07-26 | kim75503 | 탐지 manifest 구성. 제목의 leak-free 표현은 완전한 누수 제거 주장으로 인용하지 않음 |
| `b92c2cf` | 2026-07-27 | kim75503 | camera holdout pilot |
| `b15189b` | 2026-07-28 | kim75503 | 제공 수확 모델 감사 |
| `04bac5c`, `3ddbbe0` | 2026-07-28 | kim75503 | 날짜·카메라 분리 감사와 세트 구성 |
| `1f69736`, `ed1824d`, `5041bee` | 2026-07-29 | kim75503 | 건강 평가·e2e 추론·v1 연결 |
| `a1e16ee`, `6148372` | 2026-07-29 | kim75503 | 낮은 탐지 confidence 분류 중단 |
| `f0c9737` | 2026-07-30 | kim75503 | 서비스 prototype |
| `719af03` | 2026-08-23 | {kim75503} | CPU CI/CD 준비·문서·플랫폼 테스트 |

해석 한계: 변경 기록의 author와 실제 개인 작업 범위가 항상 같지는 않다. 공동 설계·AI 도구 활용·리뷰 기여는 본인 설명으로 보완해야 한다.

### VE15 — 환경·배포·테스트의 완료 범위

- `Vision_server/docs/MAC_VALIDATION.md` **S**, 5~11행: Mac 로컬 호환성과 Linux CPU/CUDA 성능 구분.
- 같은 파일 **S**, 99~110행 / 146~183행: fake-model, 실모델 opt-in, MPS preflight와 fallback.
- 같은 파일 **S**, 185~212행: 완료 체크리스트는 미체크 문서이며 실행 성적표가 아님.
- `Vision_server/docs/CI_CD_HANDOFF.md` **S**, 7~16행: Linux amd64 CPU, worker 1, probes, SHA image 계약.
- 같은 파일 **S**, 24~41행: CI/CD 단계. 146~152행: memory/latency 실측·외부 fixture 등 후속 개선.
- 이번 문서화에서 확인한 것은 코드/설정/테스트 존재와 배포 준비 이력이다. 최신 CI·실시간 서비스·전체 플랫폼 테스트 완료는 확인하지 않았다.

### VE16 — 원 학습 로그와 외부 사례 미확보

- 현재 저장소 파일 목록 및 R tree에서 `results.csv`, `args.yaml`을 찾지 못했다. 다른 저장소·외장 디스크까지 광범위하게 검색하지 않았다.
- 상세 epoch/batch/optimizer/장비/평균 지연시간은 미확인이다.
- 외부 사진 실패는 대화 맥락에 있으나 이번 조사에서는 사진·정답·모델 버전·응답이 묶인 원본을 확보하지 못했다.
- V12의 학습 방식 설명 혼동과 V13의 부실한 피드백 의견은 사용자 대화 근거다. 실제 원문 피드백/변경 후 결과가 없어 원인·개선 완료로 단정하지 않았다.

## 3. 오래된 계획을 현재 기능으로 옮기지 않기

`reports/health_model_scope_decision.md` **R**, 5~7행은 ‘확정 기능’이라는 제목 아래 병해 종류 후보의 후속 2단계까지 적고 있다. 그러나 현재 모델 manifest와 `docs/MODEL_CARD.md` **S**, 170~180행은 특정 병명·병반·해충·수확 판단을 지원하지 않는다. 따라서 이 오래된 문서의 제목만 보고 병해 종류 모델을 완성했다고 쓰지 않는다.

동일하게 다음을 분리한다.

- 분할 **계획**과 실제 구성·평가 보고서.
- 후보 데이터 전체 manifest와 모델에 실제 입력한 학습 subset.
- 모델 승인 artifact와 현재 운영 Pod의 모델.
- 작성된 테스트와 실행 통과 기록.
- API에 전달되는 경고와 사용자가 실제로 이해한 정도.

## 4. 검수 결과

- [x] JSON 수·학습 수·모델별 평가 지표 분리.
- [x] 내부 Val과 별도 Test·외부 일반화 분리.
- [x] crop은 병반이 아닌 분석 대상 영역임을 명시.
- [x] `UNCERTAIN`은 학습 클래스가 아닌 서비스 정책임을 명시.
- [x] 15% padding·0.70 threshold를 최적성/실제 발병 확률로 표현하지 않음.
- [x] 개인 기여는 사용자 설명+author 범위 교차 확인으로 제한.
- [x] 문서·코드 존재와 실행·배포 성공을 분리.
- [x] 비밀값·개인 이메일·운영 접속 주소를 결과 문서에 포함하지 않음.
- [ ] 최신 운영 artifact·실행 로그·개인 공동기여 최종 확인은 후속 보완.
