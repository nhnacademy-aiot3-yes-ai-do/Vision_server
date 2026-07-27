# AIHub 수확 모델과 현재 YOLO11n 통합 판단

## 결론

현재 AIHub 패키지는 그대로 통합할 수 없다.

핵심 차단 요인은 다음과 같다.

1. 학습 완료 수확 가중치가 없다.
2. `_1/_2/_3`의 생육·수확 의미가 없다.
3. 양송이 수확 클래스가 없다.
4. confidence, 최종 NMS IoU, resize/letterbox 등 배포 파라미터가 없다.
5. 수확 단계→READY/NOT_READY 후처리가 없다.
6. 학습 분할과 성능·외부 일반화 자료가 없다.
7. 패키지의 실행 환경은 legacy Windows/CUDA Darknet이고 설정 경로도 불완전하다.

ONNX 변환은 해결책의 첫 단계가 아니다. 변환할 최종 `.weights`가 없기 때문이다.

## 현재 프로젝트와의 결합 가능성

현재 프로젝트의 YOLO11n은 5품종 bbox, 개수, confidence를 반환한다. 요청한 구조는 개념적으로 가능하다.

```text
사진
→ YOLO11n 품종·객체 탐지
→ 버섯 객체 또는 주변 문맥 crop
→ 생육/수확 모델
→ stage + readiness + confidence
```

그러나 AIHub `harv`는 crop 분류기가 아니라 자체 bbox를 출력하는 full-image detector 정의다. 최종 weight가 확보돼도 다음 두 방식을 비교 검증해야 한다.

| 방식 | 장점 | 위험 |
| --- | --- | --- |
| 원본 이미지에 YOLO11n과 AIHub detector를 각각 실행 후 bbox IoU matching | AIHub가 학습한 전체 이미지 문맥을 유지 | 두 detector 연산과 bbox matching 필요 |
| YOLO11n crop을 AIHub detector에 입력 | 객체별 API 구성 단순 | 학습 입력과 다른 crop/배율/배경으로 domain shift 가능 |

현재 증거만으로 crop 입력 호환성을 확정할 수 없다. 배경 암기 가능성을 줄이려면 crop이 유리할 수 있지만, 수확 판단에 필요한 군집·재배실·크기 문맥을 잃을 수도 있다.

## 품종 지원

| 품종 | 현재 YOLO11n | AIHub harv 정의 | 결론 |
| --- | --- | --- | --- |
| 느타리 | 지원 | 지원 | 단계 의미·weight 확보 후 검증 가능 |
| 양송이 | 지원 | **미지원** | 신규 라벨/모델 필요 |
| 큰느타리 | 지원 | 지원 | 단계 의미·weight 확보 후 검증 가능 |
| 팽이 | 지원 | 지원 | 단계 의미·weight 확보 후 검증 가능 |
| 표고 | 지원 | 지원 | 단계 의미·weight 확보 후 검증 가능 |

## 권장 의사결정

### 1단계: 원 제작기관에서 누락 산출물 확보

필수 요청 목록:

- YOLOv3-tiny 또는 YOLOv4 `harv` 최종 `.weights`
- `_1/_2/_3`의 정확한 한국어 정의
- 어느 단계가 수확 적기/과숙인지에 대한 표
- 양송이 수확 모델 존재 여부
- 원본 추론 명령 또는 source
- RGB/BGR, normalization, resize/letterbox
- confidence 및 NMS IoU
- 학습/검증/평가 목록과 클래스 수량
- 카메라·날짜·재배 객체 분리 방식
- mAP/precision/recall과 외부 스마트폰 평가

이 자료를 확보하기 전에는 `model_source=AIHub`인 실제 수확 예측을 서비스하지 않는다.

### 2단계: 확보 성공 시 제한적 재현

- 원본과 격리된 legacy Windows 환경에서 공식 hash와 의존성을 검증한다.
- 소량의 전문가 검수 이미지로 원 Darknet 결과를 재현한다.
- 원본 전체 이미지와 YOLO11n crop 두 입력을 비교한다.
- 클래스 번호를 임의로 READY에 매핑하지 않고 공식 의미표를 사용한다.
- 카메라 홀드아웃 및 외부 스마트폰 사진으로 성능을 재평가한다.
- 이후에만 ONNX 변환을 검토하고 원본 Darknet과 수치 동등성을 확인한다.

### 3단계: 확보 실패 시 신규 모델

가장 적절한 방안은 전문가가 정의한 수확 단계/READY 라벨로 현대 모델을 재학습하는 것이다.

- 5품종 전체에 품종별 수확 기준을 정의한다.
- 미성숙·수확 적기·과숙 등 상호 배타적인 단계 ontology를 고정한다.
- 동일 객체의 시간 순서를 확보하면 날짜/카메라/재배 단위로 누수 없이 분리한다.
- 현재 프로젝트 분석에서 별도 생육 단계 필드가 없고 형태값도 전부 결측이므로 기존 라벨만으로 수확 모델을 자동 생성하지 않는다.
- bbox가 필요하면 YOLO11 detector, 객체 crop 단일 상태 판단이면 classifier를 비교한다.

## 그대로 사용·변환·재학습 판단

| 선택 | 현재 판단 |
| --- | --- |
| 그대로 사용 | 불가 |
| ONNX 변환 | 최종 weight 확보 전 불가; 확보 후 동등성 검증 조건부 |
| 신규 YOLO11 detector | 전문가 bbox+stage 라벨이 있다면 가능 |
| crop classifier | 객체별 stage 라벨과 충분한 주변 문맥 설계가 있다면 유력 |
| AIHub pseudo-label | 현재 불가; weight+stage mapping+검증 성능 확보 후 전문가 검수 조건부 |

## 전문가 검수가 필요한 사항

- 품종별 `_1/_2/_3`의 실제 의미
- 수확 적기와 과숙의 시각적 경계
- 군집 내 혼합 단계 처리
- 스마트폰 거리·각도·조명 변화
- 양송이 수확 단계 추가
- confidence가 낮거나 여러 단계가 검출되는 경우의 정책
- 날짜나 남은 일수 산출에 필요한 시계열 자료

## 지금 구현 가능한 API

현 시점에는 신뢰 가능한 수확 예측 대신 모델 비가용 상태만 반환할 수 있다.

```json
{
  "growth_stage": null,
  "harvest_readiness": "UNKNOWN",
  "confidence": null,
  "expected_harvest_date": null,
  "expected_harvest_days": null,
  "model_source": "AIHub",
  "model_status": "UNAVAILABLE",
  "reason": "trained weights and stage semantics are unavailable"
}
```

최종 weight와 공식 의미표를 확보하고 재현·일반화 검증을 통과한 뒤에만 다음 형태를 고려한다.

```json
{
  "growth_stage": "공식 단계명",
  "harvest_readiness": "READY",
  "confidence": 0.91,
  "expected_harvest_date": null,
  "expected_harvest_days": null,
  "model_source": "AIHub",
  "model_status": "AVAILABLE"
}
```

정확한 날짜와 남은 일수는 현재 모델 출력 근거가 없으므로 후속 버전에서도 별도 시계열/회귀 모델 없이는 `null`로 유지해야 한다.
