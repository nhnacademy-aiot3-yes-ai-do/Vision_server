# 버섯 건강 체크 v1 모델 범위 결정

## 확정 기능

1. 기존 YOLO11n: 5품종, bbox, 개수, confidence
2. 건강 1단계: `HEALTHY` / `DISEASE_SUSPECTED` / `UNCERTAIN`
3. 건강 2단계: `DISEASE_SUSPECTED`일 때만 품종 조건부 병해 종류 후보와 confidence

수확 적기, 생육 단계, 예상 수확일, 병반 위치, 해충, 실제 갓·대 크기는 v1에서 제외한다.

## 모델 후보 비교

| 후보 | 장점 | 핵심 위험 | v1 판단 |
| --- | --- | --- | --- |
| 전체 이미지 이진 분류 | 문맥 최대 | 작업·카메라·재배실 암기 | ablation만 |
| context crop 이진 분류 | 객체와 배지 문맥 균형 | crop 정책 의존 | **권장 입력** |
| 품종별 이진 모델 5개 | 품종별 외형 최적화 | 데이터·운영 분절 | 보조 실험 |
| 공통 이진 모델 1개 | 데이터 공유·배포 단순 | 품종별 성능 차이 | **권장 1단계** |
| 건강+병해 단일 분류 | 단일 호출 | 불균형·품종 shortcut·오류 전파 불투명 | 비권장 |
| 이진 후 병해 2단계 | 건강과 병해 후보를 분리 | 1단계 오류 전파 | **권장 구조** |
| species+disease 복합 클래스 | 허용 조합 표현 쉬움 | 종 외형만으로 병해를 맞히는 shortcut | 단일 flat head는 비권장 |

## 병해 종류 편향

- 전역 다수 병해만 고르면 40.08%다.
- 품종별 다수 병해 lookup은 51.92%다.
- 세균성검은썩음병은 팽이에만, 솜털곰팡이병은 양송이에만 존재한다.
- 따라서 병해 종류 모델은 품종 외형을 지름길로 사용할 수 있다.

## 권장 구조

### 1단계

하나의 공통 이진 classifier를 15% context union crop으로 학습한다. 품종 균형 sampling과 품종별 recall/F1을 보고한다. Validation에서 calibration한 두 threshold 사이를 `UNCERTAIN`으로 둔다.

### 2단계

공유 backbone과 **품종별 disease head 또는 species-conditioned class mask**를 사용한다. 알려진 품종에서 가능한 병해만 후보로 반환하고 품종별 macro F1·recall·혼동행렬을 평가한다. 단일 전역 flat class 정확도만 보고하지 않는다.

## 개발 가능성

- 건강 이진 모델: 수량상 개발 가능. 생육 정상 220,230개, 병해 83,160개다.
- 병해 종류 모델: 5종 모두 수천 건으로 후보 모델 개발은 가능하지만 최대/최소 불균형과 품종·카메라·날짜 편향 때문에 제한적 후보 출력으로 시작해야 한다.
- 병반 위치 모델: 위치 annotation 부재로 개발 근거가 없다.

## API 상태

```json
{
  "health_status": "HEALTHY | DISEASE_SUSPECTED | UNCERTAIN",
  "health_confidence": 0.0,
  "disease_candidates": [],
  "unsupported": [
    "harvest_readiness",
    "growth_stage",
    "expected_harvest_date",
    "physical_cap_stipe_size",
    "pest_detection"
  ]
}
```

병해 후보는 `DISEASE_SUSPECTED`일 때만 채우며 threshold는 Test가 아닌 Validation에서 결정한다.
