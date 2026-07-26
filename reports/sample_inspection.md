# 제한 샘플 검사

- 작업별·라벨 ZIP별 JSON 상한: 20
- 라벨/원천 ZIP 쌍별 이미지 매칭 상한: 20
- JSON 샘플 성공/실패: 560/0
- 인코딩: `{"utf-8-sig": 560}`
- 이미지 매칭 상태: `{"matched": 200}`
- 이미지 매칭 방법: `{"full_path": 200}`

## JSON 샘플 분포

| 구분 | 품종 | 작업 | 수 |
|---|---|---|---:|
| train | 느타리 | 배양 | 20 |
| train | 느타리 | 병해 | 20 |
| train | 느타리 | 생육 | 20 |
| train | 양송이 | 병해 | 20 |
| train | 양송이 | 생육 | 20 |
| train | 큰느타리 | 배양 | 20 |
| train | 큰느타리 | 병해 | 20 |
| train | 큰느타리 | 생육 | 20 |
| train | 팽이 | 배양 | 20 |
| train | 팽이 | 병해 | 20 |
| train | 팽이 | 생육 | 20 |
| train | 표고 | 배양 | 20 |
| train | 표고 | 병해 | 20 |
| train | 표고 | 생육 | 20 |
| validation | 느타리 | 배양 | 20 |
| validation | 느타리 | 병해 | 20 |
| validation | 느타리 | 생육 | 20 |
| validation | 양송이 | 병해 | 20 |
| validation | 양송이 | 생육 | 20 |
| validation | 큰느타리 | 배양 | 20 |
| validation | 큰느타리 | 병해 | 20 |
| validation | 큰느타리 | 생육 | 20 |
| validation | 팽이 | 배양 | 20 |
| validation | 팽이 | 병해 | 20 |
| validation | 팽이 | 생육 | 20 |
| validation | 표고 | 배양 | 20 |
| validation | 표고 | 병해 | 20 |
| validation | 표고 | 생육 | 20 |

## 이미지 샘플 매칭

| 구분 | 품종 | JSON | 이미지 필드 | 방법 | 결과 | 추출 위치 |
|---|---|---|---|---|---|---|
| train | 느타리 | `배양/느타리_배양실_10_10076954.json` | `느타리_배양실_10_10076954.jpg` | full_path | matched | `artifacts/sample_extract/train/TS1_느타리/배양/느타리_배양실_10_10076954.jpg` |
| train | 느타리 | `생육/느타리_생육실_10_10075496.json` | `느타리_생육실_10_10075496.jpg` | full_path | matched | `artifacts/sample_extract/train/TS1_느타리/생육/느타리_생육실_10_10075496.jpg` |
| train | 느타리 | `병해/느타리_생육실1_11_16100327.json` | `느타리_생육실1_11_16100327.jpg` | full_path | matched | `artifacts/sample_extract/train/TS1_느타리/병해/느타리_생육실1_11_16100327.jpg` |
| train | 느타리 | `배양/느타리_배양실_10_10697323.json` | `느타리_배양실_10_10697323.jpg` | full_path | matched | `artifacts/sample_extract/train/TS1_느타리/배양/느타리_배양실_10_10697323.jpg` |
| train | 느타리 | `생육/느타리_생육실_11_10143913.json` | `느타리_생육실_11_10143913.jpg` | full_path | matched | `artifacts/sample_extract/train/TS1_느타리/생육/느타리_생육실_11_10143913.jpg` |
| train | 느타리 | `병해/느타리_생육실1_13_15705294.json` | `느타리_생육실1_13_15705294.jpg` | full_path | matched | `artifacts/sample_extract/train/TS1_느타리/병해/느타리_생육실1_13_15705294.jpg` |
| train | 느타리 | `배양/느타리_배양실_1_10077743.json` | `느타리_배양실_1_10077743.jpg` | full_path | matched | `artifacts/sample_extract/train/TS1_느타리/배양/느타리_배양실_1_10077743.jpg` |
| train | 느타리 | `생육/느타리_생육실_12_10193852.json` | `느타리_생육실_12_10193852.jpg` | full_path | matched | `artifacts/sample_extract/train/TS1_느타리/생육/느타리_생육실_12_10193852.jpg` |
| train | 느타리 | `병해/느타리_생육실1_13_15713160.json` | `느타리_생육실1_13_15713160.jpg` | full_path | matched | `artifacts/sample_extract/train/TS1_느타리/병해/느타리_생육실1_13_15713160.jpg` |
| train | 느타리 | `배양/느타리_배양실_1_10697347.json` | `느타리_배양실_1_10697347.jpg` | full_path | matched | `artifacts/sample_extract/train/TS1_느타리/배양/느타리_배양실_1_10697347.jpg` |
| train | 느타리 | `생육/느타리_생육실_13_10192723.json` | `느타리_생육실_13_10192723.jpg` | full_path | matched | `artifacts/sample_extract/train/TS1_느타리/생육/느타리_생육실_13_10192723.jpg` |
| train | 느타리 | `병해/느타리_생육실1_13_16121627.json` | `느타리_생육실1_13_16121627.jpg` | full_path | matched | `artifacts/sample_extract/train/TS1_느타리/병해/느타리_생육실1_13_16121627.jpg` |
| train | 느타리 | `배양/느타리_배양실_2_10077753.json` | `느타리_배양실_2_10077753.jpg` | full_path | matched | `artifacts/sample_extract/train/TS1_느타리/배양/느타리_배양실_2_10077753.jpg` |
| train | 느타리 | `생육/느타리_생육실_14_10193961.json` | `느타리_생육실_14_10193961.jpg` | full_path | matched | `artifacts/sample_extract/train/TS1_느타리/생육/느타리_생육실_14_10193961.jpg` |
| train | 느타리 | `병해/느타리_생육실1_13_16172178.json` | `느타리_생육실1_13_16172178.jpg` | full_path | matched | `artifacts/sample_extract/train/TS1_느타리/병해/느타리_생육실1_13_16172178.jpg` |
| train | 느타리 | `배양/느타리_배양실_2_10697142.json` | `느타리_배양실_2_10697142.jpg` | full_path | matched | `artifacts/sample_extract/train/TS1_느타리/배양/느타리_배양실_2_10697142.jpg` |
| train | 느타리 | `생육/느타리_생육실_15_10694136.json` | `느타리_생육실_15_10694136.jpg` | full_path | matched | `artifacts/sample_extract/train/TS1_느타리/생육/느타리_생육실_15_10694136.jpg` |
| train | 느타리 | `병해/느타리_생육실1_14_15705792.json` | `느타리_생육실1_14_15705792.jpg` | full_path | matched | `artifacts/sample_extract/train/TS1_느타리/병해/느타리_생육실1_14_15705792.jpg` |
| train | 느타리 | `배양/느타리_배양실_3_10077204.json` | `느타리_배양실_3_10077204.jpg` | full_path | matched | `artifacts/sample_extract/train/TS1_느타리/배양/느타리_배양실_3_10077204.jpg` |
| train | 느타리 | `생육/느타리_생육실_16_12420248.json` | `느타리_생육실_16_12420248.jpg` | full_path | matched | `artifacts/sample_extract/train/TS1_느타리/생육/느타리_생육실_16_12420248.jpg` |
| train | 양송이 | `생육/양송이_생육실_10_10733411.json` | `양송이_생육실_10_10733411.jpg` | full_path | matched | `artifacts/sample_extract/train/TS2_양송이/생육/양송이_생육실_10_10733411.jpg` |
| train | 양송이 | `병해/양송이_생육실1_10_16117179.json` | `양송이_생육실1_10_16117179.jpg` | full_path | matched | `artifacts/sample_extract/train/TS2_양송이/병해/양송이_생육실1_10_16117179.jpg` |
| train | 양송이 | `생육/양송이_생육실_11_15641029.json` | `양송이_생육실_11_15641029.jpg` | full_path | matched | `artifacts/sample_extract/train/TS2_양송이/생육/양송이_생육실_11_15641029.jpg` |
| train | 양송이 | `병해/양송이_생육실1_10_18098974.json` | `양송이_생육실1_10_18098974.jpg` | full_path | matched | `artifacts/sample_extract/train/TS2_양송이/병해/양송이_생육실1_10_18098974.jpg` |
| train | 양송이 | `생육/양송이_생육실_13_10734819.json` | `양송이_생육실_13_10734819.jpg` | full_path | matched | `artifacts/sample_extract/train/TS2_양송이/생육/양송이_생육실_13_10734819.jpg` |
| train | 양송이 | `병해/양송이_생육실1_11_17185473.json` | `양송이_생육실1_11_17185473.jpg` | full_path | matched | `artifacts/sample_extract/train/TS2_양송이/병해/양송이_생육실1_11_17185473.jpg` |
| train | 양송이 | `생육/양송이_생육실_14_12497467.json` | `양송이_생육실_14_12497467.jpg` | full_path | matched | `artifacts/sample_extract/train/TS2_양송이/생육/양송이_생육실_14_12497467.jpg` |
| train | 양송이 | `병해/양송이_생육실1_12_17182874.json` | `양송이_생육실1_12_17182874.jpg` | full_path | matched | `artifacts/sample_extract/train/TS2_양송이/병해/양송이_생육실1_12_17182874.jpg` |
| train | 양송이 | `생육/양송이_생육실_15_16180101.json` | `양송이_생육실_15_16180101.jpg` | full_path | matched | `artifacts/sample_extract/train/TS2_양송이/생육/양송이_생육실_15_16180101.jpg` |
| train | 양송이 | `병해/양송이_생육실1_1_16350530.json` | `양송이_생육실1_1_16350530.jpg` | full_path | matched | `artifacts/sample_extract/train/TS2_양송이/병해/양송이_생육실1_1_16350530.jpg` |
| train | 양송이 | `생육/양송이_생육실_16_12499411.json` | `양송이_생육실_16_12499411.jpg` | full_path | matched | `artifacts/sample_extract/train/TS2_양송이/생육/양송이_생육실_16_12499411.jpg` |
| train | 양송이 | `병해/양송이_생육실1_1_17183900.json` | `양송이_생육실1_1_17183900.jpg` | full_path | matched | `artifacts/sample_extract/train/TS2_양송이/병해/양송이_생육실1_1_17183900.jpg` |
| train | 양송이 | `생육/양송이_생육실_17_12501486.json` | `양송이_생육실_17_12501486.jpg` | full_path | matched | `artifacts/sample_extract/train/TS2_양송이/생육/양송이_생육실_17_12501486.jpg` |
| train | 양송이 | `병해/양송이_생육실1_2_17072766.json` | `양송이_생육실1_2_17072766.jpg` | full_path | matched | `artifacts/sample_extract/train/TS2_양송이/병해/양송이_생육실1_2_17072766.jpg` |
| train | 양송이 | `생육/양송이_생육실_17_16181220.json` | `양송이_생육실_17_16181220.jpg` | full_path | matched | `artifacts/sample_extract/train/TS2_양송이/생육/양송이_생육실_17_16181220.jpg` |
| train | 양송이 | `병해/양송이_생육실1_2_17184986.json` | `양송이_생육실1_2_17184986.jpg` | full_path | matched | `artifacts/sample_extract/train/TS2_양송이/병해/양송이_생육실1_2_17184986.jpg` |
| train | 양송이 | `생육/양송이_생육실_1_12511007.json` | `양송이_생육실_1_12511007.jpg` | full_path | matched | `artifacts/sample_extract/train/TS2_양송이/생육/양송이_생육실_1_12511007.jpg` |
| train | 양송이 | `병해/양송이_생육실1_3_17073936.json` | `양송이_생육실1_3_17073936.jpg` | full_path | matched | `artifacts/sample_extract/train/TS2_양송이/병해/양송이_생육실1_3_17073936.jpg` |
| train | 양송이 | `생육/양송이_생육실_1_12513536.json` | `양송이_생육실_1_12513536.jpg` | full_path | matched | `artifacts/sample_extract/train/TS2_양송이/생육/양송이_생육실_1_12513536.jpg` |
| train | 양송이 | `병해/양송이_생육실1_3_18573868.json` | `양송이_생육실1_3_18573868.jpg` | full_path | matched | `artifacts/sample_extract/train/TS2_양송이/병해/양송이_생육실1_3_18573868.jpg` |
| train | 큰느타리 | `배양/큰느타리_배양실_10_10699338.json` | `큰느타리_배양실_10_10699338.jpg` | full_path | matched | `artifacts/sample_extract/train/TS3_큰느타리/배양/큰느타리_배양실_10_10699338.jpg` |
| train | 큰느타리 | `생육/큰느타리_생육실_10_10697761.json` | `큰느타리_생육실_10_10697761.jpg` | full_path | matched | `artifacts/sample_extract/train/TS3_큰느타리/생육/큰느타리_생육실_10_10697761.jpg` |
| train | 큰느타리 | `병해/큰느타리_생육실1_13_16174760.json` | `큰느타리_생육실1_13_16174760.jpg` | full_path | matched | `artifacts/sample_extract/train/TS3_큰느타리/병해/큰느타리_생육실1_13_16174760.jpg` |
| train | 큰느타리 | `배양/큰느타리_배양실_10_12588940.json` | `큰느타리_배양실_10_12588940.jpg` | full_path | matched | `artifacts/sample_extract/train/TS3_큰느타리/배양/큰느타리_배양실_10_12588940.jpg` |
| train | 큰느타리 | `생육/큰느타리_생육실_11_12407897.json` | `큰느타리_생육실_11_12407897.jpg` | full_path | matched | `artifacts/sample_extract/train/TS3_큰느타리/생육/큰느타리_생육실_11_12407897.jpg` |
| train | 큰느타리 | `병해/큰느타리_생육실1_13_17165035.json` | `큰느타리_생육실1_13_17165035.jpg` | full_path | matched | `artifacts/sample_extract/train/TS3_큰느타리/병해/큰느타리_생육실1_13_17165035.jpg` |
| train | 큰느타리 | `배양/큰느타리_배양실_1_10723982.json` | `큰느타리_배양실_1_10723982.jpg` | full_path | matched | `artifacts/sample_extract/train/TS3_큰느타리/배양/큰느타리_배양실_1_10723982.jpg` |
| train | 큰느타리 | `생육/큰느타리_생육실_11_14671270.json` | `큰느타리_생육실_11_14671270.jpg` | full_path | matched | `artifacts/sample_extract/train/TS3_큰느타리/생육/큰느타리_생육실_11_14671270.jpg` |
| train | 큰느타리 | `병해/큰느타리_생육실1_13_17187135.json` | `큰느타리_생육실1_13_17187135.jpg` | full_path | matched | `artifacts/sample_extract/train/TS3_큰느타리/병해/큰느타리_생육실1_13_17187135.jpg` |
| train | 큰느타리 | `배양/큰느타리_배양실_1_14648039.json` | `큰느타리_배양실_1_14648039.jpg` | full_path | matched | `artifacts/sample_extract/train/TS3_큰느타리/배양/큰느타리_배양실_1_14648039.jpg` |
| train | 큰느타리 | `생육/큰느타리_생육실_12_14673068.json` | `큰느타리_생육실_12_14673068.jpg` | full_path | matched | `artifacts/sample_extract/train/TS3_큰느타리/생육/큰느타리_생육실_12_14673068.jpg` |
| train | 큰느타리 | `병해/큰느타리_생육실1_14_16175180.json` | `큰느타리_생육실1_14_16175180.jpg` | full_path | matched | `artifacts/sample_extract/train/TS3_큰느타리/병해/큰느타리_생육실1_14_16175180.jpg` |
| train | 큰느타리 | `배양/큰느타리_배양실_2_10724475.json` | `큰느타리_배양실_2_10724475.jpg` | full_path | matched | `artifacts/sample_extract/train/TS3_큰느타리/배양/큰느타리_배양실_2_10724475.jpg` |
| train | 큰느타리 | `생육/큰느타리_생육실_13_14680681.json` | `큰느타리_생육실_13_14680681.jpg` | full_path | matched | `artifacts/sample_extract/train/TS3_큰느타리/생육/큰느타리_생육실_13_14680681.jpg` |
| train | 큰느타리 | `병해/큰느타리_생육실1_14_17169975.json` | `큰느타리_생육실1_14_17169975.jpg` | full_path | matched | `artifacts/sample_extract/train/TS3_큰느타리/병해/큰느타리_생육실1_14_17169975.jpg` |
| train | 큰느타리 | `배양/큰느타리_배양실_2_15637744.json` | `큰느타리_배양실_2_15637744.jpg` | full_path | matched | `artifacts/sample_extract/train/TS3_큰느타리/배양/큰느타리_배양실_2_15637744.jpg` |
| train | 큰느타리 | `생육/큰느타리_생육실_15_12431965.json` | `큰느타리_생육실_15_12431965.jpg` | full_path | matched | `artifacts/sample_extract/train/TS3_큰느타리/생육/큰느타리_생육실_15_12431965.jpg` |
| train | 큰느타리 | `병해/큰느타리_생육실1_14_17190205.json` | `큰느타리_생육실1_14_17190205.jpg` | full_path | matched | `artifacts/sample_extract/train/TS3_큰느타리/병해/큰느타리_생육실1_14_17190205.jpg` |
| train | 큰느타리 | `배양/큰느타리_배양실_3_12411135.json` | `큰느타리_배양실_3_12411135.jpg` | full_path | matched | `artifacts/sample_extract/train/TS3_큰느타리/배양/큰느타리_배양실_3_12411135.jpg` |
| train | 큰느타리 | `생육/큰느타리_생육실_16_14684805.json` | `큰느타리_생육실_16_14684805.jpg` | full_path | matched | `artifacts/sample_extract/train/TS3_큰느타리/생육/큰느타리_생육실_16_14684805.jpg` |
| train | 팽이 | `배양/팽이_배양실_10_10129563.json` | `팽이_배양실_10_10129563.jpg` | full_path | matched | `artifacts/sample_extract/train/TS4_팽이/배양/팽이_배양실_10_10129563.jpg` |
| train | 팽이 | `생육/팽이_생육실_10_10128415.json` | `팽이_생육실_10_10128415.jpg` | full_path | matched | `artifacts/sample_extract/train/TS4_팽이/생육/팽이_생육실_10_10128415.jpg` |
| train | 팽이 | `병해/팽이_생육실1_10_15714436.json` | `팽이_생육실1_10_15714436.jpg` | full_path | matched | `artifacts/sample_extract/train/TS4_팽이/병해/팽이_생육실1_10_15714436.jpg` |
| train | 팽이 | `배양/팽이_배양실_10_10729063.json` | `팽이_배양실_10_10729063.jpg` | full_path | matched | `artifacts/sample_extract/train/TS4_팽이/배양/팽이_배양실_10_10729063.jpg` |
| train | 팽이 | `생육/팽이_생육실_10_14648279.json` | `팽이_생육실_10_14648279.jpg` | full_path | matched | `artifacts/sample_extract/train/TS4_팽이/생육/팽이_생육실_10_14648279.jpg` |
| train | 팽이 | `병해/팽이_생육실1_1_15722078.json` | `팽이_생육실1_1_15722078.jpg` | full_path | matched | `artifacts/sample_extract/train/TS4_팽이/병해/팽이_생육실1_1_15722078.jpg` |
| train | 팽이 | `배양/팽이_배양실_1_10137459.json` | `팽이_배양실_1_10137459.jpg` | full_path | matched | `artifacts/sample_extract/train/TS4_팽이/배양/팽이_배양실_1_10137459.jpg` |
| train | 팽이 | `생육/팽이_생육실_11_12484339.json` | `팽이_생육실_11_12484339.jpg` | full_path | matched | `artifacts/sample_extract/train/TS4_팽이/생육/팽이_생육실_11_12484339.jpg` |
| train | 팽이 | `병해/팽이_생육실1_1_16115030.json` | `팽이_생육실1_1_16115030.jpg` | full_path | matched | `artifacts/sample_extract/train/TS4_팽이/병해/팽이_생육실1_1_16115030.jpg` |
| train | 팽이 | `배양/팽이_배양실_1_10701867.json` | `팽이_배양실_1_10701867.jpg` | full_path | matched | `artifacts/sample_extract/train/TS4_팽이/배양/팽이_배양실_1_10701867.jpg` |
| train | 팽이 | `생육/팽이_생육실_12_12485427.json` | `팽이_생육실_12_12485427.jpg` | full_path | matched | `artifacts/sample_extract/train/TS4_팽이/생육/팽이_생육실_12_12485427.jpg` |
| train | 팽이 | `병해/팽이_생육실1_1_16124148.json` | `팽이_생육실1_1_16124148.jpg` | full_path | matched | `artifacts/sample_extract/train/TS4_팽이/병해/팽이_생육실1_1_16124148.jpg` |
| train | 팽이 | `배양/팽이_배양실_1_17054124.json` | `팽이_배양실_1_17054124.jpg` | full_path | matched | `artifacts/sample_extract/train/TS4_팽이/배양/팽이_배양실_1_17054124.jpg` |
| train | 팽이 | `생육/팽이_생육실_13_12473793.json` | `팽이_생육실_13_12473793.jpg` | full_path | matched | `artifacts/sample_extract/train/TS4_팽이/생육/팽이_생육실_13_12473793.jpg` |
| train | 팽이 | `병해/팽이_생육실1_1_16126237.json` | `팽이_생육실1_1_16126237.jpg` | full_path | matched | `artifacts/sample_extract/train/TS4_팽이/병해/팽이_생육실1_1_16126237.jpg` |
| train | 팽이 | `배양/팽이_배양실_2_10198578.json` | `팽이_배양실_2_10198578.jpg` | full_path | matched | `artifacts/sample_extract/train/TS4_팽이/배양/팽이_배양실_2_10198578.jpg` |
| train | 팽이 | `생육/팽이_생육실_14_12474945.json` | `팽이_생육실_14_12474945.jpg` | full_path | matched | `artifacts/sample_extract/train/TS4_팽이/생육/팽이_생육실_14_12474945.jpg` |
| train | 팽이 | `병해/팽이_생육실1_2_15724703.json` | `팽이_생육실1_2_15724703.jpg` | full_path | matched | `artifacts/sample_extract/train/TS4_팽이/병해/팽이_생육실1_2_15724703.jpg` |
| train | 팽이 | `배양/팽이_배양실_2_15656525.json` | `팽이_배양실_2_15656525.jpg` | full_path | matched | `artifacts/sample_extract/train/TS4_팽이/배양/팽이_배양실_2_15656525.jpg` |
| train | 팽이 | `생육/팽이_생육실_15_12476143.json` | `팽이_생육실_15_12476143.jpg` | full_path | matched | `artifacts/sample_extract/train/TS4_팽이/생육/팽이_생육실_15_12476143.jpg` |
| train | 표고 | `배양/표고_배양실_10_15640925.json` | `표고_배양실_10_15640925.jpg` | full_path | matched | `artifacts/sample_extract/train/TS5_표고/배양/표고_배양실_10_15640925.jpg` |
| train | 표고 | `생육/표고_생육실_10_10139536.json` | `표고_생육실_10_10139536.jpg` | full_path | matched | `artifacts/sample_extract/train/TS5_표고/생육/표고_생육실_10_10139536.jpg` |
| train | 표고 | `병해/표고_생육실1_10_18574358.json` | `표고_생육실1_10_18574358.jpg` | full_path | matched | `artifacts/sample_extract/train/TS5_표고/병해/표고_생육실1_10_18574358.jpg` |
| train | 표고 | `배양/표고_배양실_10_17071056.json` | `표고_배양실_10_17071056.jpg` | full_path | matched | `artifacts/sample_extract/train/TS5_표고/배양/표고_배양실_10_17071056.jpg` |
| train | 표고 | `생육/표고_생육실_10_14652045.json` | `표고_생육실_10_14652045.jpg` | full_path | matched | `artifacts/sample_extract/train/TS5_표고/생육/표고_생육실_10_14652045.jpg` |
| train | 표고 | `병해/표고_생육실1_10_18586715.json` | `표고_생육실1_10_18586715.jpg` | full_path | matched | `artifacts/sample_extract/train/TS5_표고/병해/표고_생육실1_10_18586715.jpg` |
| train | 표고 | `배양/표고_배양실_1_15659698.json` | `표고_배양실_1_15659698.jpg` | full_path | matched | `artifacts/sample_extract/train/TS5_표고/배양/표고_배양실_1_15659698.jpg` |
| train | 표고 | `생육/표고_생육실_11_12536160.json` | `표고_생육실_11_12536160.jpg` | full_path | matched | `artifacts/sample_extract/train/TS5_표고/생육/표고_생육실_11_12536160.jpg` |
| train | 표고 | `병해/표고_생육실1_11_18585110.json` | `표고_생육실1_11_18585110.jpg` | full_path | matched | `artifacts/sample_extract/train/TS5_표고/병해/표고_생육실1_11_18585110.jpg` |
| train | 표고 | `배양/표고_배양실_1_17071327.json` | `표고_배양실_1_17071327.jpg` | full_path | matched | `artifacts/sample_extract/train/TS5_표고/배양/표고_배양실_1_17071327.jpg` |
| train | 표고 | `생육/표고_생육실_12_10729513.json` | `표고_생육실_12_10729513.jpg` | full_path | matched | `artifacts/sample_extract/train/TS5_표고/생육/표고_생육실_12_10729513.jpg` |
| train | 표고 | `병해/표고_생육실1_12_18582083.json` | `표고_생육실1_12_18582083.jpg` | full_path | matched | `artifacts/sample_extract/train/TS5_표고/병해/표고_생육실1_12_18582083.jpg` |
| train | 표고 | `배양/표고_배양실_2_15659736.json` | `표고_배양실_2_15659736.jpg` | full_path | matched | `artifacts/sample_extract/train/TS5_표고/배양/표고_배양실_2_15659736.jpg` |
| train | 표고 | `생육/표고_생육실_13_10704988.json` | `표고_생육실_13_10704988.jpg` | full_path | matched | `artifacts/sample_extract/train/TS5_표고/생육/표고_생육실_13_10704988.jpg` |
| train | 표고 | `병해/표고_생육실1_1_18576365.json` | `표고_생육실1_1_18576365.jpg` | full_path | matched | `artifacts/sample_extract/train/TS5_표고/병해/표고_생육실1_1_18576365.jpg` |
| train | 표고 | `배양/표고_배양실_2_17071592.json` | `표고_배양실_2_17071592.jpg` | full_path | matched | `artifacts/sample_extract/train/TS5_표고/배양/표고_배양실_2_17071592.jpg` |
| train | 표고 | `생육/표고_생육실_14_10616135.json` | `표고_생육실_14_10616135.jpg` | full_path | matched | `artifacts/sample_extract/train/TS5_표고/생육/표고_생육실_14_10616135.jpg` |
| train | 표고 | `병해/표고_생육실1_1_18583801.json` | `표고_생육실1_1_18583801.jpg` | full_path | matched | `artifacts/sample_extract/train/TS5_표고/병해/표고_생육실1_1_18583801.jpg` |
| train | 표고 | `배양/표고_배양실_3_16100140.json` | `표고_배양실_3_16100140.jpg` | full_path | matched | `artifacts/sample_extract/train/TS5_표고/배양/표고_배양실_3_16100140.jpg` |
| train | 표고 | `생육/표고_생육실_16_10147402.json` | `표고_생육실_16_10147402.jpg` | full_path | matched | `artifacts/sample_extract/train/TS5_표고/생육/표고_생육실_16_10147402.jpg` |
| validation | 느타리 | `배양/느타리_배양실_10_16098170.json` | `느타리_배양실_10_16098170.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS1_느타리/배양/느타리_배양실_10_16098170.jpg` |
| validation | 느타리 | `생육/느타리_생육실_10_12404827.json` | `느타리_생육실_10_12404827.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS1_느타리/생육/느타리_생육실_10_12404827.jpg` |
| validation | 느타리 | `병해/느타리_생육실1_11_16100358.json` | `느타리_생육실1_11_16100358.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS1_느타리/병해/느타리_생육실1_11_16100358.jpg` |
| validation | 느타리 | `배양/느타리_배양실_10_17069047.json` | `느타리_배양실_10_17069047.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS1_느타리/배양/느타리_배양실_10_17069047.jpg` |
| validation | 느타리 | `생육/느타리_생육실_10_12450837.json` | `느타리_생육실_10_12450837.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS1_느타리/생육/느타리_생육실_10_12450837.jpg` |
| validation | 느타리 | `병해/느타리_생육실1_13_15705285.json` | `느타리_생육실1_13_15705285.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS1_느타리/병해/느타리_생육실1_13_15705285.jpg` |
| validation | 느타리 | `배양/느타리_배양실_10_18572968.json` | `느타리_배양실_10_18572968.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS1_느타리/배양/느타리_배양실_10_18572968.jpg` |
| validation | 느타리 | `생육/느타리_생육실_11_12452685.json` | `느타리_생육실_11_12452685.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS1_느타리/생육/느타리_생육실_11_12452685.jpg` |
| validation | 느타리 | `병해/느타리_생육실1_13_15708251.json` | `느타리_생육실1_13_15708251.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS1_느타리/병해/느타리_생육실1_13_15708251.jpg` |
| validation | 느타리 | `배양/느타리_배양실_1_17068873.json` | `느타리_배양실_1_17068873.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS1_느타리/배양/느타리_배양실_1_17068873.jpg` |
| validation | 느타리 | `생육/느타리_생육실_14_12406015.json` | `느타리_생육실_14_12406015.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS1_느타리/생육/느타리_생육실_14_12406015.jpg` |
| validation | 느타리 | `병해/느타리_생육실1_13_15712853.json` | `느타리_생육실1_13_15712853.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS1_느타리/병해/느타리_생육실1_13_15712853.jpg` |
| validation | 느타리 | `배양/느타리_배양실_2_17046558.json` | `느타리_배양실_2_17046558.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS1_느타리/배양/느타리_배양실_2_17046558.jpg` |
| validation | 느타리 | `생육/느타리_생육실_1_12406467.json` | `느타리_생육실_1_12406467.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS1_느타리/생육/느타리_생육실_1_12406467.jpg` |
| validation | 느타리 | `병해/느타리_생육실1_14_15702889.json` | `느타리_생육실1_14_15702889.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS1_느타리/병해/느타리_생육실1_14_15702889.jpg` |
| validation | 느타리 | `배양/느타리_배양실_2_17069111.json` | `느타리_배양실_2_17069111.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS1_느타리/배양/느타리_배양실_2_17069111.jpg` |
| validation | 느타리 | `생육/느타리_생육실_1_12462522.json` | `느타리_생육실_1_12462522.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS1_느타리/생육/느타리_생육실_1_12462522.jpg` |
| validation | 느타리 | `병해/느타리_생육실1_14_15711061.json` | `느타리_생육실1_14_15711061.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS1_느타리/병해/느타리_생육실1_14_15711061.jpg` |
| validation | 느타리 | `배양/느타리_배양실_3_17052412.json` | `느타리_배양실_3_17052412.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS1_느타리/배양/느타리_배양실_3_17052412.jpg` |
| validation | 느타리 | `생육/느타리_생육실_20_12448414.json` | `느타리_생육실_20_12448414.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS1_느타리/생육/느타리_생육실_20_12448414.jpg` |
| validation | 양송이 | `생육/양송이_생육실_10_12504979.json` | `양송이_생육실_10_12504979.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS2_양송이/생육/양송이_생육실_10_12504979.jpg` |
| validation | 양송이 | `병해/양송이_생육실1_10_16117181.json` | `양송이_생육실1_10_16117181.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS2_양송이/병해/양송이_생육실1_10_16117181.jpg` |
| validation | 양송이 | `생육/양송이_생육실_11_17174445.json` | `양송이_생육실_11_17174445.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS2_양송이/생육/양송이_생육실_11_17174445.jpg` |
| validation | 양송이 | `병해/양송이_생육실1_1_17167800.json` | `양송이_생육실1_1_17167800.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS2_양송이/병해/양송이_생육실1_1_17167800.jpg` |
| validation | 양송이 | `생육/양송이_생육실_12_17174920.json` | `양송이_생육실_12_17174920.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS2_양송이/생육/양송이_생육실_12_17174920.jpg` |
| validation | 양송이 | `병해/양송이_생육실1_1_19374420.json` | `양송이_생육실1_1_19374420.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS2_양송이/병해/양송이_생육실1_1_19374420.jpg` |
| validation | 양송이 | `생육/양송이_생육실_13_17172692.json` | `양송이_생육실_13_17172692.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS2_양송이/생육/양송이_생육실_13_17172692.jpg` |
| validation | 양송이 | `병해/양송이_생육실1_1_19375821.json` | `양송이_생육실1_1_19375821.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS2_양송이/병해/양송이_생육실1_1_19375821.jpg` |
| validation | 양송이 | `생육/양송이_생육실_14_17172968.json` | `양송이_생육실_14_17172968.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS2_양송이/생육/양송이_생육실_14_17172968.jpg` |
| validation | 양송이 | `병해/양송이_생육실1_2_17168069.json` | `양송이_생육실1_2_17168069.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS2_양송이/병해/양송이_생육실1_2_17168069.jpg` |
| validation | 양송이 | `생육/양송이_생육실_16_12499664.json` | `양송이_생육실_16_12499664.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS2_양송이/생육/양송이_생육실_16_12499664.jpg` |
| validation | 양송이 | `병해/양송이_생육실1_2_19375166.json` | `양송이_생육실1_2_19375166.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS2_양송이/병해/양송이_생육실1_2_19375166.jpg` |
| validation | 양송이 | `생육/양송이_생육실_17_12502823.json` | `양송이_생육실_17_12502823.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS2_양송이/생육/양송이_생육실_17_12502823.jpg` |
| validation | 양송이 | `병해/양송이_생육실1_2_19376656.json` | `양송이_생육실1_2_19376656.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS2_양송이/병해/양송이_생육실1_2_19376656.jpg` |
| validation | 양송이 | `생육/양송이_생육실_17_17173464.json` | `양송이_생육실_17_17173464.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS2_양송이/생육/양송이_생육실_17_17173464.jpg` |
| validation | 양송이 | `병해/양송이_생육실1_3_19374570.json` | `양송이_생육실1_3_19374570.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS2_양송이/병해/양송이_생육실1_3_19374570.jpg` |
| validation | 양송이 | `생육/양송이_생육실_19_17173745.json` | `양송이_생육실_19_17173745.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS2_양송이/생육/양송이_생육실_19_17173745.jpg` |
| validation | 양송이 | `병해/양송이_생육실1_3_19374765.json` | `양송이_생육실1_3_19374765.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS2_양송이/병해/양송이_생육실1_3_19374765.jpg` |
| validation | 양송이 | `생육/양송이_생육실_1_12511627.json` | `양송이_생육실_1_12511627.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS2_양송이/생육/양송이_생육실_1_12511627.jpg` |
| validation | 양송이 | `병해/양송이_생육실1_3_19376171.json` | `양송이_생육실1_3_19376171.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS2_양송이/병해/양송이_생육실1_3_19376171.jpg` |
| validation | 큰느타리 | `배양/큰느타리_배양실_10_10723945.json` | `큰느타리_배양실_10_10723945.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS3_큰느타리/배양/큰느타리_배양실_10_10723945.jpg` |
| validation | 큰느타리 | `생육/큰느타리_생육실_10_10697770.json` | `큰느타리_생육실_10_10697770.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS3_큰느타리/생육/큰느타리_생육실_10_10697770.jpg` |
| validation | 큰느타리 | `병해/큰느타리_생육실1_13_17165008.json` | `큰느타리_생육실1_13_17165008.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS3_큰느타리/병해/큰느타리_생육실1_13_17165008.jpg` |
| validation | 큰느타리 | `배양/큰느타리_배양실_10_17058348.json` | `큰느타리_배양실_10_17058348.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS3_큰느타리/배양/큰느타리_배양실_10_17058348.jpg` |
| validation | 큰느타리 | `생육/큰느타리_생육실_13_14679619.json` | `큰느타리_생육실_13_14679619.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS3_큰느타리/생육/큰느타리_생육실_13_14679619.jpg` |
| validation | 큰느타리 | `병해/큰느타리_생육실1_13_17169913.json` | `큰느타리_생육실1_13_17169913.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS3_큰느타리/병해/큰느타리_생육실1_13_17169913.jpg` |
| validation | 큰느타리 | `배양/큰느타리_배양실_10_17070053.json` | `큰느타리_배양실_10_17070053.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS3_큰느타리/배양/큰느타리_배양실_10_17070053.jpg` |
| validation | 큰느타리 | `생육/큰느타리_생육실_13_14680878.json` | `큰느타리_생육실_13_14680878.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS3_큰느타리/생육/큰느타리_생육실_13_14680878.jpg` |
| validation | 큰느타리 | `병해/큰느타리_생육실1_13_17187263.json` | `큰느타리_생육실1_13_17187263.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS3_큰느타리/병해/큰느타리_생육실1_13_17187263.jpg` |
| validation | 큰느타리 | `배양/큰느타리_배양실_1_17058357.json` | `큰느타리_배양실_1_17058357.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS3_큰느타리/배양/큰느타리_배양실_1_17058357.jpg` |
| validation | 큰느타리 | `생육/큰느타리_생육실_14_14681812.json` | `큰느타리_생육실_14_14681812.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS3_큰느타리/생육/큰느타리_생육실_14_14681812.jpg` |
| validation | 큰느타리 | `병해/큰느타리_생육실1_14_17165076.json` | `큰느타리_생육실1_14_17165076.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS3_큰느타리/병해/큰느타리_생육실1_14_17165076.jpg` |
| validation | 큰느타리 | `배양/큰느타리_배양실_1_17069836.json` | `큰느타리_배양실_1_17069836.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS3_큰느타리/배양/큰느타리_배양실_1_17069836.jpg` |
| validation | 큰느타리 | `생육/큰느타리_생육실_15_14682803.json` | `큰느타리_생육실_15_14682803.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS3_큰느타리/생육/큰느타리_생육실_15_14682803.jpg` |
| validation | 큰느타리 | `병해/큰느타리_생육실1_14_17170241.json` | `큰느타리_생육실1_14_17170241.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS3_큰느타리/병해/큰느타리_생육실1_14_17170241.jpg` |
| validation | 큰느타리 | `배양/큰느타리_배양실_2_17058389.json` | `큰느타리_배양실_2_17058389.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS3_큰느타리/배양/큰느타리_배양실_2_17058389.jpg` |
| validation | 큰느타리 | `생육/큰느타리_생육실_15_14683331.json` | `큰느타리_생육실_15_14683331.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS3_큰느타리/생육/큰느타리_생육실_15_14683331.jpg` |
| validation | 큰느타리 | `병해/큰느타리_생육실1_14_17187556.json` | `큰느타리_생육실1_14_17187556.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS3_큰느타리/병해/큰느타리_생육실1_14_17187556.jpg` |
| validation | 큰느타리 | `배양/큰느타리_배양실_2_17069861.json` | `큰느타리_배양실_2_17069861.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS3_큰느타리/배양/큰느타리_배양실_2_17069861.jpg` |
| validation | 큰느타리 | `생육/큰느타리_생육실_16_14684335.json` | `큰느타리_생육실_16_14684335.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS3_큰느타리/생육/큰느타리_생육실_16_14684335.jpg` |
| validation | 팽이 | `배양/팽이_배양실_10_16099366.json` | `팽이_배양실_10_16099366.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS4_팽이/배양/팽이_배양실_10_16099366.jpg` |
| validation | 팽이 | `생육/팽이_생육실_10_12411769.json` | `팽이_생육실_10_12411769.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS4_팽이/생육/팽이_생육실_10_12411769.jpg` |
| validation | 팽이 | `병해/팽이_생육실1_10_15714540.json` | `팽이_생육실1_10_15714540.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS4_팽이/병해/팽이_생육실1_10_15714540.jpg` |
| validation | 팽이 | `배양/팽이_배양실_10_17070301.json` | `팽이_배양실_10_17070301.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS4_팽이/배양/팽이_배양실_10_17070301.jpg` |
| validation | 팽이 | `생육/팽이_생육실_13_12412016.json` | `팽이_생육실_13_12412016.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS4_팽이/생육/팽이_생육실_13_12412016.jpg` |
| validation | 팽이 | `병해/팽이_생육실1_1_15727020.json` | `팽이_생육실1_1_15727020.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS4_팽이/병해/팽이_생육실1_1_15727020.jpg` |
| validation | 팽이 | `배양/팽이_배양실_1_17047714.json` | `팽이_배양실_1_17047714.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS4_팽이/배양/팽이_배양실_1_17047714.jpg` |
| validation | 팽이 | `생육/팽이_생육실_15_12533901.json` | `팽이_생육실_15_12533901.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS4_팽이/생육/팽이_생육실_15_12533901.jpg` |
| validation | 팽이 | `병해/팽이_생육실1_1_15732269.json` | `팽이_생육실1_1_15732269.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS4_팽이/병해/팽이_생육실1_1_15732269.jpg` |
| validation | 팽이 | `배양/팽이_배양실_2_16099414.json` | `팽이_배양실_2_16099414.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS4_팽이/배양/팽이_배양실_2_16099414.jpg` |
| validation | 팽이 | `생육/팽이_생육실_18_12479719.json` | `팽이_생육실_18_12479719.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS4_팽이/생육/팽이_생육실_18_12479719.jpg` |
| validation | 팽이 | `병해/팽이_생육실1_2_15727217.json` | `팽이_생육실1_2_15727217.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS4_팽이/병해/팽이_생육실1_2_15727217.jpg` |
| validation | 팽이 | `배양/팽이_배양실_2_17070361.json` | `팽이_배양실_2_17070361.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS4_팽이/배양/팽이_배양실_2_17070361.jpg` |
| validation | 팽이 | `생육/팽이_생육실_1_12490682.json` | `팽이_생육실_1_12490682.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS4_팽이/생육/팽이_생육실_1_12490682.jpg` |
| validation | 팽이 | `병해/팽이_생육실1_2_15731237.json` | `팽이_생육실1_2_15731237.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS4_팽이/병해/팽이_생육실1_2_15731237.jpg` |
| validation | 팽이 | `배양/팽이_배양실_3_17047771.json` | `팽이_배양실_3_17047771.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS4_팽이/배양/팽이_배양실_3_17047771.jpg` |
| validation | 팽이 | `생육/팽이_생육실_1_12491366.json` | `팽이_생육실_1_12491366.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS4_팽이/생육/팽이_생육실_1_12491366.jpg` |
| validation | 팽이 | `병해/팽이_생육실1_3_15727303.json` | `팽이_생육실1_3_15727303.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS4_팽이/병해/팽이_생육실1_3_15727303.jpg` |
| validation | 팽이 | `배양/팽이_배양실_3_17070633.json` | `팽이_배양실_3_17070633.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS4_팽이/배양/팽이_배양실_3_17070633.jpg` |
| validation | 팽이 | `생육/팽이_생육실_2_12491575.json` | `팽이_생육실_2_12491575.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS4_팽이/생육/팽이_생육실_2_12491575.jpg` |
| validation | 표고 | `배양/표고_배양실_10_18098269.json` | `표고_배양실_10_18098269.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS5_표고/배양/표고_배양실_10_18098269.jpg` |
| validation | 표고 | `생육/표고_생육실_10_15639486.json` | `표고_생육실_10_15639486.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS5_표고/생육/표고_생육실_10_15639486.jpg` |
| validation | 표고 | `병해/표고_생육실1_10_18574371.json` | `표고_생육실1_10_18574371.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS5_표고/병해/표고_생육실1_10_18574371.jpg` |
| validation | 표고 | `배양/표고_배양실_10_18573304.json` | `표고_배양실_10_18573304.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS5_표고/배양/표고_배양실_10_18573304.jpg` |
| validation | 표고 | `생육/표고_생육실_10_15646296.json` | `표고_생육실_10_15646296.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS5_표고/생육/표고_생육실_10_15646296.jpg` |
| validation | 표고 | `병해/표고_생육실1_10_18582836.json` | `표고_생육실1_10_18582836.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS5_표고/병해/표고_생육실1_10_18582836.jpg` |
| validation | 표고 | `배양/표고_배양실_1_18098295.json` | `표고_배양실_1_18098295.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS5_표고/배양/표고_배양실_1_18098295.jpg` |
| validation | 표고 | `생육/표고_생육실_11_15646980.json` | `표고_생육실_11_15646980.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS5_표고/생육/표고_생육실_11_15646980.jpg` |
| validation | 표고 | `병해/표고_생육실1_11_18577357.json` | `표고_생육실1_11_18577357.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS5_표고/병해/표고_생육실1_11_18577357.jpg` |
| validation | 표고 | `배양/표고_배양실_1_18573336.json` | `표고_배양실_1_18573336.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS5_표고/배양/표고_배양실_1_18573336.jpg` |
| validation | 표고 | `생육/표고_생육실_11_15647859.json` | `표고_생육실_11_15647859.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS5_표고/생육/표고_생육실_11_15647859.jpg` |
| validation | 표고 | `병해/표고_생육실1_12_18575018.json` | `표고_생육실1_12_18575018.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS5_표고/병해/표고_생육실1_12_18575018.jpg` |
| validation | 표고 | `배양/표고_배양실_2_18573104.json` | `표고_배양실_2_18573104.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS5_표고/배양/표고_배양실_2_18573104.jpg` |
| validation | 표고 | `생육/표고_생육실_12_15648441.json` | `표고_생육실_12_15648441.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS5_표고/생육/표고_생육실_12_15648441.jpg` |
| validation | 표고 | `병해/표고_생육실1_12_18585455.json` | `표고_생육실1_12_18585455.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS5_표고/병해/표고_생육실1_12_18585455.jpg` |
| validation | 표고 | `배양/표고_배양실_2_18573352.json` | `표고_배양실_2_18573352.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS5_표고/배양/표고_배양실_2_18573352.jpg` |
| validation | 표고 | `생육/표고_생육실_13_15642230.json` | `표고_생육실_13_15642230.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS5_표고/생육/표고_생육실_13_15642230.jpg` |
| validation | 표고 | `병해/표고_생육실1_1_18583749.json` | `표고_생육실1_1_18583749.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS5_표고/병해/표고_생육실1_1_18583749.jpg` |
| validation | 표고 | `배양/표고_배양실_3_18098347.json` | `표고_배양실_3_18098347.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS5_표고/배양/표고_배양실_3_18098347.jpg` |
| validation | 표고 | `생육/표고_생육실_13_15643131.json` | `표고_생육실_13_15643131.jpg` | full_path | matched | `artifacts/sample_extract/validation/VS5_표고/생육/표고_생육실_13_15643131.jpg` |

이미지 파일은 ZIP 멤버를 바이트 단위로 복사했으며 디코딩하지 않았습니다.
