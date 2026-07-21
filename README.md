# FiLM Few-shot Anomaly Detection (leave-5-out on MVTec AD)

제안서(*Local-preserving Self-supervised Representation + Task-aware Meta-learning*)의
4개 모형을 충실히 구현하고, 설계 감사에서 찾은 결함을 수정한 뒤 **직접 측정한** 결과를 정리한다.
모든 수치는 우리 실측이며, 제안서 초안의 결과표(89.4/93.2/91.8)는 근거 없는 값이므로 사용하지 않는다.

## 프로토콜
- MVTec AD 15종 → **학습 10종 / 테스트 5종(wood·cable·zipper·transistor·hazelnut)** leave-5-out.
- 백본 WRN-50-2, layer2+layer3 (28×28×1536). Image-AUROC, 1/5-shot, support=정상만.

## 4개 모형
- **모형1 Local Feature Extractor (SSL)** — `defect_ssl.py`(결함민감형, 최종), `dense_contrastive_ssl.py`, `contrastive_ssl.py`, `ssl_and_extract.py`(CutPaste)
- **모형2 Task Encoder** — 순열불변 채널 mean+var → z (`film_ad_full.py`, `film_ad_fixed.py`)
- **모형3 Task-Aware Learner (FiLM)** — `film_ad_full.py`(정본), `film_ad_fixed.py`(특징보존형 수정)
- **모형4 Anomaly Scoring** — 패치 프로토타입 최근접(PatchCore식)

## 설계 감사에서 찾은 결함과 수정
1. **모형1: contrastive 불변성 ≠ 결함 민감성.** 국소교란(노이즈·erase)에 불변 학습 →
   결함이 곧 국소교란이라 결함에 둔감(유효랭크 719→397 붕괴). **수정(`defect_ssl.py`)**:
   nuisance 불변 + 합성결함 위치는 '멀게'(민감)·비결함은 '가깝게' → 유효랭크 662 회복, 0.929.
2. **모형3: `proj(1536→512)` 랜덤초기화**로 좋은 입력특징을 버리고 재학습 → 과적합·붕괴.
   **수정(`film_ad_fixed.py`)**: residual + `conv2` zero-init → 초기 항등(출력=입력특징,
   검증 ‖ml(x)−x‖≈3e-8), 붕괴 방지.

## 측정 결과 (held-out 5종, Image-AUROC 5-shot)
| 구성 | 5-shot | 비고 |
|---|---|---|
| ImageNet(동결) + 패치-NN | 0.947 | 참고(동결) |
| **결함민감 SSL + 패치-NN (모형1 수정)** | **0.929** | 비동결 최선 |
| CutPaste + 패치-NN | 0.926 | |
| dense 대조 + 패치-NN | 0.917 | |
| 결함민감 SSL + 특징보존 FiLM (완전체 수정) | 0.776 | 항등 0.855서 하락 |
| (수정 전) 완전체 | 0.651 | 붕괴 |

## 결론
- **모형2·4는 충실·정상.** 모형1 수정(결함민감 SSL)은 학습형 SSL 최고(0.929)를 달성.
- **모형3 FiLM 메타학습은 held-out에 도움이 안 됨**: 특징보존으로 붕괴는 막았으나, 학습 시작 즉시
  항등(입력특징)에서 아래로 이동(0.855→0.747). 학습 태스크가 10종뿐이라 특화가 근본 원인.
- **비동결 최선 = 결함민감 SSL + 패치-NN(모형4), FiLM 미학습 = 0.929.** 태스크-수 스케일링
  실험(2→10종: 0.723→0.801)상 데이터셋 확장(더 많은 태스크)이 남은 유일한 지렛대.

## 실행
```bash
python defect_ssl.py                                    # 모형1(결함민감 SSL) 학습·추출·평가
FILMAD_CACHE=feature_cache_wrn_defect python film_ad_fixed.py   # 모형1수정+모형3수정 완전체
python film_task_scaling.py                             # 태스크-수 스케일링 진단
```
특징 캐시(.pt)는 용량이 커 저장소에서 제외한다(`cache_features.py`로 재생성).
