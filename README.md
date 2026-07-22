# FiLM 기반 비지도 Few-shot 이상탐지

**자기지도 표현학습 + 태스크 적응형 메타학습(FiLM)을 통한 Unsupervised Few-shot Anomaly Detection**
· 데이터셋: MVTec AD · 프로토콜: leave-5-out (학습 10종 / 테스트 5종, 한 번도 안 본 카테고리)

> 본 저장소의 모든 성능 수치는 **직접 측정한 값**입니다. (제안서 본문의 베이스라인 비교표 89.4/93.2/91.8은 근거 없는 값이라 사용하지 않았습니다.)

---

## 1. 연구 아이디어 — 제안서의 4개 모델

정상 데이터만으로, 소수(1~5장)의 정상 참조만 주어진 **처음 보는 카테고리**의 이상을 탐지·국소화한다.
핵심은 "학습 10종을 외우는" 것이 아니라 **"적응하는 법을 배우는"** 메타학습이다.

| 모델 | 역할 | 구현 |
|---|---|---|
| **Model 1 — 국소 자기지도 백본** | 정상 이미지에서 결함을 **스스로 합성(CutPaste)** 해 국소 패치 단위로 정상/이상을 구분하는 표현을 학습 | `ssl/ssl_and_extract.py` (CutPaste 다중분류, 제안서 충실판)<br>`ssl/defect_ssl.py` (결함민감 dense-SSL, 학습형 최고) |
| **Model 2 — 태스크 인코더** | support(정상 K장)의 채널별 통계(mean+var, 순열불변) → MLP → 태스크 벡터 `z` | `src/film_ad.py:TaskEncoder` |
| **Model 3 — 태스크 적응 FiLM** | `z`로 특징을 채널 변조: `Adapt(H) = γ(z)·H + β(z)` | `src/film_ad.py:FiLM`, `src/film_ad_fixed.py`(특징보존형) |
| **Model 4 — 패치 메모리 정렬** | 적응된 특징에서 support 메모리뱅크에 대한 패치별 최근접 거리로 채점 | `src/film_ad.py:patch_scores` |

**흐름:** 이미지 → (Model1 백본) 28×28×1536 특징 → (Model2) `z` → (Model3 FiLM) 적응 → (Model4) 패치 최근접 거리 → Image/Pixel 이상점수.

---

## 2. 측정 결과 (전부 자체 측정)

### 2.1 Model 1 (백본) 옵션별 — 테스트 5종, patch-NN 5-shot Image-AUROC
| 백본 학습 방식 | Image-AUROC | 비고 |
|---|---|---|
| ImageNet 동결 (WRN-50-2) | **0.947** | 학습 안 함(상한 참조용) |
| 결함민감 dense-SSL (`defect_ssl.py`) | **0.929** | 학습형 중 최고 |
| CutPaste 다중분류 (`ssl_and_extract.py`) | 0.926 | 제안서 충실판 |
| dense contrastive (`dense_contrastive_ssl.py`) | 0.917 | ablation |
| pooled contrastive (`contrastive_ssl.py`) | 0.832 | ⚠️ 표현 붕괴(음성 결과) |

### 2.2 제안서-충실 전체 파이프라인 (Model 1~4, 결함SSL + 특징보존·정규화 FiLM) — 테스트 5종
| shot | Image-AUROC | Pixel-AUROC |
|---|---|---|
| 1-shot | 0.830 | 0.851 |
| **5-shot** | **0.918** | 0.871 |

### 2.3 FiLM 학습의 핵심 발견 — 특화억제 정규화
FiLM은 용량이 커서 **학습 10종에 과적합(암기)** 하는 경향이 있다 (FiLM 원논문 CoGenT의 "조합 암기"와 동일 현상).
입력특징에서 벗어난 정도에 벌점을 주는 **특화억제 정규화**를 넣자, 학습된 FiLM이 **처음으로 무학습을 상회**했다.

| split | 무학습(항등) | 최적 λ | 학습 후 정점(held-out 1-shot) |
|---|---|---|---|
| 10/5 | 0.855 | λ=0.5 | **0.885 (+0.030)** |
| 12/3 | — | λ=2.0 | +0.011 (안정) |

→ `src/film_ad_meta.py` (정규화 강도 λ 스윕)

### 2.4 베이스라인 3-way 비교 (전부 자체 측정, 동일 5종 held-out)
**Image-AUROC (탐지)**
| 모델 | 프로토콜 | K=1 | K=2 | K=5 |
|---|---|---|---|---|
| **본 제안 (충실판)** | leave-5-out (학습 10종) | 0.830 | — | **0.918** |
| WinCLIP+ | 동일 프로토콜(같은 5종·시드) | 0.823 | — | 0.852 |
| RegAD (per_domain) | leave-1-out (학습 14종) | — | 0.798 | — |
| RegAD (joint) | leave-1-out (학습 14종) | — | 0.830 | — |

*WinCLIP zero-shot = 0.797.*

**Pixel-AUROC (국소화):** 본 제안 0.871 · RegAD 0.937~0.944

**요약:**
- **이상 "탐지"(Image-AUROC)** — 본 제안이 WinCLIP·RegAD를 **모두 상회**. K가 커질수록 격차 확대(K=5 0.918). RegAD가 오히려 더 많은 학습 카테고리(14종)를 쓰는데도 우세.
- **이상 "국소화"(Pixel-AUROC)** — 등록(registration) 기반인 RegAD가 우세. 향후 과제.

---

## 3. 주요 발견 · 한계 (정직)
1. **에피소드를 늘려도 성능은 오르지 않는다.** 붕괴는 특징보존 설계(conv2 zero-init → 초기 항등, ‖ml(x)−x‖≈3e-8)로 막았으나 정점은 초반에 도달하고 이후 진동. 병목은 계산량/에피소드가 아니라 **태스크 다양성**이며 MVTec은 학습 카테고리가 10~12종으로 제한됨(태스크-수 스케일링 2→10종: 0.723→0.801).
2. **무학습(동결 ImageNet patch-NN, 0.947)이 강력하다.** few-shot·소수 태스크 환경에서 메타학습이 이를 크게 넘기 어려움 — 구현 결함이 아니라 구조적 상한. 그럼에도 §2.4처럼 학습형 전체 파이프라인이 동종 베이스라인(WinCLIP·RegAD)은 탐지에서 상회.
3. **FiLM 원논문과 정합.** §5 참조.

---

## 4. 저장소 구조 · 실행
```
src/        FiLM 메타학습 파이프라인 (Model 2·3·4) + 실험 — 서로 import하므로 한 폴더
  film_ad.py            핵심: TaskEncoder / FiLM / patch-NN + 평가 + 데이터 로더
  film_ad_fixed.py      Model 3 특징보존형 FiLM (메인 파이프라인)
  film_ad_meta.py       특화억제 정규화 강도(λ) 스윕 (핵심 결과)
  film_ad_full.py       FiLM 원논문 정본 ResBlock (1×1→3×3→BN끄기→FiLM→ReLU+잔차 ×4)
  film_ad_real.py       실제 결함 query로 메타학습하는 변형
  film_12_3.py          12/3 split 실험
  film_task_scaling.py  학습 태스크 수 스케일링 실험
  ssl_cause.py          SSL 표현 붕괴 원인 진단
ssl/        Model 1 (자기지도 백본 학습) — 독립 실행
  ssl_and_extract.py    CutPaste 다중분류 (제안서 충실판)
  defect_ssl.py         결함민감 dense-SSL (학습형 최고)
  dense_contrastive_ssl.py / contrastive_ssl.py   ablation
baselines/
  winclip.py            WinCLIP / WinCLIP+ (CLIP ViT-B/16) 자체 측정
viz/
  viz_detection.py      탐지 시각화(피처맵 PCA-RGB + patch-NN 히트맵)
```

**실행 (백본 특징 캐시가 준비된 환경 기준):**
```bash
# 1) Model 1: 자기지도 백본 학습 → 28×28×1536 특징 캐시 생성
python ssl/defect_ssl.py            # 또는 ssl/ssl_and_extract.py (CutPaste 충실판)

# 2) Model 2·3·4: 메타학습 + 평가 (테스트 5종)
cd src
FILMAD_CACHE=feature_cache_wrn_defect REG_LAM=0.5 AUG_STD=0.1 python film_ad_fixed.py  # 전체 파이프라인
FILMAD_CACHE=feature_cache_wrn_defect python film_ad_meta.py                            # 정규화 λ 스윕

# 3) 베이스라인
python baselines/winclip.py
```
> 무거운 백본 특징추출은 **1회만** 수행해 디스크에 캐시(수 GB)하며, 메타학습(수만 스텝)은 이 경량 특징 위에서 초경량 어댑터(≈8.5M 파라미터, 백본 68.9M은 동결·미포함)만 갱신하므로 수 분 내 수렴한다.

---

## 5. FiLM 원논문(Perez et al., 2018) 대조
| 항목 | 원논문 | 본 구현 |
|---|---|---|
| FiLM 공식 | `γ·F + β` (Eq 2) | 동일 |
| generator | affine projection | `Linear(z)→γ,β` |
| ResBlock | 1×1 → 3×3, BN(직전) 끄기, ×4, 잔차 | `film_ad_full.py` 동일 |
| γ 초기화 | `1+Δγ` | `g.bias=1` → γ=1 |
| 좌표맵 | concat | 생략 (patch-NN이 공간 담당) |
| 특징보존 zero-init | (없음) | few-shot 과적합 대응으로 `conv2` zero-init 추가(의도적) |

원논문은 **"FiLM은 용량이 커 정규화가 필요"**, **"CoGenT에서 조합을 암기"** 라고 명시한다 — 본 연구의 과적합 진단과 특화억제 정규화가 원저자 관찰과 정확히 부합한다.

---

## 데이터 · 캐시
MVTec AD (15종). 학습 10종 / 테스트 5종(`wood, cable, zipper, transistor, hazelnut`).
데이터·특징 캐시 경로는 각 스크립트 상단 상수(`REPO`, `DATA`, `FILMAD_CACHE`)로 지정. 특징 캐시(.pt, 수 GB)는 저장소에서 제외한다.
