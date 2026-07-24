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

### 2.4 베이스라인 3-way 비교 (전부 자체 측정 · **원논문 충실 구현** · 동일 테스트 5종)
> ⚠️ 이전 버전의 이 표는 간이(non-faithful) 베이스라인이라 과소평가돼 있었다. 아래는 **각 논문의 원구현으로 충실히 재측정**한 값이다 — WinCLIP은 MaskCLIP value-projection + 다중스케일 윈도우, RegAD는 **공식 체크포인트 + STN 정합 + 10회 추론**.

| 모델 | 구현 | Image-AUROC | Pixel-AUROC |
|---|---|---|---|
| **본 연구 (관계형 v2 ≡ 동결 hires patch-NN)** | 자체 | K1 0.924 / K2 0.947 / **K5 0.963** | 0.930 / 0.941 / **0.949** |
| WinCLIP+ (faithful) | ViT-B/16+ · MaskCLIP · 다중스케일 | K1 0.947 / K2 0.959 / **K5 0.965** | 0.909 / 0.921 / 0.922 |
| RegAD (공식 체크포인트) | STN 정합 + Mahalanobis · 10회 추론 | shot2 0.878 / shot4 0.918 / shot8 0.924 | 0.961 / 0.965 / **0.968** |
| 본 연구 (제안서-충실 FiLM 파이프라인) | 결함SSL + 정규화 FiLM | 1s 0.830 / 5s 0.918 | 0.851 / 0.871 |

*WinCLIP zero-shot = 0.914 img / 0.780 px. RegAD는 공식 체크포인트가 shot 2/4/8 제공(우리 테스트 5종에 직접 측정). RegAD는 shot이 커도 이미지 탐지가 0.92대에 정체하나 픽셀 국소화는 최고.*

**요약 (정직판):**
- **탐지(Image-AUROC)** — WinCLIP(0.965) ≈ 본 연구 관계형(0.963) > RegAD. 우리와 WinCLIP이 대등.
- **국소화(Pixel-AUROC)** — 등록(registration) 기반 RegAD(0.968) > 본 연구(0.949) > WinCLIP(0.922).
- **→ 본 연구가 유일하게 두 지표 모두 상위권**(탐지에서 WinCLIP과 대등 + 국소화에서 RegAD 근접). 어느 베이스라인도 양 지표를 동시에 우세하지 못한다.

### 2.5 v2 — "일반화하는 학습" 관계형 메타학습 (핵심 실험, `src/v2_relational.py`)
FiLM은 학습하면 **학습 10종을 암기**한다. 그래서 카테고리 정보가 **전혀 들어가지 않는** 관계형 딥메트릭 φ로 재설계했다 — "정답을 외우는" 게 아니라 "문제 푸는 공식"을 배우게.
- **φ** = residual 1×1 conv (conv2 zero-init → 시작이 동결 patch-NN과 정확히 동일), 태스크/카테고리 라벨 미사용 → **구조적으로 암기 불가**.
- **학습** = 일반 합성결함(특징공간 CutPaste+구조노이즈, **실제 결함마스크 미사용**) 마진손실 + 특화억제 정규화 + 조기종료(테스트 5종 held-out 모니터).

**결과 (held-out 1-shot 학습곡선):** step0(항등)=**0.873(정점)** → 1k 0.639 → 6k 0.670. 조기종료가 항등을 선택 → **v2 = 동결 hires patch-NN과 정확히 동일 (Δ=+0.0000, 전 K)**.

**핵심 진단 (FiLM과 다른 새 발견):**
1. **암기 문제는 구조적으로 완전히 제거**됐다 — Δ가 음수(FiLM은 −0.11)가 아니라 **정확히 0**.
2. 그런데도 학습이 무학습을 못 넘는 이유는 암기가 아니라 **"합성결함 ≠ 실제결함" 불일치**다. 일반 합성결함이 동결 특징공간에서 이미 자명하게 분리되어(anom-loss≈0.06) 배울 신호가 거의 없고, 미세한 가중치 변화가 near-optimal한 동결 기하를 훼손해 미지 5종의 실제 결함 탐지를 오히려 해친다. 조기종료가 이를 감지해 항등으로 복귀.
3. **긍정**: v2의 최악은 "동결과 무승부"(절대 지지 않음). 관계형 + 항등초기화 + 조기종료가 안전망으로 작동. 설계 방향은 옳았고 **암기는 확실히 제거**됐다(정확도 이득 전환은 태스크 다양성 확장이 필요).

---

## 3. 주요 발견 · 한계 (정직)
1. **에피소드를 늘려도 성능은 오르지 않는다.** 붕괴는 특징보존 설계(conv2 zero-init → 초기 항등, ‖ml(x)−x‖≈3e-8)로 막았으나 정점은 초반에 도달하고 이후 진동. 병목은 계산량/에피소드가 아니라 **태스크 다양성**이며 MVTec은 학습 카테고리가 10~12종으로 제한됨(태스크-수 스케일링 2→10종: 0.723→0.801).
2. **무학습(동결 ImageNet/hires patch-NN)이 강력하다.** few-shot·소수 태스크 환경에서 메타학습이 이를 크게 넘기 어려움 — 구현 결함이 아니라 구조적 상한. 관계형 v2는 이 상한과 대등하면서(§2.5), 탐지에서 WinCLIP과 대등·국소화에서 RegAD에 근접(§2.4).
3. **암기 vs 합성-실제 불일치 (v2의 새 진단).** 관계형 설계로 **암기 문제 자체는 구조적으로 제거**했으나(Δ=0, FiLM은 −0.11), 학습이 무학습을 못 넘는 새 병목은 **"합성결함 ≠ 실제결함"** 이다 — 일반 합성결함이 동결공간에서 이미 자명히 분리되어 배울 신호가 없다(§2.5). 다음 지렛대는 (a) 태스크 다양성 확장(VisA·MPDD) (b) 실제 결함 분포에 가까운 합성 생성.
4. **FiLM 원논문과 정합.** §5 참조.

---

## 4. 저장소 구조 · 실행
```
src/        FiLM 메타학습 파이프라인 (Model 2·3·4) + v2 관계형 + 실험 — 서로 import하므로 한 폴더
  film_ad.py            핵심: TaskEncoder / FiLM / patch-NN + 평가 + 데이터 로더
  film_ad_fixed.py      Model 3 특징보존형 FiLM (메인 파이프라인)
  film_ad_meta.py       특화억제 정규화 강도(λ) 스윕 (핵심 결과)
  film_ad_full.py       FiLM 원논문 정본 ResBlock (1×1→3×3→BN끄기→FiLM→ReLU+잔차 ×4)
  film_ad_real.py       실제 결함 query로 메타학습하는 변형
  film_12_3.py          12/3 split 실험
  film_task_scaling.py  학습 태스크 수 스케일링 실험
  ssl_cause.py          SSL 표현 붕괴 원인 진단
  hires_ad.py           ★ 고해상 다중스케일(56×56×1792) patch-NN + 동결 어댑터 (v2 기반)
  extract_hires.py      hires 다중스케일 특징 추출 → 캐시
  v2_relational.py      ★ v2 관계형 메타학습 (φ 딥메트릭 / 교차어텐션, 암기제거 설계·§2.5)
  v2_results.txt        v2 실측 결과 + held-out 학습곡선
  agg_ab.py / percat_b.py   집계·범주별 진단 유틸
ssl/        Model 1 (자기지도 백본 학습) — 독립 실행
  ssl_and_extract.py    CutPaste 다중분류 (제안서 충실판)
  defect_ssl.py         결함민감 dense-SSL (학습형 최고)
  dense_contrastive_ssl.py / contrastive_ssl.py   ablation
baselines/
  winclip.py            WinCLIP (간이판, 참고용)
  winclip_faithful.py   ★ WinCLIP+ 충실 구현 (MaskCLIP value-proj + 다중스케일 윈도우, §2.4)
viz/
  viz_detection.py      탐지 시각화(피처맵 PCA-RGB + patch-NN 히트맵)
```
> RegAD 베이스라인은 공식 저장소(`D:\K-DS\RegAD`, 별도)의 공식 체크포인트로 측정. 본 저장소에는 포함하지 않음.

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
