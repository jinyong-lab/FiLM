# Amortized Subspace for Few-Shot Anomaly Detection (support-span variant)

Meta-learned few-shot anomaly detection on **MVTec AD** and **MVTec LOCO AD**.
Given a few normal ("support") images of a new category, the model predicts that
category's **normal subspace** `(mu, U)` and scores a query image by how far its
patch features fall **outside** that subspace.

This repository packages the method that fixed our meta-learner's collapse:
**the global subspace anchor (`base_U`) is removed and the subspace is built from
the support features via attention**. With the data in place, everything below
runs end-to-end.

---

## Method

```
Model 1  frozen backbone (DINOv2 ViT-B/14)     ->  patch features  [P, D=768]
Model 3  SpanSubspacePredictor (this repo)      ->  (mu, U)  normal subspace of THIS task
Scorer   residual outside the subspace          ->  per-patch anomaly score
Aggregate  max (local) or mean (global)         ->  image-level score
```

**Why "support-span".** An earlier head predicted `U = base_U + dU(z)`. Training
collapsed to a single global subspace shared by every task (cross-task
principal-angle cosine = `1.0000`): the global anchor `base_U` was an easy
degenerate optimum, and the support was ignored. The fix
([`src/subspace.py`](src/subspace.py) → `SpanSubspacePredictor`) removes `base_U`
entirely and builds `U` only from the support features via attention
(`r` learnable basis queries attend over the support patches → QR). With no
global constant term, `U` is **structurally forced** to depend on the support, so
collapse is impossible. After the fix, cross-task cosine drops to ~`0.43` (real
adaptation) and image AUROC on MVTec recovers from `0.58` to `0.90`.

**Query stays passive.** The query is only *measured* against the predicted
subspace — it is never transformed by content-dependent parameters. This avoids
the "self-normalization" failure (an anomalous query rewriting itself to look
normal) that sank an earlier FiLM-style feature-modulation head.

**Baseline `A` = per-task PCA** (training-free, SubspaceAD-style). `A` and `B`
share the *same* scorer; the only difference is how `(mu, U)` is obtained — PCA
vs. the meta-learned head. This isolates the contribution of meta-learning.

---

## Install

```bash
pip install -r requirements.txt
# DINOv2 weights are fetched once via torch.hub on first run (needs internet once).
```

## Get the data

Download from MVTec (free, registration required) and unzip:

- **MVTec AD** — https://www.mvtec.com/company/research/datasets/mvtec-ad
- **MVTec LOCO AD** — https://www.mvtec.com/company/research/datasets/mvtec-loco

Expected layout (folder = label):

```
<DATA>/<category>/train/good/*.png            # normal only (support pool)
<DATA>/<category>/test/good/*.png             # normal test
<DATA>/<category>/test/<defect>/*.png         # anomalous test
# LOCO defect folders are exactly: logical_anomalies/  structural_anomalies/
```

## Run

```bash
# 1) cache frozen DINOv2 patch features (once per dataset)
python -m src.cache_features --data <MVTEC_LOCO_DIR> --dataset loco  --out feature_cache_loco_dino
python -m src.cache_features --data <MVTEC_DIR>      --dataset mvtec --out feature_cache_dino

# 2) meta-train (leave-one-category-out) + evaluate our model (B) vs PCA (A)
python -m src.meta_train --cache feature_cache_loco_dino --dataset loco    # logical vs structural crossover
python -m src.meta_train --cache feature_cache_dino      --dataset mvtec   # per-shot image AUROC

# quick end-to-end sanity check (1 category, 300 steps)
python -m src.cache_features --data <MVTEC_LOCO_DIR> --dataset loco --smoke --out smoke_cache
python -m src.meta_train --cache smoke_cache --dataset loco --smoke
```

Everything is CPU-runnable; a CUDA GPU is used automatically if available
(features were cached on an RTX 3060).

---

## Repo layout

| file | what it does |
|------|--------------|
| [`src/subspace.py`](src/subspace.py) | `SpanSubspacePredictor` (the method), `subspace_residual` (scorer), `direct_pca_subspace` (baseline A), `image_score` (max/mean aggregation) |
| [`src/cache_features.py`](src/cache_features.py) | frozen DINOv2 patch-feature extraction for MVTec / LOCO (`test_defect_type`: 0 good / 1 logical / 2 structural) |
| [`src/meta_train.py`](src/meta_train.py) | episodic leave-one-out meta-training + evaluation, A (PCA) vs B (B_span), logical/structural split, max/mean aggregation |

---

## Findings baked into the code (honest notes)

- **Collapse is fixed** — removing `base_U` restores per-task adaptation
  (cross-task cosine `1.0000 → 0.43`; MVTec AUROC `0.58 → 0.90`).
- **On MVTec, `B` ties per-task PCA** at every shot count (Δ within ±0.01). PCA
  already finds a near-optimal subspace, so learning the *subspace* does not beat
  it — MVTec is saturated for this family.
- **The real lever is the aggregation, not the subspace.** On LOCO **logical**
  anomalies, switching `max → mean` lifts AUROC by ~`+0.11` (0.59 → 0.70) for
  *both* A and B. Logical anomalies perturb many patches slightly, so the
  worst-single-patch `max` misses them. `B` still ties `A` here — the next step
  is to move the meta-learning from the subspace to a **learned, task-adaptive
  aggregation** (WIP).
- Absolute LOCO numbers (~0.6–0.7) are a **mechanism study**, below purpose-built
  LOCO SOTA. `image_score(..., agg="mean")` is the recommended aggregation for
  logical-heavy data.

Backbone note: cached results use **DINOv2 ViT-B/14** (`--backbone dinov2`).
`--backbone dinov2_giant` (ViT-g/14) is also supported.
