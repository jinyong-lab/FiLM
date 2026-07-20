"""Step 2: meta-train the support-span subspace predictor and evaluate it
against the training-free per-task PCA baseline.

Protocol (leave-one-category-out):
  - Meta-train on all categories EXCEPT the target (episodic: each step samples a
    training category, builds U from a K-shot support, pushes same-category patches
    INTO the subspace and other-category patches OUT via a margin loss).
  - Evaluate on the held-out target: draw K normal support images, build (mu, U)
    with A=PCA and B=B_span, score every test image, report image-level AUROC.
  - Rotate the target over all categories.

For LOCO, AUROC is reported separately for logical vs structural anomalies (the
crossover), under both max and mean aggregation.

Usage:
  python -m src.meta_train --cache feature_cache_dino      --dataset mvtec
  python -m src.meta_train --cache feature_cache_loco_dino --dataset loco
  python -m src.meta_train --cache feature_cache_loco_dino --dataset loco --smoke
"""
import os
import sys
import json
import random
import argparse
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.subspace import (SpanSubspacePredictor, subspace_residual, direct_pca_subspace,
                          img_patches, l2n)

SHOTS = (1, 2, 4, 8)


def load_cache(cache_dir):
    cats = {}
    for f in sorted(os.listdir(cache_dir)):
        if f.endswith(".pt"):
            cats[f[:-3]] = torch.load(os.path.join(cache_dir, f), map_location="cpu", weights_only=False)
    return cats


def meta_train(cats, target, device, steps=2000, r=32, margin=0.5, lam=1.0, lr=1e-4, seed=0):
    """Episodic meta-training of B_span on every category except `target`."""
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    train_cats = [c for c in cats if c != target]
    D = cats[target]["train_feats"].shape[-1]
    sub = SpanSubspacePredictor(feat_dim=D, rank=r).to(device)
    opt = torch.optim.AdamW(sub.parameters(), lr=lr)
    for _ in range(steps):
        t = random.choice(train_cats); ft = cats[t]["train_feats"]; n = ft.shape[0]
        K = random.choice(SHOTS)
        if n < K + 2:
            continue
        perm = torch.randperm(n)
        supp = img_patches(ft, perm[:K]).to(device)
        pos = img_patches(ft, perm[K:K + 2]).to(device)
        t2 = random.choice([c for c in train_cats if c != t]); ft2 = cats[t2]["train_feats"]
        neg = img_patches(ft2, torch.randperm(ft2.shape[0])[:2]).to(device)
        mu, U = sub(supp)
        loss = (subspace_residual(pos, mu, U).mean()
                + lam * torch.relu(margin - subspace_residual(neg, mu, U)).mean())
        opt.zero_grad(); loss.backward(); opt.step()
    sub.eval()
    return sub


@torch.no_grad()
def scores(qf, mu, U, device):
    r = subspace_residual(l2n(qf.float().to(device)), mu, U)
    return r.max().item(), r.mean().item()   # (max-agg, mean-agg)


@torch.no_grad()
def evaluate(cats, target, sub, device, r=32, n_draws=10, loco=False):
    d = cats[target]; pool = d["train_feats"]; test = d["test_feats"]
    lab = d["test_img_label"].numpy()
    dt = d["test_defect_type"].numpy()
    idx_log = np.where(dt == 1)[0]; idx_str = np.where(dt == 2)[0]; idx_good = np.where(dt == 0)[0]

    out = {}
    for K in SHOTS:
        acc = {m: {a: {"all": [], "logic": [], "struct": []} for a in ["max", "mean"]}
               for m in ["A", "B"]}
        for draw in range(n_draws):
            g = torch.Generator().manual_seed(1000 + 7 * K + draw)
            idx = torch.randperm(pool.shape[0], generator=g)[:K]
            supp = img_patches(pool, idx).to(device)
            muA, UA = direct_pca_subspace(supp, rank=r)
            muB, UB = sub(supp)
            for key, (mu, U) in [("A", (muA, UA)), ("B", (muB, UB))]:
                s = np.array([scores(test[j], mu, U, device) for j in range(len(test))])  # [N,2]
                for ai, a in enumerate(["max", "mean"]):
                    acc[key][a]["all"].append(roc_auc_score(lab, s[:, ai]))
                    if loco and len(idx_log) and len(idx_str):
                        sg = s[idx_good, ai]
                        acc[key][a]["logic"].append(
                            roc_auc_score([0]*len(sg)+[1]*len(idx_log), list(sg)+list(s[idx_log, ai])))
                        acc[key][a]["struct"].append(
                            roc_auc_score([0]*len(sg)+[1]*len(idx_str), list(sg)+list(s[idx_str, ai])))
        out[K] = {key: {a: {g: (float(np.mean(v)) if v else None) for g, v in acc[key][a].items()}
                        for a in ["max", "mean"]} for key in ["A", "B"]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--dataset", choices=["mvtec", "loco"], default="loco")
    ap.add_argument("--out", default=None)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--draws", type=int, default=10)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loco = args.dataset == "loco"
    out_dir = args.out or (args.cache.rstrip("/\\") + "_results")
    os.makedirs(out_dir, exist_ok=True)
    print(f"device={device} dataset={args.dataset} cache={args.cache}")

    cats = load_cache(args.cache)
    targets = list(cats.keys())
    if args.smoke:
        targets = targets[:1]; args.steps = 300; args.draws = 3

    allres = {}
    for target in targets:
        sub = meta_train(cats, target, device, steps=args.steps, r=args.rank)
        res = evaluate(cats, target, sub, device, r=args.rank, n_draws=args.draws, loco=loco)
        allres[target] = res
        json.dump(res, open(os.path.join(out_dir, target + ".json"), "w"), indent=2)
        k4 = res[4]
        if loco:
            print(f"  {target}: [mean] logical A={k4['A']['mean']['logic']:.3f} "
                  f"B={k4['B']['mean']['logic']:.3f} | structural "
                  f"A={k4['A']['mean']['struct']:.3f} B={k4['B']['mean']['struct']:.3f}")
        else:
            print(f"  {target}: K=4 image-AUROC  A(PCA)={k4['A']['max']['all']:.3f} "
                  f"B(span)={k4['B']['max']['all']:.3f}")

    # ---- aggregate table ----
    print("\n" + "=" * 70)
    if loco:
        print("LOCO crossover (mean of categories) - A=PCA vs B=B_span")
        print(f"{'shot':<5}{'agg':<6}{'model':<8}{'structural':>12}{'logical':>10}")
        for K in SHOTS:
            for a in ["max", "mean"]:
                for key in ["A", "B"]:
                    st = np.mean([allres[t][K][key][a]["struct"] for t in targets])
                    lo = np.mean([allres[t][K][key][a]["logic"] for t in targets])
                    print(f"{K:<5}{a:<6}{key:<8}{st:>12.3f}{lo:>10.3f}")
    else:
        print("MVTec image-AUROC (mean of 15 categories) - A=PCA vs B=B_span")
        print(f"{'shot':<5}{'A(PCA,max)':>12}{'B(span,max)':>13}{'Δ':>8}")
        for K in SHOTS:
            a = np.mean([allres[t][K]["A"]["max"]["all"] for t in targets])
            b = np.mean([allres[t][K]["B"]["max"]["all"] for t in targets])
            print(f"{K:<5}{a:>12.3f}{b:>13.3f}{b-a:>+8.3f}")
    print("=" * 70)
    print("results ->", out_dir)


if __name__ == "__main__":
    main()
