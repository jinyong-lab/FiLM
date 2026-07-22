"""SSL 특징 하락 원인 진단: CutPaste 파인튜닝이 특징을 '학습 도메인+합성태스크'에 특화시켜
held-out 전이를 해치는가? patch-NN(무-FiLM)을 in-domain(학습10종) vs out-domain(테스트5종)에서
ImageNet vs SSL 특징으로 비교. SSL이 in-domain↑ & out-domain↓ 이면 특화/일반성상실 확정.
"""
import os
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from film_ad import to_map, patch_scores, dev, TRAIN_CATS, TEST_CATS, REPO


@torch.no_grad()
def eval_cache(cache_dir, cats, K=5, draws=5):
    outs = []
    for cat in cats:
        d = torch.load(os.path.join(cache_dir, cat + ".pt"), map_location="cpu", weights_only=False)
        pool, test, lab = d["train_feats"], d["test_feats"], d["test_img_label"].numpy()
        ca = []
        for draw in range(draws):
            g = torch.Generator().manual_seed(7 * K + draw)
            sidx = torch.randperm(pool.shape[0], generator=g)[:K]
            supp = torch.stack([to_map(pool[i]) for i in sidx]).to(dev)
            img_s = [patch_scores(supp, to_map(test[j]).to(dev)).max().item() for j in range(len(test))]
            ca.append(roc_auc_score(lab, img_s))
        outs.append(float(np.mean(ca)))
    return float(np.mean(outs)), outs


def feat_stats(cache_dir, cats):
    """특징 일반성 지표: 패치 특징의 유효 랭크(다양성). 낮을수록 특화/붕괴."""
    eff = []
    for cat in cats:
        d = torch.load(os.path.join(cache_dir, cat + ".pt"), map_location="cpu", weights_only=False)
        X = d["train_feats"][:20].reshape(-1, d["train_feats"].shape[-1]).float()
        X = X - X.mean(0)
        s = torch.linalg.svdvals(X[torch.randperm(X.shape[0])[:2000]])
        p = s / s.sum()
        eff.append(float(torch.exp(-(p * (p + 1e-12).log()).sum())))   # 유효 랭크(엔트로피)
    return float(np.mean(eff))


def main():
    print("=" * 66)
    print("SSL 하락 원인 진단 (patch-NN 무-FiLM, K=5)")
    print("=" * 66)
    for name, cache in [("ImageNet", "feature_cache_wrn"), ("SSL(CutPaste)", "feature_cache_wrn_ssl")]:
        cd = os.path.join(REPO, cache)
        tr, _ = eval_cache(cd, TRAIN_CATS)
        te, te_list = eval_cache(cd, TEST_CATS)
        er_tr = feat_stats(cd, TRAIN_CATS); er_te = feat_stats(cd, TEST_CATS)
        print(f"\n[{name}]")
        print(f"  Image-AUROC  in-domain(학습10종)={tr:.3f}   held-out(테스트5종)={te:.3f}   격차={tr-te:+.3f}")
        print(f"  유효랭크(특징다양성)  in-domain={er_tr:.1f}   held-out={er_te:.1f}")
        print(f"  테스트5종별: " + ", ".join(f"{c}={v:.3f}" for c, v in zip(TEST_CATS, te_list)))
    print("\n[판정] SSL이 in-domain은 유지/상승, held-out은 하락 & 유효랭크 하락이면")
    print("       = '학습도메인+합성태스크 특화로 일반성 상실'이 하락 원인.")


if __name__ == "__main__":
    main()
