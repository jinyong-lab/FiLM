"""사용자 요청: 학습 12종 / 테스트 3종. FiLM(12종 메타학습) vs baseline(동결)을 같은 3종서 비교.
- test 3종=난이도 고른 표본: cable(중)·zipper(난)·hazelnut(이). 나머지 12종 학습.
- FiLM: 12종으로 메타학습 후 학습곡선 정점 + 최종 full eval(1/5-shot). baseline: 동결 patch-NN.
- 목적: 학습 태스크 10→12로 늘리면 baseline(3종 기준)에 얼마나 다가가나.
"""
import random
import numpy as np
import torch
import torch.nn.functional as F
from film_ad import load, to_map, ALL, dev
import film_ad
import film_ad_full as FF
from film_ad_full import TaskEncoder, MetaLearner, to_bank, patch_dist, quick_heldout, BANK_SUB, MARGIN, LR

TEST3 = ["cable", "zipper", "hazelnut"]
TRAIN12 = [c for c in ALL if c not in TEST3]
STEPS, EVAL_EVERY = 16000, 2000


def train(cats, train_feats, train_test, held):
    enc, ml = TaskEncoder().to(dev), MetaLearner().to(dev)
    opt = torch.optim.Adam(list(enc.parameters()) + list(ml.parameters()), lr=LR)
    curve = []
    for step in range(STEPS):
        cat = random.choice(cats); feats = train_feats[cat]; tf, tp = train_test[cat]
        K = random.choice([1, 2, 4]); sidx = torch.randperm(feats.shape[0])[:K]
        supp = torch.stack([to_map(feats[i]) for i in sidx]).to(dev)
        qi = torch.randint(tf.shape[0], (1,)).item()
        q = to_map(tf[qi]).to(dev).unsqueeze(0)
        mask = torch.tensor(tp[qi].reshape(-1) > 0, device=dev)
        z = enc(supp)
        allml = ml(torch.cat([supp, q]), z)
        supp_ml, q_ml = allml[:K], allml[K:]
        bank = to_bank(supp_ml, BANK_SUB); qd = patch_dist(bank, to_bank(q_ml))
        loss = (qd[~mask].mean() + F.relu(MARGIN - qd[mask]).mean()) if mask.any() else qd.mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % EVAL_EVERY == 0:
            ho, _ = quick_heldout(enc, ml, held)
            curve.append((step + 1, ho))
            print(f"  step {step+1}: held-out(3종) 1-shot={ho:.4f}", flush=True)
    return enc, ml, curve


def avg(out, K, key="img"):
    return float(np.mean(out[K][key]))


def main():
    random.seed(0); torch.manual_seed(0); np.random.seed(0)
    print("=" * 64)
    print("학습 12종 / 테스트 3종 : FiLM(제안) vs 동결 baseline")
    print("학습 12종:", TRAIN12)
    print("테스트 3종:", TEST3)
    print("=" * 64, flush=True)
    train_feats = {c: load(c)["train_feats"] for c in TRAIN12}
    train_test = {c: (load(c)["test_feats"], load(c)["test_pix_label"].numpy()) for c in TRAIN12}
    held = {c: (load(c)["train_feats"], load(c)["test_feats"], load(c)["test_img_label"].numpy())
            for c in TEST3}

    print("\n[baseline: 동결 patch-NN, 3종]", flush=True)
    base = film_ad.evaluate(None, None, use_film=False, cats=TEST3, shots=(1, 5), n_draws=5)

    print("\n[제안 FiLM: 12종 메타학습]", flush=True)
    enc, ml, curve = train(TRAIN12, train_feats, train_test, held)
    peak = max(a for _, a in curve)
    filmr = FF.evaluate(enc, ml, TEST3, shots=(1, 5), draws=5)

    print("\n" + "=" * 64)
    print("결과 (테스트 3종 평균, Image-AUROC)")
    print("=" * 64)
    print(f"  동결 baseline   : 1-shot={avg(base,1):.3f}  5-shot={avg(base,5):.3f}")
    print(f"  제안 FiLM(최종) : 1-shot={avg(filmr,1):.3f}  5-shot={avg(filmr,5):.3f}")
    print(f"  제안 FiLM(정점) : 1-shot={peak:.3f} (학습곡선 최고, 상한)")
    print(f"\n  곡선: " + "  ".join(f"{s//1000}k:{a:.3f}" for s, a in curve))
    print(f"\n[비교] 5종테스트땐 FiLM 0.69~0.76 vs baseline 0.888. 12/3에선 간극이 좁혀졌나?")
    print(f"  간극(1-shot) = baseline {avg(base,1):.3f} - FiLM {avg(filmr,1):.3f} = {avg(base,1)-avg(filmr,1):+.3f}")


if __name__ == "__main__":
    main()
