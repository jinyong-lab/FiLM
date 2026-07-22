"""메타학습 핵심 가설 검정: held-out 성능 vs '학습 태스크 수'.
사용자 지적(정당): 우리 모델은 10종 맞춤이 아니라 '여러 태스크에서 태스크구분+불량패치탐지
능력'을 배우는 메타학습. 그렇다면 학습 태스크가 많아질수록 held-out(안 본 5종)이 좋아져야 함.
  - 학습 태스크 수 n ∈ {2,4,6,8,10} (같은 held-out 5종 고정)
  - 각 n에서 메타학습 후, 학습곡선의 held-out '정점(best)'을 기록(각 n의 상한, 공정 비교)
  - 우상향이면 '태스크 부족이 병목'(설계 옳음, 데이터 늘리면 됨). 평탄/하락이면 재설계 필요.
"""
import random
import numpy as np
import torch
import torch.nn.functional as F
from film_ad import load, to_map, TRAIN_CATS, TEST_CATS, dev
from film_ad_full import TaskEncoder, MetaLearner, to_bank, patch_dist, quick_heldout, BANK_SUB, MARGIN, LR

STEPS, EVAL_EVERY = 16000, 2000            # 60k실행서 정점~14k → 16k면 각 조건 정점 포착
TASK_COUNTS = [2, 4, 6, 8, 10]


def train_subset(cats, train_feats, train_test, held):
    """cats(학습 태스크 부분집합)로 메타학습. 학습곡선의 held-out 정점과 전체 곡선 반환."""
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
    best = max(a for _, a in curve)
    return best, curve


def main():
    random.seed(0); torch.manual_seed(0); np.random.seed(0)
    print("=" * 64)
    print("메타학습 가설검정: held-out(안 본 5종) vs 학습 태스크 수")
    print("학습 태스크 후보(10):", TRAIN_CATS)
    print("held-out 고정(5):", TEST_CATS)
    print("=" * 64, flush=True)
    train_feats = {c: load(c)["train_feats"] for c in TRAIN_CATS}
    train_test = {c: (load(c)["test_feats"], load(c)["test_pix_label"].numpy()) for c in TRAIN_CATS}
    held = {c: (load(c)["train_feats"], load(c)["test_feats"], load(c)["test_img_label"].numpy())
            for c in TEST_CATS}
    results = []
    for n in TASK_COUNTS:
        cats = TRAIN_CATS[:n]                          # 고정 부분집합(앞에서 n개)
        print(f"\n[학습 태스크 {n}개] {cats}", flush=True)
        best, curve = train_subset(cats, train_feats, train_test, held)
        results.append((n, best))
        pts = "  ".join(f"{s//1000}k:{a:.3f}" for s, a in curve)
        print(f"  곡선: {pts}", flush=True)
        print(f"  ==> 태스크 {n}개: held-out 정점 = {best:.4f}", flush=True)
    print("\n" + "=" * 64)
    print("[요약] 학습태스크수 -> held-out 정점 AUROC (baseline 무학습=0.947)")
    for n, b in results:
        print(f"  {n:2d}개: {b:.4f}")
    inc = all(results[i][1] <= results[i + 1][1] + 0.005 for i in range(len(results) - 1))
    trend = "우상향(태스크 늘수록↑) → 태스크 부족이 병목=설계 옳음, 데이터 확장이 답" if results[-1][1] > results[0][1] + 0.01 \
        else "평탄/하락 → MVTec 내 태스크 확장으로는 부족, 재설계 필요"
    print(f"\n[판정] {trend}")
    print("(주: 각 값은 학습곡선 정점=각 태스크수의 상한. 절대치는 baseline 0.947과 비교.)")


if __name__ == "__main__":
    main()
