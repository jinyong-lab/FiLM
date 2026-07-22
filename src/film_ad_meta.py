"""메타학습 특화 억제(사용자 지적: 학습하는 법을 배우고 싶지, 10종에 맞춰지고 싶지 않음).
수정=FiLM이 입력특징(범용)에서 벗어나는 정도에 벌점(REG_LAM) → 하드 특화 불가, 완만 적응만.
  loss = margin_loss + REG_LAM * mean||out - input||^2   (out,input 모두 패치 L2정규화)
  + 메타증강(support에 소량 잡음)으로 태스크 암기 억제.
정규화 강도 λ를 쓸어 held-out이 입력특징(무학습)을 넘는 지점이 있는지 검정.
환경변수: FILMAD_CACHE(특징), META_SPLIT=10_5 | 12_3.
"""
import os, random
import numpy as np
import torch, torch.nn.functional as F
from film_ad import load, to_map, ALL, dev, C
from film_ad_fixed import TaskEncoder, MetaLearnerPreserve, to_bank, patch_dist, quick_heldout

STEPS, EVAL_EVERY, MARGIN, LR, BANK_SUB = 12000, 2000, 0.7, 1e-3, 1024
AUG_STD = 0.1                                   # 메타증강(태스크 암기 억제)
REG_LAMS = [0.0, 0.5, 2.0, 8.0]                 # 특화 억제 강도 스윕

SPLIT = os.environ.get("META_SPLIT", "10_5")
if SPLIT == "12_3":
    TEST = ["cable", "zipper", "hazelnut"]
else:
    TEST = ["wood", "cable", "zipper", "transistor", "hazelnut"]
TRAIN = [c for c in ALL if c not in TEST]


def meta_train(train_feats, train_test, held, reg_lam):
    enc, ml = TaskEncoder().to(dev), MetaLearnerPreserve().to(dev)
    opt = torch.optim.Adam(list(enc.parameters()) + list(ml.parameters()), lr=LR)
    ho0, _ = quick_heldout(enc, ml, held)
    curve = [(0, ho0)]
    for step in range(STEPS):
        cat = random.choice(TRAIN); feats = train_feats[cat]; tf, tp = train_test[cat]
        K = random.choice([1, 2, 4]); sidx = torch.randperm(feats.shape[0])[:K]
        supp = torch.stack([to_map(feats[i]) for i in sidx]).to(dev)
        qi = torch.randint(tf.shape[0], (1,)).item()
        q = to_map(tf[qi]).to(dev).unsqueeze(0)
        mask = torch.tensor(tp[qi].reshape(-1) > 0, device=dev)
        supp_z = F.normalize(supp + AUG_STD * supp.std() * torch.randn_like(supp), dim=1)  # 메타증강
        z = enc(supp_z)
        inp = torch.cat([supp, q])
        out = ml(inp, z)
        supp_ml, q_ml = out[:K], out[K:]
        bank = to_bank(supp_ml, BANK_SUB); qd = patch_dist(bank, to_bank(q_ml))
        margin = (qd[~mask].mean() + F.relu(MARGIN - qd[mask]).mean()) if mask.any() else qd.mean()
        reg = ((out - inp) ** 2).sum(1).mean()                 # 입력특징서 벗어난 정도(특화 억제)
        loss = margin + reg_lam * reg
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % EVAL_EVERY == 0:
            ho, _ = quick_heldout(enc, ml, held)
            curve.append((step + 1, ho))
    return curve


def main():
    random.seed(0); torch.manual_seed(0); np.random.seed(0)
    cache = os.environ.get("FILMAD_CACHE", "feature_cache_wrn")
    print("=" * 64)
    print(f"메타 특화억제 스윕 | 특징={cache} | split={SPLIT} (학습{len(TRAIN)}/테스트{len(TEST)})")
    print(f"테스트: {TEST}")
    print("=" * 64, flush=True)
    train_feats = {c: load(c)["train_feats"] for c in TRAIN}
    train_test = {c: (load(c)["test_feats"], load(c)["test_pix_label"].numpy()) for c in TRAIN}
    held = {c: (load(c)["train_feats"], load(c)["test_feats"], load(c)["test_img_label"].numpy()) for c in TEST}

    ho_input, _ = quick_heldout(TaskEncoder().to(dev), MetaLearnerPreserve().to(dev), held)  # 항등=입력특징
    print(f"[입력특징(무학습, 항등) held-out] = {ho_input:.4f}\n", flush=True)

    results = []
    for lam in REG_LAMS:
        curve = meta_train(train_feats, train_test, held, lam)
        peak = max(a for _, a in curve); final = curve[-1][1]
        results.append((lam, peak, final))
        pts = " ".join(f"{s//1000}k:{a:.3f}" for s, a in curve)
        print(f"[λ={lam:>4}] 곡선 {pts}", flush=True)
        print(f"          정점={peak:.4f}  최종={final:.4f}", flush=True)
    print("\n" + "=" * 64)
    print(f"[요약] split={SPLIT}  입력특징(무학습)={ho_input:.3f}")
    for lam, peak, final in results:
        mark = " ← 입력특징 초과!" if peak > ho_input + 0.003 else ""
        print(f"  λ={lam:>4}: 정점={peak:.3f}  최종={final:.3f}{mark}")
    print("\n[판정] 어떤 λ에서 정점이 입력특징(무학습)을 넘으면 특화억제 성공=학습이 도움.")


if __name__ == "__main__":
    main()
