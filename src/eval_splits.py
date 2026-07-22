"""데이터 분할(10/5, 12/3) × shot 수(K=1,2,5)별 최종 파이프라인 성능 측정.
최종 모형 = 결함SSL 특징 + 특징보존·특화억제 정규화 FiLM(Model3) + patch-NN(Model4).
Image/Pixel AUROC를 (분할, shot)별로 출력 → 미팅자료 성능표용.
환경변수: FILMAD_CACHE(특징), REG_LAM(기본0.5), AUG_STD(기본0.1).
"""
import os, random
import numpy as np
import torch, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from film_ad import load, to_map, ALL, dev, C
from film_ad_fixed import TaskEncoder, MetaLearnerPreserve, to_bank, patch_dist

STEPS, MARGIN, LR, BANK_SUB = 20000, 0.7, 1e-3, 1024
REG_LAM = float(os.environ.get("REG_LAM", 0.5))
AUG_STD = float(os.environ.get("AUG_STD", 0.1))
SHOTS = (1, 2, 5)

SPLITS = {
    "10/5": ["wood", "cable", "zipper", "transistor", "hazelnut"],   # 테스트 5종
    "12/3": ["cable", "zipper", "hazelnut"],                          # 테스트 3종
}


def meta_train(train_cats, train_feats, train_test):
    enc, ml = TaskEncoder().to(dev), MetaLearnerPreserve().to(dev)
    opt = torch.optim.Adam(list(enc.parameters()) + list(ml.parameters()), lr=LR)
    for step in range(STEPS):
        cat = random.choice(train_cats); feats = train_feats[cat]; tf, tp = train_test[cat]
        K = random.choice([1, 2, 4]); sidx = torch.randperm(feats.shape[0])[:K]
        supp = torch.stack([to_map(feats[i]) for i in sidx]).to(dev)
        qi = torch.randint(tf.shape[0], (1,)).item()
        q = to_map(tf[qi]).to(dev).unsqueeze(0)
        mask = torch.tensor(tp[qi].reshape(-1) > 0, device=dev)
        supp_z = F.normalize(supp + AUG_STD * supp.std() * torch.randn_like(supp), dim=1) if AUG_STD > 0 else supp
        z = enc(supp_z); inp = torch.cat([supp, q]); allml = ml(inp, z)
        supp_ml, q_ml = allml[:K], allml[K:]
        bank = to_bank(supp_ml, BANK_SUB); qd = patch_dist(bank, to_bank(q_ml))
        margin = (qd[~mask].mean() + F.relu(MARGIN - qd[mask]).mean()) if mask.any() else qd.mean()
        reg = ((allml - inp) ** 2).sum(1).mean()
        loss = margin + REG_LAM * reg
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 5000 == 0:
            print(f"    step {step+1}/{STEPS} loss={loss.item():.4f}", flush=True)
    enc.eval(); ml.eval(); return enc, ml


@torch.no_grad()
def evaluate(enc, ml, cats, draws=5):
    out = {}
    for cat in cats:
        d = load(cat); pool, test = d["train_feats"], d["test_feats"]
        lab = d["test_img_label"].numpy(); pix = d["test_pix_label"].numpy().reshape(len(test), -1)
        for K in SHOTS:
            ia, pa = [], []
            for draw in range(draws):
                g = torch.Generator().manual_seed(7 * K + draw)
                sidx = torch.randperm(pool.shape[0], generator=g)[:K]
                supp = torch.stack([to_map(pool[i]) for i in sidx]).to(dev)
                z = enc(supp); bank = to_bank(ml(supp, z))
                iscore, pscore = [], []
                for j in range(len(test)):
                    dd = patch_dist(bank, to_bank(ml(to_map(test[j]).to(dev).unsqueeze(0), z)))
                    iscore.append(dd.max().item()); pscore.append(dd.cpu().numpy())
                ia.append(roc_auc_score(lab, iscore))
                pv = np.stack(pscore)
                if pix.max() > 0:
                    pa.append(roc_auc_score(pix.reshape(-1) > 0, pv.reshape(-1)))
            out.setdefault(K, {"img": [], "pix": []})
            out[K]["img"].append(float(np.mean(ia)))
            out[K]["pix"].append(float(np.mean(pa)) if pa else float("nan"))
        print(f"    {cat} done", flush=True)
    return out


def main():
    random.seed(0); torch.manual_seed(0); np.random.seed(0)
    cache = os.environ.get("FILMAD_CACHE", "feature_cache_wrn")
    print("=" * 64)
    print(f"분할×shot 성능 | 특징={cache} REG_LAM={REG_LAM} AUG_STD={AUG_STD}")
    print("=" * 64, flush=True)
    summary = {}
    for name, test_cats in SPLITS.items():
        train_cats = [c for c in ALL if c not in test_cats]
        print(f"\n[split {name}] 학습 {len(train_cats)}종 / 테스트 {len(test_cats)}종: {test_cats}", flush=True)
        train_feats = {c: load(c)["train_feats"] for c in train_cats}
        train_test = {c: (load(c)["test_feats"], load(c)["test_pix_label"].numpy()) for c in train_cats}
        enc, ml = meta_train(train_cats, train_feats, train_test)
        r = evaluate(enc, ml, test_cats)
        summary[name] = {K: (float(np.mean(r[K]["img"])), float(np.nanmean(r[K]["pix"]))) for K in SHOTS}

    print("\n" + "=" * 64)
    print("[최종 성능표] 최종 모형 = 결함SSL + 특징보존·정규화 FiLM + patch-NN")
    print("=" * 64)
    print(f"{'분할(학습/테스트)':<20}{'shot':<8}{'Image-AUROC':<14}{'Pixel-AUROC':<14}")
    for name in SPLITS:
        for K in SHOTS:
            img, pix = summary[name][K]
            tag = f"{name} (테스트 {len(SPLITS[name])}종)"
            print(f"{tag:<20}K={K:<6}{img:<14.3f}{pix:<14.3f}")


if __name__ == "__main__":
    main()
