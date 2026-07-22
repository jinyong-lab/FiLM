"""모형3 수정: 특징보존형 FiLM (감사 지적 반영).
기존 결함=proj Conv(1536->512) 랜덤초기화→좋은 입력특징 버리고 재학습→항등복귀 불가→과적합.
수정=residual 병목블록 + conv2 zero-init → 초기 output=입력특징(=baseline). FiLM은 그 위 변조만.
  블록: res=x; h=conv1(x)[C->CH]; h=BN; h=γ(z)h+β(z); h=relu; h=conv2(h)[CH->C, zero-init]; return res+h
  최종 F.normalize(dim=1)로 패치 L2정규화(baseline 특징공간과 일치). 초기 전 블록=항등 → 파이프라인이
  입력특징(patch-NN) 성능에서 출발 → 학습은 개선만 가능(못해도 입력 유지). FILMAD_CACHE로 특징 선택.
"""
import os, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from film_ad import load, to_map, TRAIN_CATS, TEST_CATS, dev, C, S

Z, CH, NB = 128, 512, 4
STEPS, MARGIN, LR, BANK_SUB = 20000, 0.7, 1e-3, 1024
EVAL_EVERY = 2000
REG_LAM = float(os.environ.get("REG_LAM", 0))      # 특화억제 정규화(입력특징서 벗어난 정도 벌점)
AUG_STD = float(os.environ.get("AUG_STD", 0))      # 메타증강(support 잡음, 태스크 암기 억제)


class TaskEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * C, 512), nn.ReLU(), nn.Linear(512, Z))

    def forward(self, supp):
        return self.mlp(torch.cat([supp.mean((0, 2, 3)), supp.var((0, 2, 3))], 0))


class FiLMResBlockPreserve(nn.Module):
    """residual 병목 + conv2 zero-init → 초기 항등(출력=입력). FiLM은 병목 활성화만 변조."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(C, CH, 1)                      # C=1536 -> CH=512 병목
        self.bn = nn.BatchNorm2d(CH, affine=False)
        self.g = nn.Linear(Z, CH); self.b = nn.Linear(Z, CH)
        self.conv2 = nn.Conv2d(CH, C, 1)                      # CH -> C=1536 복원
        nn.init.zeros_(self.g.weight); nn.init.ones_(self.g.bias)
        nn.init.zeros_(self.b.weight); nn.init.zeros_(self.b.bias)
        nn.init.zeros_(self.conv2.weight); nn.init.zeros_(self.conv2.bias)   # ★zero-init=항등

    def forward(self, x, z):
        res = x
        h = self.bn(self.conv1(x))
        h = self.g(z).view(1, CH, 1, 1) * h + self.b(z).view(1, CH, 1, 1)
        h = F.relu(h)
        h = self.conv2(h)
        return res + h                                        # 초기 conv2=0 → output=x


class MetaLearnerPreserve(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([FiLMResBlockPreserve() for _ in range(NB)])

    def forward(self, x, z):                                  # x [N,C,S,S]
        h = x
        for blk in self.blocks:
            h = blk(h, z)
        return F.normalize(h, dim=1)                          # 패치 L2정규화(baseline 공간 일치)


def to_bank(feat4d, sub=None):
    ch = feat4d.shape[1]
    b = feat4d.permute(0, 2, 3, 1).reshape(-1, ch)
    if sub and b.shape[0] > sub:
        b = b[torch.randperm(b.shape[0], device=b.device)[:sub]]
    return b


def patch_dist(bank, q):
    return torch.cdist(q, bank).min(1).values


@torch.no_grad()
def quick_heldout(enc, ml, held, K=1, draws=3):
    enc.eval(); ml.eval()
    aucs = []
    for cat, (pool, test, lab) in held.items():
        ca = []
        for draw in range(draws):
            g = torch.Generator().manual_seed(7 * K + draw)
            sidx = torch.randperm(pool.shape[0], generator=g)[:K]
            supp = torch.stack([to_map(pool[i]) for i in sidx]).to(dev)
            z = enc(supp); bank = to_bank(ml(supp, z))
            iscore = [patch_dist(bank, to_bank(ml(to_map(test[j]).to(dev).unsqueeze(0), z))).max().item()
                      for j in range(len(test))]
            ca.append(roc_auc_score(lab, iscore))
        aucs.append(float(np.mean(ca)))
    enc.train(); ml.train()
    return float(np.mean(aucs)), aucs


def meta_train(train_feats, train_test):
    enc, ml = TaskEncoder().to(dev), MetaLearnerPreserve().to(dev)
    opt = torch.optim.Adam(list(enc.parameters()) + list(ml.parameters()), lr=LR)
    held = {c: (load(c)["train_feats"], load(c)["test_feats"], load(c)["test_img_label"].numpy())
            for c in TEST_CATS}
    ho0, _ = quick_heldout(enc, ml, held)
    print(f"  >> [초기 항등 확인] step 0: held-out={ho0:.4f} (입력특징 patch-NN과 같아야 함)", flush=True)
    curve = [(0, ho0)]
    for step in range(STEPS):
        cat = random.choice(TRAIN_CATS); feats = train_feats[cat]; tf, tp = train_test[cat]
        K = random.choice([1, 2, 4]); sidx = torch.randperm(feats.shape[0])[:K]
        supp = torch.stack([to_map(feats[i]) for i in sidx]).to(dev)
        qi = torch.randint(tf.shape[0], (1,)).item()
        q = to_map(tf[qi]).to(dev).unsqueeze(0)
        mask = torch.tensor(tp[qi].reshape(-1) > 0, device=dev)
        supp_z = F.normalize(supp + AUG_STD * supp.std() * torch.randn_like(supp), dim=1) if AUG_STD > 0 else supp
        z = enc(supp_z)
        inp = torch.cat([supp, q])
        allml = ml(inp, z)
        supp_ml, q_ml = allml[:K], allml[K:]
        bank = to_bank(supp_ml, BANK_SUB); qd = patch_dist(bank, to_bank(q_ml))
        margin = (qd[~mask].mean() + F.relu(MARGIN - qd[mask]).mean()) if mask.any() else qd.mean()
        reg = ((allml - inp) ** 2).sum(1).mean()         # 특화억제: 입력특징서 벗어난 정도 벌점
        loss = margin + REG_LAM * reg
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 500 == 0:
            print(f"  step {step+1}/{STEPS} loss={loss.item():.4f}", flush=True)
        if (step + 1) % EVAL_EVERY == 0:
            ho, _ = quick_heldout(enc, ml, held)
            curve.append((step + 1, ho))
            print(f"  >> [학습곡선] step {step+1}: held-out={ho:.4f}", flush=True)
    print("\n[held-out 곡선] " + "  ".join(f"{s//1000}k:{a:.3f}" for s, a in curve), flush=True)
    enc.eval(); ml.eval(); return enc, ml, max(a for _, a in curve)


@torch.no_grad()
def evaluate(enc, ml, cats, shots=(1, 5), draws=5):
    out = {}
    for cat in cats:
        d = load(cat); pool, test = d["train_feats"], d["test_feats"]
        lab = d["test_img_label"].numpy(); pix = d["test_pix_label"].numpy().reshape(len(test), -1)
        for K in shots:
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
        print(f"  {cat} done", flush=True)
    return out


def main():
    import os
    random.seed(0); torch.manual_seed(0); np.random.seed(0)
    cache = os.environ.get("FILMAD_CACHE", "feature_cache_wrn")
    print("=" * 64); print(f"모형3 수정: 특징보존형 FiLM | 특징={cache} | 10/5"); print("=" * 64)
    train_feats = {c: load(c)["train_feats"] for c in TRAIN_CATS}
    train_test = {c: (load(c)["test_feats"], load(c)["test_pix_label"].numpy()) for c in TRAIN_CATS}
    print("[메타학습 시작]", flush=True)
    enc, ml, peak = meta_train(train_feats, train_test)
    print("\n[평가: 테스트 5종]", flush=True)
    r = evaluate(enc, ml, TEST_CATS)
    print("\n----- 모형3수정(특징보존 FiLM) (테스트 5종 평균) -----")
    for K in (1, 5):
        print(f"  {K}-shot: Image-AUROC={np.mean(r[K]['img']):.3f}  Pixel-AUROC={np.nanmean(r[K]['pix']):.3f}")
    print(f"  학습곡선 정점(held-out 1-shot) = {peak:.3f}")
    print("\n[판정] 초기 항등이 입력특징 성능과 같고, 최종이 그 이상이면 특징보존 FiLM 성공.")


if __name__ == "__main__":
    main()
