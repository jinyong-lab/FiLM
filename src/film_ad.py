"""FiLM 기반 Unsupervised Few-shot Anomaly Detection (제안 논문 구현).
- 백본: WRN-50-2 (feature_cache_wrn, 28x28x1536). ImageNet 동결 특징 사용(Local-SSL 단계는 별도).
- Model2 Task Encoder: 채널별 mean+var -> MLP -> z
- Model3 FiLM: gamma(z), beta(z) 채널 변조 (항등 초기화)
- 채점: FiLM 적응 특징에 대한 support 메모리뱅크 최근접(Euclidean) 패치 거리
- 데이터: MVTec 15종 -> 학습 10종 / 테스트 5종(한번도 안 본 카테고리)
- 메타학습: 에피소드(support 정상 K + query 정상 + 특징공간 합성이상), 마진 손실. 백본 동결.
- 평가: 테스트 5종 실제 이상, Image-AUROC + Pixel-AUROC, K=1/5. FiLM vs 무-FiLM(baseline) 대조.
"""
import os, sys, random, argparse
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score

REPO = r"C:\Users\HOSEO\Desktop\K-DS\code\AnomalyDetection"
CACHE = os.path.join(REPO, os.environ.get("FILMAD_CACHE", "feature_cache_wrn"))
dev = "cuda" if torch.cuda.is_available() else "cpu"

C, S = 1536, 28                       # 채널, 공간(28x28)
ZDIM = 128
ALL = ["bottle","cable","capsule","carpet","grid","hazelnut","leather","metal_nut",
       "pill","screw","tile","toothbrush","transistor","wood","zipper"]
TEST_CATS = ["wood","cable","zipper","transistor","hazelnut"]        # 5종(테스트 전용)
TRAIN_CATS = [c for c in ALL if c not in TEST_CATS]                  # 10종(메타학습)
STEPS, MARGIN, LR = 2000, 0.7, 1e-3
BANK_SUB = 1024                       # 학습 시 메모리뱅크 서브샘플


def l2n(x, dim=-1):
    return x / (x.norm(dim=dim, keepdim=True) + 1e-8)


def load(c):
    return torch.load(os.path.join(CACHE, c + ".pt"), map_location="cpu", weights_only=False)


def to_map(feat_PC):                  # [P=784, C] -> [C, S, S], 패치 L2정규화 먼저
    return l2n(feat_PC.float(), dim=-1).transpose(0, 1).reshape(C, S, S)


# ---------------- 모형 ----------------
class TaskEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * C, 512), nn.ReLU(), nn.Linear(512, ZDIM))

    def forward(self, supp_maps):     # [N, C, S, S]
        mean = supp_maps.mean(dim=(0, 2, 3)); var = supp_maps.var(dim=(0, 2, 3))
        return self.mlp(torch.cat([mean, var], 0))    # [ZDIM]


class FiLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_g = nn.Linear(ZDIM, C); self.to_b = nn.Linear(ZDIM, C)
        nn.init.zeros_(self.to_g.weight); nn.init.ones_(self.to_g.bias)     # gamma=1 (항등)
        nn.init.zeros_(self.to_b.weight); nn.init.zeros_(self.to_b.bias)    # beta=0

    def forward(self, maps, z):       # maps [N,C,S,S]
        g = self.to_g(z).view(1, C, 1, 1); b = self.to_b(z).view(1, C, 1, 1)
        return g * maps + b


def patch_scores(supp_ad, q_ad, sub=None):
    """supp_ad [N,C,S,S], q_ad [C,S,S] -> 쿼리 패치별 최근접 거리 [S*S]."""
    bank = supp_ad.permute(0, 2, 3, 1).reshape(-1, C)     # [N*784, C]
    if sub and bank.shape[0] > sub:
        bank = bank[torch.randperm(bank.shape[0], device=bank.device)[:sub]]
    q = q_ad.permute(1, 2, 0).reshape(-1, C)              # [784, C]
    return torch.cdist(q, bank).min(dim=1).values          # [784]


def synth_anomaly(normal_map, donor_map):
    """특징공간 CutPaste: 정상맵의 무작위 사각영역을 이질 특징과 알파블렌드(미묘한 이상).
    블렌드 후 패치 L2 재정규화로 스케일 유지. mask 반환."""
    m = normal_map.clone()
    bh, bw = random.randint(3, 9), random.randint(3, 9)
    y, x = random.randint(0, S - bh), random.randint(0, S - bw)
    sy, sx = random.randint(0, S - bh), random.randint(0, S - bw)
    a = random.uniform(0.3, 0.9)                                    # 이상 강도(미묘~강)
    blended = a * donor_map[:, sy:sy + bh, sx:sx + bw] + (1 - a) * m[:, y:y + bh, x:x + bw]
    m[:, y:y + bh, x:x + bw] = l2n(blended, dim=0)                  # 채널축 재정규화
    mask = torch.zeros(S, S, device=normal_map.device); mask[y:y + bh, x:x + bw] = 1
    return m, mask.reshape(-1)


# ---------------- 메타학습 ----------------
def meta_train(train_feats):
    enc, film = TaskEncoder().to(dev), FiLM().to(dev)
    opt = torch.optim.Adam(list(enc.parameters()) + list(film.parameters()), lr=LR)
    for step in range(STEPS):
        cat = random.choice(TRAIN_CATS); feats = train_feats[cat]; n = feats.shape[0]
        K = random.choice([1, 2, 4]); idx = torch.randperm(n)[:K + 1]
        supp = torch.stack([to_map(feats[i]) for i in idx[:K]]).to(dev)   # [K,C,S,S]
        qn = to_map(feats[idx[K]]).to(dev)                                # 정상 쿼리
        oc = random.choice([c for c in TRAIN_CATS if c != cat]); of = train_feats[oc]
        donor = to_map(of[torch.randint(of.shape[0], (1,)).item()]).to(dev)
        qa, mask = synth_anomaly(qn, donor)                               # 합성이상 쿼리
        z = enc(supp); supp_ad = film(supp, z); qa_ad = film(qa.unsqueeze(0), z)[0]
        d = patch_scores(supp_ad, qa_ad, sub=BANK_SUB)
        loss = d[mask == 0].mean() + F.relu(MARGIN - d[mask == 1]).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 500 == 0:
            print(f"  step {step+1}/{STEPS} loss={loss.item():.4f}", flush=True)
    enc.eval(); film.eval(); return enc, film


# ---------------- 평가 ----------------
@torch.no_grad()
def evaluate(enc, film, use_film, cats, shots=(1, 5), n_draws=5):
    out = {}
    for cat in cats:
        d = load(cat); pool = d["train_feats"]; test = d["test_feats"]
        lab = d["test_img_label"].numpy(); pix = d["test_pix_label"].numpy().reshape(len(test), -1)
        for K in shots:
            img_aucs, pix_aucs = [], []
            for draw in range(n_draws):
                g = torch.Generator().manual_seed(7 * K + draw)
                sidx = torch.randperm(pool.shape[0], generator=g)[:K]
                supp = torch.stack([to_map(pool[i]) for i in sidx]).to(dev)
                z = enc(supp) if use_film else None
                supp_ad = film(supp, z) if use_film else supp
                img_s, pix_s = [], []
                for j in range(len(test)):
                    qm = to_map(test[j]).to(dev)
                    q_ad = film(qm.unsqueeze(0), z)[0] if use_film else qm
                    ps = patch_scores(supp_ad, q_ad)           # [784]
                    img_s.append(ps.max().item()); pix_s.append(ps.cpu().numpy())
                img_aucs.append(roc_auc_score(lab, img_s))
                pv = np.stack(pix_s)                            # [Ntest, 784]
                if pix.max() > 0:
                    pix_aucs.append(roc_auc_score(pix.reshape(-1) > 0, pv.reshape(-1)))
            out.setdefault(K, {"img": [], "pix": []})
            out[K]["img"].append(float(np.mean(img_aucs)))
            out[K]["pix"].append(float(np.mean(pix_aucs)) if pix_aucs else float("nan"))
        print(f"  [{'FiLM' if use_film else 'base'}] {cat} done", flush=True)
    return out


def report(name, out, shots=(1, 5)):
    print(f"\n----- {name} (테스트 5종 평균) -----")
    for K in shots:
        img = np.mean(out[K]["img"]); pix = np.nanmean(out[K]["pix"])
        print(f"  {K}-shot: Image-AUROC={img:.3f}  Pixel-AUROC={pix:.3f}")


def main():
    random.seed(0); torch.manual_seed(0); np.random.seed(0)
    print("=" * 64)
    print("FiLM Few-shot AD | 학습 10종:", TRAIN_CATS)
    print("            | 테스트 5종:", TEST_CATS)
    print("=" * 64)
    train_feats = {c: load(c)["train_feats"] for c in TRAIN_CATS}
    print("[메타학습 시작]", flush=True)
    enc, film = meta_train(train_feats)
    print("\n[평가: 테스트 5종]", flush=True)
    base = evaluate(enc, film, use_film=False, cats=TEST_CATS)
    filmr = evaluate(enc, film, use_film=True, cats=TEST_CATS)
    report("baseline (무-FiLM, few-shot PatchCore식)", base)
    report("제안 (FiLM 적응)", filmr)
    print("\n[해석] FiLM - baseline 차이가 +면 FiLM 적응이 실제로 기여")


if __name__ == "__main__":
    main()
