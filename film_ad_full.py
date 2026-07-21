"""FiLM Few-shot AD — 풀 버전(논문 3.3 충실): 메타러너 CNN의 중간 블록에 FiLM 삽입.
이전(film_ad_real)은 백본 출력에 FiLM 1회 -> 메타러너 CNN 누락. 여기서는
  백본(동결) -> 메타러너[proj Conv + (Conv3x3 + FiLM(z))xN] -> 패치메모리(마할라노비스)
Task Encoder(z) -> FiLM(gamma,beta)로 메타러너 중간 활성화 변조. 백본 동결, 나머지 학습.
데이터: 학습 10종(query=실제결함) / 테스트 5종 held-out. 대조=raw특징 patch-NN(무-메타러너).
"""
import os, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from film_ad import load, to_map, TRAIN_CATS, TEST_CATS, dev, C, S

Z, CH, NB = 128, 512, 4          # FiLM 논문: ResBlock 4개
STEPS, MARGIN, LR, BANK_SUB = 20000, 0.7, 1e-3, 1024      # 완전체 결합 테스트(정점~14k 포착)
EVAL_EVERY = 2000                # held-out 학습곡선 측정 간격
MAHAL = False                    # baseline과 동일 채점(Euclidean)으로 메타러너 효과만 격리


class TaskEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * C, 512), nn.ReLU(), nn.Linear(512, Z))

    def forward(self, supp):                       # [N,C,S,S]
        return self.mlp(torch.cat([supp.mean((0, 2, 3)), supp.var((0, 2, 3))], 0))


class FiLMResBlock(nn.Module):
    """FiLM 논문 정본 ResBlock: 1x1 conv -> [3x3 conv -> BN(affine없음) -> FiLM(BN 뒤) -> ReLU] + 잔차.
    BN(affine없음)+FiLM = Conditional BatchNorm. γ,β는 z의 아핀투영(FiLM generator)."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(CH, CH, 1)                    # 1x1 conv 시작
        self.conv2 = nn.Conv2d(CH, CH, 3, padding=1)
        self.bn = nn.BatchNorm2d(CH, affine=False)           # 정규화만(아핀은 FiLM)
        self.g = nn.Linear(Z, CH); self.b = nn.Linear(Z, CH)  # FiLM generator(아핀투영)
        nn.init.zeros_(self.g.weight); nn.init.ones_(self.g.bias)    # γ~1
        nn.init.zeros_(self.b.weight); nn.init.zeros_(self.b.bias)   # β~0

    def forward(self, x, z):
        x = F.relu(self.conv1(x)); res = x
        h = self.bn(self.conv2(x))
        h = self.g(z).view(1, CH, 1, 1) * h + self.b(z).view(1, CH, 1, 1)   # FiLM: γ⊙h+β (BN 뒤)
        return F.relu(h) + res                               # 잔차 연결


class MetaLearner(nn.Module):
    """백본 특징 -> proj -> FiLM-ed ResBlock x4 -> 적응 특징 (FiLM 논문 FiLM-ed network)."""
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(C, CH, 1)
        self.blocks = nn.ModuleList([FiLMResBlock() for _ in range(NB)])

    def forward(self, x, z):                        # x [N,C,S,S]
        h = F.relu(self.proj(x))
        for blk in self.blocks:
            h = blk(h, z)
        return h                                    # [N,CH,S,S]


def to_bank(feat4d, sub=None):
    b = feat4d.permute(0, 2, 3, 1).reshape(-1, CH)
    if sub and b.shape[0] > sub:
        b = b[torch.randperm(b.shape[0], device=b.device)[:sub]]
    return b


def patch_dist(bank, q):                            # bank [M,CH], q [P,CH]
    if MAHAL:
        cov = torch.cov(bank.T) + 1e-3 * torch.eye(CH, device=bank.device)
        L = torch.linalg.cholesky(cov)
        bw = torch.linalg.solve_triangular(L, bank.T, upper=False).T
        qw = torch.linalg.solve_triangular(L, q.T, upper=False).T
        return torch.cdist(qw, bw).min(1).values
    return torch.cdist(q, bank).min(1).values


@torch.no_grad()
def quick_heldout(enc, ml, held, K=1, draws=3):
    """학습 도중 held-out 5종의 Image-AUROC를 싸게 측정(학습곡선용). 1-shot, 3draw."""
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
    enc, ml = TaskEncoder().to(dev), MetaLearner().to(dev)
    opt = torch.optim.Adam(list(enc.parameters()) + list(ml.parameters()), lr=LR)
    held = {c: (load(c)["train_feats"], load(c)["test_feats"], load(c)["test_img_label"].numpy())
            for c in TEST_CATS}                          # 학습곡선용 held-out 미리 로드
    curve = []
    for step in range(STEPS):
        cat = random.choice(TRAIN_CATS); feats = train_feats[cat]; tf, tp = train_test[cat]
        K = random.choice([1, 2, 4]); sidx = torch.randperm(feats.shape[0])[:K]
        supp = torch.stack([to_map(feats[i]) for i in sidx]).to(dev)
        qi = torch.randint(tf.shape[0], (1,)).item()
        q = to_map(tf[qi]).to(dev).unsqueeze(0)
        mask = torch.tensor(tp[qi].reshape(-1) > 0, device=dev)
        z = enc(supp)
        allml = ml(torch.cat([supp, q]), z)              # BN 통계 일관되게 함께 forward
        supp_ml, q_ml = allml[:K], allml[K:]
        bank = to_bank(supp_ml, BANK_SUB); qd = patch_dist(bank, to_bank(q_ml))
        loss = (qd[~mask].mean() + F.relu(MARGIN - qd[mask]).mean()) if mask.any() else qd.mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 500 == 0:
            print(f"  step {step+1}/{STEPS} loss={loss.item():.4f}", flush=True)
        if (step + 1) % EVAL_EVERY == 0:
            ho, _ = quick_heldout(enc, ml, held)
            curve.append((step + 1, ho))
            print(f"  >> [학습곡선] step {step+1}: held-out 5종 1-shot Img-AUROC={ho:.4f}  (baseline무FiLM=0.855)", flush=True)
    print("\n[held-out 학습곡선 요약] step:AUROC")
    print("  " + "  ".join(f"{s//1000}k:{a:.3f}" for s, a in curve), flush=True)
    enc.eval(); ml.eval(); return enc, ml


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
                z = enc(supp); supp_ml = ml(supp, z); bank = to_bank(supp_ml)
                iscore, pscore = [], []
                for j in range(len(test)):
                    q_ml = ml(to_map(test[j]).to(dev).unsqueeze(0), z)
                    dd = patch_dist(bank, to_bank(q_ml))
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
    random.seed(0); torch.manual_seed(0); np.random.seed(0)
    print("=" * 64); print("FiLM 풀버전(메타러너 CNN + 중간 FiLM + 마할라노비스) | 10/5"); print("=" * 64)
    train_feats = {c: load(c)["train_feats"] for c in TRAIN_CATS}
    train_test = {c: (load(c)["test_feats"], load(c)["test_pix_label"].numpy()) for c in TRAIN_CATS}
    print("[메타학습 시작]", flush=True)
    enc, ml = meta_train(train_feats, train_test)
    print("\n[평가: 테스트 5종]", flush=True)
    r = evaluate(enc, ml, TEST_CATS)
    print("\n----- 제안 풀버전 (메타러너+FiLM+마할라노비스) (테스트 5종 평균) -----")
    for K in (1, 5):
        print(f"  {K}-shot: Image-AUROC={np.mean(r[K]['img']):.3f}  Pixel-AUROC={np.nanmean(r[K]['pix']):.3f}")
    print("\n대조: raw특징 patch-NN(무-메타러너) = 0.888/0.947 (1/5-shot). 이걸 넘으면 메타러너+FiLM 기여.")


if __name__ == "__main__":
    main()
