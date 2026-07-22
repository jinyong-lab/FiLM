"""모형1 수정: 결함민감형 dense SSL (감사 지적 반영: contrastive 불변성 → 결함 민감성).
문제(감사): 국소교란(노이즈·erase)에 '불변' 학습 → 결함이 곧 국소교란이라 결함에 둔감.
수정: (1) clean 두 뷰 = nuisance-only 증강(erase 제거)로 dense contrastive = nuisance 불변.
      (2) 합성결함 뷰(CutPaste patch/scar) + 격자마스크 → 결함 위치는 clean과 '멀게'(민감),
          비결함 위치는 clean과 '가깝게'(국소성). = nuisance 불변 + 결함 민감 동시.
백본 layer2 동결·layer3 학습(랭크 보존), VICReg 분산항, 추출·평가는 dense판과 동일.
"""
import os, glob, math, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
from sklearn.metrics import roc_auc_score

DATA = r"C:\Users\HOSEO\Desktop\K-DS\datasets\mvtec_ad\mvtech_anomaly_detection"
OUT = r"C:\Users\HOSEO\Desktop\K-DS\code\AnomalyDetection\feature_cache_wrn_defect"
IM_MEAN, IM_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
IMG, MAX_TRAIN, BS = 224, 200, 16
EPOCHS, LR, TEMP, N_LOC, VIC, LAM, MDEF = 10, 1e-5, 0.2, 64, 1.0, 1.0, 0.5
ALL = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut",
       "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"]
TEST_CATS = ["wood", "cable", "zipper", "transistor", "hazelnut"]
TRAIN_CATS = [c for c in ALL if c not in TEST_CATS]
dev = "cuda" if torch.cuda.is_available() else "cpu"
GRID = 28

det = T.Compose([T.Resize(IMG, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(IMG), T.ToTensor()])
norm = T.Normalize(IM_MEAN, IM_STD)


class AddNoise:
    def __init__(self, s=0.015): self.s = s
    def __call__(self, x): return x + self.s * torch.randn_like(x)


# nuisance-only(결함 모사 증강 erase 제거): 광도 지터·블러·미세노이즈만
nuis = T.Compose([
    T.RandomApply([T.ColorJitter(0.2, 0.2, 0.2, 0.03)], p=0.6),
    T.RandomApply([T.GaussianBlur(3, (0.1, 1.2))], p=0.3),
    AddNoise(0.015),
])


def nuis_view(base):
    return norm(nuis(base.clone()))


def cutpaste_defect(base):
    """합성결함(patch 또는 scar) + 28x28 격자 결함마스크. base [3,224,224] in [0,1]."""
    out = base.clone(); mpx = torch.zeros(IMG, IMG)
    if random.random() < 0.5:                                # patch
        area = random.uniform(0.02, 0.06) * IMG * IMG
        ar = math.exp(random.uniform(math.log(0.3), math.log(3.3)))
        ph = max(4, min(120, int((area * ar) ** 0.5))); pw = max(4, min(120, int((area / ar) ** 0.5)))
        sy, sx = random.randint(0, IMG - ph), random.randint(0, IMG - pw)
        dy, dx = random.randint(0, IMG - ph), random.randint(0, IMG - pw)
        out[:, dy:dy + ph, dx:dx + pw] = base[:, sy:sy + ph, sx:sx + pw].clone()
        mpx[dy:dy + ph, dx:dx + pw] = 1
    else:                                                    # scar
        pw, ph = random.randint(2, 16), random.randint(10, 25)
        sy, sx = random.randint(0, IMG - ph), random.randint(0, IMG - pw)
        dy, dx = random.randint(0, IMG - ph), random.randint(0, IMG - pw)
        out[:, dy:dy + ph, dx:dx + pw] = TF.rotate(base[:, sy:sy + ph, sx:sx + pw].clone(),
                                                    random.uniform(-45, 45))
        mpx[dy:dy + ph, dx:dx + pw] = 1
    cell = IMG // GRID
    gm = mpx.reshape(GRID, cell, GRID, cell).amax(dim=(1, 3))   # [28,28] 결함 셀=1
    return norm(nuis(out)), gm


def load_bases(cat):
    files = sorted(glob.glob(os.path.join(DATA, cat, "train", "good", "*.png")))[:MAX_TRAIN]
    return [det(Image.open(f).convert("RGB")) for f in files]


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        m = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        m.fc = nn.Identity()
        for n, p in m.named_parameters():
            if n.startswith(("conv1", "bn1", "layer1.", "layer2.")):
                p.requires_grad_(False)
        self.m = m; self.feats = {}
        m.layer2.register_forward_hook(lambda mm, i, o: self.feats.__setitem__("l2", o))
        m.layer3.register_forward_hook(lambda mm, i, o: self.feats.__setitem__("l3", o))
        self.proj = nn.Conv2d(1536, 128, 1)

    def fmap(self, x):
        self.m(x)
        l2, l3 = self.feats["l2"], self.feats["l3"]
        l3 = F.interpolate(l3, size=l2.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([l2, l3], 1)

    def forward(self, x):
        return self.proj(self.fmap(x))


def dense_ntxent(v1, v2, temp=TEMP, n_loc=N_LOC):
    b, d, h, w = v1.shape
    v1 = v1.reshape(b, d, h * w); v2 = v2.reshape(b, d, h * w)
    idx = torch.randperm(h * w, device=v1.device)[:n_loc]
    z1 = F.normalize(v1[:, :, idx].permute(0, 2, 1).reshape(b * n_loc, d), dim=1)
    z2 = F.normalize(v2[:, :, idx].permute(0, 2, 1).reshape(b * n_loc, d), dim=1)
    n = z1.shape[0]; z = torch.cat([z1, z2], 0)
    sim = z @ z.T / temp; sim.fill_diagonal_(-1e9)
    targets = torch.cat([torch.arange(n) + n, torch.arange(n)]).to(v1.device)
    return F.cross_entropy(sim, targets), z1, z2


def vicreg_var(z, eps=1e-4):
    return torch.mean(F.relu(1.0 - torch.sqrt(z.var(0) + eps)))


def defect_loss(e_def, e_clean, gm):
    """결함 위치: e_def를 e_clean에서 멀게(민감). 비결함 위치: 가깝게(국소성)."""
    ed = F.normalize(e_def, dim=1); ec = F.normalize(e_clean, dim=1)
    d = 1.0 - (ed * ec).sum(1)                               # [B,28,28] 코사인거리
    gm = gm.to(d.device); defect = gm > 0.5; clean = ~defect
    lo = d.new_tensor(0.0)
    if defect.any():
        lo = lo + F.relu(MDEF - d[defect]).mean()            # 결함: 멀게
    if clean.any():
        lo = lo + d[clean].mean()                            # 비결함: 가깝게
    return lo


def train():
    print("[결함민감SSL] 학습 이미지 로딩...", flush=True)
    bases = []
    for c in TRAIN_CATS:
        bases += load_bases(c)
    print(f"[결함민감SSL] 총 {len(bases)}장. nuisance불변 + 결함민감 dense SSL "
          f"(LR={LR}, epochs={EPOCHS}, layer2동결, LAM={LAM}, VICReg={VIC})", flush=True)
    enc = Encoder().to(dev)
    opt = torch.optim.Adam([p for p in enc.parameters() if p.requires_grad], lr=LR)
    for ep in range(1, EPOCHS + 1):
        random.shuffle(bases); tc, td, nb = 0.0, 0.0, 0
        enc.train()
        for i in range(0, len(bases), BS):
            b = bases[i:i + BS]
            if len(b) < 2:
                continue
            v1 = torch.stack([nuis_view(im) for im in b]).to(dev)
            v2 = torch.stack([nuis_view(im) for im in b]).to(dev)
            dpairs = [cutpaste_defect(im) for im in b]
            vdef = torch.stack([p[0] for p in dpairs]).to(dev)
            gm = torch.stack([p[1] for p in dpairs])
            e1, e2, edef = enc(v1), enc(v2), enc(vdef)
            closs, z1, z2 = dense_ntxent(e1, e2)
            dl = defect_loss(edef, e1, gm)
            loss = closs + LAM * dl + VIC * (vicreg_var(z1) + vicreg_var(z2))
            opt.zero_grad(); loss.backward(); opt.step()
            tc += closs.item(); td += dl.item(); nb += 1
        print(f"  epoch {ep}/{EPOCHS}  contrast={tc/nb:.4f}  defect={td/nb:.4f}", flush=True)
    enc.eval()
    return enc


def mask_grid(path, ps):
    if path is None or not os.path.exists(path):
        return np.zeros((ps, ps), np.uint8)
    mm = np.array(Image.open(path).convert("L").resize((IMG, IMG))) > 0
    return mm.reshape(ps, IMG // ps, ps, IMG // ps).max(axis=(1, 3)).astype(np.uint8)


@torch.no_grad()
def extract_all(enc):
    os.makedirs(OUT, exist_ok=True)

    def extract(imgs):
        fm = enc.fmap(imgs.to(dev))
        b, c, h, w = fm.shape
        return fm.permute(0, 2, 3, 1).reshape(b, h * w, c).to(torch.float16).cpu()

    for c in ALL:
        cd = os.path.join(DATA, c)
        tr = sorted(glob.glob(os.path.join(cd, "train", "good", "*.png")))[:MAX_TRAIN]
        trf = []
        for i in range(0, len(tr), BS):
            trf.append(extract(torch.stack([norm(det(Image.open(f).convert("RGB"))) for f in tr[i:i+BS]])))
        trf = torch.cat(trf); ps = int(round(math.sqrt(trf.shape[1])))
        tef, il, pl = [], [], []
        for dd in sorted(glob.glob(os.path.join(cd, "test", "*"))):
            defect = os.path.basename(dd); files = sorted(glob.glob(os.path.join(dd, "*.png")))
            for i in range(0, len(files), BS):
                ch = files[i:i + BS]
                tef.append(extract(torch.stack([norm(det(Image.open(f).convert("RGB"))) for f in ch])))
                for f in ch:
                    if defect == "good":
                        il.append(0); pl.append(np.zeros((ps, ps), np.uint8))
                    else:
                        il.append(1)
                        gt = f.replace(os.sep+"test"+os.sep, os.sep+"ground_truth"+os.sep).replace(".png", "_mask.png")
                        pl.append(mask_grid(gt, ps))
        torch.save({"train_feats": trf, "test_feats": torch.cat(tef),
                    "test_img_label": torch.tensor(il, dtype=torch.int8),
                    "test_pix_label": torch.tensor(np.stack(pl)), "patch_hw": (ps, ps)},
                   os.path.join(OUT, c + ".pt"))
        print(f"  [추출] {c}: train {tuple(trf.shape)}", flush=True)


def _to_map(feat_pc):
    x = feat_pc.float(); x = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
    c = x.shape[-1]; s = int(round(math.sqrt(x.shape[0])))
    return x.transpose(0, 1).reshape(c, s, s)


@torch.no_grad()
def eval_cache(cats, K=5, draws=5):
    outs = []
    for cat in cats:
        d = torch.load(os.path.join(OUT, cat + ".pt"), map_location="cpu", weights_only=False)
        pool, test, lab = d["train_feats"], d["test_feats"], d["test_img_label"].numpy()
        ca = []
        for draw in range(draws):
            g = torch.Generator().manual_seed(7 * K + draw)
            sidx = torch.randperm(pool.shape[0], generator=g)[:K]
            supp = torch.stack([_to_map(pool[i]) for i in sidx]).to(dev)
            bank = supp.permute(0, 2, 3, 1).reshape(-1, supp.shape[1])
            img_s = []
            for j in range(len(test)):
                q = _to_map(test[j]).to(dev).permute(1, 2, 0).reshape(-1, supp.shape[1])
                img_s.append(torch.cdist(q, bank).min(1).values.max().item())
            ca.append(roc_auc_score(lab, img_s))
        outs.append(float(np.mean(ca)))
    return float(np.mean(outs)), outs


def eff_rank(cats):
    eff = []
    for cat in cats:
        d = torch.load(os.path.join(OUT, cat + ".pt"), map_location="cpu", weights_only=False)
        X = d["train_feats"][:20].reshape(-1, d["train_feats"].shape[-1]).float(); X = X - X.mean(0)
        s = torch.linalg.svdvals(X[torch.randperm(X.shape[0])[:2000]]); p = s / s.sum()
        eff.append(float(torch.exp(-(p * (p + 1e-12).log()).sum())))
    return float(np.mean(eff))


def main():
    random.seed(0); torch.manual_seed(0); np.random.seed(0)
    print("=" * 64); print("모형1 수정: 결함민감형 dense SSL"); print("=" * 64)
    enc = train()
    print("\n[재추출] 15종 → feature_cache_wrn_defect", flush=True)
    extract_all(enc)
    print("\n[평가: patch-NN K=5 + 유효랭크]", flush=True)
    tr, _ = eval_cache(TRAIN_CATS); te, te_list = eval_cache(TEST_CATS); er = eff_rank(TEST_CATS)
    print("\n" + "=" * 64); print("결과 (Image-AUROC K=5)"); print("=" * 64)
    print(f"  결함민감SSL : in={tr:.3f}  held-out={te:.3f}  유효랭크={er:.1f}")
    print(f"  (대조) ImageNet 0.947 / CutPaste 0.926 / dense대조 0.917 / pooled 0.832")
    print(f"  테스트5종별: " + ", ".join(f"{c}={v:.3f}" for c, v in zip(TEST_CATS, te_list)))


if __name__ == "__main__":
    main()
