"""모형1 재설계: dense(패치 단위) 국소 Contrastive SSL.
- 이전 pooled(전역) NT-Xent는 전역 instance-discrimination→특징 랭크 붕괴(719→397)로 국소 유실.
- 재설계: (1)dense 대조=특징맵 공간위치별 대조(국소보존, #1) (2)정렬보존 증강(기하변형X,
  국소 광도교란만 다르게→위치 대응 유지) (3)랭크보존=LR1e-5·8에폭·layer2동결·VICReg 분산항.
- 추출은 ImageNet 캐시와 동일 파이프라인 → feature_cache_wrn_dense. patch-NN+유효랭크로 대조.
"""
import os, glob, math, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
from sklearn.metrics import roc_auc_score

DATA = r"C:\Users\HOSEO\Desktop\K-DS\datasets\mvtec_ad\mvtech_anomaly_detection"
OUT = r"C:\Users\HOSEO\Desktop\K-DS\code\AnomalyDetection\feature_cache_wrn_dense"
IM_MEAN, IM_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
IMG, MAX_TRAIN, BS = 224, 200, 16
EPOCHS, LR, TEMP, N_LOC, VIC = 8, 1e-5, 0.2, 64, 1.0
ALL = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut",
       "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"]
TEST_CATS = ["wood", "cable", "zipper", "transistor", "hazelnut"]
TRAIN_CATS = [c for c in ALL if c not in TEST_CATS]
dev = "cuda" if torch.cuda.is_available() else "cpu"

det = T.Compose([T.Resize(IMG, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(IMG), T.ToTensor()])
norm = T.Normalize(IM_MEAN, IM_STD)


class AddNoise:
    def __init__(self, s=0.02): self.s = s
    def __call__(self, x): return x + self.s * torch.randn_like(x)


# 정렬보존 광도 교란(기하변형 없음 → 위치 p↔p 대응 유지)
photo = T.Compose([
    T.RandomApply([T.ColorJitter(0.2, 0.2, 0.2, 0.03)], p=0.6),
    T.RandomApply([T.GaussianBlur(3, (0.1, 1.2))], p=0.3),
    AddNoise(0.02),
    T.RandomErasing(p=0.5, scale=(0.01, 0.05), ratio=(0.5, 2.0)),
])


def two_views(base):                       # base [3,224,224] in [0,1]
    return norm(photo(base.clone())), norm(photo(base.clone()))


def load_bases(cat):
    files = sorted(glob.glob(os.path.join(DATA, cat, "train", "good", "*.png")))[:MAX_TRAIN]
    return [det(Image.open(f).convert("RGB")) for f in files]


class Encoder(nn.Module):
    """WRN-50-2(layer2 동결, layer3 학습) → layer2+layer3 concat 특징맵 → 1x1 proj(dense emb)."""
    def __init__(self):
        super().__init__()
        m = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        m.fc = nn.Identity()
        for n, p in m.named_parameters():
            if n.startswith(("conv1", "bn1", "layer1.", "layer2.")):   # layer2까지 동결(랭크 보존)
                p.requires_grad_(False)
        self.m = m
        self.feats = {}
        m.layer2.register_forward_hook(lambda mm, i, o: self.feats.__setitem__("l2", o))
        m.layer3.register_forward_hook(lambda mm, i, o: self.feats.__setitem__("l3", o))
        self.proj = nn.Conv2d(1536, 128, 1)

    def fmap(self, x):                      # [B,3,224,224] -> concat 특징맵 [B,1536,28,28]
        self.m(x)
        l2, l3 = self.feats["l2"], self.feats["l3"]
        l3 = F.interpolate(l3, size=l2.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([l2, l3], 1)

    def forward(self, x):                   # dense embedding [B,128,28,28]
        return self.proj(self.fmap(x))


def dense_ntxent(v1, v2, temp=TEMP, n_loc=N_LOC):
    b, d, h, w = v1.shape
    v1 = v1.reshape(b, d, h * w); v2 = v2.reshape(b, d, h * w)
    idx = torch.randperm(h * w, device=v1.device)[:n_loc]           # 공유 위치 서브샘플
    z1 = F.normalize(v1[:, :, idx].permute(0, 2, 1).reshape(b * n_loc, d), dim=1)
    z2 = F.normalize(v2[:, :, idx].permute(0, 2, 1).reshape(b * n_loc, d), dim=1)
    n = z1.shape[0]
    z = torch.cat([z1, z2], 0)                                      # [2n, d]
    sim = z @ z.T / temp
    sim.fill_diagonal_(-1e9)
    targets = torch.cat([torch.arange(n) + n, torch.arange(n)]).to(v1.device)  # 같은 (b,loc) 대응
    return F.cross_entropy(sim, targets), z1, z2


def vicreg_var(z, eps=1e-4):                # 분산 붕괴 방지(각 차원 std>=1 유도)
    std = torch.sqrt(z.var(0) + eps)
    return torch.mean(F.relu(1.0 - std))


def train():
    print(f"[dense대조] 학습 이미지 로딩...", flush=True)
    bases = []
    for c in TRAIN_CATS:
        bases += load_bases(c)
    print(f"[dense대조] 총 {len(bases)}장. dense 국소 contrastive 시작 "
          f"(LR={LR}, epochs={EPOCHS}, layer2동결, VICReg={VIC})", flush=True)
    enc = Encoder().to(dev)
    params = [p for p in enc.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=LR)
    for ep in range(1, EPOCHS + 1):
        random.shuffle(bases); tot, nb = 0.0, 0
        enc.train()
        for i in range(0, len(bases), BS):
            b = bases[i:i + BS]
            if len(b) < 2:
                continue
            v1 = torch.stack([two_views(im)[0] for im in b]).to(dev)
            v2 = torch.stack([two_views(im)[1] for im in b]).to(dev)
            e1, e2 = enc(v1), enc(v2)
            closs, z1, z2 = dense_ntxent(e1, e2)
            loss = closs + VIC * (vicreg_var(z1) + vicreg_var(z2))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        print(f"  epoch {ep}/{EPOCHS}  loss={tot/nb:.4f}", flush=True)
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

    def extract(imgs):                      # 추출=concat 특징맵(proj 이전, ImageNet 캐시와 동일 형식)
        fm = enc.fmap(imgs.to(dev))
        b, c, h, w = fm.shape
        return fm.permute(0, 2, 3, 1).reshape(b, h * w, c).to(torch.float16).cpu()

    for c in ALL:
        cd = os.path.join(DATA, c)
        tr = sorted(glob.glob(os.path.join(cd, "train", "good", "*.png")))[:MAX_TRAIN]
        trf = []
        for i in range(0, len(tr), BS):
            trf.append(extract(torch.stack([norm(det(Image.open(f).convert("RGB"))) for f in tr[i:i+BS]])))
        trf = torch.cat(trf)
        ps = int(round(math.sqrt(trf.shape[1])))
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
    print("=" * 64); print("모형1 재설계: dense(패치단위) 국소 Contrastive SSL"); print("=" * 64)
    enc = train()
    print("\n[재추출] 15종 → feature_cache_wrn_dense", flush=True)
    extract_all(enc)
    print("\n[평가: patch-NN K=5 + 유효랭크]", flush=True)
    tr, _ = eval_cache(TRAIN_CATS); te, te_list = eval_cache(TEST_CATS)
    er_te = eff_rank(TEST_CATS)
    print("\n" + "=" * 64)
    print("결과 (Image-AUROC K=5)")
    print("=" * 64)
    print(f"  dense대조 : in-domain={tr:.3f}  held-out={te:.3f}   유효랭크(held-out)={er_te:.1f}")
    print(f"  ImageNet  : in=0.883  held-out=0.947   유효랭크 719")
    print(f"  CutPaste  : in=0.894  held-out=0.926   유효랭크 690")
    print(f"  pooled대조: in=0.765  held-out=0.832   유효랭크 397 (붕괴)")
    print(f"  테스트5종별: " + ", ".join(f"{c}={v:.3f}" for c, v in zip(TEST_CATS, te_list)))
    print(f"\n[판정] 유효랭크가 690+로 회복되고 held-out>0.926이면 dense 재설계 성공.")


if __name__ == "__main__":
    main()
