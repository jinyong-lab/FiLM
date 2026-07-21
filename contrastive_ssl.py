"""모형1: 제안서 원의도 그대로의 '국소 보존형 contrastive SSL'(CutPaste 분류가 아님).
- 제안서 3.1: 데이터증강 기반 contrastive learning, 같은 데이터에 서로다른 '국소' 증강을
  가해 positive pair 구성 → 국소 결함(질감 불연속)까지 캡처하는 유도편향. 이후 백본 동결.
- 구현: WRN-50-2(ImageNet) layer2+ 파인튜닝(conv1/bn1/layer1 동결), NT-Xent(InfoNCE).
  국소보존 증강=RandomResizedCrop(0.7~1.0)+미세노이즈+약지터+국소erase(거시크롭·플립 배제).
- 그 백본으로 15종 재추출(ImageNet 캐시와 동일 파이프라인) → feature_cache_wrn_contrastive.
- patch-NN으로 held-out5/train10 평가 → ImageNet(0.883/0.947)·CutPaste(0.894/0.926)와 대조.
"""
import os, glob, math, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
from sklearn.metrics import roc_auc_score

DATA = r"C:\Users\HOSEO\Desktop\K-DS\datasets\mvtec_ad\mvtech_anomaly_detection"
OUT = r"C:\Users\HOSEO\Desktop\K-DS\code\AnomalyDetection\feature_cache_wrn_contrastive"
IM_MEAN, IM_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
IMG, MAX_TRAIN, BS = 224, 200, 24
EPOCHS, LR, TEMP = 25, 3e-4, 0.2
ALL = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut",
       "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"]
TEST_CATS = ["wood", "cable", "zipper", "transistor", "hazelnut"]
TRAIN_CATS = [c for c in ALL if c not in TEST_CATS]
dev = "cuda" if torch.cuda.is_available() else "cpu"

# 결정적 추출 파이프라인(ImageNet 캐시와 동일 — 공정비교 필수)
det = T.Compose([T.Resize(IMG, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(IMG), T.ToTensor()])
norm = T.Normalize(IM_MEAN, IM_STD)


class AddNoise:
    def __init__(self, s=0.02): self.s = s
    def __call__(self, x): return x + self.s * torch.randn_like(x)


# 국소 보존형 증강(2뷰): 대부분 영역 유지(0.7~1.0), 미세노이즈·약지터·국소erase. 거시크롭/플립 배제.
aug = T.Compose([
    T.RandomResizedCrop(IMG, scale=(0.7, 1.0), ratio=(0.85, 1.18),
                        interpolation=T.InterpolationMode.BICUBIC),
    T.RandomApply([T.ColorJitter(0.2, 0.2, 0.2, 0.03)], p=0.6),
    T.RandomApply([T.GaussianBlur(3, (0.1, 1.2))], p=0.3),
    T.ToTensor(),
    AddNoise(0.02),
    T.RandomErasing(p=0.4, scale=(0.01, 0.04), ratio=(0.5, 2.0)),  # 국소 결함 모사
    norm,
])


def load_pils(cat):
    files = sorted(glob.glob(os.path.join(DATA, cat, "train", "good", "*.png")))[:MAX_TRAIN]
    return [Image.open(f).convert("RGB").resize((256, 256), Image.BICUBIC) for f in files]


def build():
    m = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
    m.fc = nn.Identity()
    for n, p in m.named_parameters():
        if n.startswith(("conv1", "bn1", "layer1.")):
            p.requires_grad_(False)
    proj = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Linear(512, 128))
    return m.to(dev), proj.to(dev)


def nt_xent(z1, z2, temp=TEMP):
    b = z1.shape[0]
    z = F.normalize(torch.cat([z1, z2], 0), dim=1)          # [2b, d]
    sim = z @ z.T / temp                                    # [2b, 2b]
    sim.fill_diagonal_(-1e9)                                # 자기 자신 제외
    targets = torch.cat([torch.arange(b) + b, torch.arange(b)]).to(dev)  # i <-> i+b
    return F.cross_entropy(sim, targets)


def contrastive_train():
    print("[대조SSL] 10개 학습 카테고리 정상 이미지 로딩...", flush=True)
    imgs = []
    for c in TRAIN_CATS:
        imgs += load_pils(c)
    print(f"[대조SSL] 총 {len(imgs)}장. 국소보존 contrastive(NT-Xent) 시작 (epochs={EPOCHS}, BS={BS})", flush=True)
    m, proj = build()
    params = [p for p in m.parameters() if p.requires_grad] + list(proj.parameters())
    opt = torch.optim.Adam(params, lr=LR)
    for ep in range(1, EPOCHS + 1):
        random.shuffle(imgs); tot, nb = 0.0, 0
        m.train(); proj.train()
        for i in range(0, len(imgs), BS):
            b = imgs[i:i + BS]
            if len(b) < 2:
                continue
            v1 = torch.stack([aug(im) for im in b]).to(dev)
            v2 = torch.stack([aug(im) for im in b]).to(dev)
            z1 = proj(m(v1)); z2 = proj(m(v2))
            loss = nt_xent(z1, z2)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        print(f"  epoch {ep}/{EPOCHS}  NT-Xent={tot/nb:.4f}", flush=True)
    m.eval()
    return m


def make_extract(model):
    feats = {}
    model.layer2.register_forward_hook(lambda mm, i, o: feats.__setitem__("l2", o))
    model.layer3.register_forward_hook(lambda mm, i, o: feats.__setitem__("l3", o))

    @torch.no_grad()
    def extract(imgs):
        model(imgs.to(dev))
        l2, l3 = feats["l2"], feats["l3"]
        l3 = F.interpolate(l3, size=l2.shape[-2:], mode="bilinear", align_corners=False)
        fm = torch.cat([l2, l3], 1)
        b, c, h, w = fm.shape
        return fm.permute(0, 2, 3, 1).reshape(b, h * w, c).to(torch.float16).cpu()
    return extract


def mask_grid(path, ps):
    if path is None or not os.path.exists(path):
        return np.zeros((ps, ps), np.uint8)
    mm = np.array(Image.open(path).convert("L").resize((IMG, IMG))) > 0
    return mm.reshape(ps, IMG // ps, ps, IMG // ps).max(axis=(1, 3)).astype(np.uint8)


def extract_all(model):
    os.makedirs(OUT, exist_ok=True)
    extract = make_extract(model)
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
        print(f"  [추출] {c}: train {tuple(trf.shape)} test {tuple(torch.cat(tef).shape)}", flush=True)


@torch.no_grad()
def eval_cache(cats, K=5, draws=5):
    def to_map(feat_pc):
        x = feat_pc.float(); x = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
        c = x.shape[-1]; s = int(round(math.sqrt(x.shape[0])))
        return x.transpose(0, 1).reshape(c, s, s)
    outs = []
    for cat in cats:
        d = torch.load(os.path.join(OUT, cat + ".pt"), map_location="cpu", weights_only=False)
        pool, test, lab = d["train_feats"], d["test_feats"], d["test_img_label"].numpy()
        ca = []
        for draw in range(draws):
            g = torch.Generator().manual_seed(7 * K + draw)
            sidx = torch.randperm(pool.shape[0], generator=g)[:K]
            supp = torch.stack([to_map(pool[i]) for i in sidx]).to(dev)
            bank = supp.permute(0, 2, 3, 1).reshape(-1, supp.shape[1])
            img_s = []
            for j in range(len(test)):
                q = to_map(test[j]).to(dev).permute(1, 2, 0).reshape(-1, supp.shape[1])
                img_s.append(torch.cdist(q, bank).min(1).values.max().item())
            ca.append(roc_auc_score(lab, img_s))
        outs.append(float(np.mean(ca)))
    return float(np.mean(outs)), outs


def main():
    random.seed(0); torch.manual_seed(0); np.random.seed(0)
    print("=" * 64); print("모형1: 국소보존 Contrastive SSL (제안서 3.1 원의도)"); print("=" * 64)
    model = contrastive_train()
    print("\n[재추출] 15종 → feature_cache_wrn_contrastive", flush=True)
    extract_all(model)
    print("\n[평가: patch-NN, K=5]", flush=True)
    tr, _ = eval_cache(TRAIN_CATS)
    te, te_list = eval_cache(TEST_CATS)
    print("\n" + "=" * 64)
    print("결과 (Image-AUROC, K=5)")
    print("=" * 64)
    print(f"  대조SSL   : in-domain(학습10)={tr:.3f}  held-out(테스트5)={te:.3f}")
    print(f"  (대조군) ImageNet : in=0.883  held-out=0.947")
    print(f"  (대조군) CutPaste : in=0.894  held-out=0.926")
    print(f"  테스트5종별: " + ", ".join(f"{c}={v:.3f}" for c, v in zip(TEST_CATS, te_list)))
    print(f"\n[판정] held-out이 0.947 넘으면 대조SSL이 ImageNet 개선. 미달이면 CutPaste처럼 특화.")


if __name__ == "__main__":
    main()
