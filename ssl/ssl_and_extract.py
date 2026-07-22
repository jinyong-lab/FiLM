"""Stage 1: Local SSL (CutPaste) 백본 사전학습 + 동일방식 특징 재추출.
- WRN-50-2(ImageNet) 백본을 10개 학습 카테고리 정상 이미지에 CutPaste 3-way 분류로 파인튜닝
  (normal/cutpaste/scar) -> 백본이 '국소 불연속(이상)'에 민감해지도록 유도편향 주입.
- 그 SSL 백본으로 15종 전체 재추출(layer2+layer3, 224, 28x28x1536) -> feature_cache_wrn_ssl.
- 이후 film_ad_real.py를 이 캐시로 재실행하면 논문 완전 충실(SSL+메타학습).
"""
import os, glob, math, random, argparse
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights

DATA = r"C:\Users\HOSEO\Desktop\K-DS\datasets\mvtec_ad\mvtech_anomaly_detection"
OUT = r"C:\Users\HOSEO\Desktop\K-DS\code\AnomalyDetection\feature_cache_wrn_ssl"
IM_MEAN, IM_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
IMG, MAX_TRAIN, BS = 224, 200, 16
EPOCHS, LR = 8, 1e-4
ALL = ["bottle","cable","capsule","carpet","grid","hazelnut","leather","metal_nut",
       "pill","screw","tile","toothbrush","transistor","wood","zipper"]
TEST_CATS = ["wood","cable","zipper","transistor","hazelnut"]
TRAIN_CATS = [c for c in ALL if c not in TEST_CATS]
dev = "cuda" if torch.cuda.is_available() else "cpu"

resize = T.Compose([T.Resize(IMG, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(IMG), T.ToTensor()])
norm = T.Normalize(IM_MEAN, IM_STD)


def cutpaste_patch(img):                       # img [3,H,W] in [0,1]
    _, H, W = img.shape
    area = random.uniform(0.02, 0.15) * H * W; ar = random.uniform(0.3, 3.3)
    ph = max(1, min(H - 1, int(round(math.sqrt(area * ar)))))
    pw = max(1, min(W - 1, int(round(math.sqrt(area / ar)))))
    sy, sx = random.randint(0, H - ph), random.randint(0, W - pw)
    patch = img[:, sy:sy + ph, sx:sx + pw].clone()
    dy, dx = random.randint(0, H - ph), random.randint(0, W - pw)
    out = img.clone(); out[:, dy:dy + ph, dx:dx + pw] = patch
    return out


def cutpaste_scar(img):
    _, H, W = img.shape
    pw, ph = random.randint(2, 16), random.randint(10, 25)
    sy, sx = random.randint(0, H - ph), random.randint(0, W - pw)
    patch = TF.rotate(img[:, sy:sy + ph, sx:sx + pw].clone(), random.uniform(-45, 45))
    dy, dx = random.randint(0, H - ph), random.randint(0, W - pw)
    out = img.clone(); out[:, dy:dy + ph, dx:dx + pw] = patch
    return out


def load_imgs(cat, split_glob):
    files = sorted(glob.glob(split_glob))[:MAX_TRAIN]
    return [resize(Image.open(f).convert("RGB")) for f in files]


def build_model():
    m = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(2048, 3)
    # 저수준 보존: conv1/bn1/layer1 동결, layer2+ 학습
    for n, p in m.named_parameters():
        if n.startswith(("conv1", "bn1", "layer1.")):
            p.requires_grad_(False)
    return m.to(dev)


def ssl_train():
    print("[SSL] 10개 학습 카테고리 정상 이미지 로딩...", flush=True)
    imgs = []
    for c in TRAIN_CATS:
        imgs += load_imgs(c, os.path.join(DATA, c, "train", "good", "*.png"))
    print(f"[SSL] 총 {len(imgs)}장. CutPaste 3-way 파인튜닝 시작 (epochs={EPOCHS})", flush=True)
    m = build_model()
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=LR)
    ce = nn.CrossEntropyLoss()
    for ep in range(1, EPOCHS + 1):
        random.shuffle(imgs); tot, nb = 0.0, 0
        m.train()
        for i in range(0, len(imgs), BS):
            b = imgs[i:i + BS]
            x0 = torch.stack([norm(im) for im in b])
            x1 = torch.stack([norm(cutpaste_patch(im)) for im in b])
            x2 = torch.stack([norm(cutpaste_scar(im)) for im in b])
            X = torch.cat([x0, x1, x2]).to(dev)
            y = torch.cat([torch.zeros(len(b)), torch.ones(len(b)), 2 * torch.ones(len(b))]).long().to(dev)
            logits = m(X); loss = ce(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        print(f"  epoch {ep}/{EPOCHS}  CE={tot/nb:.4f}", flush=True)
    m.eval()
    return m


def make_extract(model):
    feats = {}
    model.layer2.register_forward_hook(lambda mm, i, o: feats.__setitem__("l2", o))
    model.layer3.register_forward_hook(lambda mm, i, o: feats.__setitem__("l3", o))

    @torch.no_grad()
    def extract(imgs):                          # imgs [B,3,H,W] normalized
        model(imgs.to(dev))
        l2, l3 = feats["l2"], feats["l3"]
        l3 = F.interpolate(l3, size=l2.shape[-2:], mode="bilinear", align_corners=False)
        fm = torch.cat([l2, l3], 1)
        B, C, H, W = fm.shape
        return fm.permute(0, 2, 3, 1).reshape(B, H * W, C).to(torch.float16).cpu()
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
            trf.append(extract(torch.stack([norm(resize(Image.open(f).convert("RGB"))) for f in tr[i:i+BS]])))
        trf = torch.cat(trf)
        ps = int(round(math.sqrt(trf.shape[1])))
        tef, il, pl = [], [], []
        for dd in sorted(glob.glob(os.path.join(cd, "test", "*"))):
            defect = os.path.basename(dd); files = sorted(glob.glob(os.path.join(dd, "*.png")))
            for i in range(0, len(files), BS):
                ch = files[i:i + BS]
                tef.append(extract(torch.stack([norm(resize(Image.open(f).convert("RGB"))) for f in ch])))
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
        print(f"  [추출] {c}: train {tuple(trf.shape)} test {tuple(torch.cat(tef).shape)} grid {ps}x{ps}", flush=True)


if __name__ == "__main__":
    random.seed(0); torch.manual_seed(0); np.random.seed(0)
    print("=" * 60); print("Stage 1: Local SSL(CutPaste) + 재추출"); print("=" * 60)
    model = ssl_train()
    print("\n[재추출] 15종 전체 SSL 백본으로 특징 추출 ->", OUT, flush=True)
    extract_all(model)
    print("Done. film_ad_real.py의 CACHE를 feature_cache_wrn_ssl로 바꿔 재실행하세요.")
