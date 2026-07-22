"""미팅용 시각화: '우리가 어떻게 탐지하는가' — 피처맵 + patch-NN 이상 히트맵.
테스트(held-out) 카테고리의 실제 결함 이미지에 대해:
  원본 | 피처맵(1536d→PCA 3d→RGB) | 이상 히트맵(패치별 최근접거리) | 오버레이(+GT 윤곽)
결함민감 SSL 특징 캐시(feature_cache_wrn_defect) 사용. 결과 PNG를 figs/에 저장.
"""
import os, glob, math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

REPO = r"C:\Users\HOSEO\Desktop\K-DS\code\AnomalyDetection"
DATA = r"C:\Users\HOSEO\Desktop\K-DS\datasets\mvtec_ad\mvtech_anomaly_detection"
CACHE = os.path.join(REPO, "feature_cache_wrn_defect")
OUTDIR = r"C:\Users\HOSEO\Desktop\K-DS\figs"
os.makedirs(OUTDIR, exist_ok=True)
from PIL import Image
S, IMG, C = 28, 224, 1536
CATS = ["hazelnut", "wood", "cable"]


def to_map(feat_pc):
    x = feat_pc.float(); x = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
    return x.transpose(0, 1).reshape(C, S, S)


def test_paths(cat):
    out = []
    for dd in sorted(glob.glob(os.path.join(DATA, cat, "test", "*"))):
        defect = os.path.basename(dd)
        for f in sorted(glob.glob(os.path.join(dd, "*.png"))):
            out.append((f, defect))
    return out


def load_img(path):
    im = Image.open(path).convert("RGB").resize((IMG, IMG), Image.BICUBIC)
    return np.asarray(im) / 255.0


def pca_rgb(qflat):                         # [784, C] -> [28,28,3] in [0,1]
    x = qflat - qflat.mean(0)
    u, s, v = torch.pca_lowrank(x, q=3)
    p = x @ v[:, :3]
    p = (p - p.min(0).values) / (p.max(0).values - p.min(0).values + 1e-8)
    return p.reshape(S, S, 3).numpy()


def main():
    d0 = {c: torch.load(os.path.join(CACHE, c + ".pt"), map_location="cpu", weights_only=False) for c in CATS}
    fig, axes = plt.subplots(len(CATS), 4, figsize=(13, 3.2 * len(CATS)))
    col_titles = ["Input (defect)", "Feature map (PCA-RGB)", "Anomaly heatmap (patch-NN)", "Overlay + GT(green)"]
    for r, cat in enumerate(CATS):
        d = d0[cat]; pool, test = d["train_feats"], d["test_feats"]
        lab = d["test_img_label"].numpy(); pix = d["test_pix_label"].numpy()
        paths = test_paths(cat)
        # 결함 마스크가 큰 예시 선택
        cand = [i for i in range(len(test)) if lab[i] == 1 and pix[i].sum() > 6]
        qi = max(cand, key=lambda i: pix[i].sum()) if cand else int(np.where(lab == 1)[0][0])
        # support bank(정상 K=5)
        g = torch.Generator().manual_seed(0)
        sidx = torch.randperm(pool.shape[0], generator=g)[:5]
        supp = torch.stack([to_map(pool[i]) for i in sidx])
        bank = supp.permute(0, 2, 3, 1).reshape(-1, C)
        qmap = to_map(test[qi]); qflat = qmap.permute(1, 2, 0).reshape(-1, C)
        dist = torch.cdist(qflat, bank).min(1).values.reshape(S, S).numpy()
        amap = (dist - dist.min()) / (dist.max() - dist.min() + 1e-8)
        amap_up = np.asarray(Image.fromarray((amap * 255).astype(np.uint8)).resize((IMG, IMG), Image.BICUBIC)) / 255.0
        img = load_img(paths[qi][0])
        gt = pix[qi]
        gt_up = np.asarray(Image.fromarray((gt * 255).astype(np.uint8)).resize((IMG, IMG), Image.NEAREST)) / 255.0

        axes[r, 0].imshow(img)
        axes[r, 1].imshow(pca_rgb(qflat))
        axes[r, 2].imshow(amap_up, cmap="jet")
        axes[r, 3].imshow(img)
        axes[r, 3].imshow(cm.jet(amap_up)[..., :3], alpha=0.45)
        axes[r, 3].contour(gt_up, levels=[0.5], colors="lime", linewidths=1.8)
        axes[r, 0].set_ylabel(f"{cat}\n({paths[qi][1]})", fontsize=11)
        for c in range(4):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(col_titles[c], fontsize=11)
    plt.tight_layout()
    out = os.path.join(OUTDIR, "detection_examples.png")
    plt.savefig(out, dpi=110, bbox_inches="tight")
    print("SAVED:", out)


if __name__ == "__main__":
    main()
