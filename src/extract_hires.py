"""Hi-res multi-scale feature extractor (Q2: sharper magnifying glass).

FROZEN WRN-50-2 (ImageNet), requires_grad=False, eval. For each 224x224 image we grab
  layer1 (56x56, 256ch), layer2 (28x28, 512ch), layer3 (14x14, 1024ch).
We store the THREE layers at their NATIVE resolution in fp16 (disk-efficient: 2.8 MB/img
vs 11 MB/img if we pre-upsampled to 56x56). On load (hires_ad.py) we bilinear-upsample
l2,l3 to 56x56, concat -> [56,56,1792] multi-scale map, then L2-normalize per patch.
This is mathematically identical to caching the pre-upsampled 56x56 map, just 4x smaller.

Also stores full-resolution (224x224) test pixel masks so pixel-AUROC can be measured at
full resolution (NOT downsampled to the 28/56 grid). Same preprocessing as defect_ssl.py.

Output: feature_cache_wrn_hires/{cat}.pt with keys
  train_l1 [N,56,56,256] fp16, train_l2 [N,28,28,512] fp16, train_l3 [N,14,14,1024] fp16
  test_l1, test_l2, test_l3 (same shapes for the test split)
  test_img_label [Ntest] int8, test_pix_label [Ntest,224,224] uint8 (full-res GT)
"""
import os, glob, math, time
import numpy as np
import torch, torch.nn as nn
from PIL import Image
import torchvision.transforms as T
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights

DATA = r"C:\Users\HOSEO\Desktop\K-DS\datasets\mvtec_ad\mvtech_anomaly_detection"
OUT = r"C:\Users\HOSEO\Desktop\K-DS\code\AnomalyDetection\feature_cache_wrn_hires"
IM_MEAN, IM_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
IMG, MAX_TRAIN, BS = 224, 200, 8
ALL = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut",
       "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"]
dev = "cuda" if torch.cuda.is_available() else "cpu"

# SAME preprocessing as defect_ssl.py: BICUBIC resize -> center crop -> tensor -> ImageNet norm.
det = T.Compose([T.Resize(IMG, interpolation=T.InterpolationMode.BICUBIC),
                 T.CenterCrop(IMG), T.ToTensor()])
norm = T.Normalize(IM_MEAN, IM_STD)
# masks: same geometry (resize/crop) but NEAREST to keep them binary; kept at full 224 res.
mask_tf = T.Compose([T.Resize(IMG, interpolation=T.InterpolationMode.NEAREST), T.CenterCrop(IMG)])


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        m = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        m.fc = nn.Identity()
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        self.m = m
        self.feats = {}
        m.layer1.register_forward_hook(lambda mm, i, o: self.feats.__setitem__("l1", o))
        m.layer2.register_forward_hook(lambda mm, i, o: self.feats.__setitem__("l2", o))
        m.layer3.register_forward_hook(lambda mm, i, o: self.feats.__setitem__("l3", o))

    @torch.no_grad()
    def forward(self, x):
        self.m(x.to(dev))
        # -> [b,H,W,C] fp16 on cpu, native resolution per layer
        def nhwc(o):
            return o.permute(0, 2, 3, 1).contiguous().to(torch.float16).cpu()
        return nhwc(self.feats["l1"]), nhwc(self.feats["l2"]), nhwc(self.feats["l3"])


def load_batch(files):
    return torch.stack([norm(det(Image.open(f).convert("RGB"))) for f in files])


def full_mask(gt_path):
    if gt_path is None or not os.path.exists(gt_path):
        return np.zeros((IMG, IMG), np.uint8)
    m = mask_tf(Image.open(gt_path).convert("L"))
    return (np.array(m) > 0).astype(np.uint8)


@torch.no_grad()
def extract_split(bb, files, with_masks):
    l1s, l2s, l3s, ilab, plab = [], [], [], [], []
    for i in range(0, len(files), BS):
        chunk = files[i:i + BS]
        a, b, c = bb(load_batch(chunk))
        l1s.append(a); l2s.append(b); l3s.append(c)
        if with_masks is not None:
            for f in chunk:
                defect = with_masks(f)
                if defect is None:            # good
                    ilab.append(0); plab.append(np.zeros((IMG, IMG), np.uint8))
                else:
                    ilab.append(1); plab.append(defect)
    out = {"l1": torch.cat(l1s), "l2": torch.cat(l2s), "l3": torch.cat(l3s)}
    if with_masks is not None:
        out["img_label"] = torch.tensor(ilab, dtype=torch.int8)
        out["pix_label"] = torch.from_numpy(np.stack(plab))   # [N,224,224] uint8
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    bb = Backbone().to(dev)
    # sanity: confirm channel/spatial dims
    a, b, c = bb(load_batch(sorted(glob.glob(os.path.join(DATA, "wood", "train", "good", "*.png")))[:2]))
    print(f"[dim check] l1={tuple(a.shape)} l2={tuple(b.shape)} l3={tuple(c.shape)}", flush=True)
    assert a.shape[1:] == (56, 56, 256) and b.shape[1:] == (28, 28, 512) and c.shape[1:] == (14, 14, 1024)
    for ci, cat in enumerate(ALL):
        t0 = time.time()
        cd = os.path.join(DATA, cat)
        tr_files = sorted(glob.glob(os.path.join(cd, "train", "good", "*.png")))[:MAX_TRAIN]
        train = extract_split(bb, tr_files, with_masks=None)

        te_files = []
        for dd in sorted(glob.glob(os.path.join(cd, "test", "*"))):
            te_files += [(f, os.path.basename(dd)) for f in sorted(glob.glob(os.path.join(dd, "*.png")))]
        flat = [f for f, _ in te_files]

        def mask_of(f):
            defect = dict(te_files)[f]
            if defect == "good":
                return None
            gt = f.replace(os.sep + "test" + os.sep, os.sep + "ground_truth" + os.sep).replace(".png", "_mask.png")
            return full_mask(gt)

        test = extract_split(bb, flat, with_masks=mask_of)

        torch.save({
            "train_l1": train["l1"], "train_l2": train["l2"], "train_l3": train["l3"],
            "test_l1": test["l1"], "test_l2": test["l2"], "test_l3": test["l3"],
            "test_img_label": test["img_label"], "test_pix_label": test["pix_label"],
        }, os.path.join(OUT, cat + ".pt"))
        print(f"[{ci+1}/15] {cat}: train {tuple(train['l1'].shape[:1])} test {tuple(test['l1'].shape[:1])} "
              f"pix {tuple(test['pix_label'].shape)}  ({time.time()-t0:.1f}s)", flush=True)
    print("DONE extract_hires", flush=True)


if __name__ == "__main__":
    main()
