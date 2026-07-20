"""Step 1: cache frozen-backbone patch features for every category.

Run ONCE per (dataset, backbone). The backbone is frozen, so features are reused
by every experiment. Works for MVTec AD and MVTec LOCO AD (folder layouts below).

MVTec AD layout:
    <data>/<category>/train/good/*.png
    <data>/<category>/test/good/*.png
    <data>/<category>/test/<defect>/*.png
MVTec LOCO AD layout (has the logical/structural split we care about):
    <data>/<category>/train/good/*.png
    <data>/<category>/test/good/*.png
    <data>/<category>/test/logical_anomalies/*.png
    <data>/<category>/test/structural_anomalies/*.png

Usage:
    python -m src.cache_features --data <MVTEC_DIR>       --dataset mvtec --out feature_cache_dino
    python -m src.cache_features --data <MVTEC_LOCO_DIR>  --dataset loco  --out feature_cache_loco_dino
    python -m src.cache_features --data <DIR> --dataset loco --smoke
"""
import os
import glob
import argparse
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

IM_MEAN = [0.485, 0.456, 0.406]
IM_STD = [0.229, 0.224, 0.225]

# defect folder name -> defect_type code. good=0, logical=1, structural=2.
# Anything else that is not "good" is treated as a generic anomaly (type 1).
DEFECT_CODE = {"good": 0, "logical_anomalies": 1, "structural_anomalies": 2}


def build_transform(img_size, squash):
    """squash=True keeps the WHOLE image (resize to square) — important for LOCO
    logical anomalies (arrangement/count spread across the frame). squash=False
    resizes the short side then center-crops (classic MVTec pipeline)."""
    if squash:
        resize = [T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC)]
    else:
        resize = [T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC),
                  T.CenterCrop(img_size)]
    return T.Compose(resize + [T.ToTensor(), T.Normalize(IM_MEAN, IM_STD)])


def make_extractor(name, device):
    """Return extract(imgs[B,3,H,W]) -> patch features [B, P, D] (fp16, cpu)."""
    if name == "dinov2":
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")  # ViT-B/14, D=768
    elif name == "dinov2_giant":
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitg14_reg")  # ViT-g/14, D=1536
    else:
        raise ValueError("unknown backbone " + name)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    @torch.no_grad()
    def extract(imgs):
        out = model.forward_features(imgs.to(device))
        return out["x_norm_patchtokens"].to(torch.float16).cpu()
    return extract


def batched_extract(extract, files, tfm, bs):
    outs = []
    for i in range(0, len(files), bs):
        batch = torch.stack([tfm(Image.open(f).convert("RGB")) for f in files[i:i + bs]])
        outs.append(extract(batch))
    return torch.cat(outs) if outs else torch.empty(0)


def process_category(extract, cat_dir, tfm, max_train, bs):
    train_files = sorted(glob.glob(os.path.join(cat_dir, "train", "good", "*.png")))[:max_train]
    train_feats = batched_extract(extract, train_files, tfm, bs)

    test_files, img_lab, defect_type = [], [], []
    for defect_dir in sorted(glob.glob(os.path.join(cat_dir, "test", "*"))):
        defect = os.path.basename(defect_dir)
        files = sorted(glob.glob(os.path.join(defect_dir, "*.png")))
        test_files += files
        img_lab += [0 if defect == "good" else 1] * len(files)
        defect_type += [DEFECT_CODE.get(defect, 1)] * len(files)
    test_feats = batched_extract(extract, test_files, tfm, bs)
    p = train_feats.shape[1] if len(train_feats) else (test_feats.shape[1] if len(test_feats) else 0)
    return {
        "train_feats": train_feats,                                   # [N, P, D]
        "test_feats": test_feats,                                     # [M, P, D]
        "test_img_label": torch.tensor(img_lab, dtype=torch.int8),    # 0 normal / 1 anomaly
        "test_defect_type": torch.tensor(defect_type, dtype=torch.int8),  # 0 good/1 logical/2 structural
        "patch_hw": (int(round(p ** 0.5)), int(round(p ** 0.5))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset root (contains category folders)")
    ap.add_argument("--out", required=True, help="output cache dir")
    ap.add_argument("--dataset", choices=["mvtec", "loco"], default="loco")
    ap.add_argument("--backbone", choices=["dinov2", "dinov2_giant"], default="dinov2")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--max_train", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--squash", action="store_true",
                    help="force whole-image resize (default: on for loco, off for mvtec)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    squash = args.squash or (args.dataset == "loco")
    print(f"device={device} torch={torch.__version__} backbone={args.backbone} "
          f"dataset={args.dataset} squash={squash}")
    os.makedirs(args.out, exist_ok=True)
    tfm = build_transform(args.img_size, squash)
    extract = make_extractor(args.backbone, device)

    cats = sorted(d for d in os.listdir(args.data) if os.path.isdir(os.path.join(args.data, d)))
    if args.smoke:
        cats = cats[:1]
        args.max_train = 8
    for c in cats:
        data = process_category(extract, os.path.join(args.data, c), tfm, args.max_train, args.batch_size)
        torch.save(data, os.path.join(args.out, c + ".pt"))
        dt = data["test_defect_type"]
        print(f"  {c}: train {tuple(data['train_feats'].shape)} test {tuple(data['test_feats'].shape)} "
              f"good/log/str = {(dt==0).sum().item()}/{(dt==1).sum().item()}/{(dt==2).sum().item()}")
    print("Done ->", args.out)


if __name__ == "__main__":
    main()
