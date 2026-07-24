"""
Faithful reproduction of WinCLIP and WinCLIP+ (Jeong et al., CVPR 2023,
"WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation").

Measured on MVTec AD: wood, cable, zipper, transistor, hazelnut.

Key faithful components
-----------------------
1. Backbone: open_clip CLIP ViT-B-16-plus-240 / laion400m_e32 (240 input, joint
   dim 640), fallback ViT-B-16 / openai (224, dim 512). model.eval(), cuda.

2. Text-aligned DENSE patch features. Raw last-layer patch tokens of a CLIP ViT
   are NOT aligned to the text space (they give ~random zero-shot localization).
   We therefore extract dense features with the MaskCLIP value-projection: for the
   last transformer block, replace the self-attention output by the value
   projection passed through the attention output projection (no query-key
   attention mixing, no residual, no MLP), then apply ln_post and the visual proj.
   This yields per-patch embeddings in CLIP's joint space that ARE text-aligned,
   which is exactly what makes WinCLIP's masked-window forward text-alignable.
   Patch embeddings are L2-normalized. Grid is 15x15 for 240 input.

3. Multi-scale sliding windows over the patch grid at scales 1x1, 2x2, 3x3. A
   window embedding = L2-normalized mean of its member patch embeddings.

4. Compositional text-prompt ensemble: normal-state and anomaly-state templates
   crossed with CLIP inspection templates; each set mean-pooled + L2-normalized to
   one NORMAL and one ANOMALY text vector.

5. Per-window anomaly probability = softmax over temperature-scaled
   [cos(win, normal), cos(win, anomaly)] (temperature = model.logit_scale.exp()).

6. Hierarchical HARMONIC aggregation: within a scale, each grid location is the
   harmonic mean of all windows of that scale covering it; the final map is the
   harmonic mean across the three scales.

7. Zero-shot WinCLIP:
     - dense anomaly map from (6).
     - image score = whole-image CLIP embedding vs text (the largest / whole-image
       scale of WinCLIP), i.e. the text-aligned pooled embedding's anomaly prob.
     - pixel score = dense map upsampled (bilinear) to eval resolution + light
       Gaussian smoothing (sigma ~ 4 px).

8. WinCLIP+ (few-shot, K normal references):
     - reference memory of dense MULTI-SCALE window embeddings (standard projected
       patch tokens, which are best for patch-vs-patch matching) from K normals.
     - reference dense map = 1 - max cosine similarity to the bank, per window,
       harmonic-aggregated across scales (same scheme as the text branch).
     - PIXEL: average the (category z-scored) text and reference maps, then
       upsample + smooth.
     - IMAGE: average of the zero-shot image score and the few-shot image score
       (max of the reference map), each category z-scored.

Measurement
-----------
Shots K in {0, 1, 2, 5}. For each K>=1: 5 draws with seed = 7*K + draw, each
sampling K images from {cat}/train/good. Metrics: Image-AUROC and Pixel-AUROC
(pixels flattened over anomalous + normal test images), averaged over draws then
categories. Prints per-category and 5-category-mean tables.
"""

import os
import glob
import time
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter
import open_clip

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT = r"C:\Users\HOSEO\Desktop\K-DS\datasets\mvtec_ad\mvtech_anomaly_detection"
CATEGORIES = ["wood", "cable", "zipper", "transistor", "hazelnut"]
SHOTS = [0, 1, 2, 5]
N_DRAWS = 5
EVAL_RES = 256          # resolution for pixel-AUROC evaluation (map + mask)
GAUSS_SIGMA = 4.0       # Gaussian smoothing sigma (in eval-res pixels)
BATCH = 32
SCALES = (1, 2, 3)      # multi-scale window sizes over the patch grid

# Compositional prompt ensemble (WinCLIP CPE)
NORMAL_STATES = [
    "{}",
    "flawless {}",
    "perfect {}",
    "unblemished {}",
    "{} without flaw",
    "{} without defect",
    "{} without damage",
]
ANOMALY_STATES = [
    "damaged {}",
    "broken {}",
    "{} with flaw",
    "{} with defect",
    "{} with damage",
    "{} with a scratch",
]
TEMPLATES = [
    "a cropped photo of a {}.",
    "a close-up photo of a {}.",
    "a bright photo of a {}.",
    "a dark photo of a {}.",
    "a photo of a {} for inspection.",
    "a photo of a {} for anomaly detection.",
    "there is a {} in the scene.",
]

CLASS_NAME = {
    "wood": "wood", "cable": "cable", "zipper": "zipper",
    "transistor": "transistor", "hazelnut": "hazelnut",
}


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def load_model():
    force = os.environ.get("WINCLIP_BACKBONE", "").lower()  # '', 'plus', 'openai'
    try:
        if force == "openai":
            raise RuntimeError("forced openai fallback")
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16-plus-240", pretrained="laion400m_e32")
        tokenizer = open_clip.get_tokenizer("ViT-B-16-plus-240")
        backbone, input_res = "ViT-B-16-plus-240 / laion400m_e32", 240
    except Exception as e:  # pragma: no cover
        if force != "openai":
            print(f"[warn] primary backbone failed ({e}); falling back to ViT-B-16/openai")
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained="openai")
        tokenizer = open_clip.get_tokenizer("ViT-B-16")
        backbone, input_res = "ViT-B-16 / openai", 224
    model = model.eval().to(DEVICE)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, preprocess, tokenizer, backbone, input_res


@torch.no_grad()
def build_text_embeds(model, tokenizer, class_name):
    """Return (normal_vec, anomaly_vec), each L2-normalized [D]."""
    def encode(states):
        prompts = [t.format(s.format(class_name)) for s in states for t in TEMPLATES]
        emb = model.encode_text(tokenizer(prompts).to(DEVICE))
        emb = F.normalize(emb, dim=-1)   # normalize each prompt embedding
        emb = emb.mean(dim=0)            # mean-pool the ensemble
        return F.normalize(emb, dim=0)   # renormalize
    return encode(NORMAL_STATES), encode(ANOMALY_STATES)


class DenseExtractor:
    """
    One shared forward returning, per image:
      pooled       [B, D]      whole-image CLIP embedding (ln_post + proj), L2-norm
      std_tokens   [B, N, D]   standard projected patch tokens, L2-norm  (reference)
      mask_tokens  [B, N, D]   MaskCLIP value-projection patch tokens, L2-norm (text)
    """
    def __init__(self, model):
        self.model = model
        self.v = model.visual
        self.grid = None

    @torch.no_grad()
    def __call__(self, images):
        v = self.v
        x = v._embeds(images)                       # conv + cls + pos + ln_pre
        blocks = v.transformer.resblocks
        for blk in blocks[:-1]:
            x = blk(x)
        last = blocks[-1]
        emb_dim = x.shape[-1]

        # --- standard path: run the last block normally ---
        x_full = v.ln_post(last(x))
        pooled = x_full[:, 0]
        std_tokens = x_full[:, 1:]
        if v.proj is not None:
            pooled = pooled @ v.proj
            std_tokens = std_tokens @ v.proj

        # --- MaskCLIP value path: value projection only, no attn mix / residual / mlp ---
        x_ln = last.ln_1(x)
        w_v = last.attn.in_proj_weight[2 * emb_dim:3 * emb_dim]
        b_v = last.attn.in_proj_bias[2 * emb_dim:3 * emb_dim]
        val = x_ln @ w_v.t() + b_v
        out = val @ last.attn.out_proj.weight.t() + last.attn.out_proj.bias
        out = v.ln_post(out)
        mask_tokens = out[:, 1:]
        if v.proj is not None:
            mask_tokens = mask_tokens @ v.proj

        if self.grid is None:
            n = std_tokens.shape[1]
            g = int(round(n ** 0.5))
            assert g * g == n, f"non-square token grid n={n}"
            self.grid = g

        return (F.normalize(pooled, dim=-1),
                F.normalize(std_tokens, dim=-1),
                F.normalize(mask_tokens, dim=-1))


# ----------------------------------------------------------------------------
# Multi-scale windows + harmonic aggregation
# ----------------------------------------------------------------------------
def window_mean_embeds(patch_grid, k):
    """patch_grid [B,g,g,D] -> L2-normalized mean over kxk windows [B,g-k+1,g-k+1,D]."""
    x = patch_grid.permute(0, 3, 1, 2)               # [B,D,g,g]
    x = F.avg_pool2d(x, kernel_size=k, stride=1)     # mean over window members
    x = x.permute(0, 2, 3, 1)
    return F.normalize(x, dim=-1)


def window_anom_prob(win_embeds, normal_vec, anomaly_vec, logit_scale):
    """win_embeds [B,h,w,D] -> anomaly prob [B,h,w] via temperature-scaled softmax."""
    cn = win_embeds @ normal_vec
    ca = win_embeds @ anomaly_vec
    logits = torch.stack([cn, ca], dim=-1) * logit_scale
    return torch.softmax(logits, dim=-1)[..., 1]


def harmonic_per_scale(win_score, k, g):
    """
    win_score [B, g-k+1, g-k+1] -> per-scale harmonic map [B,g,g]: harmonic mean
    over all kxk windows covering each location (transposed conv scatters recip.).
    """
    eps = 1e-8
    s = win_score.clamp(min=eps).unsqueeze(1)                       # [B,1,h,w]
    ones_k = torch.ones(1, 1, k, k, device=s.device, dtype=s.dtype)
    recip_sum = F.conv_transpose2d(1.0 / s, ones_k, stride=1)       # [B,1,g,g]
    count = F.conv_transpose2d(torch.ones_like(s), ones_k, stride=1)
    return (count / recip_sum).squeeze(1)


def _harmonic_across(maps):
    eps = 1e-8
    return len(maps) / sum(1.0 / m.clamp(min=eps) for m in maps)


def text_dense_map(grid, normal_vec, anomaly_vec, logit_scale, scales=SCALES):
    """grid [B,g,g,D] MaskCLIP patches -> zero-shot text anomaly map [B,g,g]."""
    g = grid.shape[1]
    scale_maps = []
    for k in scales:
        we = window_mean_embeds(grid, k)
        ws = window_anom_prob(we, normal_vec, anomaly_vec, logit_scale)
        scale_maps.append(harmonic_per_scale(ws, k, g))
    return _harmonic_across(scale_maps)


def reference_dense_map(query_grid, bank_grids, scales=SCALES):
    """
    query_grid [B,g,g,D] standard patches; bank_grids [K,g,g,D] standard patches.
    Multi-scale reference association: per window, distance = 1 - max cosine to the
    bank's windows of the same scale; harmonic-aggregate within and across scales.
    Returns [B,g,g].
    """
    B, g, _, D = query_grid.shape
    K = bank_grids.shape[0]
    scale_maps = []
    for k in scales:
        qwe = window_mean_embeds(query_grid, k)      # [B,h,w,D]
        bwe = window_mean_embeds(bank_grids, k)      # [K,h,w,D]
        h, w = qwe.shape[1], qwe.shape[2]
        qf = qwe.reshape(B, h * w, D)
        bf = bwe.reshape(K * h * w, D)
        sim = torch.einsum("bnd,md->bnm", qf, bf)    # cosine (all L2-normalized)
        dist = (1.0 - sim.max(dim=-1).values).reshape(B, h, w)
        scale_maps.append(harmonic_per_scale(dist, k, g))
    return _harmonic_across(scale_maps)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def list_test_images(cat):
    """Return list of (path, label, mask_path_or_None)."""
    test_dir = os.path.join(DATA_ROOT, cat, "test")
    gt_dir = os.path.join(DATA_ROOT, cat, "ground_truth")
    items = []
    for defect in sorted(os.listdir(test_dir)):
        ddir = os.path.join(test_dir, defect)
        if not os.path.isdir(ddir):
            continue
        for img in sorted(glob.glob(os.path.join(ddir, "*.png"))):
            if defect == "good":
                items.append((img, 0, None))
            else:
                name = os.path.splitext(os.path.basename(img))[0]
                mask = os.path.join(gt_dir, defect, name + "_mask.png")
                items.append((img, 1, mask if os.path.exists(mask) else None))
    return items


def list_train_normals(cat):
    return sorted(glob.glob(os.path.join(DATA_ROOT, cat, "train", "good", "*.png")))


def load_mask(mask_path, res):
    if mask_path is None:
        return np.zeros((res, res), dtype=np.uint8)
    m = Image.open(mask_path).convert("L").resize((res, res), Image.NEAREST)
    return (np.array(m) > 0).astype(np.uint8)


@torch.no_grad()
def extract_dataset(paths, preprocess, extractor, want=("pooled", "std", "mask")):
    """Return dict of stacked CPU float32 features for the requested streams."""
    acc = {k: [] for k in want}
    for i in range(0, len(paths), BATCH):
        imgs = torch.stack([preprocess(Image.open(p).convert("RGB"))
                            for p in paths[i:i + BATCH]]).to(DEVICE)
        pooled, std_tok, mask_tok = extractor(imgs)
        if "pooled" in acc:
            acc["pooled"].append(pooled.float().cpu())
        if "std" in acc:
            acc["std"].append(std_tok.float().cpu())
        if "mask" in acc:
            acc["mask"].append(mask_tok.float().cpu())
    return {k: torch.cat(v, dim=0) for k, v in acc.items()}


# ----------------------------------------------------------------------------
# Scoring helpers
# ----------------------------------------------------------------------------
def upsample_and_smooth(dense_map, res, sigma):
    """dense_map [B,g,g] tensor -> np [B,res,res] bilinear-upsampled + Gaussian."""
    m = dense_map.unsqueeze(1)
    up = F.interpolate(m, size=(res, res), mode="bilinear", align_corners=False)
    up = up.squeeze(1).cpu().numpy()
    for i in range(up.shape[0]):
        up[i] = gaussian_filter(up[i], sigma=sigma)
    return up


def zscore(a):
    return (a - a.mean()) / (a.std() + 1e-8)


# ----------------------------------------------------------------------------
# Main measurement
# ----------------------------------------------------------------------------
def run():
    t0 = time.time()
    model, preprocess, tokenizer, backbone, input_res = load_model()
    logit_scale = model.logit_scale.exp().item()
    extractor = DenseExtractor(model)
    with torch.no_grad():
        text_dim = model.encode_text(tokenizer(["a photo"]).to(DEVICE)).shape[-1]
    print(f"[info] backbone={backbone} input_res={input_res} text_dim={text_dim} "
          f"logit_scale={logit_scale:.1f} device={DEVICE} eval_res={EVAL_RES}", flush=True)

    settings = ["zero"] + [f"k{k}" for k in SHOTS if k > 0]
    img_res = {s: {} for s in settings}
    pix_res = {s: {} for s in settings}

    for cat in CATEGORIES:
        tc = time.time()
        normal_vec, anomaly_vec = build_text_embeds(model, tokenizer, CLASS_NAME[cat])

        items = list_test_images(cat)
        paths = [x[0] for x in items]
        labels = np.array([x[1] for x in items])
        masks = np.stack([load_mask(x[2], EVAL_RES) for x in items])
        gt_flat = (masks.reshape(len(items), -1) > 0).astype(np.uint8).reshape(-1)

        feats = extract_dataset(paths, preprocess, extractor)
        g, D = extractor.grid, feats["std"].shape[-1]
        pooled = feats["pooled"].to(DEVICE)                       # [n,D]
        mask_grid = feats["mask"].view(-1, g, g, D).to(DEVICE)    # [n,g,g,D]
        std_grid = feats["std"].view(-1, g, g, D)                 # CPU [n,g,g,D]

        # ---------- zero-shot ----------
        text_map = text_dense_map(mask_grid, normal_vec, anomaly_vec, logit_scale)  # [n,g,g]
        pool_prob = torch.softmax(
            torch.stack([pooled @ normal_vec, pooled @ anomaly_vec], -1) * logit_scale,
            dim=-1)[:, 1].cpu().numpy()
        img_res["zero"][cat] = roc_auc_score(labels, pool_prob)
        pix_res["zero"][cat] = roc_auc_score(
            gt_flat, upsample_and_smooth(text_map, EVAL_RES, GAUSS_SIGMA).reshape(-1))
        text_map_np = text_map.cpu().numpy()

        # ---------- few-shot: precompute all train-normal std features ----------
        train_paths = list_train_normals(cat)
        train_std = extract_dataset(train_paths, preprocess, extractor, want=("std",))["std"]
        train_std_grid = train_std.view(-1, g, g, D)
        n_train = len(train_paths)

        for K in [k for k in SHOTS if k > 0]:
            ias, pas = [], []
            for draw in range(N_DRAWS):
                rng = random.Random(7 * K + draw)
                idx = rng.sample(range(n_train), K)
                bank = train_std_grid[idx].to(DEVICE)              # [K,g,g,D]

                ref_maps = []
                for i in range(0, std_grid.shape[0], BATCH):
                    q = std_grid[i:i + BATCH].to(DEVICE)
                    ref_maps.append(reference_dense_map(q, bank).cpu())
                ref_map_np = torch.cat(ref_maps, dim=0).numpy()    # [n,g,g]
                ref_max = ref_map_np.reshape(ref_map_np.shape[0], -1).max(axis=1)

                # IMAGE: average zero-shot (pool) and few-shot (ref-max) scores
                img_score = zscore(pool_prob) + zscore(ref_max)
                # PIXEL: average category z-scored text and reference maps
                comb = 0.5 * (zscore(text_map_np) + zscore(ref_map_np))
                pix_map = upsample_and_smooth(torch.from_numpy(comb), EVAL_RES, GAUSS_SIGMA)

                ias.append(roc_auc_score(labels, img_score))
                pas.append(roc_auc_score(gt_flat, pix_map.reshape(-1)))
            img_res[f"k{K}"][cat] = float(np.mean(ias))
            pix_res[f"k{K}"][cat] = float(np.mean(pas))

        print(f"[cat] {cat:11s} {time.time()-tc:5.1f}s  "
              f"zero(img={img_res['zero'][cat]:.3f} pix={pix_res['zero'][cat]:.3f})  "
              + " ".join(f"k{K}(img={img_res[f'k{K}'][cat]:.3f} pix={pix_res[f'k{K}'][cat]:.3f})"
                         for K in SHOTS if K > 0), flush=True)

    print_table("IMAGE-AUROC", img_res, settings)
    print_table("PIXEL-AUROC", pix_res, settings)
    print(f"\n[info] backbone used: {backbone}")
    print(f"[info] total time {time.time()-t0:.1f}s")
    return img_res, pix_res, backbone


def _row_label(s):
    return {"zero": "WinCLIP zero-shot", "k1": "WinCLIP+ K=1",
            "k2": "WinCLIP+ K=2", "k5": "WinCLIP+ K=5"}.get(s, s)


def print_table(title, res, settings):
    print(f"\n===== {title} =====")
    header = f"{'setting':20s}" + "".join(f"{c:>11s}" for c in CATEGORIES) + f"{'MEAN':>9s}"
    print(header)
    print("-" * len(header))
    for s in settings:
        vals = [res[s][c] for c in CATEGORIES]
        row = (f"{_row_label(s):20s}" + "".join(f"{v:>11.3f}" for v in vals)
               + f"{np.mean(vals):>9.3f}")
        print(row)


if __name__ == "__main__":
    run()
