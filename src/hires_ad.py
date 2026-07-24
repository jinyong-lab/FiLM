"""Hi-res multi-scale + meta-learned frozen adapter for few-shot anomaly detection.

Combines the two approved ideas:
  (Q1) FROZEN WRN-50-2 backbone + a lightweight FiLM adapter META-LEARNED on cached features
       (backbone never fine-tuned). Adapter = feature-preserving residual FiLM (film_ad_fixed
       design): conv1->BN->gamma(z)*h+beta(z)->relu->conv2(zero-init), residual, 4 blocks,
       final L2-normalize; specialization-suppression reg (REG_LAM=0.5) + support-noise
       meta-aug (AUG_STD=0.1). Episodic training K in {1,2,4}, margin 0.7, ~20000 steps, lr 1e-3.
  (Q2) "sharper magnifying glass": hi-res multi-scale 56x56 patch features (layer1 + upsampled
       layer2 + upsampled layer3), Gaussian smoothing (sigma=4) of the upsampled 224 anomaly
       map, and FULL-RESOLUTION (224) pixel evaluation.

Ablations (isolate each contribution), reported at K=1/2/5 (5 test cats, 5 draws):
  A) base 28x28 frozen + patch-NN (reproduce current frozen baseline)
  B) A -> hi-res multi-scale 56x56 + smoothing (no adapter)
  C) B + meta-learned FiLM adapter (full proposed method)
For pixel we report BOTH grid-eval and full-res 224-eval to expose the resolution effect.

Reads: feature_cache_wrn_hires/{cat}.pt (see extract_hires.py).
"""
import os, sys, time, random, argparse
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score

CACHE = r"D:\K-DS\code\AnomalyDetection\feature_cache_wrn_hires"
dev = "cuda" if torch.cuda.is_available() else "cpu"

ALL = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut",
       "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"]
TEST_CATS = ["wood", "cable", "zipper", "transistor", "hazelnut"]
TRAIN_CATS = [c for c in ALL if c not in TEST_CATS]

# feature configs
C_HIRES, S_HIRES = 1792, 56       # layer1(256)+layer2(512)+layer3(1024)
C_BASE, S_BASE = 1536, 28         # layer2(512)+layer3(1024)  == existing feature_cache_wrn
Z, CH, NB = 128, 512, 4
STEPS, MARGIN, LR, BANK_SUB = 20000, 0.7, 1e-3, 1024
REG_LAM, AUG_STD = 0.5, 0.1
SIGMA = 4
EVAL_RES = 224


def l2n(x, dim=-1):
    return x / (x.norm(dim=dim, keepdim=True) + 1e-8)


_raw = {}
def load_raw(cat):
    if cat not in _raw:
        _raw[cat] = torch.load(os.path.join(CACHE, cat + ".pt"), map_location="cpu", weights_only=False)
    return _raw[cat]


def n_imgs(cat, split):
    return load_raw(cat)[f"{split}_l1"].shape[0]


def make_map(cat, split, i, mode):
    """Assemble ONE image's normalized patch map [C,S,S] on device (memory-efficient).
    mode='hires' -> l1 + up(l2) + up(l3) at 56x56 (1792ch);
    mode='base28' -> l2 + up(l3) at 28x28 (1536ch, == existing feature_cache_wrn).
    L2-normalize per patch (over channel dim) => baseline feature space. Done on the fly to
    avoid materializing giant [N,P,C] float32 tensors (hi-res would be ~4.5GB/category)."""
    d = load_raw(cat)

    def up(key, S):
        x = d[f"{split}_{key}"][i:i + 1].to(dev).permute(0, 3, 1, 2).float()   # [1,Cc,h,w]
        if x.shape[-1] != S:
            x = F.interpolate(x, size=(S, S), mode="bilinear", align_corners=False)
        return x

    if mode == "hires":
        S = S_HIRES
        feat = torch.cat([up("l1", S), up("l2", S), up("l3", S)], 1)[0]        # [1792,56,56]
    else:
        S = S_BASE
        feat = torch.cat([up("l2", S), up("l3", S)], 1)[0]                     # [1536,28,28]
    return feat / (feat.norm(dim=0, keepdim=True) + 1e-8)                      # [C,S,S] on dev


def stack_maps(cat, split, idxs, mode):
    return torch.stack([make_map(cat, split, int(i), mode) for i in idxs])     # [K,C,S,S]


# ---------------- meta-learned adapter (film_ad_fixed design, parametrized by C) ----------------
class TaskEncoder(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * C, 512), nn.ReLU(), nn.Linear(512, Z))

    def forward(self, supp):                       # [N,C,S,S]
        return self.mlp(torch.cat([supp.mean((0, 2, 3)), supp.var((0, 2, 3))], 0))


class FiLMResBlockPreserve(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.conv1 = nn.Conv2d(C, CH, 1)
        self.bn = nn.BatchNorm2d(CH, affine=False)
        self.g = nn.Linear(Z, CH); self.b = nn.Linear(Z, CH)
        self.conv2 = nn.Conv2d(CH, C, 1)
        nn.init.zeros_(self.g.weight); nn.init.ones_(self.g.bias)
        nn.init.zeros_(self.b.weight); nn.init.zeros_(self.b.bias)
        nn.init.zeros_(self.conv2.weight); nn.init.zeros_(self.conv2.bias)   # zero-init -> identity

    def forward(self, x, z):
        h = self.bn(self.conv1(x))
        h = self.g(z).view(1, CH, 1, 1) * h + self.b(z).view(1, CH, 1, 1)
        h = F.relu(h)
        return x + self.conv2(h)                    # residual; initially conv2=0 => output=x


class MetaLearnerPreserve(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.blocks = nn.ModuleList([FiLMResBlockPreserve(C) for _ in range(NB)])

    def forward(self, x, z):
        h = x
        for blk in self.blocks:
            h = blk(h, z)
        return F.normalize(h, dim=1)                # per-patch L2 (matches baseline space)


def to_bank(feat4d, sub=None):
    ch = feat4d.shape[1]
    b = feat4d.permute(0, 2, 3, 1).reshape(-1, ch)
    if sub and b.shape[0] > sub:
        b = b[torch.randperm(b.shape[0], device=b.device)[:sub]]
    return b


def patch_dist(bank, q):
    return torch.cdist(q, bank).min(1).values


# ---------------- meta-training (hi-res features) ----------------
def meta_train():
    C, S = C_HIRES, S_HIRES
    enc, ml = TaskEncoder(C).to(dev), MetaLearnerPreserve(C).to(dev)
    opt = torch.optim.Adam(list(enc.parameters()) + list(ml.parameters()), lr=LR)
    ntr = {c: n_imgs(c, "train") for c in TRAIN_CATS}
    nte = {c: n_imgs(c, "test") for c in TRAIN_CATS}
    qpix = {c: load_raw(c)["test_pix_label"] for c in TRAIN_CATS}    # [Nt,224,224]
    cell = EVAL_RES // S
    for step in range(STEPS):
        cat = random.choice(TRAIN_CATS)
        K = random.choice([1, 2, 4])
        sidx = torch.randperm(ntr[cat])[:K]
        supp = stack_maps(cat, "train", sidx, "hires")                      # [K,C,S,S] on dev
        qi = torch.randint(nte[cat], (1,)).item()
        q = make_map(cat, "test", qi, "hires").unsqueeze(0)                 # [1,C,S,S] on dev
        # real defect mask of the query (from OTHER cats' anomalies), max-pooled to the SxS grid
        pm = qpix[cat][qi].numpy()
        grid = pm.reshape(S, cell, S, cell).max(axis=(1, 3)).reshape(-1)
        mask = torch.tensor(grid > 0, device=dev)
        supp_z = l2n(supp + AUG_STD * supp.std() * torch.randn_like(supp), dim=1) if AUG_STD > 0 else supp
        z = enc(supp_z)
        inp = torch.cat([supp, q])
        allml = ml(inp, z)
        supp_ml, q_ml = allml[:K], allml[K:]
        bank = to_bank(supp_ml, BANK_SUB); qd = patch_dist(bank, to_bank(q_ml))
        margin = (qd[~mask].mean() + F.relu(MARGIN - qd[mask]).mean()) if mask.any() else qd.mean()
        reg = ((allml - inp) ** 2).sum(1).mean()
        loss = margin + REG_LAM * reg
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 2000 == 0:
            print(f"  [meta] step {step+1}/{STEPS} loss={loss.item():.4f}", flush=True)
    enc.eval(); ml.eval()
    return enc, ml


# ---------------- scoring ----------------
def upsample_smooth(dist_flat, S, sigma=SIGMA, smooth=True):
    """[S*S] patch distances -> [224,224] map (bilinear upsample, optional gaussian smooth)."""
    m = dist_flat.view(1, 1, S, S).float()
    m = F.interpolate(m, size=(EVAL_RES, EVAL_RES), mode="bilinear", align_corners=False)
    arr = m[0, 0].cpu().numpy()
    if smooth:
        arr = gaussian_filter(arr, sigma=sigma)
    return arr


def grid_maxpool_gt(pix_224, S):
    cell = EVAL_RES // S
    return pix_224.reshape(-1, S, cell, S, cell).max(axis=(2, 4))   # [N,S,S]


# ---------------- evaluation of one config ----------------
@torch.no_grad()
def eval_config(mode, adapter, K_list=(1, 2, 5), draws=5, smooth=True, topk_frac=0.01, cats=TEST_CATS):
    """Returns nested dict: res[cat][K] = {img_max, img_smax, img_topk, pix_grid, pix_224}."""
    S = S_HIRES if mode == "hires" else S_BASE
    enc, ml = (adapter if adapter else (None, None))
    res = {}
    for cat in cats:
        d = load_raw(cat)
        npool = n_imgs(cat, "train"); ntest = n_imgs(cat, "test")
        lab = d["test_img_label"].numpy()
        pix224 = d["test_pix_label"].numpy().astype(np.uint8)          # [Nt,224,224]
        gt_grid = grid_maxpool_gt(pix224, S).reshape(ntest, -1)        # [Nt,S*S]
        has_pix = pix224.max() > 0
        res[cat] = {}
        for K in K_list:
            agg = {k: [] for k in ["img_max", "img_smax", "img_topk", "pix_grid", "pix_224"]}
            for draw in range(draws):
                g = torch.Generator().manual_seed(7 * K + draw)
                sidx = torch.randperm(npool, generator=g)[:K]
                supp = stack_maps(cat, "train", sidx, mode)            # [K,C,S,S] on dev
                if adapter:
                    z = enc(supp); bank = to_bank(ml(supp, z))
                else:
                    z = None; bank = to_bank(supp)
                img_max, img_smax, img_topk = [], [], []
                grid_maps, smooth_maps = [], []
                ntop = max(1, int(topk_frac * EVAL_RES * EVAL_RES))
                for j in range(ntest):
                    qm = make_map(cat, "test", j, mode).unsqueeze(0)   # [1,C,S,S] on dev
                    q = to_bank(ml(qm, z)) if adapter else to_bank(qm)
                    dd = patch_dist(bank, q)                            # [S*S]
                    grid_maps.append(dd.view(S, S).cpu().numpy())
                    sm = upsample_smooth(dd, S, smooth=smooth)          # [224,224]
                    smooth_maps.append(sm)
                    img_max.append(float(dd.max().item()))              # raw grid max (baseline style)
                    img_smax.append(float(sm.max()))                   # smoothed 224 max
                    flat = np.partition(sm.reshape(-1), -ntop)[-ntop:]
                    img_topk.append(float(flat.mean()))                # smoothed top-k mean
                agg["img_max"].append(roc_auc_score(lab, img_max))
                agg["img_smax"].append(roc_auc_score(lab, img_smax))
                agg["img_topk"].append(roc_auc_score(lab, img_topk))
                if has_pix:
                    gm = np.stack(grid_maps).reshape(ntest, -1)
                    sm = np.stack(smooth_maps).reshape(ntest, -1)
                    agg["pix_grid"].append(roc_auc_score(gt_grid.reshape(-1) > 0, gm.reshape(-1)))
                    agg["pix_224"].append(roc_auc_score(pix224.reshape(-1) > 0, sm.reshape(-1)))
            res[cat][K] = {k: (float(np.mean(v)) if v else float("nan")) for k, v in agg.items()}
        print(f"    [{mode}{'+adapter' if adapter else ''}] {cat} done", flush=True)
    return res


def mean_over_cats(res, K, key):
    vals = [res[c][K][key] for c in res if not np.isnan(res[c][K][key])]
    return float(np.mean(vals)) if vals else float("nan")


def print_table(name, res, img_key, K_list=(1, 2, 5)):
    print(f"\n----- {name} (5-cat mean) -----")
    hdr = "  K   Image-AUROC   Pixel-AUROC@224   Pixel-AUROC@grid"
    print(hdr)
    for K in K_list:
        im = mean_over_cats(res, K, img_key)
        p2 = mean_over_cats(res, K, "pix_224")
        pg = mean_over_cats(res, K, "pix_grid")
        print(f"  {K}     {im:.4f}        {p2:.4f}            {pg:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="A only, K=5, 2 draws (sanity)")
    ap.add_argument("--configs", default="A,B,C")
    ap.add_argument("--draws", type=int, default=5)
    args = ap.parse_args()
    random.seed(0); torch.manual_seed(0); np.random.seed(0)
    K_list = (5,) if args.quick else (1, 2, 5)
    draws = 2 if args.quick else args.draws
    want = args.configs.split(",")

    print("=" * 70)
    print("Hi-res multi-scale + meta frozen adapter | test cats:", TEST_CATS)
    print("=" * 70)

    results = {}
    if "A" in want:
        print("\n[A] base 28x28 frozen + patch-NN (baseline reproduction)", flush=True)
        t0 = time.time()
        results["A"] = eval_config("base28", None, K_list=K_list, draws=draws)
        print_table("A base28 frozen patch-NN", results["A"], img_key="img_max", K_list=K_list)
        print(f"  (A elapsed {time.time()-t0:.0f}s)", flush=True)

    if "B" in want:
        print("\n[B] hi-res multi-scale 56x56 + smoothing (no adapter)", flush=True)
        t0 = time.time()
        results["B"] = eval_config("hires", None, K_list=K_list, draws=draws, smooth=True)
        print_table("B hires-multiscale + smooth", results["B"], img_key="img_smax", K_list=K_list)
        print(f"  (B elapsed {time.time()-t0:.0f}s)", flush=True)

    if "C" in want:
        print("\n[C] hi-res multi-scale + META-LEARNED FiLM adapter (full proposed)", flush=True)
        print("  meta-training adapter on 10 train cats (hi-res)...", flush=True)
        t0 = time.time()
        enc, ml = meta_train()
        print(f"  (meta-train {time.time()-t0:.0f}s)", flush=True)
        t0 = time.time()
        results["C"] = eval_config("hires", (enc, ml), K_list=K_list, draws=draws, smooth=True)
        print_table("C hires + meta adapter", results["C"], img_key="img_smax", K_list=K_list)
        print(f"  (C elapsed {time.time()-t0:.0f}s)", flush=True)

    # ---- combined summary ----
    print("\n" + "=" * 70)
    print("SUMMARY  (Image-AUROC / Pixel-AUROC@224full)  5-cat mean")
    print("=" * 70)
    print(f"{'config':<30}" + "".join(f"K={k:<12}" for k in K_list))
    keymap = {"A": ("img_max", "A base28 patch-NN"),
              "B": ("img_smax", "B hires-multiscale+smooth"),
              "C": ("img_smax", "C +meta adapter")}
    for cfg in ["A", "B", "C"]:
        if cfg not in results:
            continue
        ik, lbl = keymap[cfg]
        row = f"{lbl:<30}"
        for K in K_list:
            im = mean_over_cats(results[cfg], K, ik)
            p2 = mean_over_cats(results[cfg], K, "pix_224")
            row += f"{im:.3f}/{p2:.3f}    "
        print(row)

    # per-category for best config present (prefer C, else B, else A)
    best = "C" if "C" in results else ("B" if "B" in results else "A")
    ik = keymap[best][0]
    print(f"\nPer-category for best config [{best}={keymap[best][1]}]:")
    print(f"{'cat':<12}" + "".join(f"K={k} Img/Pix224   " for k in K_list))
    for cat in TEST_CATS:
        row = f"{cat:<12}"
        for K in K_list:
            row += f"{results[best][cat][K][ik]:.3f}/{results[best][cat][K]['pix_224']:.3f}      "
        print(row)

    # image-score aggregation comparison for best config
    print(f"\n[image-score aggregation, best config {best}, 5-cat mean]:")
    for K in K_list:
        mx = mean_over_cats(results[best], K, "img_max")
        smx = mean_over_cats(results[best], K, "img_smax")
        tk = mean_over_cats(results[best], K, "img_topk")
        print(f"  K={K}: raw-grid-max={mx:.3f}  smoothed-max={smx:.3f}  smoothed-top1%={tk:.3f}")


if __name__ == "__main__":
    main()
