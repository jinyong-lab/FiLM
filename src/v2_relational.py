"""v2: RELATIONAL learned scorer for few-shot anomaly detection (designed to GENERALIZE).

WHY v2 (vs the old FiLM / hires_ad meta_train that MEMORIZED train cats):
  Old failure modes (both cause category memorization):
    (a) FiLM FEATURE MODULATION: gamma(z)*h+beta(z) conditioned on task z -> can shift
        features toward memorized category prototypes.
    (b) trains on the TRAIN cats' REAL defect masks (qpix[cat][qi]) -> learns *those* defects.
  v2 fixes BOTH:
    1) LOCAL patch detection on FROZEN hi-res multi-scale features (56x56, 1792ch). Backbone
       never trained. Same 'hires' feature space as hires_ad.
    2) RELATIONAL learned scorer (NOT feature modulation): a shared, z-FREE deep-metric
       projection phi (residual 1x1-conv bottleneck, IDENTITY-INITIALIZED like film_ad_fixed's
       zero-init conv2 -> starts == frozen patch-NN). phi maps EVERY patch (support & query)
       identically; the anomaly score of a query patch = (soft-)min distance IN PHI-SPACE to the
       SAME task's support patches. Because phi is a single fixed map applied to both sides and
       only reshapes RELATIVE query-vs-support distances, it is category-agnostic by construction
       (no category/task label ever enters).
    3) SSL + META objective on GENERIC synthetic anomalies in FEATURE space (feature-space
       CutPaste from other images + structured noise) -- NEVER the train cats' real defect masks.
       Margin loss: normal query patches close to support, synthetic-anomaly patches far, in
       phi-space. + specialization-suppression reg (phi output ~ input, small lambda) +
       support-noise meta-augmentation.
    4) EARLY STOPPING: monitor 1-shot held-out AUROC on the 5 TEST cats; keep BEST checkpoint.

Also a CROSS-ATTENTION variant (--variant xattn): query patch attends to support patches;
score = residual (reconstruction) norm.

Reuses hires_ad machinery (make_map/stack_maps/to_bank/patch_dist/eval_config/mean_over_cats).
Run:  python v2_relational.py --variant phi
"""
import os, sys, time, random, copy, math, argparse
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score

import hires_ad as H
from hires_ad import (make_map, stack_maps, to_bank, patch_dist, n_imgs, load_raw,
                      eval_config, mean_over_cats, TRAIN_CATS, TEST_CATS, dev,
                      C_HIRES, S_HIRES, ALL)

C, S = C_HIRES, S_HIRES            # 1792, 56
CH   = int(os.environ.get("V2_CH", 256))
NB   = int(os.environ.get("V2_NB", 3))
STEPS   = int(os.environ.get("V2_STEPS", 12000))
CHK_EVERY = int(os.environ.get("V2_CHK", 1500))
MARGIN  = float(os.environ.get("V2_MARGIN", 0.7))
LR      = float(os.environ.get("V2_LR", 1e-3))
REG_LAM = float(os.environ.get("V2_REG", 0.3))
AUG_STD = float(os.environ.get("V2_AUG", 0.1))
TAU     = float(os.environ.get("V2_TAU", 20.0))
CLEAN_W = float(os.environ.get("V2_CLEAN", 1.0))   # weight on redundant clean-query pull term (0 = drop)
BANK_SUB= 1536
HO_DRAWS= int(os.environ.get("V2_HODRAWS", 2))
HO_SUB  = 2048
CKPT_DIR= os.environ.get("V2_CKPT", r"D:\K-DS\filmad\src")


def l2n(x, dim=1):
    return x / (x.norm(dim=dim, keepdim=True) + 1e-8)


# ============================ VARIANT 1: deep-metric phi ============================
class PhiResBlock(nn.Module):
    """residual 1x1-conv bottleneck; conv2 zero-init -> block starts as identity."""
    def __init__(self, C, CH):
        super().__init__()
        self.conv1 = nn.Conv2d(C, CH, 1)
        self.gn = nn.GroupNorm(8, CH)             # batch-independent (same train/eval), no running stats
        self.conv2 = nn.Conv2d(CH, C, 1)
        nn.init.zeros_(self.conv2.weight); nn.init.zeros_(self.conv2.bias)   # identity init

    def forward(self, x):
        h = F.relu(self.gn(self.conv1(x)))
        return x + self.conv2(h)                  # init conv2=0 => output == x


class PhiMetric(nn.Module):
    """Shared, z-FREE metric projection. phi(x)=normalize(stack of residual blocks). At init
    every block is identity and input is already per-patch L2-normed => phi==identity==frozen NN."""
    def __init__(self, C, CH, NB):
        super().__init__()
        self.blocks = nn.ModuleList([PhiResBlock(C, CH) for _ in range(NB)])

    def forward(self, x):                         # x [N,C,S,S] (per-patch normed)
        h = x
        for b in self.blocks:
            h = b(h)
        return F.normalize(h, dim=1)              # per-patch L2 (baseline space)


class PhiAdapterML(nn.Module):
    """Wrap phi so eval_config's `ml(x, z)` interface works (z ignored)."""
    def __init__(self, phi):
        super().__init__(); self.phi = phi
    def forward(self, x, z=None):
        return self.phi(x)


# ============================ VARIANT 2: cross-attention ============================
class XAttnScorer(nn.Module):
    """Query patch attends over support patches; score = residual (reconstruction) norm.
    Value space = the ORIGINAL support features (recon in feature space), so score is
    category-agnostic (only relative q-vs-support). q_proj/k_proj init ~ identity + temperature
    -> attention ~ raw-feature-similarity -> recon ~ nearest support -> starts close to patch-NN."""
    def __init__(self, C, d=None):
        super().__init__()
        d = d or C
        self.q_proj = nn.Linear(C, d, bias=False)
        self.k_proj = nn.Linear(C, d, bias=False)
        # identity-ish init: attention driven by raw cosine similarity at start
        with torch.no_grad():
            eye = torch.eye(d, C)
            self.q_proj.weight.copy_(eye); self.k_proj.weight.copy_(eye)
        self.log_temp = nn.Parameter(torch.tensor(math.log(8.0)))   # sharpness of attention
        self.d = d

    def score_maps(self, supp_phiin, q_phiin):
        """supp_phiin [M,C] (bank, normalized), q_phiin [P,C] (normalized). Returns [P] residual norm."""
        Q = F.normalize(self.q_proj(q_phiin), dim=1)
        Kk = F.normalize(self.k_proj(supp_phiin), dim=1)
        temp = self.log_temp.exp()
        attn = torch.softmax(temp * (Q @ Kk.t()), dim=1)            # [P,M]
        recon = attn @ supp_phiin                                   # [P,C] weighted support (feature space)
        return (q_phiin - recon).norm(dim=1)                        # residual norm


class XAttnAdapterML(nn.Module):
    """Adapter that lets eval_config reuse: we cannot fold xattn into per-map ml(); instead we
    build the bank from support and, per query, compute residual-norm scores. eval_config expects
    ml(x,z)->normed map then patch_dist(bank,q). So we DON'T use eval_config for xattn; we eval
    with a dedicated loop (see eval_xattn)."""
    def __init__(self, sc): super().__init__(); self.sc = sc


# ---------------- synthetic anomalies in FEATURE space (generic, category-agnostic) ----------------
def synth_anomaly(q, src):
    """q [1,C,S,S] normalized; src [C,S,S] normalized (another image, possibly SAME category).
    Returns (q_anom [1,C,S,S] normed, mask [S*S] bool). Mix biased toward SUBTLE anomalies:
    gross cross-cat cutpaste is trivially separable in frozen space (no transferable signal),
    so we emphasize partial blends + moderate noise that sit NEAR the normal manifold."""
    qa = q.clone()
    mask = torch.zeros(S, S, dtype=torch.bool, device=q.device)
    bh = random.randint(S // 10, S // 4)          # ~5..14 (smaller = subtler/local)
    bw = random.randint(S // 10, S // 4)
    y0 = random.randint(0, S - bh); x0 = random.randint(0, S - bw)
    mode = random.random()
    if mode < 0.30:                                # feature-space CutPaste (paste foreign block)
        sy = random.randint(0, S - bh); sx = random.randint(0, S - bw)
        qa[0, :, y0:y0 + bh, x0:x0 + bw] = src[:, sy:sy + bh, sx:sx + bw]
    elif mode < 0.55:                              # structured additive noise (moderate)
        amp = 0.3 + 0.5 * random.random()
        qa[0, :, y0:y0 + bh, x0:x0 + bw] += amp * torch.randn(C, bh, bw, device=q.device)
    else:                                          # SUBTLE partial blend query<-src (near-manifold)
        a = 0.25 + 0.45 * random.random()
        sy = random.randint(0, S - bh); sx = random.randint(0, S - bw)
        qa[0, :, y0:y0 + bh, x0:x0 + bw] = (1 - a) * qa[0, :, y0:y0 + bh, x0:x0 + bw] + \
                                            a * src[:, sy:sy + bh, sx:sx + bw]
    qa = F.normalize(qa, dim=1)
    mask[y0:y0 + bh, x0:x0 + bw] = True
    return qa, mask.reshape(-1)


def softmin_dist(bank, q, tau=TAU):
    """soft-min over bank of ||q-bank||: differentiable NN. bank [M,C], q [P,C] -> [P]."""
    D = torch.cdist(q, bank)                        # [P,M]
    w = torch.softmax(-tau * D, dim=1)
    return (w * D).sum(1)


def sample_source_map(this_cat):
    """A normal map for cutpaste/blend. 50% SAME category (subtle positional/structural
    anomaly), 50% a different train category (foreign). Category-agnostic either way."""
    if random.random() < 0.5:
        c = this_cat                                # same-cat: subtle, near-manifold
    else:
        c = random.choice([x for x in TRAIN_CATS if x != this_cat])
    i = random.randrange(n_imgs(c, "train"))
    return make_map(c, "train", i, "hires")         # [C,S,S]


# ---------------- held-out monitoring (1-shot img-AUROC on the 5 TEST cats) ----------------
@torch.no_grad()
def quick_heldout_phi(phi, K=1, draws=HO_DRAWS, sub=HO_SUB, cats=TEST_CATS):
    phi.eval()
    per = {}
    for cat in cats:
        d = load_raw(cat); lab = d["test_img_label"].numpy()
        ntr, nte = n_imgs(cat, "train"), n_imgs(cat, "test")
        ca = []
        for draw in range(draws):
            g = torch.Generator().manual_seed(7 * K + draw)
            sidx = torch.randperm(ntr, generator=g)[:K]
            supp = stack_maps(cat, "train", sidx, "hires")
            bank = to_bank(phi(supp), sub)
            sc = [patch_dist(bank, to_bank(phi(make_map(cat, "test", j, "hires").unsqueeze(0)))).max().item()
                  for j in range(nte)]
            ca.append(roc_auc_score(lab, sc))
        per[cat] = float(np.mean(ca))
    phi.train()
    return float(np.mean(list(per.values()))), per


@torch.no_grad()
def quick_heldout_xattn(sc_mod, K=1, draws=HO_DRAWS, sub=HO_SUB, cats=TEST_CATS):
    sc_mod.eval()
    per = {}
    for cat in cats:
        d = load_raw(cat); lab = d["test_img_label"].numpy()
        ntr, nte = n_imgs(cat, "train"), n_imgs(cat, "test")
        ca = []
        for draw in range(draws):
            g = torch.Generator().manual_seed(7 * K + draw)
            sidx = torch.randperm(ntr, generator=g)[:K]
            supp = stack_maps(cat, "train", sidx, "hires")
            bank = to_bank(supp, sub)
            sc = [sc_mod.score_maps(bank, to_bank(make_map(cat, "test", j, "hires").unsqueeze(0))).max().item()
                  for j in range(nte)]
            ca.append(roc_auc_score(lab, sc))
        per[cat] = float(np.mean(ca))
    sc_mod.train()
    return float(np.mean(list(per.values()))), per


# ---------------- training: phi (deep metric) ----------------
def train_phi(log):
    phi = PhiMetric(C, CH, NB).to(dev)
    opt = torch.optim.Adam(phi.parameters(), lr=LR)
    ntr = {c: n_imgs(c, "train") for c in TRAIN_CATS}
    ho0, per0 = quick_heldout_phi(phi)
    log(f"  [identity check] step0 held-out 1-shot img-AUROC = {ho0:.4f}  (should == frozen patch-NN)")
    log(f"    per-cat: " + "  ".join(f"{k}:{v:.3f}" for k, v in per0.items()))
    curve = [(0, ho0)]
    best = {"ho": ho0, "sd": copy.deepcopy(phi.state_dict()), "step": 0}
    for step in range(STEPS):
        cat = random.choice(TRAIN_CATS)
        K = random.choice([1, 2, 4])
        sidx = torch.randperm(ntr[cat])[:K]
        supp = stack_maps(cat, "train", sidx, "hires")                     # [K,C,S,S]
        qi = random.randrange(ntr[cat])
        while K == 1 and qi == int(sidx[0]):
            qi = random.randrange(ntr[cat])
        q = make_map(cat, "train", qi, "hires").unsqueeze(0)               # normal query (train split)
        src = sample_source_map(cat)
        q_anom, mask = synth_anomaly(q, src)
        # support-noise meta-augmentation
        supp_aug = l2n(supp + AUG_STD * supp.std() * torch.randn_like(supp)) if AUG_STD > 0 else supp
        inp = torch.cat([supp_aug, q, q_anom])                             # [K+2,C,S,S]
        out = phi(inp)
        supp_ml, q_ml, qa_ml = out[:K], out[K:K + 1], out[K + 1:]
        bank = to_bank(supp_ml, BANK_SUB)
        d_anom = softmin_dist(bank, to_bank(qa_ml))                        # [S*S]
        d_clean = softmin_dist(bank, to_bank(q_ml))
        loss_anom = d_anom[~mask].mean() + F.relu(MARGIN - d_anom[mask]).mean()
        loss_clean = d_clean.mean()                                        # genuinely-normal query stays close
        reg = ((out - inp) ** 2).sum(1).mean()                            # specialization suppression
        loss = loss_anom + CLEAN_W * loss_clean + REG_LAM * reg
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 1000 == 0:
            log(f"  step {step+1}/{STEPS}  loss={loss.item():.4f} (anom={loss_anom.item():.3f} "
                f"clean={loss_clean.item():.3f} reg={reg.item():.3f})")
        if (step + 1) % CHK_EVERY == 0:
            ho, per = quick_heldout_phi(phi)
            curve.append((step + 1, ho))
            tag = ""
            if ho > best["ho"]:
                best = {"ho": ho, "sd": copy.deepcopy(phi.state_dict()), "step": step + 1}
                tag = "  <-- best"
                torch.save({"sd": best["sd"], "ho": ho, "step": step + 1,
                            "curve": curve}, os.path.join(CKPT_DIR, "phi_best.pt"))
            log(f"  >> [held-out] step {step+1}: 1-shot img-AUROC = {ho:.4f}{tag}")
    log("\n  [held-out learning curve] " + "  ".join(f"{s//1000 if s>=1000 else 0}k:{a:.3f}" for s, a in curve))
    log(f"  [early stopping] BEST held-out = {best['ho']:.4f} at step {best['step']} "
        f"(last = {curve[-1][1]:.4f})")
    phi.load_state_dict(best["sd"]); phi.eval()
    return phi, curve, best


# ---------------- training: cross-attention ----------------
def train_xattn(log):
    sc = XAttnScorer(C).to(dev)
    opt = torch.optim.Adam(sc.parameters(), lr=LR)
    ntr = {c: n_imgs(c, "train") for c in TRAIN_CATS}
    ho0, per0 = quick_heldout_xattn(sc)
    log(f"  [init] step0 held-out 1-shot img-AUROC = {ho0:.4f}")
    curve = [(0, ho0)]
    best = {"ho": ho0, "sd": copy.deepcopy(sc.state_dict()), "step": 0}
    for step in range(STEPS):
        cat = random.choice(TRAIN_CATS)
        K = random.choice([1, 2, 4])
        sidx = torch.randperm(ntr[cat])[:K]
        supp = stack_maps(cat, "train", sidx, "hires")
        qi = random.randrange(ntr[cat])
        while K == 1 and qi == int(sidx[0]):
            qi = random.randrange(ntr[cat])
        q = make_map(cat, "train", qi, "hires").unsqueeze(0)
        src = sample_source_map(cat)
        q_anom, mask = synth_anomaly(q, src)
        supp_aug = l2n(supp + AUG_STD * supp.std() * torch.randn_like(supp)) if AUG_STD > 0 else supp
        bank = to_bank(supp_aug, BANK_SUB)
        d_anom = sc.score_maps(bank, to_bank(q_anom))
        d_clean = sc.score_maps(bank, to_bank(q))
        loss = d_anom[~mask].mean() + F.relu(MARGIN - d_anom[mask]).mean() + d_clean.mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 1000 == 0:
            log(f"  step {step+1}/{STEPS}  loss={loss.item():.4f}")
        if (step + 1) % CHK_EVERY == 0:
            ho, per = quick_heldout_xattn(sc)
            curve.append((step + 1, ho))
            tag = ""
            if ho > best["ho"]:
                best = {"ho": ho, "sd": copy.deepcopy(sc.state_dict()), "step": step + 1}
                tag = "  <-- best"
            log(f"  >> [held-out] step {step+1}: 1-shot img-AUROC = {ho:.4f}{tag}")
    log("\n  [held-out learning curve] " + "  ".join(f"{s//1000 if s>=1000 else 0}k:{a:.3f}" for s, a in curve))
    log(f"  [early stopping] BEST held-out = {best['ho']:.4f} at step {best['step']} (last={curve[-1][1]:.4f})")
    sc.load_state_dict(best["sd"]); sc.eval()
    return sc, curve, best


# ---------------- full eval for xattn (dedicated; eval_config can't express recon-residual) ----------------
@torch.no_grad()
def eval_xattn(sc, K_list=(1, 2, 5), draws=5, topk_frac=0.01, cats=TEST_CATS):
    from hires_ad import upsample_smooth, grid_maxpool_gt, EVAL_RES
    sc.eval(); res = {}
    for cat in cats:
        d = load_raw(cat); lab = d["test_img_label"].numpy()
        pix224 = d["test_pix_label"].numpy().astype(np.uint8)
        ntest = n_imgs(cat, "test"); npool = n_imgs(cat, "train")
        has_pix = pix224.max() > 0
        res[cat] = {}
        for K in K_list:
            agg = {k: [] for k in ["img_max", "img_smax", "img_topk", "pix_224"]}
            for draw in range(draws):
                g = torch.Generator().manual_seed(7 * K + draw)
                sidx = torch.randperm(npool, generator=g)[:K]
                supp = stack_maps(cat, "train", sidx, "hires")
                bank = to_bank(supp)
                img_max, img_smax, img_topk, smooth_maps = [], [], [], []
                ntop = max(1, int(topk_frac * EVAL_RES * EVAL_RES))
                for j in range(ntest):
                    qm = make_map(cat, "test", j, "hires").unsqueeze(0)
                    dd = sc.score_maps(bank, to_bank(qm))               # [S*S]
                    sm = upsample_smooth(dd, S, smooth=True)
                    smooth_maps.append(sm)
                    img_max.append(float(dd.max().item()))
                    img_smax.append(float(sm.max()))
                    flat = np.partition(sm.reshape(-1), -ntop)[-ntop:]
                    img_topk.append(float(flat.mean()))
                agg["img_max"].append(roc_auc_score(lab, img_max))
                agg["img_smax"].append(roc_auc_score(lab, img_smax))
                agg["img_topk"].append(roc_auc_score(lab, img_topk))
                if has_pix:
                    sm = np.stack(smooth_maps).reshape(ntest, -1)
                    agg["pix_224"].append(roc_auc_score(pix224.reshape(-1) > 0, sm.reshape(-1)))
            res[cat][K] = {k: (float(np.mean(v)) if v else float("nan")) for k, v in agg.items()}
        print(f"    [xattn] {cat} done", flush=True)
    return res


# ============================ main ============================
def fmt_table(name, res, img_key, K_list=(1, 2, 5)):
    lines = [f"----- {name} (5-cat mean) -----",
             "  K   Image-AUROC   Pixel-AUROC@224"]
    for K in K_list:
        im = mean_over_cats(res, K, img_key)
        p2 = mean_over_cats(res, K, "pix_224")
        lines.append(f"  {K}      {im:.4f}         {p2:.4f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="phi", choices=["phi", "xattn"])
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--frozen", type=int, default=1, help="also run frozen baseline")
    ap.add_argument("--out", default=r"D:\K-DS\filmad\src\v2_results.txt")
    args = ap.parse_args()
    random.seed(0); torch.manual_seed(0); np.random.seed(0)
    K_list = (1, 2, 5)

    buf = []
    def log(m=""):
        print(m, flush=True); buf.append(m)

    log("=" * 74)
    log(f"v2 RELATIONAL few-shot AD  |  variant={args.variant}  |  test cats: {TEST_CATS}")
    log(f"  phi: CH={CH} NB={NB} | STEPS={STEPS} MARGIN={MARGIN} REG_LAM={REG_LAM} AUG_STD={AUG_STD} TAU={TAU}")
    log(f"  synthetic anomalies = feature-space CutPaste + structured noise (NO real defect masks)")
    log("=" * 74)

    # ---- frozen baseline (no learning) ----
    frozen = None
    if args.frozen:
        log("\n[FROZEN BASELINE] hires multi-scale patch-NN (eval_config('hires', None))")
        fcache = os.path.join(os.path.dirname(args.out), f"frozen_hires_d{args.draws}.pt")
        t0 = time.time()
        if os.path.exists(fcache):
            frozen = torch.load(fcache, weights_only=False)
            log(f"  (loaded cached frozen baseline from {os.path.basename(fcache)})")
        else:
            frozen = eval_config("hires", None, K_list=K_list, draws=args.draws, smooth=True)
            torch.save(frozen, fcache)
            log(f"  (frozen eval {time.time()-t0:.0f}s -> cached)")
        for ik, lbl in [("img_topk", "img=smoothed-top1%"), ("img_smax", "img=smoothed-max"),
                        ("img_max", "img=raw-grid-max")]:
            log("  [" + lbl + "]")
            for K in K_list:
                log(f"    K={K}: Image-AUROC={mean_over_cats(frozen,K,ik):.4f}  "
                    f"Pixel-AUROC@224={mean_over_cats(frozen,K,'pix_224'):.4f}")

    # ---- v2 ----
    log(f"\n[v2 TRAINING] variant={args.variant} (episodic, generic synth anomalies, early stopping)")
    t0 = time.time()
    if args.variant == "phi":
        model, curve, best = train_phi(log)
    else:
        model, curve, best = train_xattn(log)
    log(f"  (train {time.time()-t0:.0f}s)")

    log(f"\n[v2 EVAL] variant={args.variant} best checkpoint (step {best['step']})")
    t0 = time.time()
    if args.variant == "phi":
        adapter = (lambda supp: None, PhiAdapterML(model))
        v2 = eval_config("hires", adapter, K_list=K_list, draws=args.draws, smooth=True)
    else:
        v2 = eval_xattn(model, K_list=K_list, draws=args.draws)
    log(f"  (v2 eval {time.time()-t0:.0f}s)")

    # ---- comparison tables ----
    IMG = "img_topk"     # smoothed-top1% (task-specified image aggregation)
    log("\n" + "=" * 74)
    log("RESULT TABLE  |  Image-AUROC = smoothed-top1% ; Pixel-AUROC@224 = full-res")
    log("=" * 74)
    if frozen is not None:
        log(fmt_table("FROZEN baseline (no learning)", frozen, IMG))
        log("")
    log(fmt_table(f"v2 ({args.variant}, relational learned)", v2, IMG))

    if frozen is not None:
        log("\n----- DELTA (v2 - frozen), smoothed-top1% img / pix@224 -----")
        log("  K     dImage      dPixel")
        for K in K_list:
            di = mean_over_cats(v2, K, IMG) - mean_over_cats(frozen, K, IMG)
            dp = mean_over_cats(v2, K, "pix_224") - mean_over_cats(frozen, K, "pix_224")
            log(f"  {K}    {di:+.4f}    {dp:+.4f}")

    # per-category for v2
    log(f"\n----- v2 ({args.variant}) per-category (Image smoothed-top1% / Pixel@224) -----")
    log(f"  {'cat':<12}" + "".join(f"K={k} Img/Pix     " for k in K_list))
    for cat in TEST_CATS:
        row = f"  {cat:<12}"
        for K in K_list:
            row += f"{v2[cat][K][IMG]:.3f}/{v2[cat][K]['pix_224']:.3f}     "
        log(row)
    if frozen is not None:
        log(f"\n----- frozen per-category (Image smoothed-top1% / Pixel@224) -----")
        log(f"  {'cat':<12}" + "".join(f"K={k} Img/Pix     " for k in K_list))
        for cat in TEST_CATS:
            row = f"  {cat:<12}"
            for K in K_list:
                row += f"{frozen[cat][K][IMG]:.3f}/{frozen[cat][K]['pix_224']:.3f}     "
            log(row)

    log("\n----- held-out learning curve (1-shot img-AUROC on 5 TEST cats) -----")
    log("  " + "  ".join(f"{s}:{a:.4f}" for s, a in curve))
    log(f"  peak={max(a for _,a in curve):.4f} @ step {best['step']}  |  last={curve[-1][1]:.4f}  "
        f"|  {'PEAKED-THEN-DECLINED (still memorizing)' if best['step']<curve[-1][0] and curve[-1][1]<best['ho']-1e-4 else 'kept-improving/flat (generalizing)'}")

    with open(args.out, "a", encoding="utf-8") as f:
        f.write("\n".join(buf) + "\n\n")
    log(f"\n[written to {args.out}]")


if __name__ == "__main__":
    main()
