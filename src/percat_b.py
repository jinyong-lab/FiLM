"""Per-category (5-draw) numbers for the best config B (hires multiscale + smoothing)."""
import os, sys, random, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hires_ad import eval_config, TEST_CATS

random.seed(0); torch.manual_seed(0); np.random.seed(0)
r = eval_config("hires", None, K_list=(1, 2, 5), draws=5, smooth=True)
print("\n== B (hires+smooth) per-category, 5-draw ==")
print(f"{'cat':<12} " + " ".join(f"K={k}:Img_top1%/Img_smax/Pix@224" for k in (1, 2, 5)))
for c in TEST_CATS:
    row = f"{c:<12} "
    for K in (1, 2, 5):
        d = r[c][K]
        row += f"{d['img_topk']:.3f}/{d['img_smax']:.3f}/{d['pix_224']:.3f}   "
    print(row)
# means
print(f"{'MEAN':<12} " + "  ".join(
    f"{np.mean([r[c][K]['img_topk'] for c in TEST_CATS]):.3f}/"
    f"{np.mean([r[c][K]['img_smax'] for c in TEST_CATS]):.3f}/"
    f"{np.mean([r[c][K]['pix_224'] for c in TEST_CATS]):.3f}" for K in (1, 2, 5)))
print("PERCAT_DONE")
