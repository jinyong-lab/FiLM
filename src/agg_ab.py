"""Supplementary: dump 5-draw image-score aggregation (raw-grid-max / smoothed-max /
smoothed-top1%) for configs A (base28) and B (hires) at K=1/2/5, so the image comparison
is apples-to-apples with C. Reuses hires_ad eval machinery + the same cache."""
import random, numpy as np, torch
from hires_ad import eval_config, mean_over_cats, TEST_CATS

random.seed(0); torch.manual_seed(0); np.random.seed(0)
K_list = (1, 2, 5)
for tag, mode in [("A base28", "base28"), ("B hires+smooth", "hires")]:
    r = eval_config(mode, None, K_list=K_list, draws=5, smooth=True)
    print(f"\n== {tag} : 5-draw image aggregation (5-cat mean) ==")
    for K in K_list:
        mx = mean_over_cats(r, K, "img_max")
        sm = mean_over_cats(r, K, "img_smax")
        tk = mean_over_cats(r, K, "img_topk")
        p2 = mean_over_cats(r, K, "pix_224")
        pg = mean_over_cats(r, K, "pix_grid")
        print(f"  K={K}: raw-max={mx:.4f}  smooth-max={sm:.4f}  smooth-top1%={tk:.4f}"
              f"   | pix@224={p2:.4f} pix@grid={pg:.4f}")
print("AGG_DONE")
