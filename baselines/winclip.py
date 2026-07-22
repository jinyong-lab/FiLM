"""WinCLIP / WinCLIP+ 베이스라인 (우리 직접 측정, 테스트 5종).
- CLIP ViT-B/16(openai) + 정상/이상 텍스트 프롬프트 앙상블 -> zero-shot 텍스트 점수
- WinCLIP+ = 텍스트 점수 + few-shot 시각참조(K개 정상 이미지 임베딩 최근접) 결합
- Image-level AUROC, 테스트 5종(wood·cable·zipper·transistor·hazelnut), K=1/5.
"""
import os, glob, random
import numpy as np
import torch
import open_clip
from PIL import Image
from sklearn.metrics import roc_auc_score

DATA = r"C:\Users\HOSEO\Desktop\K-DS\datasets\mvtec_ad\mvtech_anomaly_detection"
TEST_CATS = ["wood", "cable", "zipper", "transistor", "hazelnut"]
dev = "cuda" if torch.cuda.is_available() else "cpu"

print("[WinCLIP] CLIP ViT-B/16(openai) 로딩...", flush=True)
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
tokenizer = open_clip.get_tokenizer("ViT-B-16")
model.eval().to(dev)

TEMPLATES = ["a photo of a {}.", "a cropped photo of a {}.", "a close-up photo of a {}.",
             "a photo of the {}.", "a bright photo of a {}.", "a dark photo of a {}.",
             "a photo of a small {}.", "a photo of a large {}."]
NORMAL = ["{}", "flawless {}", "perfect {}", "{} without defect", "good {}", "unblemished {}"]
ANOM = ["damaged {}", "{} with defect", "{} with flaw", "broken {}", "defective {}", "{} with scratch"]


@torch.no_grad()
def text_embeds(cat):
    def emb(states):
        prompts = [t.format(s.format(cat)) for s in states for t in TEMPLATES]
        te = model.encode_text(tokenizer(prompts).to(dev))
        te = te / te.norm(dim=-1, keepdim=True)
        v = te.mean(0)
        return v / v.norm()
    return emb(NORMAL), emb(ANOM)


@torch.no_grad()
def img_embed(paths):
    out = []
    for i in range(0, len(paths), 32):
        b = torch.stack([preprocess(Image.open(f).convert("RGB")) for f in paths[i:i + 32]]).to(dev)
        e = model.encode_image(b); e = e / e.norm(dim=-1, keepdim=True)
        out.append(e.cpu())
    return torch.cat(out)


def z(x):
    x = np.asarray(x); return (x - x.mean()) / (x.std() + 1e-9)


def main():
    random.seed(0); torch.manual_seed(0)
    res = {"WinCLIP(zero)": [], "WinCLIP+(1)": [], "WinCLIP+(2)": [], "WinCLIP+(5)": []}
    for cat in TEST_CATS:
        cd = os.path.join(DATA, cat)
        test_paths, labels = [], []
        for dd in sorted(glob.glob(os.path.join(cd, "test", "*"))):
            defect = os.path.basename(dd)
            for f in sorted(glob.glob(os.path.join(dd, "*.png"))):
                test_paths.append(f); labels.append(0 if defect == "good" else 1)
        labels = np.array(labels)
        emb_test = img_embed(test_paths)                    # [N, D]
        n_txt, a_txt = text_embeds(cat)
        s_txt = (emb_test @ a_txt.cpu()) - (emb_test @ n_txt.cpu())   # zero-shot 텍스트 점수
        res["WinCLIP(zero)"].append(roc_auc_score(labels, s_txt.numpy()))

        train_paths = sorted(glob.glob(os.path.join(cd, "train", "good", "*.png")))
        for K, key in [(1, "WinCLIP+(1)"), (2, "WinCLIP+(2)"), (5, "WinCLIP+(5)")]:
            aucs = []
            for draw in range(5):
                g = random.Random(7 * K + draw)
                sidx = g.sample(range(len(train_paths)), K)
                emb_ref = img_embed([train_paths[i] for i in sidx])   # [K, D]
                s_vis = 1 - (emb_test @ emb_ref.T).max(dim=1).values.numpy()   # 1 - max cos
                s = z(s_txt.numpy()) + z(s_vis)                       # WinCLIP+ 결합
                aucs.append(roc_auc_score(labels, s))
            res[key].append(float(np.mean(aucs)))
        print(f"  {cat} done", flush=True)

    print("\n===== WinCLIP 베이스라인 (테스트 5종 평균, Image-AUROC, 우리 측정) =====")
    for k, v in res.items():
        print(f"  {k:16s}: {np.mean(v):.3f}")


if __name__ == "__main__":
    main()
