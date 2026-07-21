"""FiLM Few-shot AD — 메타학습 query를 '실제 이상'으로 (논문 충실판).
이전(film_ad.py)은 합성(특징공간 CutPaste) 이상으로 학습. 여기서는 10개 학습 카테고리의
실제 test 결함(픽셀 마스크)을 query로 사용 = 지도적 메타학습 신호(더 강함/충실).
support=정상만(비지도), 테스트 5종은 여전히 한번도 안 본 카테고리.
"""
import random
import numpy as np
import torch, torch.nn.functional as F
from film_ad import (TaskEncoder, FiLM, to_map, patch_scores, load, evaluate, report,
                     TRAIN_CATS, TEST_CATS, dev, MARGIN, STEPS, LR, BANK_SUB)


def meta_train_real(train_feats, train_test):
    enc, film = TaskEncoder().to(dev), FiLM().to(dev)
    opt = torch.optim.Adam(list(enc.parameters()) + list(film.parameters()), lr=LR)
    for step in range(STEPS):
        cat = random.choice(TRAIN_CATS)
        feats = train_feats[cat]; tf, tp = train_test[cat]
        K = random.choice([1, 2, 4]); sidx = torch.randperm(feats.shape[0])[:K]
        supp = torch.stack([to_map(feats[i]) for i in sidx]).to(dev)     # 정상 support
        qi = torch.randint(tf.shape[0], (1,)).item()                     # 실제 test 이미지(정상 or 이상)
        q = to_map(tf[qi]).to(dev)
        mask = torch.tensor(tp[qi].reshape(-1) > 0, device=dev)          # 실제 결함 마스크 [784]
        z = enc(supp); supp_ad = film(supp, z); q_ad = film(q.unsqueeze(0), z)[0]
        d = patch_scores(supp_ad, q_ad, sub=BANK_SUB)
        if mask.any():
            loss = d[~mask].mean() + F.relu(MARGIN - d[mask]).mean()
        else:
            loss = d.mean()                                              # 정상 쿼리: 전부 가깝게
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 500 == 0:
            print(f"  step {step+1}/{STEPS} loss={loss.item():.4f}", flush=True)
    enc.eval(); film.eval(); return enc, film


def main():
    random.seed(0); torch.manual_seed(0); np.random.seed(0)
    print("=" * 64)
    print("FiLM Few-shot AD (실제이상 메타학습) | 학습 10 / 테스트 5")
    print("  테스트 5종:", TEST_CATS)
    print("=" * 64)
    train_feats = {c: load(c)["train_feats"] for c in TRAIN_CATS}
    train_test = {c: (load(c)["test_feats"], load(c)["test_pix_label"].numpy()) for c in TRAIN_CATS}
    print("[메타학습 시작: 실제 결함 query]", flush=True)
    enc, film = meta_train_real(train_feats, train_test)
    print("\n[평가: 테스트 5종]", flush=True)
    base_r = evaluate(enc, film, use_film=False, cats=TEST_CATS)
    film_r = evaluate(enc, film, use_film=True, cats=TEST_CATS)
    report("baseline (무-FiLM, few-shot PatchCore식)", base_r)
    report("제안 (FiLM, 실제이상 학습)", film_r)
    print("\n[해석] FiLM - baseline 차이가 +면 (실제이상으로 학습한) FiLM이 실제 기여")


if __name__ == "__main__":
    main()
