"""Amortized subspace model — support-span variant (global anchor removed).

Lineage: this project started from a FiLM-based task-conditioning head that
*modulated query features*. That collapsed via "self-normalization": an
anomalous query could rewrite its own features to look normal. We pivoted to
predicting a *normal subspace* (mu, U) from the support and measuring the query
by its residual OUTSIDE that subspace — the query is passive, so an anomaly
cannot pull the subspace toward itself.

A later diagnosis showed the meta-learned head was still ignoring the support:
`U = base_U + dU(z)` had a global anchor `base_U`, and training collapsed to a
single global subspace shared by every task (cross-task principal-angle
cosine = 1.0000). This file is the FIX:

    SpanSubspacePredictor:  U is built ONLY from the support features via
    attention (no base_U). With no global constant term, U is *structurally*
    forced to be a function of the support -> collapse is impossible.

Pipeline:
    Model 1 (frozen backbone, e.g. DINOv2)  -> patch features
    Model 3 (SpanSubspacePredictor)         -> (mu, U) for THIS task
    Scorer  -> ||(x-mu) - U Uᵀ(x-mu)||²      -> per-patch anomaly score

Baseline for comparison: direct_pca_subspace (per-task PCA = SubspaceAD-style,
training-free). A and B share the SAME scorer; the only difference is how (mu, U)
is obtained.
"""
from __future__ import annotations
import torch
import torch.nn as nn


def l2n(x: torch.Tensor) -> torch.Tensor:
    """L2-normalize the last dimension."""
    return x / (x.norm(dim=-1, keepdim=True) + 1e-6)


def img_patches(feats: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather selected images' patches into [M, D] and L2-normalize.

    feats: [N, P, D] cached patch features. idx: image indices to use as support.
    """
    return l2n(feats[idx].reshape(-1, feats.shape[-1]).float())


class SpanSubspacePredictor(nn.Module):
    """Model 3 (support-span): support features -> (mu, U). No global base_U.

    U is an attention-weighted combination of the support features themselves:
    `r` learnable basis queries attend over the support patches; each query
    yields one direction; QR orthonormalizes them. Because there is no global
    constant term, U MUST depend on the support -> the degenerate "one global
    subspace" optimum is structurally unreachable.
    """

    def __init__(self, feat_dim: int = 768, rank: int = 32, d_attn: int = 256):
        super().__init__()
        self.rank = rank
        self.q = nn.Parameter(torch.randn(rank, d_attn) * 0.02)  # r learnable basis queries
        self.Wk = nn.Linear(feat_dim, d_attn)                    # support -> keys

    def forward(self, support_feats: torch.Tensor):
        # support_feats: [M, D]  (M = #support images * patches-per-image)
        mu = support_feats.mean(0)                               # [D] task center
        k = self.Wk(support_feats)                               # [M, d_attn]
        attn = torch.softmax(self.q @ k.transpose(0, 1) / (k.shape[-1] ** 0.5), dim=-1)  # [r, M]
        u_raw = attn @ support_feats                             # [r, D] task-specific dirs
        U, _ = torch.linalg.qr(u_raw.transpose(0, 1))            # [D, r] orthonormal
        return mu, U


def subspace_residual(x: torch.Tensor, mu: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    """Per-patch anomaly score = squared residual outside the normal subspace.

    x: [P, D] query patches. Returns [P]. U is orthonormal so the projection is
    U Uᵀ (x - mu).
    """
    xc = x - mu                                                  # [P, D]
    proj = (xc @ U) @ U.transpose(0, 1)                          # in-subspace part
    resid = xc - proj                                            # out-of-subspace part
    return (resid ** 2).sum(dim=-1)                              # [P]


@torch.no_grad()
def direct_pca_subspace(sup_feat: torch.Tensor, rank: int = 32):
    """Baseline A: training-free per-task PCA subspace (SubspaceAD-style).

    Same scorer as B; the only difference is that (mu, U) come from PCA here.
    """
    mu = sup_feat.mean(0)
    Xc = sup_feat - mu
    _, _, Vt = torch.linalg.svd(Xc, full_matrices=False)
    U_basis = Vt[:rank].transpose(0, 1)                          # top-r principal directions
    return mu, U_basis


def image_score(residual_map: torch.Tensor, agg: str = "max") -> float:
    """Aggregate per-patch residuals into one image-level score.

    agg="max": worst single patch (good for LOCAL/structural anomalies).
    agg="mean": average (much better for LOGICAL anomalies, which perturb many
    patches slightly). On MVTec-LOCO logical anomalies, mean beats max by ~+0.11
    AUROC — the aggregation, not the subspace, is the real lever.
    """
    if agg == "max":
        return residual_map.max().item()
    if agg == "mean":
        return residual_map.mean().item()
    raise ValueError("unknown agg: " + agg)
