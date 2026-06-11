#!/usr/bin/env python
"""ATE for SPCOM camera-ready: RMSE of VGGT-aligned train camera centers vs reference COLMAP.

For each (scene, n) cell of vggt_seedfilter0_10k_sh3:
  - read train cameras from the cell's dataset/sparse/0/images.txt (test images excluded via test.txt;
    test poses are copied from reference GT so including them would bias ATE toward 0)
  - read reference centers from data/mipnerf360/{scene}/sparse/0/images.bin
  - standard ATE protocol: Sim(3) Umeyama align centers, report RMSE (in reference/COLMAP units)
"""
import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "repos/gaussian-splatting"))
from scene.colmap_loader import read_extrinsics_binary, qvec2rotmat  # noqa: E402

COND = "vggt_seedfilter0_10k_sh3"
SCENES = ["bicycle", "garden", "stump", "bonsai", "counter", "kitchen", "room"]
NS = [3, 6, 9, 12]


def read_images_txt(path):
    """name -> camera center C = -R^T t from COLMAP images.txt.

    VGGT-written models have EMPTY points2D lines, so naive 2-line pairing desyncs.
    Detect pose lines structurally instead: 10 fields, last one a non-numeric filename.
    """
    centers = {}
    for line in path.read_text().splitlines():
        f = line.strip().split()
        if len(f) != 10 or f[0].startswith("#"):
            continue
        try:
            float(f[9])
            continue  # last field numeric -> a points2D line, not a pose line
        except ValueError:
            pass
        qvec = np.array(list(map(float, f[1:5])))
        tvec = np.array(list(map(float, f[5:8])))
        R = qvec2rotmat(qvec)
        centers[f[9]] = -R.T @ tvec
    return centers


def umeyama_align(src, dst):
    """Sim(3) aligning src->dst (N x 3 each); returns residual RMSE after alignment."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    cov = dc.T @ sc / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_s = (sc ** 2).sum() / len(src)
    s = np.trace(np.diag(D) @ S) / var_s
    t = mu_d - s * R @ mu_s
    res = (s * (R @ src.T)).T + t - dst
    return float(np.sqrt((res ** 2).sum(1).mean()))


ref_cache = {}
print(f"{'scene':10s} " + " ".join(f"n={n:<8d}" for n in NS))
rows = []
for scene in SCENES:
    if scene not in ref_cache:
        ext = read_extrinsics_binary(ROOT / f"data/mipnerf360/{scene}/sparse/0/images.bin")
        ref_cache[scene] = {im.name: -qvec2rotmat(im.qvec).T @ im.tvec for im in ext.values()}
    ref = ref_cache[scene]
    vals = []
    for n in NS:
        cell = ROOT / f"outputs_spcom_camera_ready/paper_eval/mipnerf360/{scene}/fixed_test_n{n}/{COND}/dataset/sparse/0"
        test_names = set((cell / "test.txt").read_text().split())
        cam = read_images_txt(cell / "images.txt")
        train = sorted(k for k in cam if k not in test_names)
        assert len(train) == n, f"{scene} n={n}: {len(train)} train cams"
        src = np.stack([cam[k] for k in train])
        dst = np.stack([ref[k] for k in train])
        ate = umeyama_align(src, dst)
        scale = float(np.sqrt(((dst - dst.mean(0)) ** 2).sum(1).mean()))  # RMS trajectory radius
        vals.append(ate)
        rows.append((scene, n, ate, ate / scale))
    print(f"{scene:10s} " + " ".join(f"{v:<10.4f}" for v in vals))

print()
for n in NS:
    m = np.mean([a for s, nn, a, r in rows if nn == n])
    mr = np.mean([r for s, nn, a, r in rows if nn == n])
    print(f"mean ATE n={n:2d}: {m:.4f} (COLMAP units)   relative to RMS trajectory radius: {mr*100:.2f}%")
