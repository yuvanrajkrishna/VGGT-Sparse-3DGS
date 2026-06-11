#!/usr/bin/env python
"""RPE for SPCOM camera-ready: relative pose error of VGGT train poses vs reference COLMAP.

Complements compute_ate_spcom.py (same 28 cells of vggt_seedfilter0_10k_sh3,
7 scenes x n in {3,6,9,12}, train cameras only — test poses are copied from
reference GT so including them would bias toward 0).

For all train pairs i<j:
  RPE-rot   = geodesic angle (deg) of (R_j^vggt R_i^vggt^T) (R_j^ref R_i^ref^T)^T.
              Relative rotations are gauge-invariant: no alignment needed.
  RPE-trans = angle (deg) between relative center directions. VGGT center
              differences (C_j - C_i) are first rotated by the Umeyama Sim(3)
              rotation aligning VGGT centers to reference centers (scale and
              translation do not affect directions). Pairs where either
              baseline norm < 1e-8 are skipped.
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
    """name -> (R, C) from COLMAP images.txt; R world-to-cam, C = -R^T t.

    VGGT-written models have EMPTY points2D lines, so naive 2-line pairing desyncs.
    Detect pose lines structurally instead: 10 fields, last one a non-numeric filename.
    """
    poses = {}
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
        poses[f[9]] = (R, -R.T @ tvec)
    return poses


def umeyama_rotation(src, dst):
    """Rotation of the Sim(3) Umeyama transform aligning src->dst (N x 3 each)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    cov = dc.T @ sc / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    return U @ S @ Vt


def rot_angle_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))))


ref_cache = {}
rows = []  # (scene, n, rpe_rot_mean, rpe_trans_mean, n_pairs, n_skipped)
for scene in SCENES:
    if scene not in ref_cache:
        ext = read_extrinsics_binary(ROOT / f"data/mipnerf360/{scene}/sparse/0/images.bin")
        ref_cache[scene] = {
            im.name: (qvec2rotmat(im.qvec), -qvec2rotmat(im.qvec).T @ im.tvec)
            for im in ext.values()
        }
    ref = ref_cache[scene]
    for n in NS:
        cell = ROOT / f"outputs_spcom_camera_ready/paper_eval/mipnerf360/{scene}/fixed_test_n{n}/{COND}/dataset/sparse/0"
        test_names = set((cell / "test.txt").read_text().split())
        cam = read_images_txt(cell / "images.txt")
        train = sorted(k for k in cam if k not in test_names)
        assert len(train) == n, f"{scene} n={n}: {len(train)} train cams"

        Rv = [cam[k][0] for k in train]
        Cv = np.stack([cam[k][1] for k in train])
        Rr = [ref[k][0] for k in train]
        Cr = np.stack([ref[k][1] for k in train])

        R_align = umeyama_rotation(Cv, Cr)  # VGGT -> reference

        rot_errs, trans_errs, skipped = [], [], 0
        for i in range(n):
            for j in range(i + 1, n):
                # rotation: gauge-invariant relative rotation discrepancy
                Erel = (Rv[j] @ Rv[i].T) @ (Rr[j] @ Rr[i].T).T
                rot_errs.append(rot_angle_deg(Erel))
                # translation: angle between relative center directions
                dv = R_align @ (Cv[j] - Cv[i])
                dr = Cr[j] - Cr[i]
                nv, nr = np.linalg.norm(dv), np.linalg.norm(dr)
                if nv < 1e-8 or nr < 1e-8:
                    skipped += 1
                    continue
                cosang = np.clip(np.dot(dv, dr) / (nv * nr), -1.0, 1.0)
                trans_errs.append(float(np.degrees(np.arccos(cosang))))
        rows.append((scene, n, float(np.mean(rot_errs)), float(np.mean(trans_errs)),
                     len(rot_errs), skipped))

print("Per-scene RPE (mean over all train pairs i<j, degrees)")
print(f"{'scene':10s} " + " ".join(f"n={n}: rot / trans   " for n in NS))
for scene in SCENES:
    cells = {nn: (ro, tr) for s, nn, ro, tr, np_, sk in rows if s == scene}
    print(f"{scene:10s} " + " ".join(f"{cells[n][0]:7.3f} / {cells[n][1]:7.3f}  " for n in NS))

total_skipped = sum(sk for *_, sk in rows)
print(f"\nskipped degenerate-baseline pairs (norm < 1e-8): {total_skipped}")

print("\nPer-n means over scenes (degrees)")
print(f"{'':22s} {'RPE-rot':>10s} {'RPE-trans':>10s}")
for n in NS:
    mr = np.mean([ro for s, nn, ro, tr, *_ in rows if nn == n])
    mt = np.mean([tr for s, nn, ro, tr, *_ in rows if nn == n])
    print(f"mean (all 7 scenes) n={n:2d} {mr:10.3f} {mt:10.3f}")

print("\nPer-n means EXCLUDING bicycle n=9 outlier (degrees)")
for n in NS:
    sel = [(ro, tr) for s, nn, ro, tr, *_ in rows
           if nn == n and not (s == "bicycle" and nn == 9)]
    mr = np.mean([ro for ro, tr in sel])
    mt = np.mean([tr for ro, tr in sel])
    print(f"mean ({len(sel)} scenes)      n={n:2d} {mr:10.3f} {mt:10.3f}")
