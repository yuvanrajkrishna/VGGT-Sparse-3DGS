# Reproducing the SPCOM 2026 results

## Condition-name semantics (important)

The runner (`scripts/run_view_sweep.py`) resolves behavior from the **condition name**:

- `vggt_seedfilter0_10k_sh3` — the paper method. Three name rules combine:
  1. suffix `_10k_sh3` → 10,000 iterations, SH degree 3, matched 3DGS hyperparameters, `-r 8`;
  2. the name is in **no** extra-args table → plain L1 + D-SSIM training (λ = 0.2), no auxiliary losses;
  3. the substring `filter` activates confidence-aware seeding: the VGGT confidence
     sidecar (`points3D_conf.npy`) stays visible to the fork, which initializes each
     Gaussian's opacity as α = 0.05 + 0.45·c (c = per-scene min–max-normalized confidence).
     **Do not rename this condition** — without a matching substring the sidecar is
     hidden and seeding silently turns off (uniform α = 0.1).
- `colmap_trainonly_10k_sh3` — COLMAP SfM from scratch on only the n training images
  (CPU SIFT), Sim(3)-aligned to the reference model, then identical vanilla training.
  COLMAP failures (no model) are expected at sparse n — they ARE the result.

## Expected outputs per cell

`outputs_spcom_camera_ready/paper_eval/mipnerf360/<scene>/fixed_test_n<k>/<condition>/`
- `model/results.json` — `{"ours_10000": {"PSNR", "SSIM", "LPIPS"}}` (LPIPS = vgg)
- `dataset/sparse/0/test.txt` — held-out views (every 8th image; identical at every n)
- `run_metadata.json` — stage timings, exact training command, split documentation

## Expected numbers (full per-scene table)

7-scene means (PSNR / SSIM / LPIPS), condition `vggt_seedfilter0_10k_sh3`:
- n=3: 12.34 / 0.167 / 0.607 · n=6: 14.12 / 0.228 / 0.523
- n=9: 14.88 / 0.257 / 0.502 · n=12: 15.96 / 0.287 / 0.454

COLMAP train-only succeeds on: n=3 {room}; n=6 {garden, kitchen}; n=9,12 {all but stump}.

## Runtime / memory (RTX 4080 SUPER 16 GB)

VGGT inference + export ≈ 10 s; full cell (init → train 10k → render → metrics)
≈ 101–166 s (n = 3–12). Peak GPU memory ≈ 14 GB during the VGGT pass (bf16);
3DGS training typically < 2 GB (up to ~5 GB on dense outdoor scenes).

## Pose-accuracy metrics

After running the 28 Mip-NeRF 360 cells: `python eval/compute_ate_spcom.py` (ATE,
Sim(3)-aligned, vs. the dataset's full-scene reference COLMAP) and
`python eval/compute_rpe_spcom.py` (relative rotation / translation-direction errors).
