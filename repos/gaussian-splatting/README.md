# Vendored 3D Gaussian Splatting fork

Fork of [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)
(upstream base `54c035f`) with the paper's modifications:

- `scene/gaussian_model.py` — confidence-seeded initial opacity, α = 0.05 + 0.45·c
- `scene/dataset_readers.py` — `points3D_conf.npy` sidecar loading; `test.txt` split support
- `utils/graphics_utils.py`, `arguments/__init__.py`, `train.py` — plumbing for the above

License: Inria/MPII Gaussian-Splatting research license (`LICENSE.md`).
CUDA extensions (`diff-gaussian-rasterization` branch `dr_aa`, `simple-knn`) are installed
from upstream — see the root README's Installation section.
