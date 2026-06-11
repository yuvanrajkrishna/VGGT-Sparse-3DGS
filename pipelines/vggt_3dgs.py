import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import json as _json

import cv2
import numpy as np
from PIL import Image

from .base import PipelineBase, PipelineConfig, ensure_dir, resolve_binary, run_command
from scripts.vggt_export import run as vggt_run
import torch


def _get(data: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    raise KeyError(f"None of the keys {keys} found in VGGT output")


def _rotation_to_quaternion(R: np.ndarray) -> Tuple[float, float, float, float]:
    """Convert a 3x3 rotation matrix to (qw, qx, qy, qz)."""
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    trace = m00 + m11 + m22

    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (m21 - m12) * s
        qy = (m02 - m20) * s
        qz = (m10 - m01) * s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
        qw = (m21 - m12) / s
        qx = 0.25 * s
        qy = (m01 + m10) / s
        qz = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
        qw = (m02 - m20) / s
        qx = (m01 + m10) / s
        qy = 0.25 * s
        qz = (m12 + m21) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
        qw = (m10 - m01) / s
        qx = (m02 + m20) / s
        qy = (m12 + m21) / s
        qz = 0.25 * s
    return float(qw), float(qx), float(qy), float(qz)


class VGGTThreeDGSPipeline(PipelineBase):
    """VGGT initialised 3DGS pipeline (slides 6–8 & 14–16)."""

    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.vggt_cfg = config.vggt
        self.vggt_repo = Path(self.vggt_cfg.get("repo", os.environ.get("VGGT_REPO", "./repos/vggt")))
        checkpoint = self.vggt_cfg.get("checkpoint", "facebook/VGGT-1B")
        self.checkpoint = checkpoint

        self.vggt_output = ensure_dir(self.output_dir / "vggt")
        self.dataset_dir = ensure_dir(self.output_dir / "dataset")
        self.dataset_sparse_dir = ensure_dir(self.dataset_dir / "sparse" / "0")
        self._ensure_images_link()

        self.gaussian_train_script = resolve_binary(
            "gaussian_splatting",
            "GAUSSIAN_SPLATTING_BINARY",
            config.training.get("script", "train.py"),
        )

    # ---- Pipeline orchestration -------------------------------------------------
    def steps(self) -> List[Any]:
        return [
            self.run_vggt_inference,
            self.convert_to_colmap,
            self.inspect_pointcloud,
            self.launch_3dgs,
        ]

    # ---- Individual stages ------------------------------------------------------
    def run_vggt_inference(self) -> None:
        output_npz = self.vggt_output / "predictions.npz"
        if output_npz.exists() and not self.vggt_cfg.get("force", False):
            print("[PIPELINE] VGGT output already present, skipping inference.")
            return

        force_cpu = bool(self.vggt_cfg.get("force_cpu", False))
        device = "cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
        if device == "cuda":
            major, _ = torch.cuda.get_device_capability()
            dtype = torch.bfloat16 if major >= 8 else torch.float16
        else:
            dtype = torch.float32

        vggt_run(
            self.images_dir,
            self.vggt_output,
            str(self.vggt_cfg.get("checkpoint")),
            int(self.vggt_cfg.get("batch_size", 6)),
            device,
            dtype,
            self.vggt_cfg.get("max_images"),
        )

    def convert_to_colmap(self) -> None:
        predictions = self.vggt_output / "predictions.npz"
        if not predictions.exists():
            # Fall back to most recent npz in folder.
            npzs = sorted(self.vggt_output.glob("*.npz"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not npzs:
                raise FileNotFoundError("VGGT output (.npz) not found; did inference run correctly?")
            predictions = npzs[0]

        data = np.load(predictions, allow_pickle=True)
        Ks = np.asarray(_get(data, ("Ks", "intrinsics", "K")), dtype=float)
        poses = np.asarray(_get(data, ("extrinsics", "w2c", "poses_w2c")), dtype=float)
        points = np.asarray(_get(data, ("points3d", "points", "xyz")), dtype=float)
        rgb = data.get("rgb")
        if rgb is not None:
            rgb = np.asarray(rgb, dtype=float)

        # Extract confidence values if available
        points_conf = data.get("points_conf")
        if points_conf is not None:
            points_conf = np.asarray(points_conf, dtype=float)

        # Extract per-camera depth confidence maps for pose refinement
        depth_conf = data.get("depth_conf")
        if depth_conf is not None:
            depth_conf = np.asarray(depth_conf, dtype=float)

        # Extract per-camera depth maps for depth regularization
        depth_maps = data.get("depths")
        if depth_maps is not None:
            depth_maps = np.asarray(depth_maps, dtype=float)

        meta_path = self.vggt_output / "metadata.json"
        if meta_path.exists():
            import json

            names = json.loads(meta_path.read_text()).get("images", [])
            if not names:
                raise ValueError("metadata.json contains no image names")
            image_paths = [self.images_dir / name for name in names]
        else:
            image_paths = sorted(self.images_dir.iterdir())

        if points.ndim >= 4:
            points = points.reshape(-1, points.shape[-1])
        if points.shape[-1] != 3:
            raise ValueError("3D points must have shape (*, 3)")

        w2c_mats = []
        for pose in poses:
            if pose.shape == (4, 4):
                w2c = pose
            elif pose.shape == (3, 4):
                w2c = np.eye(4)
                w2c[:3, :] = pose
            else:
                raise ValueError("Extrinsics must have shape 3x4 or 4x4")
            w2c_mats.append(w2c)
        w2c_mats = np.stack(w2c_mats)

        if self.config.vggt.get("disable_median_norm", False):
            scale = 1.0
            print("[PIPELINE] Median-norm DISABLED (bridge ablation B_s)")
        else:
            scale = self._normalise_scale(points, w2c_mats)
            print(f"[PIPELINE] Applied global scale factor: {scale:.4f}")

        # Persist COLMAP files.
        width, height = self._image_size(image_paths[0])
        camera_ids = self._write_cameras_txt(Ks, width, height)
        self._write_images_txt(image_paths, w2c_mats, camera_ids)
        self._write_points_txt(points, rgb, points_conf)

        # Save per-camera depth confidence maps for pose refinement
        if depth_conf is not None:
            self._write_depth_conf_maps(depth_conf, image_paths)

        # Export VGGT depth maps for depth regularization
        if depth_maps is not None:
            self._write_depth_maps(depth_maps, image_paths, scale)

    def inspect_pointcloud(self) -> None:
        points_file = self.dataset_sparse_dir / "points3D.txt"
        if not points_file.exists():
            print("[PIPELINE] No points3D.txt to inspect.")
            return
        pts = []
        with points_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                pts.append([float(parts[1]), float(parts[2]), float(parts[3])])
        if not pts:
            print("[PIPELINE] points3D.txt has no entries")
            return
        pts = np.array(pts)
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        print(
            f"[PIPELINE] Point cloud stats — count: {len(pts)}, bounds: X[{mins[0]:.2f}, {maxs[0]:.2f}] "
            f"Y[{mins[1]:.2f}, {maxs[1]:.2f}] Z[{mins[2]:.2f}, {maxs[2]:.2f}]"
        )

    def launch_3dgs(self) -> None:
        if self.config.training.get("skip", False):
            print("[PIPELINE] Skipping 3DGS training (training.skip=true in config)")
            return
        args = [
            sys.executable,
            str(self.gaussian_train_script),
            "-s",
            str(self.dataset_dir),
            "-m",
            str(self.output_dir),
        ]
        iterations = self.config.training.get("iterations")
        if iterations:
            args.extend(["--iterations", str(iterations)])
        # Pose refinement flags
        if self.config.training.get("refine_poses", False):
            args.append("--refine_poses")
            pose_params = {
                "pose_lr_init": "pose_lr_init",
                "pose_lr_final": "pose_lr_final",
                "pose_lr_max_steps": "pose_lr_max_steps",
                "pose_reg_lambda": "pose_reg_lambda",
                "pose_grad_clip": "pose_grad_clip",
                "pose_start_iter": "pose_start_iter",
                "pose_end_iter": "pose_end_iter",
                "pose_huber_delta": "pose_huber_delta",
                "pose_sh_promote_interval": "pose_sh_promote_interval",
            }
            for config_key, cli_key in pose_params.items():
                val = self.config.training.get(config_key)
                if val is not None:
                    args.extend([f"--{cli_key}", str(val)])
            if not self.config.training.get("pose_use_camp", True):
                pass  # pose_use_camp defaults to True; only need to pass if False
                # Note: bool args use store_true, so we can't easily disable via CLI.
                # Users can set it via extra_args if needed.

        extra_args = self.config.training.get("extra_args")
        if extra_args:
            args.extend(extra_args)
        run_command(args)

    # ---- Helpers ----------------------------------------------------------------
    def _ensure_images_link(self) -> None:
        target = self.dataset_dir / "images"
        if target.exists():
            return
        try:
            os.symlink(self.images_dir, target, target_is_directory=True)
        except (AttributeError, NotImplementedError, OSError):
            shutil.copytree(self.images_dir, target)

    def _image_size(self, sample_path: Optional[Path] = None) -> Tuple[int, int]:
        sample = sample_path or next(iter(self.images_dir.glob("*")), None)
        if not sample:
            raise FileNotFoundError(f"No images found under {self.images_dir}")
        with Image.open(sample) as im:
            width, height = im.size
        return width, height

    def _write_cameras_txt(self, Ks: np.ndarray, width: int, height: int) -> List[int]:
        path = self.dataset_sparse_dir / "cameras.txt"
        with path.open("w", encoding="utf-8") as fh:
            fh.write("# Camera list with one line of data per camera:\n")
            fh.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            fh.write(f"# Number of cameras: {len(Ks)}\n")
            camera_ids = []
            for idx, K in enumerate(Ks, start=1):
                fx = float(K[0, 0])
                fy = float(K[1, 1])
                cx = float(K[0, 2])
                cy = float(K[1, 2])
                fh.write(f"{idx} PINHOLE {width} {height} {fx} {fy} {cx} {cy}\n")
                camera_ids.append(idx)
        return camera_ids

    def _write_images_txt(self, image_paths: List[Path], w2c_mats: np.ndarray, camera_ids: List[int]) -> None:
        path = self.dataset_sparse_dir / "images.txt"
        with path.open("w", encoding="utf-8") as fh:
            fh.write("# Image list with two lines of data per image:\n")
            fh.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            fh.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
            for idx, (img_path, w2c, cam_id) in enumerate(zip(image_paths, w2c_mats, camera_ids), start=1):
                R = w2c[:3, :3]
                t = w2c[:3, 3]
                qw, qx, qy, qz = _rotation_to_quaternion(R)
                fh.write(
                    f"{idx} {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]} {cam_id} {img_path.name}\n\n"
                )

    def _write_points_txt(self, points: np.ndarray, rgb: Optional[np.ndarray], confidences: Optional[np.ndarray] = None) -> None:
        path = self.dataset_sparse_dir / "points3D.txt"
        subsample = int(self.vggt_cfg.get("point_subsample", 20))
        pts = points[::subsample]
        if rgb is not None:
            colours = rgb[::subsample]
            if colours.shape[-1] == 1:
                colours = np.repeat(colours, 3, axis=-1)
            if colours.max() <= 1.0:
                colours = (colours * 255.0).clip(0, 255)
        else:
            colours = np.full((len(pts), 3), 128)

        # Handle confidence subsampling and sidecar save
        conf_subsampled = None
        if confidences is not None:
            # Flatten confidence map so it's 1:1 with flattened point list
            confidences = np.asarray(confidences, dtype=float).reshape(-1)
            if len(confidences) != len(points):
                raise ValueError(
                    f"Confidence count ({len(confidences)}) does not match points ({len(points)}) before subsampling"
                )
            # Normalise to [0, 1] so downstream mapping remains bounded.
            conf_min = confidences.min()
            conf_max = confidences.max()
            if conf_max > conf_min:
                confidences = (confidences - conf_min) / (conf_max - conf_min)
            else:
                confidences = np.clip(confidences, 0.0, 1.0)
            conf_subsampled = confidences[::subsample]

            # Save confidence sidecar (1:1 aligned with points after subsampling)
            conf_path = self.dataset_sparse_dir / "points3D_conf.npy"
            np.save(conf_path, conf_subsampled)
            print(
                "[PIPELINE] Saved confidence sidecar: "
                f"{conf_path} ({len(conf_subsampled)} points, "
                f"mean={conf_subsampled.mean():.3f}, min={conf_subsampled.min():.3f}, max={conf_subsampled.max():.3f})"
            )

        with path.open("w", encoding="utf-8") as fh:
            fh.write("# 3D point list with one line of data per point:\n")
            fh.write("#   POINT_ID, X, Y, Z, R, G, B, ERROR, TRACKS[]\n")
            for pid, (pt, col) in enumerate(zip(pts, colours), start=1):
                fh.write(
                    f"{pid} {pt[0]} {pt[1]} {pt[2]} {int(col[0])} {int(col[1])} {int(col[2])} 1.0\n"
                )

    def _write_depth_conf_maps(self, depth_conf: np.ndarray, image_paths: List[Path]) -> None:
        """Save per-camera depth confidence maps as .npy files."""
        conf_dir = self.dataset_sparse_dir / "depth_conf"
        conf_dir.mkdir(parents=True, exist_ok=True)

        # depth_conf shape: (N_images, H, W) or (N_images, H, W, 1)
        if depth_conf.ndim == 4 and depth_conf.shape[-1] == 1:
            depth_conf = depth_conf[..., 0]

        n_saved = 0
        for idx, img_path in enumerate(image_paths):
            if idx >= len(depth_conf):
                break
            stem = img_path.stem
            conf_path = conf_dir / f"{stem}.npy"
            np.save(conf_path, depth_conf[idx].astype(np.float32))
            n_saved += 1

        print(f"[PIPELINE] Saved {n_saved} depth confidence maps to {conf_dir}")

    def _write_depth_maps(self, depth_maps: np.ndarray, image_paths: List[Path], scale: float) -> None:
        """Export VGGT depth maps as 16-bit inverse depth PNGs for depth regularization.

        The 3DGS depth regularization pipeline expects:
          - depths/{stem}.png  — 16-bit uint, storing invdepth * 2^16
          - sparse/0/depth_params.json — per-image {"scale": ..., "offset": ...}

        VGGT depth maps are metric depth in the original coordinate frame.
        After _normalise_scale, coordinates are multiplied by ``scale``, so
        depth values must also be multiplied by ``scale`` to stay consistent.
        """
        depths_dir = self.dataset_dir / "depths"
        depths_dir.mkdir(parents=True, exist_ok=True)

        if depth_maps.ndim == 4 and depth_maps.shape[-1] == 1:
            depth_maps = depth_maps[..., 0]

        depth_params = {}
        n_saved = 0
        for idx, img_path in enumerate(image_paths):
            if idx >= len(depth_maps):
                break
            stem = img_path.stem
            d = depth_maps[idx]  # (H, W) metric depth

            # Apply the same normalisation scale used for points / translations
            d_scaled = d * scale

            # Convert metric depth → inverse depth
            inv_depth = np.where(d_scaled > 1e-6, 1.0 / d_scaled, 0.0)

            # Save as 16-bit PNG (loaded back as float32 / 2^16)
            inv_depth_u16 = np.clip(inv_depth * (2 ** 16), 0, 2 ** 16 - 1).astype(np.uint16)
            cv2.imwrite(str(depths_dir / f"{stem}.png"), inv_depth_u16)

            # Identity depth_params — depth is already in the correct scale
            depth_params[stem] = {"scale": 1.0, "offset": 0.0}
            n_saved += 1

        # Write depth_params.json to sparse/0/
        params_path = self.dataset_sparse_dir / "depth_params.json"
        with open(params_path, "w") as f:
            _json.dump(depth_params, f, indent=2)

        print(f"[PIPELINE] Saved {n_saved} depth maps to {depths_dir}")
        print(f"[PIPELINE] Saved depth_params.json to {params_path}")

    def _normalise_scale(self, points: np.ndarray, w2c: np.ndarray) -> float:
        med = np.median(np.linalg.norm(points, axis=1))
        if med <= 0:
            return 1.0
        scale = 1.0 / med
        points *= scale
        for mat in w2c:
            mat[:3, 3] *= scale
        return scale
