import os
import shutil
from pathlib import Path
from typing import Any, List

from .base import PipelineBase, PipelineConfig, ensure_dir, resolve_binary, run_command
import sys


class Colmap3DGSPipeline(PipelineBase):
    """Baseline pipeline replicating the COLMAP-initialised 3DGS workflow (slides 4 & 13)."""

    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        colmap_cfg = config.colmap
        # Ensure COLMAP can run headless (Wayland / SSH) by disabling GUI.
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        self.colmap_bin = resolve_binary("colmap", "COLMAP_BINARY", colmap_cfg.get("binary", "colmap"))
        self.colmap_available = shutil.which(self.colmap_bin) is not None or Path(self.colmap_bin).exists()
        self.use_gpu = bool(colmap_cfg.get("use_gpu", False))
        self.db_path = Path(colmap_cfg.get("database_path", self.output_dir / "colmap.db"))
        self.model_txt_dir = ensure_dir(self.output_dir / "colmap" / "model_txt")
        self.model_bin_dir = ensure_dir(self.output_dir / "colmap" / "model_bin")
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
        if self.colmap_available:
            return [
                self.feature_extraction,
                self.feature_matching,
                self.run_mapper,
                self.export_model_txt,
                self.launch_3dgs,
            ]
        return [
            self.prepare_from_existing,
            self.launch_3dgs,
        ]

    # ---- Individual stages ------------------------------------------------------
    def feature_extraction(self) -> None:
        self._colmap(
            "feature_extractor",
            "--database_path",
            str(self.db_path),
            "--image_path",
            str(self.images_dir),
            "--ImageReader.single_camera",
            "1",
            "--ImageReader.camera_model",
            "PINHOLE",
            "--SiftExtraction.use_gpu",
            "1" if self.use_gpu else "0",
        )

    def feature_matching(self) -> None:
        matcher = self.config.colmap.get("matcher", "exhaustive_matcher")
        self._colmap(
            matcher,
            "--database_path",
            str(self.db_path),
            "--SiftMatching.use_gpu",
            "1" if self.use_gpu else "0",
        )

    def run_mapper(self) -> None:
        workspace = ensure_dir(self.output_dir / "colmap" / "workspace")
        self._colmap(
            "mapper",
            "--database_path",
            str(self.db_path),
            "--image_path",
            str(self.images_dir),
            "--output_path",
            str(workspace),
        )

        reconstructions = [p for p in workspace.iterdir() if p.is_dir()]
        if not reconstructions:
            raise RuntimeError("COLMAP mapper produced no sparse reconstructions.")
        # Pick the largest model by number of images.
        reconstructions.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        latest = reconstructions[0]
        ensure_dir(self.model_bin_dir)
        ensure_dir(self.dataset_sparse_dir)
        for file in latest.glob("*.bin"):
            shutil.copy(file, self.model_bin_dir / file.name)
            shutil.copy(file, self.dataset_sparse_dir / file.name)

    def prepare_from_existing(self) -> None:
        src = self.scene_dir / "sparse" / "0"
        if not src.exists():
            raise FileNotFoundError(
                "COLMAP binary not available and no precomputed sparse model found at "
                f"{src}. Please provide COLMAP outputs."
            )
        dst = ensure_dir(self.dataset_sparse_dir)
        for file in src.glob("*"):
            if file.is_file():
                shutil.copy(file, dst / file.name)
        # Ensure images symlink exists already
        self._ensure_images_link()

    def export_model_txt(self) -> None:
        self._colmap(
            "model_converter",
            "--input_path",
            str(self.model_bin_dir),
            "--output_path",
            str(self.model_txt_dir),
            "--output_type",
            "TXT",
        )
        for txt_file in self.model_txt_dir.glob("*.txt"):
            shutil.copy(txt_file, self.dataset_sparse_dir / txt_file.name)

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
        extra_args = self.config.training.get("extra_args")
        if extra_args:
            args.extend(extra_args)
        run_command(args)

    # ---- Helpers ----------------------------------------------------------------
    def _colmap(self, command: str, *args: str) -> None:
        run_command([self.colmap_bin, command, *args])

    def _ensure_images_link(self) -> None:
        target = self.dataset_dir / "images"
        if target.exists():
            return
        try:
            os.symlink(self.images_dir, target, target_is_directory=True)
        except (AttributeError, NotImplementedError, OSError):
            # Fallback to copying if symlinks are unavailable.
            shutil.copytree(self.images_dir, target)
