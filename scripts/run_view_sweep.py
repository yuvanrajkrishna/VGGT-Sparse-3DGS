#!/usr/bin/env python3
from __future__ import annotations

"""
Automated view count sweep for COLMAP vs VGGT comparison.

For each view count:
1. Select N views using intelligent pose-based sampling (farthest-point)
2. Run COLMAP + 3DGS (baseline)
3. Run VGGT vanilla + 3DGS
4. Run VGGT confidence-aware + 3DGS
5. Collect metrics (PSNR, SSIM, LPIPS)
"""

import argparse
import csv
import hashlib
import json
import textwrap
import os
import random
import shutil
import subprocess
import sys
import time
import threading
from pathlib import Path

import numpy as np
import re
from collections import defaultdict
from typing import Optional

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None

import yaml

from smart_view_selection import select_views_smart, load_poses_bounds

PSEUDO_VIEW_CONDITIONS = {
    "vggt_vanilla_pseudo",
    "vggt_vanilla_pseudo_v2",
    "vggt_improved_v2_frozenpos_pseudo_v2_10k_sh3",
}


PAPER_GS_ARGS = {
    "iterations": 30000,
    "test_iterations": [7000, 30000],
    "save_iterations": [7000, 30000],
    "sh_degree": 3,
    "lambda_dssim": 0.2,
    "densify_from_iter": 500,
    "densify_until_iter": 15000,
    "densification_interval": 100,
    "densify_grad_threshold": 0.0002,
    "opacity_reset_interval": 3000,
    "percent_dense": 0.01,
    "feature_lr": 0.0025,
    "opacity_lr": 0.05,
    "scaling_lr": 0.005,
    "rotation_lr": 0.001,
    "position_lr_init": 0.00016,
    "position_lr_final": 0.0000016,
    "position_lr_delay_mult": 0.01,
    "position_lr_max_steps": 30000,
}

# Sparse-view optimized args: shorter training, lower SH, tighter densification
SPARSE_GS_ARGS = {
    "iterations": 10000,
    "test_iterations": [10000],
    "save_iterations": [10000],
    "sh_degree": 1,
    "lambda_dssim": 0.2,
    "densify_from_iter": 500,
    "densify_until_iter": 7000,
    "densification_interval": 100,
    "densify_grad_threshold": 0.0002,
    "opacity_reset_interval": 2000,
    "percent_dense": 0.01,
    "feature_lr": 0.0025,
    "opacity_lr": 0.05,
    "scaling_lr": 0.005,
    "rotation_lr": 0.001,
    "position_lr_init": 0.00016,
    "position_lr_final": 0.0000016,
    "position_lr_delay_mult": 0.01,
    "position_lr_max_steps": 10000,
}

# Profiling args: same as SPARSE but trains to 30K with eval every 2K
# Only iterations/test/save/lr_steps change — densification/DropGaussian stay identical
PROFILING_GS_ARGS = {
    **SPARSE_GS_ARGS,
    "iterations": 30000,
    "test_iterations": list(range(2000, 30001, 2000)),
    "save_iterations": [30000],
    "position_lr_max_steps": 30000,
}


def get_optimal_iterations(n_train: int) -> int:
    """Empirical formula for optimal training iterations based on view count.

    Derived from profiling sweep (bicycle + bonsai, 30K iters with eval every 2K).
    Post-densification test PSNR peaks early for high view counts (SPARSE_GS_ARGS
    learning rates are too aggressive), but needs longer training for very sparse views.
    """
    if n_train <= 3:
        return 26000
    elif n_train <= 6:
        return 12000
    else:
        return 8000


def scale_drop_schedule(extra_args: dict, total_iters: int) -> dict:
    """Scale DropGaussian absolute iteration counts proportionally to total training iterations.

    The drop schedule was originally designed for 30K iterations. When using adaptive
    iteration counts (8K-26K via get_optimal_iterations), the absolute peak_iter/end_iter
    values become misaligned. This rescales them to maintain the same fraction of training.
    """
    args = dict(extra_args)
    if "drop_prob_peak_iter" in args and "drop_prob_end_iter" in args:
        peak = args["drop_prob_peak_iter"]
        end = args["drop_prob_end_iter"]
        if peak < 0 or end < 0:
            # Sentinel: monotonic schedule — peak at final iter, no ramp-down
            args["drop_prob_peak_iter"] = total_iters
            args["drop_prob_end_iter"] = total_iters + 1
        else:
            base_iters = 30000
            peak_frac = peak / base_iters
            end_frac = end / base_iters
            args["drop_prob_peak_iter"] = max(500, int(total_iters * peak_frac))
            args["drop_prob_end_iter"] = max(1000, int(total_iters * end_frac))
    return args


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch and hasattr(torch, "manual_seed"):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def _poll_gpu_memory(proc: subprocess.Popen, stop_event: threading.Event, gpu_index: int, result: dict) -> None:
    """Poll nvidia-smi for peak memory usage while proc is running."""
    peak = 0
    while not stop_event.is_set():
        if proc.poll() is not None:
            break
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                    "-i",
                    str(gpu_index),
                ],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=2,
            )
            val = int(out.strip().splitlines()[0])
            peak = max(peak, val)
        except Exception:
            # Ignore polling errors (e.g., no GPU or nvidia-smi missing)
            pass
        stop_event.wait(1.0)
    result["peak_vram_mb"] = peak


def run_monitored_command(
    cmd: list[str],
    cwd: Optional[Path] = None,
    env: Optional[dict] = None,
    log_path: Optional[Path] = None,
    stage_name: str = "",
    gpu_index: int = 0,
) -> dict:
    """Run a command, capturing stdout, wallclock, and peak VRAM via nvidia-smi polling."""
    start = time.time()
    log_file = None
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    stop_event = threading.Event()
    vram_result = {"peak_vram_mb": 0}
    monitor_thread = threading.Thread(target=_poll_gpu_memory, args=(proc, stop_event, gpu_index, vram_result))
    monitor_thread.daemon = True
    monitor_thread.start()

    stdout_lines: list[str] = []
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            stdout_lines.append(line)
            print(line, end="")
            if log_file:
                log_file.write(line)
    finally:
        proc.wait()
        stop_event.set()
        monitor_thread.join()
        if log_file:
            log_file.close()
    elapsed = time.time() - start
    return {
        "code": proc.returncode,
        "elapsed_s": elapsed,
        "peak_vram_mb": vram_result["peak_vram_mb"],
        "stdout": "".join(stdout_lines),
        "stage": stage_name,
        "cmd": cmd,
    }


def collect_gpu_info() -> dict:
    info = {
        "gpu_name": None,
        "driver_version": None,
        "cuda_runtime": torch.version.cuda if torch else None,
        "torch_version": torch.__version__ if torch else None,
    }
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()
        if out:
            parts = [p.strip() for p in out[0].split(",")]
            if len(parts) >= 2:
                info["gpu_name"] = parts[0]
                info["driver_version"] = parts[1]
    except Exception:
        # nvidia-smi not available
        pass
    return info


def ensure_dir_clean(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def prepare_image_subset(source_dir: Path, dest_dir: Path, k: Optional[int] = None) -> list[Path]:
    """Prepare images directory. If k is None, symlink/copy full set; else copy first k sorted."""
    ensure_dir_clean(dest_dir)
    images = sorted([p for p in source_dir.iterdir() if p.is_file()])
    if not images:
        raise FileNotFoundError(f"No images found in {source_dir}")
    if k is None:
        try:
            os.symlink(source_dir, dest_dir, target_is_directory=True)
            return images
        except Exception:
            shutil.copytree(source_dir, dest_dir, dirs_exist_ok=True)
            return images
    if len(images) < k:
        raise ValueError(f"Requested {k} images but only {len(images)} found in {source_dir}")
    selected = images[:k]
    for img in selected:
        shutil.copy2(img, dest_dir / img.name)
    return selected


def list_image_files(source_dir: Path) -> list[Path]:
    return sorted(
        [
            p
            for p in source_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
    )


def select_uniform_subset(items: list[Path], n_items: int) -> list[Path]:
    if n_items <= 0:
        return []
    if len(items) < n_items:
        raise ValueError(f"Requested {n_items} items but only {len(items)} available.")
    positions = np.linspace(0, len(items) - 1, n_items)
    indices = [int(round(pos)) for pos in positions]
    seen = set()
    unique = []
    for idx in indices:
        if idx not in seen:
            unique.append(idx)
            seen.add(idx)
    if len(unique) < n_items:
        for idx in range(len(items)):
            if idx not in seen:
                unique.append(idx)
                seen.add(idx)
                if len(unique) == n_items:
                    break
    return [items[idx] for idx in unique]


def stage_sparse_fixed_test_dataset(
    source_dir: Path,
    stage_dir: Path,
    holdout: int,
    n_train: int,
) -> dict:
    """Stage a sparse train set with fixed test list derived from the full trajectory."""
    source_dir = source_dir.resolve()
    ensure_dir_clean(stage_dir)
    images_dir = stage_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    all_images = list_image_files(source_dir)
    if not all_images:
        raise FileNotFoundError(f"No images found in {source_dir}")

    if holdout <= 0:
        test_images = []
    else:
        test_images = [img for idx, img in enumerate(all_images) if idx % holdout == 0]
    train_pool = [img for idx, img in enumerate(all_images) if idx % holdout != 0]
    train_subset = select_uniform_subset(train_pool, n_train)

    for img in test_images + train_subset:
        target = img.resolve()
        dest = images_dir / target.name
        try:
            os.symlink(target, dest)
        except Exception:
            shutil.copy2(target, dest)

    test_txt = stage_dir / "sparse" / "0" / "test.txt"
    test_txt.parent.mkdir(parents=True, exist_ok=True)
    test_txt.write_text("\n".join([p.name for p in test_images]) + "\n", encoding="utf-8")

    return {
        "all_images": all_images,
        "test_images": test_images,
        "train_pool": train_pool,
        "train_subset": train_subset,
        "test_txt": test_txt,
    }


def _link_images(images: list[Path], dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    for img in images:
        target = img.resolve()
        dest = dest_dir / target.name
        try:
            os.symlink(target, dest)
        except Exception:
            shutil.copy2(target, dest)


def stage_sparse_fixed_trainonly_datasets(
    source_dir: Path,
    stage_root: Path,
    holdout: int,
    n_train: int,
) -> dict:
    """Stage train-only init dataset + train+test final dataset for fixed-test protocol."""
    source_dir = source_dir.resolve()
    init_dir = stage_root / "init"
    final_dir = stage_root / "final"
    ensure_dir_clean(init_dir)
    ensure_dir_clean(final_dir)

    all_images = list_image_files(source_dir)
    if not all_images:
        raise FileNotFoundError(f"No images found in {source_dir}")

    if holdout <= 0:
        test_images = []
    else:
        test_images = [img for idx, img in enumerate(all_images) if idx % holdout == 0]
    train_pool = [img for idx, img in enumerate(all_images) if idx % holdout != 0]
    train_subset = select_uniform_subset(train_pool, n_train)

    _link_images(train_subset, init_dir / "images")
    _link_images(train_subset + test_images, final_dir / "images")

    test_txt = final_dir / "sparse" / "0" / "test.txt"
    test_txt.parent.mkdir(parents=True, exist_ok=True)
    test_txt.write_text("\n".join([p.name for p in test_images]) + "\n", encoding="utf-8")

    init_names = {p.name for p in (init_dir / "images").iterdir() if p.is_file()}
    test_names = {p.name for p in test_images}
    if init_names & test_names:
        raise AssertionError("Init dataset contains test images; expected train-only subset.")

    return {
        "all_images": all_images,
        "test_images": test_images,
        "train_pool": train_pool,
        "train_subset": train_subset,
        "init_dir": init_dir,
        "final_dir": final_dir,
        "test_txt": test_txt,
    }


def _parse_json_list(value: str, field: str) -> list:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON list in manifest field {field}: {exc}") from exc
    if not isinstance(parsed, list):
        raise ValueError(f"Manifest field {field} must be a JSON list.")
    return parsed


def _first_number_in_stem(name: str) -> int:
    match = re.search(r"\d+", Path(name).stem)
    return int(match.group(0)) if match else 10**18


def _sort_image_paths_like_instantsplat(paths: list[Path]) -> list[Path]:
    return sorted(paths, key=lambda p: (_first_number_in_stem(p.name), p.name))


def instant_splat_official_indices(total_images: int, n_train: int) -> tuple[list[int], list[int]]:
    if total_images != 24:
        raise ValueError(f"InstantSplat official 24_views split expects 24 images, got {total_images}")
    test_idx = [int(v) for v in np.linspace(1, total_images - 2, num=12, dtype=int)]
    test_set = set(test_idx)
    train_pool = [idx for idx in range(total_images) if idx not in test_set]
    sparse_positions = [int(v) for v in np.linspace(0, len(train_pool) - 1, num=n_train, dtype=int)]
    train_idx = [train_pool[idx] for idx in sparse_positions]
    return train_idx, test_idx


def _manifest_row_sha256(row: dict) -> str:
    proof = {
        "dataset": row.get("dataset"),
        "scene": row.get("scene"),
        "n_train": row.get("n_train"),
        "source_dir": row.get("source_dir"),
        "image_dir": row.get("image_dir"),
        "sorted_images": row.get("sorted_images"),
        "train_indices_0based": row.get("train_indices_0based"),
        "test_indices_0based": row.get("test_indices_0based"),
        "train_images": row.get("train_images"),
        "test_images": row.get("test_images"),
        "split_rule": row.get("split_rule"),
    }
    payload = json.dumps(proof, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_explicit_split_manifest_row(
    manifest_path: Path,
    dataset: str,
    scene: str,
    n_train: int,
) -> dict:
    """Load one exact train/test split row from a camera-ready manifest CSV."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Split manifest not found: {manifest_path}")
    matches: list[dict] = []
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "dataset",
            "scene",
            "n_train",
            "source_dir",
            "image_dir",
            "sorted_images",
            "train_indices_0based",
            "test_indices_0based",
            "train_images",
            "test_images",
            "split_rule",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Split manifest missing required fields: {sorted(missing)}")
        for row in reader:
            try:
                row_n_train = int(row.get("n_train", ""))
            except ValueError:
                continue
            if row.get("dataset") == dataset and row.get("scene") == scene and row_n_train == n_train:
                matches.append(row)
    if not matches:
        raise ValueError(
            f"No split manifest row for dataset={dataset!r}, scene={scene!r}, n_train={n_train}. "
            "Pass --split_dataset/--split_scene if the runner dataset key or scene path differs "
            "from the manifest labels."
        )
    if len(matches) > 1:
        raise ValueError(f"Ambiguous split manifest rows for dataset={dataset}, scene={scene}, n_train={n_train}")

    row = dict(matches[0])
    row["n_train"] = int(row["n_train"])
    row["sorted_images"] = [str(name) for name in _parse_json_list(row["sorted_images"], "sorted_images")]
    row["train_images"] = [str(name) for name in _parse_json_list(row["train_images"], "train_images")]
    row["test_images"] = [str(name) for name in _parse_json_list(row["test_images"], "test_images")]
    row["train_indices_0based"] = [int(idx) for idx in _parse_json_list(row["train_indices_0based"], "train_indices_0based")]
    row["test_indices_0based"] = [int(idx) for idx in _parse_json_list(row["test_indices_0based"], "test_indices_0based")]

    if len(row["sorted_images"]) != 24:
        raise ValueError(f"Manifest row must contain exactly 24 sorted_images, got {len(row['sorted_images'])}")
    if len(row["train_images"]) != n_train:
        raise ValueError(f"Manifest train_images length {len(row['train_images'])} != n_train {n_train}")
    if len(row["test_images"]) != 12:
        raise ValueError(f"Manifest test_images length {len(row['test_images'])} != 12")
    if len(set(row["sorted_images"])) != 24:
        raise ValueError("Manifest sorted_images contains duplicates")
    if len(set(row["train_images"])) != len(row["train_images"]):
        raise ValueError("Manifest train_images contains duplicates")
    if len(set(row["test_images"])) != len(row["test_images"]):
        raise ValueError("Manifest test_images contains duplicates")
    if set(row["train_images"]) & set(row["test_images"]):
        raise ValueError("Manifest leaks train/test images")

    expected_train_idx, expected_test_idx = instant_splat_official_indices(len(row["sorted_images"]), n_train)
    if row["train_indices_0based"] != expected_train_idx:
        raise ValueError(
            f"Manifest train indices {row['train_indices_0based']} do not match "
            f"official InstantSplat indices {expected_train_idx}"
        )
    if row["test_indices_0based"] != expected_test_idx:
        raise ValueError(
            f"Manifest test indices {row['test_indices_0based']} do not match "
            f"official InstantSplat indices {expected_test_idx}"
        )
    expected_train_images = [row["sorted_images"][idx] for idx in expected_train_idx]
    expected_test_images = [row["sorted_images"][idx] for idx in expected_test_idx]
    if row["train_images"] != expected_train_images:
        raise ValueError(f"Manifest train_images do not match official indices: {row['train_images']} != {expected_train_images}")
    if row["test_images"] != expected_test_images:
        raise ValueError(f"Manifest test_images do not match official indices: {row['test_images']} != {expected_test_images}")
    if "np.linspace" not in str(row.get("split_rule", "")):
        raise ValueError("Manifest split_rule does not record the expected InstantSplat linspace rule")

    row["manifest_row_sha256"] = _manifest_row_sha256(row)
    return row


def _resolve_manifest_path(value: str, base_dir: Path) -> Optional[Path]:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return path


def validate_explicit_split_sources(
    split_row: dict,
    scene_root: Path,
    source_images_dir: Path,
    reference_model_dir: Path,
) -> dict:
    manifest_source_dir = _resolve_manifest_path(str(split_row.get("source_dir", "")), Path.cwd())
    manifest_image_dir = _resolve_manifest_path(str(split_row.get("image_dir", "")), Path.cwd())
    if manifest_source_dir is None or manifest_image_dir is None:
        raise ValueError("Manifest row must provide source_dir and image_dir")
    if not manifest_source_dir.exists():
        raise FileNotFoundError(f"Manifest source_dir does not exist: {manifest_source_dir}")
    if not manifest_image_dir.exists():
        raise FileNotFoundError(f"Manifest image_dir does not exist: {manifest_image_dir}")
    if manifest_source_dir.resolve() != scene_root.resolve():
        raise ValueError(
            "Manifest source_dir must match scene_root exactly for camera-ready InstantSplat runs: "
            f"{manifest_source_dir.resolve()} != {scene_root.resolve()}"
        )
    if manifest_image_dir.resolve() != source_images_dir.resolve():
        raise ValueError(
            "Manifest image_dir must match source_images_dir exactly: "
            f"{manifest_image_dir.resolve()} != {source_images_dir.resolve()}"
        )
    expected_ref = manifest_source_dir / "sparse" / "0"
    if reference_model_dir.resolve() != expected_ref.resolve():
        raise ValueError(
            "Reference sparse model must be the official 24_views source sparse/0: "
            f"{reference_model_dir.resolve()} != {expected_ref.resolve()}"
        )

    actual_sorted = [
        p.name
        for p in _sort_image_paths_like_instantsplat(
            [p for p in source_images_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        )
    ]
    if actual_sorted != split_row["sorted_images"]:
        raise ValueError("Manifest sorted_images do not match numeric-sorted files in source_images_dir")

    return {
        "manifest_source_dir_resolved": str(manifest_source_dir.resolve()),
        "manifest_image_dir_resolved": str(manifest_image_dir.resolve()),
        "reference_model_dir_resolved": str(reference_model_dir.resolve()),
        "source_image_count": len(actual_sorted),
        "source_images_sha256": _names_sha256(actual_sorted),
        "train_images_sha256": _names_sha256(split_row["train_images"]),
        "test_images_sha256": _names_sha256(split_row["test_images"]),
    }


def build_staging_proof(
    *,
    staged: dict,
    train_names: list[str],
    test_names: list[str],
    source_images_dir: Path,
    reference_model_dir: Path,
    camera_preprocess_meta: Optional[dict],
    expected_sorted_names: Optional[list[str]] = None,
) -> dict:
    init_names = [
        p.name for p in _sort_image_paths_like_instantsplat(list_image_files(staged["init_dir"] / "images"))
    ]
    final_names = [
        p.name for p in _sort_image_paths_like_instantsplat(list_image_files(staged["final_dir"] / "images"))
    ]
    leaked_init_test = sorted(set(init_names) & set(test_names))
    missing_train = sorted(set(train_names) - set(init_names))
    missing_final = sorted((set(train_names) | set(test_names)) - set(final_names))
    extra_final = sorted(set(final_names) - (set(train_names) | set(test_names)))
    if leaked_init_test or missing_train or missing_final or extra_final:
        raise ValueError(
            "Staging proof failed: "
            f"leaked_init_test={leaked_init_test[:5]} missing_train={missing_train[:5]} "
            f"missing_final={missing_final[:5]} extra_final={extra_final[:5]}"
        )
    source_names = [
        p.name for p in _sort_image_paths_like_instantsplat(list_image_files(source_images_dir))
    ]
    if expected_sorted_names and source_names != expected_sorted_names:
        raise ValueError("Active source image names do not match expected official sorted image names")
    test_txt = staged["test_txt"]
    test_txt_names = [line.strip() for line in test_txt.read_text(encoding="utf-8").splitlines() if line.strip()]
    if test_txt_names != test_names:
        raise ValueError(f"test.txt names do not match split test names: {test_txt_names} != {test_names}")
    return {
        "code": 0,
        "active_source_images_dir": str(source_images_dir.resolve()),
        "active_reference_model_dir": str(reference_model_dir.resolve()),
        "init_images_dir": str((staged["init_dir"] / "images").resolve()),
        "final_images_dir": str((staged["final_dir"] / "images").resolve()),
        "test_txt": str(test_txt.resolve()),
        "source_image_count": len(source_names),
        "source_images_sha256": _names_sha256(source_names),
        "init_image_count": len(init_names),
        "init_images": init_names,
        "init_images_sha256": _names_sha256(init_names),
        "final_image_count": len(final_names),
        "final_images_sha256": _names_sha256(final_names),
        "train_images": train_names,
        "train_images_sha256": _names_sha256(train_names),
        "test_images": test_names,
        "test_images_sha256": _names_sha256(test_names),
        "test_txt_images_sha256": _names_sha256(test_txt_names),
        "leaked_test_images_in_init": leaked_init_test,
        "camera_preprocess_required": bool(camera_preprocess_meta and camera_preprocess_meta.get("required")),
    }


def existing_run_matches_split(run_root: Path, split_row: dict) -> bool:
    meta = _load_run_metadata(run_root)
    if not meta:
        return False
    return (
        meta.get("split_row_sha256") == split_row.get("manifest_row_sha256")
        and meta.get("train_images") == split_row.get("train_images")
        and meta.get("test_images") == split_row.get("test_images")
        and meta.get("train_indices_0based") == split_row.get("train_indices_0based")
        and meta.get("test_indices_0based") == split_row.get("test_indices_0based")
    )


def existing_camera_preprocess_result_valid(run_root: Path, expected_meta: Optional[dict]) -> bool:
    """Return whether an existing result matches the current camera preprocessing contract."""
    if not expected_meta:
        return True
    expected_required = bool(expected_meta.get("required"))
    meta = _load_run_metadata(run_root)
    actual = meta.get("camera_preprocess")
    if not expected_required and not isinstance(actual, dict):
        return True
    if not isinstance(actual, dict):
        return False
    if bool(actual.get("required")) != expected_required:
        return False
    if actual.get("method") != expected_meta.get("method"):
        return False
    for key in (
        "raw_camera_models",
        "final_camera_models",
        "expected_images_sha256",
        "final_images_sha256",
        "final_model_images_sha256",
    ):
        if expected_meta.get(key) is not None and actual.get(key) != expected_meta.get(key):
            return False
    if expected_required:
        if actual.get("code") != 0:
            return False
        if actual.get("policy") != "undistort":
            return False
        if actual.get("method") != "colmap_image_undistorter":
            return False
        stages = meta.get("stages")
        if not isinstance(stages, dict):
            return False
        undistort_stage = stages.get("camera_undistortion")
        if not isinstance(undistort_stage, dict) or undistort_stage.get("code") != 0:
            return False
        final_models = set(actual.get("final_camera_models") or [])
        if not final_models or not final_models <= {"PINHOLE", "SIMPLE_PINHOLE"}:
            return False
        dim = actual.get("dimension_validation")
        if not isinstance(dim, dict) or dim.get("code") != 0 or not dim.get("checked"):
            return False
    return True


def stage_explicit_trainonly_datasets(
    source_dir: Path,
    stage_root: Path,
    train_image_names: list[str],
    test_image_names: list[str],
) -> dict:
    """Stage train-only init + train/test final datasets using exact image names."""
    source_dir = source_dir.resolve()
    init_dir = stage_root / "init"
    final_dir = stage_root / "final"
    ensure_dir_clean(init_dir)
    ensure_dir_clean(final_dir)

    all_images = list_image_files(source_dir)
    by_name = {p.name: p for p in all_images}
    train_set = set(train_image_names)
    test_set = set(test_image_names)
    overlap = sorted(train_set & test_set)
    if overlap:
        raise ValueError(f"Explicit split leaks train/test images: {overlap[:5]}")
    missing = [name for name in train_image_names + test_image_names if name not in by_name]
    if missing:
        raise FileNotFoundError(f"Explicit split references missing images in {source_dir}: {missing[:5]}")

    train_subset = [by_name[name] for name in train_image_names]
    test_images = [by_name[name] for name in test_image_names]

    _link_images(train_subset, init_dir / "images")
    _link_images(train_subset + test_images, final_dir / "images")

    test_txt = final_dir / "sparse" / "0" / "test.txt"
    test_txt.parent.mkdir(parents=True, exist_ok=True)
    test_txt.write_text("\n".join(test_image_names) + "\n", encoding="utf-8")

    init_names = {p.name for p in (init_dir / "images").iterdir() if p.is_file()}
    if init_names & test_set:
        raise AssertionError("Init dataset contains test images; expected train-only explicit subset.")

    return {
        "all_images": all_images,
        "test_images": test_images,
        "train_pool": train_subset,
        "train_subset": train_subset,
        "init_dir": init_dir,
        "final_dir": final_dir,
        "test_txt": test_txt,
        "explicit_split": True,
    }


_RWM_CACHE: dict[str, object] = {}


def _get_rwm(gaussian_repo: Path):
    key = str(gaussian_repo)
    if key not in _RWM_CACHE:
        sys.path.insert(0, str(gaussian_repo))
        from utils import read_write_model as rwm

        _RWM_CACHE[key] = rwm
    return _RWM_CACHE[key]


def find_reference_colmap_model(scene_root: Path) -> Path:
    candidates = [
        scene_root / "sparse" / "0",
        scene_root / "sparse",
    ]
    for cand in candidates:
        if (cand / "cameras.bin").exists() or (cand / "cameras.txt").exists():
            if (cand / "images.bin").exists() or (cand / "images.txt").exists():
                return cand
    raise FileNotFoundError(f"Reference COLMAP model not found under {scene_root}")


def _read_image_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
    if data[:2] == b"\xff\xd8":
        i = 2
        while i + 9 <= len(data):
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            if marker in (0xC0, 0xC1, 0xC2):
                height = int.from_bytes(data[i + 5:i + 7], "big")
                width = int.from_bytes(data[i + 7:i + 9], "big")
                return width, height
            seg_len = int.from_bytes(data[i + 2:i + 4], "big")
            if seg_len < 2:
                break
            i += 2 + seg_len
    raise ValueError(f"Unsupported image format for size read: {path}")


def get_sample_image_size(images_dir: Path) -> tuple[int, int]:
    for img in list_image_files(images_dir):
        return _read_image_size(img)
    raise FileNotFoundError(f"No images found under {images_dir}")


def validate_image_camera_dimensions(
    *,
    image_dir: Path,
    cameras: dict,
    images: dict,
    expected_names: list[str],
) -> dict:
    image_by_name = {img.name: img for img in images.values()}
    mismatches: list[dict] = []
    for name in expected_names:
        image_path = image_dir / name
        if not image_path.exists():
            raise FileNotFoundError(f"Expected image missing during camera dimension validation: {image_path}")
        if name not in image_by_name:
            raise ValueError(f"COLMAP image record missing during camera dimension validation: {name}")
        img = image_by_name[name]
        if img.camera_id not in cameras:
            raise ValueError(f"COLMAP image {name} references missing camera id {img.camera_id}")
        pixel_w, pixel_h = _read_image_size(image_path)
        cam = cameras[img.camera_id]
        if int(cam.width) != int(pixel_w) or int(cam.height) != int(pixel_h):
            mismatches.append(
                {
                    "image": name,
                    "pixel_size": [int(pixel_w), int(pixel_h)],
                    "camera_size": [int(cam.width), int(cam.height)],
                    "camera_id": int(img.camera_id),
                }
            )
    if mismatches:
        raise ValueError(f"Image/COLMAP camera dimension mismatch: {mismatches[:5]}")
    return {
        "code": 0,
        "image_count": len(expected_names),
        "image_dir": str(image_dir.resolve()),
        "images_sha256": _names_sha256(expected_names),
        "camera_ids": sorted({int(image_by_name[name].camera_id) for name in expected_names}),
        "checked": True,
    }


def _scale_camera_params(model: str, params: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    model = model.upper()
    scaled = np.array(params, dtype=float, copy=True)
    if model in {"PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "THIN_PRISM_FISHEYE", "FOV"}:
        scaled[0] *= scale_x
        scaled[1] *= scale_y
        scaled[2] *= scale_x
        scaled[3] *= scale_y
    elif model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL", "RADIAL_FISHEYE"}:
        uniform = 0.5 * (scale_x + scale_y)
        scaled[0] *= uniform
        scaled[1] *= scale_x
        scaled[2] *= scale_y
    return scaled


def scale_cameras_to_size(cameras: dict, target_w: int, target_h: int, rwm) -> tuple[dict, dict]:
    ref = next(iter(cameras.values()))
    scale_x = target_w / float(ref.width)
    scale_y = target_h / float(ref.height)
    scaled = {}
    for cid, cam in cameras.items():
        params = _scale_camera_params(cam.model, cam.params, scale_x, scale_y)
        scaled[cid] = rwm.Camera(
            id=cam.id,
            model=cam.model,
            width=target_w,
            height=target_h,
            params=params,
        )
    meta = {
        "ref_size": [int(ref.width), int(ref.height)],
        "target_size": [int(target_w), int(target_h)],
        "scale_x": scale_x,
        "scale_y": scale_y,
    }
    return scaled, meta


def _camera_center_from_image(img, rwm) -> np.ndarray:
    Rcw = rwm.qvec2rotmat(img.qvec)
    return -Rcw.T @ img.tvec


def _umeyama_alignment(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute similarity transform (scale, rotation, translation) aligning src -> dst."""
    if src.shape != dst.shape or src.shape[1] != 3:
        raise ValueError("Expected src and dst as Nx3 arrays.")
    n = src.shape[0]
    if n < 3:
        raise ValueError("Need at least 3 points to estimate Sim(3).")

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    cov = (dst_demean.T @ src_demean) / n
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    var_src = np.mean(np.sum(src_demean**2, axis=1))
    scale = (S * np.diag(D)).sum() / var_src
    t = dst_mean - scale * R @ src_mean
    return float(scale), R, t


def align_init_to_reference(
    init_model_dir: Path,
    reference_model_dir: Path,
    train_names: list[str],
    test_names: list[str],
    gaussian_repo: Path,
    target_image_size: Optional[tuple[int, int]] = None,
) -> tuple[dict, dict, dict, dict]:
    """Align init COLMAP model into reference frame using train image centers only."""
    rwm = _get_rwm(gaussian_repo)
    cameras_init, images_init, points_init = rwm.read_model(str(init_model_dir))
    cameras_ref, images_ref, _ = rwm.read_model(str(reference_model_dir))
    camera_scale_meta = None
    if target_image_size:
        target_w, target_h = target_image_size
        ref_cam = next(iter(cameras_ref.values()))
        if ref_cam.width != target_w or ref_cam.height != target_h:
            cameras_ref, camera_scale_meta = scale_cameras_to_size(cameras_ref, target_w, target_h, rwm)

    init_by_name = {img.name: img for img in images_init.values()}
    ref_by_name = {img.name: img for img in images_ref.values()}

    missing_ref = [name for name in train_names + test_names if name not in ref_by_name]
    if missing_ref:
        raise ValueError(f"Reference model missing {len(missing_ref)} images: {missing_ref[:5]}")

    shared_train = [name for name in train_names if name in init_by_name]
    missing_init = [name for name in train_names if name not in init_by_name]
    if len(shared_train) < 3:
        raise ValueError(
            f"Init model has only {len(shared_train)} train images; "
            f"need at least 3 for Sim(3). Missing examples: {missing_init[:5]}"
        )

    init_centers = np.stack([_camera_center_from_image(init_by_name[n], rwm) for n in shared_train], axis=0)
    ref_centers = np.stack([_camera_center_from_image(ref_by_name[n], rwm) for n in shared_train], axis=0)
    scale, R, t = _umeyama_alignment(init_centers, ref_centers)

    # Align init images (train-only)
    aligned_train_images = {}
    empty_xys = np.empty((0, 2))
    empty_ids = np.empty((0,), dtype=int)
    for name in shared_train:
        init_img = init_by_name[name]
        ref_img = ref_by_name[name]
        Rcw_init = rwm.qvec2rotmat(init_img.qvec)
        C_init = -Rcw_init.T @ init_img.tvec
        C_ref = scale * R @ C_init + t
        Rcw_ref = Rcw_init @ R.T
        t_ref = -Rcw_ref @ C_ref
        qvec_ref = rwm.rotmat2qvec(Rcw_ref)
        aligned_train_images[ref_img.id] = rwm.Image(
            id=ref_img.id,
            qvec=qvec_ref,
            tvec=t_ref,
            camera_id=ref_img.camera_id,
            name=ref_img.name,
            xys=empty_xys,
            point3D_ids=empty_ids,
        )

    # Use reference poses directly for test images
    final_images = dict(aligned_train_images)
    for name in test_names:
        ref_img = ref_by_name[name]
        final_images[ref_img.id] = rwm.Image(
            id=ref_img.id,
            qvec=ref_img.qvec,
            tvec=ref_img.tvec,
            camera_id=ref_img.camera_id,
            name=ref_img.name,
            xys=empty_xys,
            point3D_ids=empty_ids,
        )

    # Align points
    aligned_points = {}
    for pid, pt in sorted(points_init.items(), key=lambda item: item[0]):
        xyz_ref = scale * (R @ pt.xyz) + t
        aligned_points[pid] = rwm.Point3D(
            id=pid,
            xyz=xyz_ref,
            rgb=pt.rgb,
            error=pt.error,
            image_ids=np.empty((0,), dtype=int),
            point2D_idxs=np.empty((0,), dtype=int),
        )

    used_camera_ids = {img.camera_id for img in final_images.values()}
    cameras_final = {cid: cam for cid, cam in cameras_ref.items() if cid in used_camera_ids}

    meta = {
        "scale": scale,
        "rotation": R.tolist(),
        "translation": t.tolist(),
        "shared_train": len(shared_train),
        "train_missing_in_init": len(missing_init),
    }
    meta["train_used"] = shared_train
    meta["train_missing"] = missing_init
    if camera_scale_meta:
        meta["camera_scale"] = camera_scale_meta
    return cameras_final, final_images, aligned_points, meta


def validate_3dgs_camera_models(cameras: dict) -> dict:
    supported = {"PINHOLE", "SIMPLE_PINHOLE"}
    models = sorted({str(cam.model) for cam in cameras.values()})
    unsupported = [model for model in models if model not in supported]
    if unsupported:
        raise ValueError(
            "Unsupported camera model(s) for this 3DGS training path: "
            f"{unsupported}. Camera-ready InstantSplat runs must be undistorted to "
            "PINHOLE/SIMPLE_PINHOLE before training; refusing to drop distortion silently."
        )
    return {"camera_models": models, "camera_count": len(cameras)}


def prepare_3dgs_camera_reference(
    *,
    raw_images_dir: Path,
    raw_reference_model_dir: Path,
    work_dir: Path,
    gaussian_repo: Path,
    colmap_bin: str,
    radial_policy: str,
    expected_sorted_names: Optional[list[str]] = None,
    log_dir: Optional[Path] = None,
    gpu_index: int = 0,
) -> tuple[Path, Path, dict]:
    """Return a 3DGS-compatible image/model pair, undistorting radial scenes when requested."""
    raw_images_dir = raw_images_dir.resolve()
    raw_reference_model_dir = raw_reference_model_dir.resolve()
    work_dir = work_dir.resolve()
    gaussian_repo = gaussian_repo.resolve()
    if log_dir is not None:
        log_dir = log_dir.resolve()
    rwm = _get_rwm(gaussian_repo)
    cameras_raw, images_raw, _ = rwm.read_model(str(raw_reference_model_dir))
    raw_models = sorted({str(cam.model) for cam in cameras_raw.values()})
    supported = {"PINHOLE", "SIMPLE_PINHOLE"}
    unsupported = [model for model in raw_models if model not in supported]
    expected_names = expected_sorted_names or [
        p.name for p in _sort_image_paths_like_instantsplat(list_image_files(raw_images_dir))
    ]
    raw_model_names = [
        img.name for img in sorted(images_raw.values(), key=lambda img: (_first_number_in_stem(img.name), img.name))
    ]
    if raw_model_names != expected_names:
        raise ValueError(
            "Reference COLMAP model image names must exactly match the official 24-view images: "
            f"{raw_model_names[:5]} ... != {expected_names[:5]} ..."
        )

    base_meta = {
        "raw_image_dir": str(raw_images_dir.resolve()),
        "raw_reference_model_dir": str(raw_reference_model_dir.resolve()),
        "raw_camera_models": raw_models,
        "raw_camera_count": len(cameras_raw),
        "raw_model_image_count": len(images_raw),
        "raw_model_images_sha256": _names_sha256(raw_model_names),
        "expected_image_count": len(expected_names),
        "expected_images_sha256": _names_sha256(expected_names),
    }
    if not unsupported:
        active_dim_meta = validate_image_camera_dimensions(
            image_dir=raw_images_dir,
            cameras=cameras_raw,
            images=images_raw,
            expected_names=expected_names,
        )
        return raw_images_dir, raw_reference_model_dir, {
            **base_meta,
            "code": 0,
            "cmd": [],
            "log_path": None,
            "required": False,
            "method": "native_pinhole_or_simple_pinhole",
            "final_image_dir": str(raw_images_dir.resolve()),
            "final_reference_model_dir": str(raw_reference_model_dir.resolve()),
            "final_camera_models": raw_models,
            "final_camera_count": len(cameras_raw),
            "final_images_sha256": _names_sha256(expected_names),
            "dimension_validation": active_dim_meta,
        }

    if any(model != "SIMPLE_RADIAL" for model in unsupported):
        raise ValueError(
            "Unsupported camera model(s) for automatic preprocessing: "
            f"{unsupported}. Only SIMPLE_RADIAL can be converted by this path."
        )
    if radial_policy != "undistort":
        raise ValueError(
            "Reference camera model is SIMPLE_RADIAL. Pass "
            "--radial_camera_policy undistort to run COLMAP image_undistorter; "
            "the runner refuses to drop radial distortion silently."
        )

    ensure_dir_clean(work_dir)
    log_path = (log_dir or work_dir) / "camera_undistortion.log"
    cmd = [
        str(colmap_bin),
        "image_undistorter",
        "--image_path",
        str(raw_images_dir),
        "--input_path",
        str(raw_reference_model_dir),
        "--output_path",
        str(work_dir),
        "--output_type",
        "COLMAP",
        "--copy_policy",
        "copy",
        "--max_image_size",
        "-1",
    ]
    rec = run_monitored_command(
        cmd,
        cwd=work_dir.parent,
        log_path=log_path,
        stage_name="camera_undistortion",
        gpu_index=gpu_index,
    )
    cmd_meta = {
        "code": rec.get("code"),
        "elapsed_s": rec.get("elapsed_s"),
        "peak_vram_mb": rec.get("peak_vram_mb"),
        "stage": rec.get("stage"),
        "cmd": rec.get("cmd"),
        "log_path": str(log_path),
        "stdout_tail": str(rec.get("stdout", ""))[-4000:],
    }
    if rec.get("code") != 0:
        raise RuntimeError(f"COLMAP image_undistorter failed; see {log_path}")

    undistorted_images_dir = work_dir / "images"
    if not undistorted_images_dir.exists():
        raise FileNotFoundError(f"COLMAP undistorter did not create images directory: {undistorted_images_dir}")
    undistorted_reference_model_dir = find_reference_colmap_model(work_dir)
    undistorted_names = [
        p.name for p in _sort_image_paths_like_instantsplat(list_image_files(undistorted_images_dir))
    ]
    if undistorted_names != expected_names:
        raise ValueError(
            "Undistorted images do not preserve the official sorted image names: "
            f"{undistorted_names[:5]} ... != {expected_names[:5]} ..."
        )
    cameras_undist, images_undist, _ = rwm.read_model(str(undistorted_reference_model_dir))
    final_model_names = [
        img.name for img in sorted(images_undist.values(), key=lambda img: (_first_number_in_stem(img.name), img.name))
    ]
    if final_model_names != expected_names:
        raise ValueError(
            "Undistorted COLMAP model image names must exactly match the official 24-view images: "
            f"{final_model_names[:5]} ... != {expected_names[:5]} ..."
        )
    camera_validation = validate_3dgs_camera_models(cameras_undist)
    active_dim_meta = validate_image_camera_dimensions(
        image_dir=undistorted_images_dir,
        cameras=cameras_undist,
        images=images_undist,
        expected_names=expected_names,
    )

    return undistorted_images_dir, undistorted_reference_model_dir, {
        **base_meta,
        "code": cmd_meta["code"],
        "cmd": cmd_meta["cmd"],
        "log_path": cmd_meta["log_path"],
        "elapsed_s": cmd_meta["elapsed_s"],
        "peak_vram_mb": cmd_meta["peak_vram_mb"],
        "required": True,
        "method": "colmap_image_undistorter",
        "policy": radial_policy,
        "command": cmd_meta,
        "final_image_dir": str(undistorted_images_dir.resolve()),
        "final_reference_model_dir": str(undistorted_reference_model_dir.resolve()),
        "final_camera_models": camera_validation["camera_models"],
        "final_camera_count": camera_validation["camera_count"],
        "final_model_image_count": len(images_undist),
        "final_images_sha256": _names_sha256(undistorted_names),
        "final_model_images_sha256": _names_sha256(final_model_names),
        "dimension_validation": active_dim_meta,
    }


def write_colmap_text_model(model_dir: Path, cameras: dict, images: dict, points: dict, gaussian_repo: Path) -> None:
    """Write COLMAP text audit files plus binary files preferred by the 3DGS loader."""
    rwm = _get_rwm(gaussian_repo)
    model_dir.mkdir(parents=True, exist_ok=True)
    rwm.write_model(cameras, images, points, str(model_dir), ext=".txt")
    rwm.write_model(cameras, images, points, str(model_dir), ext=".bin")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def update_metrics_csv(path: Path, row: dict, fieldnames: list[str]) -> None:
    rows = []
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
    updated = False
    for existing in rows:
        if (
            existing.get("dataset") == row.get("dataset")
            and existing.get("scene") == row.get("scene")
            and existing.get("setting") == row.get("setting")
            and existing.get("condition") == row.get("condition")
        ):
            existing.update(row)
            updated = True
            break
    if not updated:
        rows.append(row)
    write_csv(path, rows, fieldnames)


def select_views(
    src_images_dir: Path,
    dst_images_dir: Path,
    n_views: int,
    data_root: Path,
    scene: str,
    strategy: str = "pose",
    seed: int = 42,
    hybrid_neighbor_span: int = 2,
):
    """
    Select n_views using intelligent pose-based sampling.

    Args:
        src_images_dir: Source directory with all images
        dst_images_dir: Destination directory for selected images
        n_views: Number of views to select
        data_root: Root directory containing poses_bounds.npy
        scene: Scene name (e.g., "bicycle")
    """
    images = sorted(src_images_dir.glob('*'))
    N = len(images)

    if n_views > N:
        raise ValueError(f"Requested {n_views} views but only {N} available")

    if strategy == "pose":
        print(f"Using intelligent pose-based sampling for {n_views} views...")
        indices, selected_image_names = select_views_smart(
            data_root,
            scene,
            n_views,
            translation_weight=1.0,
            rotation_weight=0.5,
            seed=seed,
        )
    elif strategy == "consecutive":
        print(f"Using consecutive sampling for {n_views} views (seed={seed})...")
        poses_path = data_root / scene / "poses_bounds.npy"
        if not poses_path.exists():
            raise FileNotFoundError(f"poses_bounds.npy not found: {poses_path}")
        poses, _bounds, image_names = load_poses_bounds(poses_path)
        total = len(poses)
        if n_views > total:
            raise ValueError(f"Requested {n_views} views but only {total} poses available")
        rng = random.Random(seed)
        start = rng.randint(0, total - n_views)
        indices = np.arange(start, start + n_views)
        selected_image_names = [image_names[i] for i in indices]
    elif strategy == "hybrid":
        print(f"Using hybrid pose+overlap sampling for {n_views} views (seed={seed})...")
        # Use farthest-point anchors for global coverage.
        anchors, _ = select_views_smart(
            data_root,
            scene,
            min(n_views, n_views // max(hybrid_neighbor_span, 1) + 1),
            translation_weight=1.0,
            rotation_weight=0.5,
            seed=seed,
        )

        total = len(images)
        used = set()
        ordered = []

        def add_index(idx: int):
            if 0 <= idx < total and idx not in used:
                used.add(idx)
                ordered.append(idx)

        for anchor in anchors:
            add_index(int(anchor))
            if len(ordered) >= n_views:
                break

        offset = 1
        # Expand anchors with local neighbors to ensure overlap.
        while len(ordered) < n_views and offset <= hybrid_neighbor_span:
            for anchor in anchors:
                add_index(int(anchor) - offset)
                if len(ordered) >= n_views:
                    break
                add_index(int(anchor) + offset)
                if len(ordered) >= n_views:
                    break
            offset += 1

        # If still lacking coverage, fall back to consecutive sampling starting near centre.
        if len(ordered) < n_views:
            centre = int(np.median(ordered)) if ordered else total // 2
            start = max(0, centre - n_views // 2)
            for idx in range(start, min(total, start + n_views * 2)):
                add_index(idx)
                if len(ordered) >= n_views:
                    break

        ordered = sorted(ordered)[:n_views]
        if len(ordered) < n_views:
            raise RuntimeError("Hybrid selection failed to gather enough unique views")
        indices = np.array(ordered, dtype=int)
        selected_image_names = [images[i].name for i in indices]
    else:
        raise ValueError(f"Unknown selection strategy '{strategy}'.")

    dst_images_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "indices": indices.tolist(),
        "image_names": selected_image_names,
        "strategy": strategy,
        "seed": seed,
    }

    # Copy selected images
    selected_images = []
    for img_name in selected_image_names:
        src_path = src_images_dir / img_name
        if src_path.exists():
            shutil.copy2(src_path, dst_images_dir / img_name)
            selected_images.append(src_path)
        else:
            print(f"[WARNING] Image not found: {src_path}")

    # Write metadata file alongside images directory
    metadata_path = dst_images_dir.parent / "selection_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    print(f"Selected {len(selected_images)} views using {strategy} sampling")
    print(f"Coverage indices: {indices.tolist()}")
    return selected_images, indices


def run_colmap_baseline(
    scene_dir: Path,
    output_name: str,
    num_threads: int = 8,
    use_gpu: bool = True,
):
    """Run COLMAP SfM on selected views."""
    print(f"\n{'='*60}")
    print(f"RUNNING COLMAP BASELINE: {output_name}")
    print(f"{'='*60}\n")

    sparse_dir = scene_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    # Feature extraction
    gpu_flag = "1" if use_gpu else "0"

    cmd = [
        "colmap", "feature_extractor",
        "--database_path", str(scene_dir / "database.db"),
        "--image_path", str(scene_dir / "images"),
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_model", "PINHOLE",
        "--SiftExtraction.use_gpu", gpu_flag,
        "--SiftExtraction.num_threads", str(num_threads),
        "--SiftExtraction.max_image_size", "3200",
        "--SiftExtraction.max_num_features", "16384",
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[WARNING] Feature extraction failed: {result.stderr}")
        return False

    # Feature matching
    cmd = [
        "colmap", "exhaustive_matcher",
        "--database_path", str(scene_dir / "database.db"),
        "--SiftMatching.use_gpu", gpu_flag,
        "--SiftMatching.num_threads", str(num_threads),
        "--SiftMatching.max_num_matches", "32768",
        "--SiftMatching.guided_matching", "1",
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[WARNING] Feature matching failed: {result.stderr}")
        return False

    # Mapper
    cmd = [
        "colmap", "mapper",
        "--database_path", str(scene_dir / "database.db"),
        "--image_path", str(scene_dir / "images"),
        "--output_path", str(scene_dir / "sparse"),
        "--Mapper.num_threads", str(num_threads),
        "--Mapper.ba_refine_focal_length", "1",
        "--Mapper.ba_refine_principal_point", "0",
        "--Mapper.ba_refine_extra_params", "1",
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[WARNING] COLMAP mapper failed: {result.stderr}")
        return False

    # Check if reconstruction succeeded
    if not (sparse_dir / "cameras.txt").exists():
        print(f"[WARNING] COLMAP reconstruction failed - no cameras.txt")
        return False

    print(f"[SUCCESS] COLMAP baseline complete: {output_name}")
    return True


def run_vggt_inference(
    config_path: Path,
    output_name: str,
    python_exe: Path,
    force_cpu: bool = False,
):
    """Run VGGT inference."""
    print(f"\n{'='*60}")
    print(f"RUNNING VGGT INFERENCE: {output_name}")
    print(f"{'='*60}\n")

    cmd = [
        str(python_exe), "scripts/run_pipeline.py",
        "--config", str(config_path)
    ]

    env = os.environ.copy()
    env['PYTHONPATH'] = '.'
    if force_cpu:
        env['CUDA_VISIBLE_DEVICES'] = ""

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd(), env=env)

    if result.returncode != 0:
        print("[ERROR] VGGT inference failed.")
        if result.stderr:
            print("stderr:\n" + result.stderr)
        if result.stdout:
            print("stdout:\n" + result.stdout)
        return False

    print(f"[SUCCESS] VGGT inference complete: {output_name}")
    return True


def run_training(
    source_path: Path,
    model_path: Path,
    use_confidence: bool,
    python_exe: Path,
    iterations: int,
    test_iterations: list[int] | None = None,
    save_iterations: list[int] | None = None,
):
    """Run 3DGS training."""
    mode = "confidence" if use_confidence else "vanilla"
    print(f"\n{'='*60}")
    print(f"TRAINING 3DGS ({mode.upper()}): {model_path.name}")
    print(f"{'='*60}\n")

    source_path = Path(source_path).resolve()
    model_path = Path(model_path).resolve()
    model_path.mkdir(parents=True, exist_ok=True)

    # If not using confidence, temporarily rename sidecar
    conf_file = source_path / "sparse" / "0" / "points3D_conf.npy"
    conf_backup = conf_file.with_suffix(".npy.backup")

    if not use_confidence and conf_file.exists():
        conf_file.rename(conf_backup)

    cmd = [
        str(python_exe), "train.py",
        "-s", str(source_path),
        "-m", str(model_path),
        "--iterations", str(iterations),
        "--data_device", "cpu",
        "--disable_viewer",
        "--eval",
    ]

    if test_iterations:
        cmd.append("--test_iterations")
        cmd.extend(str(it) for it in test_iterations)
    if save_iterations:
        cmd.append("--save_iterations")
        cmd.extend(str(it) for it in save_iterations)

    log_path = model_path / "train.log"
    log_lines: list[str] = []
    proc = subprocess.Popen(
        cmd,
        cwd="repos/gaussian-splatting",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert proc.stdout is not None
    with open(log_path, "w", encoding="utf-8") as log_file:
        for line in proc.stdout:
            print(line, end="")
            log_file.write(line)
            log_lines.append(line)
    result = proc.wait()

    # Restore sidecar if we hid it
    if not use_confidence and conf_backup.exists():
        conf_backup.rename(conf_file)

    if result != 0:
        print(f"[ERROR] Training failed (log saved to {log_path})")
        return False, {}

    curves = parse_psnr_curves("".join(log_lines))
    print(f"[SUCCESS] Training complete ({mode}): {model_path.name}")
    return True, curves


def parse_psnr_curves(log_text: str) -> dict[str, list[dict[str, float]]]:
    """Parse PSNR entries from training stdout."""
    curves: dict[str, list[dict[str, float]]] = defaultdict(list)
    if not log_text:
        return curves
    pattern = re.compile(
        r"\[ITER\s+(\d+)\]\s+Evaluating\s+(\w+):.*?PSNR\s+([-+0-9.eE]+)",
        re.DOTALL,
    )
    for match in pattern.finditer(log_text):
        iteration = int(match.group(1))
        split = match.group(2).lower()
        try:
            value = float(match.group(3))
        except ValueError:
            continue
        curves[split].append({"iteration": iteration, "psnr": value})
    return curves


def extract_metrics(test_dir: Path | None):
    """Extract metrics curve from test directory."""
    if test_dir is None or not test_dir.exists():
        return None
    curve = []
    for results_json in sorted(test_dir.glob("ours_*")):
        if not results_json.is_dir():
            continue
        iter_str = results_json.name.split("_")[-1]
        try:
            iteration = int(iter_str)
        except ValueError:
            continue
        json_path = results_json / "results.json"
        if not json_path.exists():
            continue
        with open(json_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        curve.append({
            "iteration": iteration,
            "PSNR": data.get("PSNR"),
            "SSIM": data.get("SSIM"),
            "LPIPS": data.get("LPIPS"),
        })
    if not curve:
        return None
    curve.sort(key=lambda item: item["iteration"])
    final = curve[-1]
    return {
        "PSNR": final.get("PSNR"),
        "SSIM": final.get("SSIM"),
        "LPIPS": final.get("LPIPS"),
        "curve": curve,
    }


# -----------------------------------------------------------------------------
# Pseudo-view generation
# -----------------------------------------------------------------------------
def run_pseudo_view_generation(
    dataset_dir: Path,
    gaussian_repo: Path,
    python_exe: Path,
    gpu_index: int,
    num_views: int = 16,
    seed: int = 42,
    use_ip_adapter: bool = False,
    ip_adapter_scale: float = 0.6,
    clip_filter: bool = False,
    clip_threshold: float = 0.7,
) -> dict:
    """Run offline pseudo-view generation for a dataset."""
    log_dir = dataset_dir / "pseudo_views"
    log_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(python_exe),
        "-m", "pseudo_views.generate",
        "--scene_path", str(dataset_dir),
        "--num_views", str(num_views),
        "--seed", str(seed),
    ]
    if use_ip_adapter:
        cmd += ["--use_ip_adapter", "--ip_adapter_scale", str(ip_adapter_scale)]
    if clip_filter:
        cmd += ["--clip_filter", "--clip_threshold", str(clip_threshold)]
    meta = run_monitored_command(
        cmd,
        cwd=gaussian_repo,
        log_path=log_dir / "generate.log",
        stage_name="pseudo_gen",
        gpu_index=gpu_index,
    )
    return meta


# -----------------------------------------------------------------------------
# Self-Training Bootstrap
# -----------------------------------------------------------------------------
def run_bootstrap_render(
    model_dir: Path,
    dataset_dir: Path,
    output_dir: Path,
    gaussian_repo: Path,
    python_exe: Path,
    gpu_index: int,
    num_views: int = 16,
    iteration: int = 7000,
    seed: int = 42,
) -> dict:
    """Render novel views from a trained model for bootstrap."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(python_exe),
        "-m", "pseudo_views.bootstrap_render",
        "--model_path", str(model_dir),
        "--scene_path", str(dataset_dir),
        "--output_dir", str(output_dir),
        "--num_views", str(num_views),
        "--iteration", str(iteration),
        "--seed", str(seed),
        "--llffhold", "0",
        "--package",  # Also package as pseudo-views
    ]
    meta = run_monitored_command(
        cmd,
        cwd=gaussian_repo,
        log_path=output_dir / "bootstrap_render.log",
        stage_name="bootstrap_render",
        gpu_index=gpu_index,
    )
    return meta


def run_sd_refinement(
    bootstrap_dir: Path,
    dataset_dir: Path,
    gaussian_repo: Path,
    python_exe: Path,
    gpu_index: int,
    strength: float = 0.3,
    ip_adapter_scale: float = 0.6,
) -> dict:
    """Run SD img2img refinement on bootstrap renders, then repackage as pseudo-views."""
    cmd = [
        str(python_exe),
        "-m", "pseudo_views.refine_bootstrap",
        "--bootstrap_dir", str(bootstrap_dir),
        "--scene_path", str(dataset_dir),
        "--strength", str(strength),
        "--ip_adapter_scale", str(ip_adapter_scale),
    ]
    meta = run_monitored_command(
        cmd,
        cwd=gaussian_repo,
        log_path=bootstrap_dir / "sd_refinement.log",
        stage_name="sd_refinement",
        gpu_index=gpu_index,
    )
    return meta


# -----------------------------------------------------------------------------
# VGGT + Bundle Adjustment
# -----------------------------------------------------------------------------
def run_vggt_ba_for_scene(
    init_dir: Path,
    output_root: Path,
    gpu_index: int,
    python_exe: Path,
    vggt_repo: Path = Path("./repos/vggt"),
    ba_params: Optional[dict] = None,
) -> tuple[Optional[Path], dict]:
    """Run VGGT + VGGSfM tracking + pycolmap bundle adjustment.

    Uses the official demo_colmap.py from the VGGT repo.
    Returns (work_dir, meta) where work_dir/sparse/ contains the COLMAP model.
    """
    ba_params = ba_params or {}
    work_dir = output_root / "ba_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Symlink images from init_dir so demo_colmap.py can find them
    images_link = work_dir / "images"
    if images_link.exists():
        if images_link.is_symlink():
            images_link.unlink()
        else:
            shutil.rmtree(images_link)
    os.symlink(init_dir / "images", images_link, target_is_directory=True)

    demo_script = vggt_repo.resolve() / "demo_colmap.py"
    cmd = [
        str(python_exe),
        str(demo_script),
        "--scene_dir", str(work_dir),
        "--use_ba",
        "--max_reproj_error", str(ba_params.get("max_reproj_error", 8.0)),
        "--query_frame_num", str(ba_params.get("query_frame_num", 8)),
        "--max_query_pts", str(ba_params.get("max_query_pts", 4096)),
        "--camera_type", ba_params.get("camera_type", "SIMPLE_PINHOLE"),
        "--seed", str(ba_params.get("seed", 42)),
    ]

    meta = run_monitored_command(
        cmd,
        cwd=Path.cwd(),
        log_path=output_root / "vggt_ba.log",
        stage_name="vggt_ba",
        gpu_index=gpu_index,
    )

    sparse_dir = work_dir / "sparse"
    if not sparse_dir.exists() or not any(sparse_dir.iterdir()):
        return None, meta
    return work_dir, meta


# -----------------------------------------------------------------------------
# Paper-ready evaluation pipeline
# -----------------------------------------------------------------------------
def run_colmap_paper(
    scene_dir: Path,
    num_threads: int,
    use_gpu: bool,
    log_dir: Path,
    gpu_index: int,
    gaussian_repo: Path = Path("./repos/gaussian-splatting"),
    colmap_bin: str = "colmap",
) -> dict:
    """Run COLMAP SfM with monitoring; returns metadata."""
    log_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir = scene_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    gpu_flag = "1" if use_gpu else "0"
    input_images = sorted(p.name for p in (scene_dir / "images").iterdir() if p.is_file())

    steps = [
        (
            "feature_extractor",
            [
                colmap_bin,
                "feature_extractor",
                "--database_path",
                str(scene_dir / "database.db"),
                "--image_path",
                str(scene_dir / "images"),
                "--ImageReader.single_camera",
                "1",
                "--ImageReader.camera_model",
                "PINHOLE",
                "--SiftExtraction.use_gpu",
                gpu_flag,
                "--SiftExtraction.num_threads",
                str(num_threads),
                "--SiftExtraction.max_image_size",
                "3200",
                "--SiftExtraction.max_num_features",
                "16384",
            ],
        ),
        (
            "exhaustive_matcher",
            [
                colmap_bin,
                "exhaustive_matcher",
                "--database_path",
                str(scene_dir / "database.db"),
                "--SiftMatching.use_gpu",
                gpu_flag,
                "--SiftMatching.num_threads",
                str(num_threads),
                "--SiftMatching.max_num_matches",
                "32768",
                "--SiftMatching.guided_matching",
                "1",
            ],
        ),
        (
            "mapper",
            [
                colmap_bin,
                "mapper",
                "--database_path",
                str(scene_dir / "database.db"),
                "--image_path",
                str(scene_dir / "images"),
                "--output_path",
                str(scene_dir / "sparse"),
                "--Mapper.num_threads",
                str(num_threads),
                "--Mapper.ba_refine_focal_length",
                "1",
                "--Mapper.ba_refine_principal_point",
                "0",
                "--Mapper.ba_refine_extra_params",
                "1",
            ],
        ),
    ]

    records: list[dict] = []
    for name, cmd in steps:
        rec = run_monitored_command(
            cmd,
            cwd=scene_dir,
            log_path=log_dir / f"{name}.log",
            stage_name=name,
            gpu_index=gpu_index,
        )
        records.append(rec)
        if rec["code"] != 0:
            break

    mapper_rec = next((rec for rec in records if rec.get("stage") == "mapper"), {})
    model_summary: dict[str, object] = {
        "input_images": input_images,
        "input_image_count": len(input_images),
        "model_image_names": [],
        "model_image_count": 0,
        "point_count": 0,
        "leaked_test_images": [],
        "no_reference_model_used": True,
        "read_ok": False,
    }
    if (
        mapper_rec.get("code") == 0
        and ((sparse_dir / "cameras.txt").exists() or (sparse_dir / "cameras.bin").exists())
        and ((sparse_dir / "images.txt").exists() or (sparse_dir / "images.bin").exists())
    ):
        try:
            rwm = _get_rwm(gaussian_repo)
            _cameras, images, points = rwm.read_model(str(sparse_dir))
            model_names = sorted(img.name for img in images.values())
            input_set = set(input_images)
            model_summary.update(
                {
                    "model_image_names": model_names,
                    "model_image_count": len(model_names),
                    "point_count": len(points),
                    "leaked_test_images": sorted(name for name in model_names if name not in input_set),
                    "read_ok": True,
                }
            )
        except Exception as exc:
            model_summary["read_error"] = str(exc)

    success = (
        mapper_rec.get("code") == 0
        and model_summary["read_ok"] is True
        and int(model_summary["model_image_count"]) > 0
        and int(model_summary["point_count"]) > 0
    )
    return {
        "success": success,
        "code": 0 if success else 1,
        "backend": "colmap_trainonly",
        "steps": records,
        "elapsed_s": sum(r.get("elapsed_s", 0.0) for r in records),
        "peak_vram_mb": max([r.get("peak_vram_mb", 0) for r in records] or [0]),
        **model_summary,
    }


def run_colmap_refpose(
    scene_dir: Path,
    reference_model_dir: Path,
    train_names: list[str],
    gaussian_repo: Path,
    log_dir: Path,
    colmap_bin: str = "colmap",
    gpu_index: int = 0,
    num_threads: int = 8,
    colmap_cpu: bool = False,
) -> dict:
    """Run COLMAP with reference poses: feature extraction + matching + point_triangulator.

    Mirrors DropGaussian's colmap_360.py pipeline:
    1. Extract SIFT features from train-only images
    2. Exhaustive matching
    3. Query database for image/camera IDs
    4. Write reference poses to 'created/' with database-matched IDs and empty observations
    5. point_triangulator (re-triangulates points using known poses + matched features)

    Unlike run_colmap_paper(), this does NOT run 'mapper' (no pose estimation from scratch).
    """
    import sqlite3

    log_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir = scene_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    gpu_flag = "0" if colmap_cpu else "1"

    # Step 1-2: Feature extraction and matching (creates database.db)
    extract_match_steps = [
        (
            "feature_extractor",
            [
                colmap_bin, "feature_extractor",
                "--database_path", str(scene_dir / "database.db"),
                "--image_path", str(scene_dir / "images"),
                "--ImageReader.single_camera", "1",
                "--ImageReader.camera_model", "PINHOLE",
                "--SiftExtraction.use_gpu", gpu_flag,
                "--SiftExtraction.num_threads", str(num_threads),
                "--SiftExtraction.max_image_size", "3200",
                "--SiftExtraction.max_num_features", "16384",
            ],
        ),
        (
            "exhaustive_matcher",
            [
                colmap_bin, "exhaustive_matcher",
                "--database_path", str(scene_dir / "database.db"),
                "--SiftMatching.use_gpu", gpu_flag,
                "--SiftMatching.num_threads", str(num_threads),
                "--SiftMatching.max_num_matches", "32768",
                "--SiftMatching.guided_matching", "1",
            ],
        ),
    ]

    records: list[dict] = []
    for name, cmd in extract_match_steps:
        rec = run_monitored_command(
            cmd, cwd=scene_dir, log_path=log_dir / f"{name}.log",
            stage_name=name, gpu_index=gpu_index,
        )
        records.append(rec)
        if rec["code"] != 0:
            return {
                "success": False, "steps": records,
                "elapsed_s": sum(r.get("elapsed_s", 0.0) for r in records),
                "peak_vram_mb": max([r.get("peak_vram_mb", 0) for r in records] or [0]),
            }

    # Step 3: Query database for image/camera ID mapping
    rwm = _get_rwm(gaussian_repo)
    cameras_ref, images_ref, _ = rwm.read_model(str(reference_model_dir))
    ref_by_name = {img.name: img for img in images_ref.values()}

    db_path = scene_dir / "database.db"
    conn = sqlite3.connect(str(db_path))
    db_name_to_id = {r[1]: r[0] for r in conn.execute("SELECT image_id, name FROM images").fetchall()}
    db_cam_id = conn.execute("SELECT camera_id FROM cameras LIMIT 1").fetchone()[0]

    # Find connected images: those with at least one verified two-view geometry
    connected_ids = set()
    for row in conn.execute(
        "SELECT pair_id FROM two_view_geometries WHERE rows > 0"
    ).fetchall():
        pair_id = row[0]
        # COLMAP encodes pair_id as: image_id1 * MAX_NUM_IMAGES + image_id2
        MAX_NUM_IMAGES = 2147483647  # 2^31 - 1
        id1 = pair_id // MAX_NUM_IMAGES
        id2 = pair_id % MAX_NUM_IMAGES
        connected_ids.add(id1)
        connected_ids.add(id2)
    conn.close()

    # Step 4: Write created/ model with ONLY connected train images for triangulation
    # (disconnected images crash point_triangulator in COLMAP 3.10)
    created_dir = scene_dir / "created"
    created_dir.mkdir(parents=True, exist_ok=True)

    empty_xys = np.empty((0, 2))
    empty_ids = np.empty((0,), dtype=int)

    train_images = {}
    disconnected = []
    for img_name in train_names:
        if img_name in ref_by_name and img_name in db_name_to_id:
            db_id = db_name_to_id[img_name]
            if db_id not in connected_ids:
                disconnected.append(img_name)
                continue
            ref_img = ref_by_name[img_name]
            train_images[db_id] = rwm.Image(
                id=db_id, qvec=ref_img.qvec, tvec=ref_img.tvec,
                camera_id=db_cam_id, name=img_name,
                xys=empty_xys, point3D_ids=empty_ids,
            )
    if disconnected:
        print(f"[REFPOSE] Excluded {len(disconnected)} disconnected images from triangulation: {disconnected}")

    # Build camera with database-matched ID and reference intrinsics
    ref_cam = next(iter(cameras_ref.values()))
    train_cameras = {db_cam_id: rwm.Camera(
        id=db_cam_id, model=ref_cam.model,
        width=ref_cam.width, height=ref_cam.height, params=ref_cam.params,
    )}

    rwm.write_model(train_cameras, train_images, {}, str(created_dir), ext=".txt")
    print(f"[REFPOSE] Wrote {len(train_images)} reference poses to {created_dir} (IDs matched to database)")

    # Step 5: Point triangulator
    tri_rec = run_monitored_command(
        [
            colmap_bin, "point_triangulator",
            "--database_path", str(scene_dir / "database.db"),
            "--image_path", str(scene_dir / "images"),
            "--input_path", str(created_dir),
            "--output_path", str(sparse_dir),
        ],
        cwd=scene_dir,
        log_path=log_dir / "point_triangulator.log",
        stage_name="point_triangulator",
        gpu_index=gpu_index,
    )
    records.append(tri_rec)

    success = (
        tri_rec.get("code") == 0
        and (
            (sparse_dir / "cameras.txt").exists()
            or (sparse_dir / "cameras.bin").exists()
        )
        and (
            (sparse_dir / "images.txt").exists()
            or (sparse_dir / "images.bin").exists()
        )
    )
    return {
        "success": success,
        "code": 0 if success else 1,
        "steps": records,
        "elapsed_s": sum(r.get("elapsed_s", 0.0) for r in records),
        "peak_vram_mb": max([r.get("peak_vram_mb", 0) for r in records] or [0]),
    }


def run_dense_stereo(
    scene_dir: Path,
    log_dir: Path,
    colmap_bin: str = "colmap",
    gpu_index: int = 0,
) -> tuple[Optional[Path], dict]:
    """Run COLMAP dense stereo: image_undistorter + patch_match_stereo + stereo_fusion.

    Expects scene_dir to contain sparse/0/ (model) and images/ (source images).
    Returns (path_to_fused_ply, metadata).
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    dense_dir = scene_dir / "dense"
    if dense_dir.exists():
        if dense_dir.is_symlink() or dense_dir.is_file():
            dense_dir.unlink()
        else:
            shutil.rmtree(dense_dir)
    dense_dir.mkdir(parents=True, exist_ok=True)

    steps = [
        (
            "image_undistorter",
            [
                colmap_bin, "image_undistorter",
                "--image_path", str(scene_dir / "images"),
                "--input_path", str(scene_dir / "sparse" / "0"),
                "--output_path", str(dense_dir),
                "--output_type", "COLMAP",
            ],
        ),
        (
            "patch_match_stereo",
            [
                colmap_bin, "patch_match_stereo",
                "--workspace_path", str(dense_dir),
                "--PatchMatchStereo.geom_consistency", "true",
            ],
        ),
        (
            "stereo_fusion",
            [
                colmap_bin, "stereo_fusion",
                "--workspace_path", str(dense_dir),
                "--output_path", str(dense_dir / "fused.ply"),
            ],
        ),
    ]

    records: list[dict] = []
    for name, cmd in steps:
        rec = run_monitored_command(
            cmd,
            cwd=scene_dir,
            log_path=log_dir / f"{name}.log",
            stage_name=name,
            gpu_index=gpu_index,
        )
        records.append(rec)
        if rec["code"] != 0:
            break

    fused_ply = dense_dir / "fused.ply"
    success = fused_ply.exists() and all(record.get("code") == 0 for record in records)
    return (
        fused_ply if fused_ply.exists() else None,
        {
            "success": success,
            "code": 0 if success else 1,
            "steps": records,
            "elapsed_s": sum(r.get("elapsed_s", 0.0) for r in records),
            "peak_vram_mb": max([r.get("peak_vram_mb", 0) for r in records] or [0]),
            "fused_ply_exists": fused_ply.exists(),
        },
    )


def build_train_command(
    python_exe: Path,
    gaussian_repo: Path,
    dataset_dir: Path,
    model_dir: Path,
    resolution: Optional[int] = None,
    llffhold: Optional[int] = None,
    use_pseudo_views: bool = False,
    refine_poses: bool = False,
    pose_params: Optional[dict] = None,
    depths_dir: Optional[str] = None,
    iterations: Optional[int] = None,
    gs_args: Optional[dict] = None,
    extra_args: Optional[dict] = None,
) -> list[str]:
    gs = gs_args or PAPER_GS_ARGS
    args = [
        str(python_exe),
        str(gaussian_repo / "train.py"),
        "-s",
        str(dataset_dir),
        "-m",
        str(model_dir),
        "--eval",
        "--disable_viewer",
        "--iterations",
        str(gs["iterations"]),
        "--test_iterations",
    ] + [str(t) for t in gs["test_iterations"]] + [
        "--save_iterations",
    ] + [str(s) for s in gs["save_iterations"]] + [
        "--sh_degree",
        str(gs["sh_degree"]),
        "--lambda_dssim",
        str(gs["lambda_dssim"]),
        "--densify_from_iter",
        str(gs["densify_from_iter"]),
        "--densify_until_iter",
        str(gs["densify_until_iter"]),
        "--densification_interval",
        str(gs["densification_interval"]),
        "--densify_grad_threshold",
        str(gs["densify_grad_threshold"]),
        "--opacity_reset_interval",
        str(gs["opacity_reset_interval"]),
        "--percent_dense",
        str(gs["percent_dense"]),
        "--feature_lr",
        str(gs["feature_lr"]),
        "--opacity_lr",
        str(gs["opacity_lr"]),
        "--scaling_lr",
        str(gs["scaling_lr"]),
        "--rotation_lr",
        str(gs["rotation_lr"]),
        "--position_lr_init",
        str(gs["position_lr_init"]),
        "--position_lr_final",
        str(gs["position_lr_final"]),
        "--position_lr_delay_mult",
        str(gs["position_lr_delay_mult"]),
        "--position_lr_max_steps",
        str(gs["position_lr_max_steps"]),
    ]
    if resolution:
        args.extend(["--resolution", str(resolution)])
    if llffhold is not None:
        args.extend(["--llffhold", str(llffhold)])
    if depths_dir:
        args.extend(["--depths", depths_dir])
    if iterations:
        # Override the default iteration count
        idx = args.index("--iterations")
        args[idx + 1] = str(iterations)
    if use_pseudo_views:
        args.append("--use_pseudo_views")
    if refine_poses:
        args.append("--refine_poses")
        if pose_params:
            for key, val in pose_params.items():
                if isinstance(val, bool):
                    if val:
                        args.append(f"--{key}")
                elif val == "FLAG":
                    args.append(f"--{key}")
                else:
                    args.extend([f"--{key}", str(val)])
    if extra_args:
        for key, val in extra_args.items():
            if isinstance(val, bool):
                if val:
                    args.append(f"--{key}")
            elif val == "FLAG":
                args.append(f"--{key}")
            else:
                args.extend([f"--{key}", str(val)])
    return args


def condition_uses_point_conf(condition: str) -> bool:
    return (
        condition == "vggt_confidence"
        or condition in {
            "vggt_bridge_gate_10k_sh3",
            "vggt_bridge_full_10k_sh3",
        }
        or "xyzanchor" in condition
        or "filter" in condition
        or "confdensify" in condition
        or "dd_dafe_conf" in condition
        or "dd_conf" in condition
    )


def condition_disables_bridge_median_norm(condition: str, global_disable: bool = False) -> bool:
    return bool(global_disable) or condition in {
        "vggt_bridge_base_10k_sh3",
        "vggt_bridge_gate_10k_sh3",
    }


def condition_is_bridge_ablation(condition: str) -> bool:
    return condition in {
        "vggt_bridge_base_10k_sh3",
        "vggt_bridge_norm_10k_sh3",
        "vggt_bridge_gate_10k_sh3",
        "vggt_bridge_full_10k_sh3",
    }


def _count_ply_vertices(ply_path: Path) -> Optional[int]:
    try:
        with ply_path.open("rb") as handle:
            for raw_line in handle:
                line = raw_line.decode("ascii", "ignore").strip()
                if line.startswith("element vertex"):
                    return int(line.split()[-1])
                if line == "end_header":
                    break
    except (OSError, ValueError):
        return None
    return None


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _names_sha256(names: list[str]) -> str:
    payload = "\n".join(sorted(Path(str(name)).name for name in names)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_vggt_dense_init_ply(
    *,
    vggt_dataset_dir: Path,
    train_names: list[str],
    test_names: list[str],
    align_meta: dict,
    target_ply: Path,
    gaussian_repo: Path,
    max_points: int = 120_000,
    sample_stride: int = 2,
) -> dict:
    """Build a train-only dense initializer from VGGT dense predictions.

    VGGT predictions are in the train-only init frame. We use the same Sim(3)
    alignment already used for cameras/sparse points, then write a replacement
    3DGS initializer PLY in the final reference frame.
    """
    pred_path = vggt_dataset_dir.parent / "vggt" / "predictions.npz"
    meta_path = vggt_dataset_dir.parent / "vggt" / "metadata.json"
    if not pred_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"VGGT dense predictions missing: {pred_path}")
    with meta_path.open("r", encoding="utf-8") as handle:
        pred_meta = json.load(handle)
    image_order = [str(name) for name in pred_meta.get("images", [])]
    if not image_order:
        raise ValueError("VGGT metadata has no image order")
    train_set = set(train_names)
    test_set = set(test_names)
    pred_set = set(image_order)
    leaked = sorted(pred_set & test_set)
    missing_train = sorted(train_set - pred_set)
    extra_non_train = sorted(pred_set - train_set)
    if leaked or missing_train or extra_non_train:
        raise ValueError(
            "VGGT dense initializer provenance failed: "
            f"leaked_test={leaked[:5]} missing_train={missing_train[:5]} "
            f"extra_non_train={extra_non_train[:5]}"
        )

    data = np.load(pred_path)
    points3d = data["points3d"]
    conf = data["points_conf"] if "points_conf" in data.files else np.ones(points3d.shape[:-1], dtype=np.float32)
    if points3d.ndim != 4 or points3d.shape[-1] != 3:
        raise ValueError(f"Unexpected points3d shape: {points3d.shape}")
    if conf.shape[:3] != points3d.shape[:3]:
        raise ValueError(f"points_conf shape {conf.shape} does not match points3d {points3d.shape}")
    input_arrays_finite = bool(np.isfinite(points3d).all() and np.isfinite(conf).all())

    scale = float(align_meta["scale"])
    R = np.asarray(align_meta["rotation"], dtype=np.float64)
    t = np.asarray(align_meta["translation"], dtype=np.float64)

    from PIL import Image

    xyz_chunks: list[np.ndarray] = []
    rgb_chunks: list[np.ndarray] = []
    conf_chunks: list[np.ndarray] = []
    h, w = points3d.shape[1], points3d.shape[2]
    image_dir = vggt_dataset_dir / "images"
    for idx, name in enumerate(image_order):
        pts = points3d[idx, ::sample_stride, ::sample_stride, :].reshape(-1, 3)
        cf = conf[idx, ::sample_stride, ::sample_stride].reshape(-1)
        image_path = image_dir / name
        if not image_path.exists():
            raise FileNotFoundError(f"VGGT image missing for dense init colors: {image_path}")
        img = Image.open(image_path).convert("RGB").resize((w, h), Image.BILINEAR)
        rgb = np.asarray(img, dtype=np.uint8)[::sample_stride, ::sample_stride, :].reshape(-1, 3)
        valid = np.isfinite(pts).all(axis=1) & np.isfinite(cf) & (cf > 0)
        if np.any(valid):
            xyz_chunks.append(pts[valid].astype(np.float64, copy=False))
            rgb_chunks.append(rgb[valid])
            conf_chunks.append(cf[valid].astype(np.float32, copy=False))
    if not xyz_chunks:
        raise ValueError("No valid VGGT dense initializer points")

    xyz_init = np.concatenate(xyz_chunks, axis=0)
    rgb = np.concatenate(rgb_chunks, axis=0)
    point_conf = np.concatenate(conf_chunks, axis=0)
    xyz_ref = scale * (xyz_init @ R.T) + t
    finite = np.isfinite(xyz_ref).all(axis=1)
    xyz_ref, rgb, point_conf = xyz_ref[finite], rgb[finite], point_conf[finite]
    if xyz_ref.shape[0] < 100:
        raise ValueError(f"Too few finite VGGT dense initializer points: {xyz_ref.shape[0]}")

    center = np.median(xyz_ref, axis=0)
    dist = np.linalg.norm(xyz_ref - center[None, :], axis=1)
    radius_limit = float(np.quantile(dist, 0.995))
    inlier = dist <= radius_limit
    xyz_ref, rgb, point_conf = xyz_ref[inlier], rgb[inlier], point_conf[inlier]
    if xyz_ref.shape[0] > max_points:
        keep_idx = np.argpartition(point_conf, -max_points)[-max_points:]
        xyz_ref, rgb, point_conf = xyz_ref[keep_idx], rgb[keep_idx], point_conf[keep_idx]

    if str(gaussian_repo) not in sys.path:
        sys.path.insert(0, str(gaussian_repo))
    from scene.dataset_readers import storePly

    target_ply.parent.mkdir(parents=True, exist_ok=True)
    storePly(str(target_ply), xyz_ref.astype(np.float32), rgb.astype(np.uint8))
    out_hash = _sha256_file(target_ply)
    ply_vertices = _count_ply_vertices(target_ply)
    result = {
        "code": 0,
        "source": "train_only_vggt_points3d_denseinit",
        "alignment_source": "train_camera_sim3",
        "predictions": str(pred_path),
        "predictions_sha256": _sha256_file(pred_path),
        "source_prediction_hash": _sha256_file(pred_path),
        "metadata": str(meta_path),
        "train_images": image_order,
        "train_image_count": len(image_order),
        "train_images_sha256": _names_sha256(image_order),
        "train_image_names_hash": _names_sha256(image_order),
        "test_images_sha256": _names_sha256(test_names),
        "leaked_test_images": [],
        "points3d_shape": [int(v) for v in points3d.shape],
        "points_conf_shape": [int(v) for v in conf.shape],
        "input_arrays_finite": input_arrays_finite,
        "sample_stride": sample_stride,
        "max_points": max_points,
        "point_count": int(xyz_ref.shape[0]),
        "output_point_count": int(xyz_ref.shape[0]),
        "ply_vertex_count": ply_vertices,
        "radius_limit": radius_limit,
        "output_ply": str(target_ply),
        "output_ply_sha256": out_hash,
        "output_ply_hash": out_hash,
    }
    (target_ply.parent / "points3D_denseinit_provenance.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    return result


def _load_run_metadata(run_root: Path) -> dict:
    try:
        with (run_root / "run_metadata.json").open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def existing_point_conf_result_valid(run_root: Path, condition: str) -> bool:
    """Return whether an existing metric is safe to reuse for confidence methods."""
    if not condition_uses_point_conf(condition):
        return True

    meta = _load_run_metadata(run_root)
    stages = meta.get("stages")
    if not isinstance(stages, dict):
        return False
    sidecar = stages.get("confidence_sidecar")
    if not isinstance(sidecar, dict) or sidecar.get("error"):
        return False
    conf_value = sidecar.get("points3D_conf")
    if not conf_value:
        return False
    conf_path = Path(str(conf_value))
    if not conf_path.exists():
        conf_path = run_root / "dataset" / "sparse" / "0" / "points3D_conf.npy"
    if not conf_path.exists():
        return False
    try:
        conf_values = np.load(conf_path)
    except Exception:
        return False
    expected_points = sidecar.get("points")
    if expected_points is None:
        expected_points = _count_ply_vertices(run_root / "dataset" / "sparse" / "0" / "points3D.ply")
    if expected_points is None or int(expected_points) <= 0 or conf_values.shape[0] != int(expected_points):
        return False
    if not np.isfinite(conf_values).all():
        return False

    if "dd_dafe_conf" in condition:
        train = stages.get("train")
        render = stages.get("render")
        metrics = stages.get("metrics")
        if not all(isinstance(stage, dict) and stage.get("code") == 0 for stage in (train, render, metrics)):
            return False
        cmd = train.get("cmd", [])
        cmd_text = " ".join(str(part) for part in cmd) if isinstance(cmd, list) else str(cmd)
        required_flags = (
            "--use_dd_drop",
            "--dd_depth_weight",
            "--dd_density_weight",
            "--dd_conf_weight",
            "--lambda_far",
            "--far_mask_quantile",
            "--use_pearson_depth",
            "--depths",
        )
        if not all(flag in cmd_text for flag in required_flags):
            return False
        stdout = str(train.get("stdout", ""))
        log_text = ""
        log_path = run_root / "model" / "logs" / "train.log"
        try:
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            pass
        evidence = stdout + "\n" + log_text
        if "Loaded confidence sidecar" not in evidence and "[CONFIDENCE-AWARE] Stored VGGT confidence" not in evidence:
            return False

    return True


def existing_bridge_ablation_result_valid(run_root: Path, condition: str) -> bool:
    if not condition_is_bridge_ablation(condition):
        return True

    meta = _load_run_metadata(run_root)
    stages = meta.get("stages")
    if not isinstance(stages, dict):
        return False

    vggt_cfg_path = run_root / "pipeline" / "vggt_config.yaml"
    try:
        vggt_cfg = yaml.safe_load(vggt_cfg_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return False
    actual_disable = bool((vggt_cfg.get("vggt") or {}).get("disable_median_norm", False))
    if actual_disable != condition_disables_bridge_median_norm(condition, False):
        return False

    train = stages.get("train")
    if not isinstance(train, dict) or train.get("code") != 0:
        return False
    cmd = train.get("cmd", [])
    cmd_parts = [str(part) for part in cmd] if isinstance(cmd, list) else str(cmd).split()
    cmd_text = " ".join(cmd_parts)

    forbidden_flags = (
        "--depths",
        "--use_pearson_depth",
        "--drop_prob_init",
        "--drop_prob_peak",
        "--drop_prob_peak_iter",
        "--drop_prob_end_iter",
        "--refine_poses",
        "--use_pseudo_views",
        "--train_test_exp",
        "--use_dd_drop",
        "--lambda_far",
        "--far_mask_quantile",
        "--use_mcmc",
        "--wavelet_weight",
        "--use_xview",
        "--use_photo_conf",
        "--use_depth_conf",
    )
    if any(flag in cmd_parts or flag in cmd_text for flag in forbidden_flags):
        return False

    conf_path = run_root / "dataset" / "sparse" / "0" / "points3D_conf.npy"
    filter_expected = condition in {"vggt_bridge_gate_10k_sh3", "vggt_bridge_full_10k_sh3"}
    if filter_expected:
        if "--conf_percentile_filter" not in cmd_parts:
            return False
        try:
            idx = cmd_parts.index("--conf_percentile_filter")
            if float(cmd_parts[idx + 1]) != 0.4:
                return False
        except (ValueError, IndexError):
            return False
        if not conf_path.exists():
            return False
        evidence = str(train.get("stdout", ""))
        try:
            evidence += "\n" + (run_root / "model" / "logs" / "train.log").read_text(
                encoding="utf-8", errors="replace"
            )
        except OSError:
            pass
        if "[INIT-FILTER]" not in evidence:
            return False
        if "Loaded confidence sidecar" not in evidence and "[CONFIDENCE-AWARE] Stored VGGT confidence" not in evidence:
            return False
    else:
        if "--conf_percentile_filter" in cmd_parts:
            return False
        if conf_path.exists():
            return False
        evidence = str(train.get("stdout", ""))
        try:
            evidence += "\n" + (run_root / "model" / "logs" / "train.log").read_text(
                encoding="utf-8", errors="replace"
            )
        except OSError:
            pass
        if "[VANILLA] Uniform opacity initialization" not in evidence:
            return False

    return True


def existing_strict_trainonly_result_valid(run_root: Path, condition: str) -> bool:
    meta = _load_run_metadata(run_root)
    protocol_mode = str(meta.get("protocol_mode") or meta.get("mode") or "")
    init_protocol = str(meta.get("init_protocol") or "")
    trainonly_signature = (
        isinstance(meta.get("train_images"), list)
        and isinstance(meta.get("test_images"), list)
        and bool(meta.get("reference_model"))
        and meta.get("train_subset") is None
    )
    if not (
        protocol_mode == "sparse_fixed_trainonly_init_ref"
        or init_protocol == "train_only_vggt_init_reference_aligned"
        or trainonly_signature
    ):
        return False
    stages = meta.get("stages")
    if not isinstance(stages, dict):
        return False
    for stage_name in ("train", "render", "metrics"):
        stage = stages.get(stage_name)
        if not isinstance(stage, dict) or stage.get("code") != 0:
            return False
    if "pose_injection" in meta or "pose_noise" in meta:
        return False
    if isinstance(stages, dict) and (
        "pose_injection" in stages
        or "pose_noise" in stages
        or "colmap_refpose" in stages
    ):
        return False
    if (run_root / "model" / "cfg_args").exists():
        try:
            cfg_text = (run_root / "model" / "cfg_args").read_text(encoding="utf-8", errors="replace")
        except OSError:
            cfg_text = ""
        if "train_test_exp=True" in cfg_text:
            return False
    if not existing_point_conf_result_valid(run_root, condition):
        return False
    if not existing_bridge_ablation_result_valid(run_root, condition):
        return False
    if condition.startswith("mast3r_sfm"):
        mast3r_stage = stages.get("mast3r_sfm")
        if not isinstance(mast3r_stage, dict) or mast3r_stage.get("code") != 0:
            return False
        if mast3r_stage.get("backend") != "mast3r_sfm":
            return False
        if mast3r_stage.get("no_reference_model_used") is not True:
            return False
        if mast3r_stage.get("leaked_test_images") not in ([], None):
            return False
        if mast3r_stage.get("input_image_count") != meta.get("n_train"):
            return False
        if mast3r_stage.get("model_image_count") != meta.get("n_train"):
            return False
    return True


def archive_stale_metrics(metrics_path: Path, reason: str) -> Path:
    stale_path = metrics_path.with_name(f"metrics.stale_{reason}_{int(time.time())}.json")
    metrics_path.rename(stale_path)
    return stale_path


def run_training_and_eval(
    dataset_dir: Path,
    model_dir: Path,
    gaussian_repo: Path,
    python_exe: Path,
    gpu_index: int,
    resolution: Optional[int],
    use_confidence: bool,
    llffhold: Optional[int] = None,
    use_pseudo_views: bool = False,
    refine_poses: bool = False,
    pose_params: Optional[dict] = None,
    depths_dir: Optional[str] = None,
    iterations: Optional[int] = None,
    gs_args: Optional[dict] = None,
    extra_args: Optional[dict] = None,
) -> tuple[dict, Optional[dict]]:
    """Train + render + metrics with monitoring. Returns stage metadata and metrics dict."""
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", f"{Path.cwd()}:{env.get('PYTHONPATH', '')}")
    model_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = model_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Hide confidence sidecar for vanilla runs
    conf_file = dataset_dir / "sparse" / "0" / "points3D_conf.npy"
    conf_backup = conf_file.with_suffix(".npy.backup")
    if not use_confidence and conf_file.exists():
        conf_file.rename(conf_backup)

    train_cmd = build_train_command(
        python_exe,
        gaussian_repo,
        dataset_dir,
        model_dir,
        resolution,
        llffhold=llffhold,
        use_pseudo_views=use_pseudo_views,
        refine_poses=refine_poses,
        pose_params=pose_params,
        depths_dir=depths_dir,
        iterations=iterations,
        gs_args=gs_args,
        extra_args=extra_args,
    )
    train_meta = run_monitored_command(
        train_cmd,
        cwd=gaussian_repo,
        env=env,
        log_path=logs_dir / "train.log",
        stage_name="train",
        gpu_index=gpu_index,
    )

    # Restore sidecar
    if conf_backup.exists():
        conf_backup.rename(conf_file)

    stages = {"train": train_meta}
    if train_meta["code"] != 0:
        return stages, None

    gs = gs_args or PAPER_GS_ARGS
    render_iteration = iterations if iterations else gs["iterations"]
    render_cmd = [
        str(python_exe),
        str(gaussian_repo / "render.py"),
        "--model_path",
        str(model_dir),
        "--iteration",
        str(render_iteration),
        "--skip_train",
    ]
    render_meta = run_monitored_command(
        render_cmd,
        cwd=gaussian_repo,
        env=env,
        log_path=logs_dir / "render.log",
        stage_name="render",
        gpu_index=gpu_index,
    )
    stages["render"] = render_meta
    if render_meta["code"] != 0:
        return stages, None

    metrics_cmd = [
        str(python_exe),
        str(gaussian_repo / "metrics.py"),
        "-m",
        str(model_dir),
    ]
    metrics_meta = run_monitored_command(
        metrics_cmd,
        cwd=gaussian_repo,
        env=env,
        log_path=logs_dir / "metrics.log",
        stage_name="metrics",
        gpu_index=gpu_index,
    )
    stages["metrics"] = metrics_meta

    results_json = model_dir / "results.json"
    metrics_data = None
    if results_json.exists():
        with results_json.open("r", encoding="utf-8") as fh:
            try:
                raw = json.load(fh)
                first_key = next(iter(raw.keys()))
                metrics_data = raw[first_key] if isinstance(raw, dict) else raw
            except Exception:
                metrics_data = None
    return stages, metrics_data


def resolve_images_dir(scene_root: Path, preferred: Optional[str]) -> Optional[Path]:
    candidates = []
    if preferred:
        candidates.append(scene_root / preferred)
    candidates.extend([scene_root / "images", scene_root / "images_4", scene_root / "images_2"])
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def run_vggt_pipeline_for_scene(
    stage_scene_dir: Path,
    output_root: Path,
    vggt_cfg: dict,
    python_exe: Path,
    gpu_index: int,
    disable_median_norm: bool = False,
) -> tuple[Optional[Path], dict]:
    """Run VGGT export + COLMAP-format conversion via run_pipeline.py."""
    output_root.mkdir(parents=True, exist_ok=True)
    cfg_path = output_root / "vggt_config.yaml"
    cfg = {
        "scene": stage_scene_dir.name,
        "data_root": str(stage_scene_dir.parent),
        "output_root": str(output_root),
        "pipeline": "vggt_3dgs",
        "vggt": {
            "repo": vggt_cfg.get("repo", "./repos/vggt"),
            "checkpoint": vggt_cfg.get("checkpoint", "facebook/VGGT-1B"),
            "batch_size": int(vggt_cfg.get("batch_size", 6)),
            "max_images": None,
            "point_subsample": int(vggt_cfg.get("point_subsample", 20)),
            "force_cpu": bool(vggt_cfg.get("force_cpu", False)),
            "disable_median_norm": bool(disable_median_norm),
        },
        "training": {"skip": True},
    }
    with cfg_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)

    env = os.environ.copy()
    vggt_repo = Path(vggt_cfg.get("repo", "./repos/vggt")).resolve()
    pythonpath = [str(Path.cwd()), str(vggt_repo)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    meta = run_monitored_command(
        [str(python_exe), "scripts/run_pipeline.py", "--config", str(cfg_path)],
        cwd=Path.cwd(),
        env=env,
        log_path=output_root / "vggt.log",
        stage_name="vggt_export",
        gpu_index=gpu_index,
    )
    dataset_dir = output_root / stage_scene_dir.name / "dataset"
    if meta.get("code", 1) != 0 or not dataset_dir.exists():
        return None, meta
    return dataset_dir, meta


def build_vggt_export_proof(
    *,
    init_dir: Path,
    dataset_dir: Path,
    train_names: list[str],
    test_names: list[str],
    meta: dict,
) -> dict:
    input_dir = init_dir / "images"
    input_names = [
        p.name for p in _sort_image_paths_like_instantsplat(list_image_files(input_dir))
    ]
    leaked_test = sorted(set(input_names) & set(test_names))
    missing_train = sorted(set(train_names) - set(input_names))
    extra_non_train = sorted(set(input_names) - set(train_names))
    if leaked_test or missing_train or extra_non_train:
        raise ValueError(
            "VGGT export input proof failed: "
            f"leaked_test={leaked_test[:5]} missing_train={missing_train[:5]} "
            f"extra_non_train={extra_non_train[:5]}"
        )

    proof = dict(meta)
    proof.update(
        {
            "input_images_dir": str(input_dir.resolve()),
            "input_image_count": len(input_names),
            "input_images": input_names,
            "input_images_sha256": _names_sha256(input_names),
            "expected_train_image_count": len(train_names),
            "expected_train_images_sha256": _names_sha256(train_names),
            "test_images_sha256": _names_sha256(test_names),
            "leaked_test_images": leaked_test,
            "output_dataset_dir": str(dataset_dir.resolve()),
        }
    )

    metadata_path = dataset_dir.parent / "vggt" / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"VGGT metadata missing for proof: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        vggt_metadata = json.load(handle)
    metadata_images = [str(name) for name in vggt_metadata.get("images", [])]
    if metadata_images != input_names:
        raise ValueError(f"VGGT metadata images do not match staged train inputs: {metadata_images} != {input_names}")
    proof.update(
        {
            "metadata": str(metadata_path),
            "metadata_images": metadata_images,
            "metadata_images_sha256": _names_sha256(metadata_images),
        }
    )

    depths_dir = dataset_dir / "depths"
    depth_files = sorted(p.name for p in depths_dir.glob("*.png")) if depths_dir.exists() else []
    depth_conf_dir = dataset_dir / "sparse" / "0" / "depth_conf"
    depth_conf_files = sorted(p.name for p in depth_conf_dir.glob("*.npy")) if depth_conf_dir.exists() else []
    expected_depth_png = sorted(f"{Path(name).stem}.png" for name in train_names)
    expected_depth_conf = sorted(f"{Path(name).stem}.npy" for name in train_names)
    proof.update(
        {
            "depths_dir": str(depths_dir) if depths_dir.exists() else None,
            "depth_map_count": len(depth_files),
            "depth_map_files_sha256": _names_sha256(depth_files),
            "missing_train_depth_maps": sorted(set(expected_depth_png) - set(depth_files)),
            "extra_depth_maps": sorted(set(depth_files) - set(expected_depth_png)),
            "depth_conf_dir": str(depth_conf_dir) if depth_conf_dir.exists() else None,
            "depth_conf_map_count": len(depth_conf_files),
            "depth_conf_files_sha256": _names_sha256(depth_conf_files),
            "missing_train_depth_conf_maps": sorted(set(expected_depth_conf) - set(depth_conf_files)),
            "extra_depth_conf_maps": sorted(set(depth_conf_files) - set(expected_depth_conf)),
            "points3D_conf": str(dataset_dir / "sparse" / "0" / "points3D_conf.npy")
            if (dataset_dir / "sparse" / "0" / "points3D_conf.npy").exists()
            else None,
        }
    )
    return proof


def run_mast3r_sfm_for_scene(
    *,
    stage_scene_dir: Path,
    output_root: Path,
    mast3r_cfg: dict,
    gaussian_repo: Path,
    python_exe: Path,
    gpu_index: int,
) -> tuple[Optional[Path], dict]:
    """Run train-only MASt3R-SfM and export a COLMAP model."""
    output_dir = output_root / "mast3r_sfm"
    output_dir.mkdir(parents=True, exist_ok=True)
    script_path = Path(__file__).resolve().with_name("mast3r_sfm_export.py")
    mast3r_repo = Path(mast3r_cfg.get("repo", "./repos/G4Splat/mast3r")).resolve()
    weights_path = Path(
        mast3r_cfg.get(
            "weights_path",
            "./repos/G4Splat/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
        )
    ).resolve()

    cmd = [
        str(python_exe),
        str(script_path),
        "--images_dir",
        str(stage_scene_dir / "images"),
        "--output_dir",
        str(output_dir),
        "--mast3r_repo",
        str(mast3r_repo),
        "--gaussian_repo",
        str(gaussian_repo),
        "--weights_path",
        str(weights_path),
        "--image_size",
        str(int(mast3r_cfg.get("image_size", 512))),
        "--scene_graph",
        str(mast3r_cfg.get("scene_graph", "complete")),
        "--n_coarse_iterations",
        str(int(mast3r_cfg.get("n_coarse_iterations", 300))),
        "--n_refinement_iterations",
        str(int(mast3r_cfg.get("n_refinement_iterations", 300))),
        "--matching_conf_thr",
        str(float(mast3r_cfg.get("matching_conf_thr", 0.0))),
        "--output_conf_thr",
        str(float(mast3r_cfg.get("output_conf_thr", 0.1))),
        "--point_stride",
        str(int(mast3r_cfg.get("point_stride", 4))),
        "--gpu",
        str(gpu_index),
    ]
    if bool(mast3r_cfg.get("shared_intrinsics", False)):
        cmd.append("--shared_intrinsics")

    env = os.environ.copy()
    pythonpath = [
        str(Path.cwd()),
        str(script_path.parent),
        str(mast3r_repo.parent),
        str(mast3r_repo),
        str(mast3r_repo / "dust3r"),
        str(gaussian_repo),
    ]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)

    meta = run_monitored_command(
        cmd,
        cwd=Path.cwd(),
        env=env,
        log_path=output_dir / "mast3r_sfm.log",
        stage_name="mast3r_sfm",
        gpu_index=gpu_index,
    )
    model_dir = output_dir / "sparse" / "0"
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists():
        try:
            with metadata_path.open("r", encoding="utf-8") as handle:
                export_meta = json.load(handle)
            meta.update(
                {
                    "export_metadata": str(metadata_path),
                    "input_image_count": export_meta.get("input_image_count"),
                    "input_images": export_meta.get("input_images"),
                    "input_images_sha256": export_meta.get("input_images_sha256"),
                    "model_image_count": export_meta.get("model_image_count"),
                    "model_image_names": export_meta.get("model_image_names"),
                    "point_count": export_meta.get("point_count"),
                    "scene_graph": export_meta.get("scene_graph"),
                    "no_reference_model_used": export_meta.get("no_reference_model_used"),
                }
            )
        except (OSError, json.JSONDecodeError) as exc:
            meta["export_metadata_error"] = str(exc)
    if meta.get("code", 1) != 0 or not model_dir.exists():
        return None, meta
    return model_dir, meta


def build_mast3r_sfm_proof(
    *,
    init_dir: Path,
    model_dir: Path,
    train_names: list[str],
    test_names: list[str],
    meta: dict,
) -> dict:
    input_dir = init_dir / "images"
    input_names = [
        p.name for p in _sort_image_paths_like_instantsplat(list_image_files(input_dir))
    ]
    leaked_test = sorted(set(input_names) & set(test_names))
    missing_train = sorted(set(train_names) - set(input_names))
    extra_non_train = sorted(set(input_names) - set(train_names))
    if leaked_test or missing_train or extra_non_train:
        raise ValueError(
            "MASt3R-SfM input proof failed: "
            f"leaked_test={leaked_test[:5]} missing_train={missing_train[:5]} "
            f"extra_non_train={extra_non_train[:5]}"
        )

    metadata_path = model_dir.parent.parent / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"MASt3R-SfM metadata missing for proof: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        export_meta = json.load(handle)

    metadata_input_names = [Path(str(name)).name for name in export_meta.get("input_images", [])]
    if metadata_input_names != input_names:
        raise ValueError(
            "MASt3R-SfM metadata inputs do not match staged train inputs: "
            f"{metadata_input_names} != {input_names}"
        )
    model_names = [Path(str(name)).name for name in export_meta.get("model_image_names", [])]
    if model_names != input_names:
        raise ValueError(
            "MASt3R-SfM model image names do not match staged train inputs: "
            f"{model_names} != {input_names}"
        )
    if export_meta.get("no_reference_model_used") is not True:
        raise ValueError("MASt3R-SfM metadata must prove no reference model was used")

    proof = dict(meta)
    proof.update(
        {
            "backend": "mast3r_sfm",
            "metadata": str(metadata_path),
            "input_images_dir": str(input_dir.resolve()),
            "input_image_count": len(input_names),
            "input_images": input_names,
            "input_images_sha256": _names_sha256(input_names),
            "expected_train_image_count": len(train_names),
            "expected_train_images_sha256": _names_sha256(train_names),
            "test_images_sha256": _names_sha256(test_names),
            "leaked_test_images": leaked_test,
            "missing_train_images": missing_train,
            "extra_non_train_images": extra_non_train,
            "output_model_dir": str(model_dir.resolve()),
            "model_image_names": model_names,
            "model_image_count": len(model_names),
            "model_images_sha256": _names_sha256(model_names),
            "no_reference_model_used": True,
            "point_count": export_meta.get("point_count"),
            "scene_graph": export_meta.get("scene_graph"),
            "image_size": export_meta.get("image_size"),
            "n_coarse_iterations": export_meta.get("n_coarse_iterations"),
            "n_refinement_iterations": export_meta.get("n_refinement_iterations"),
            "matching_conf_thr": export_meta.get("matching_conf_thr"),
            "output_conf_thr": export_meta.get("output_conf_thr"),
            "point_stride": export_meta.get("point_stride"),
            "shared_intrinsics": export_meta.get("shared_intrinsics"),
        }
    )
    return proof


def run_paper_eval(args: argparse.Namespace) -> None:
    """Paper-ready evaluation: full-set + sparse consecutive K for each scene/dataset."""
    if not args.paper_config.exists():
        raise FileNotFoundError(f"Paper config not found: {args.paper_config}")

    cfg = yaml.safe_load(args.paper_config.read_text())
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    seed = int(cfg.get("seed", 0))
    set_global_seeds(seed)
    gpu_info = collect_gpu_info()
    sparse_views = cfg.get("sparse_views", [3, 6, 12, 24])
    output_root = Path(cfg.get("output_root", args.output_root)).resolve()
    work_root = output_root / "paper_eval_work"
    results_root = output_root / "paper_eval"
    gaussian_repo = Path(cfg.get("gaussian_repo", "./repos/gaussian-splatting")).resolve()
    python_exe = Path(sys.executable)
    resolution_override = cfg.get("resolution")
    colmap_cfg = cfg.get("colmap", {}) or {}
    vggt_cfg = cfg.get("vggt", {}) or {}
    mast3r_cfg = cfg.get("mast3r", {}) or {}
    if args.vggt_force_cpu:
        vggt_cfg = dict(vggt_cfg)
        vggt_cfg["force_cpu"] = True
    if args.vggt_force_cpu:
        vggt_cfg = dict(vggt_cfg)
        vggt_cfg["force_cpu"] = True

    datasets_cfg = cfg.get("datasets", {})
    dataset_entries = []
    if "mipnerf360" in datasets_cfg:
        d = datasets_cfg["mipnerf360"] or {}
        root = Path(d.get("root", "")) if d.get("root") else None
        outdoor_subdir = d.get("images_subdir", {}).get("outdoor", "images_4")
        indoor_subdir = d.get("images_subdir", {}).get("indoor", "images_2")
        for scene in d.get("outdoor_scenes", []):
            dataset_entries.append(("mipnerf360", scene, outdoor_subdir, root))
        for scene in d.get("indoor_scenes", []):
            dataset_entries.append(("mipnerf360", scene, indoor_subdir, root))
    if "tanksandtemples" in datasets_cfg:
        d = datasets_cfg["tanksandtemples"] or {}
        root = Path(d.get("root", "")) if d.get("root") else None
        subdir = d.get("images_subdir", "images")
        for scene in d.get("scenes", []):
            dataset_entries.append(("tanksandtemples", scene, subdir, root))
    if "deepblending" in datasets_cfg:
        d = datasets_cfg["deepblending"] or {}
        root = Path(d.get("root", "")) if d.get("root") else None
        subdir = d.get("images_subdir", "images")
        for scene in d.get("scenes", []):
            dataset_entries.append(("deepblending", scene, subdir, root))

    results_rows: list[dict] = []

    for dataset_name, scene, img_subdir, root in dataset_entries:
        if not root or not Path(root).exists():
            print(f"[SKIP] Dataset root missing for {dataset_name} ({root}), skipping scene {scene}")
            continue
        scene_root = Path(root) / scene
        if not scene_root.exists():
            print(f"[SKIP] Scene folder missing: {scene_root}")
            continue
        source_images_dir = resolve_images_dir(scene_root, img_subdir)
        if not source_images_dir:
            print(f"[SKIP] No images found for {scene_root} (preferred subdir {img_subdir})")
            continue

        settings = [("full", None)] + [(f"consecutive_{k}", k) for k in sparse_views]
        for setting_label, k_val in settings:
            print(f"\n=== {dataset_name} / {scene} / {setting_label} ===")
            stage_scene_dir = work_root / dataset_name / scene / setting_label
            images_dir = stage_scene_dir / "images"
            try:
                selected_images = prepare_image_subset(source_images_dir, images_dir, k_val)
            except Exception as exc:  # pragma: no cover
                print(f"[SKIP] Failed to stage images for {scene} ({setting_label}): {exc}")
                continue

            # Add alias to match dataset convention
            if img_subdir and img_subdir != "images":
                alias_dir = stage_scene_dir / img_subdir
                ensure_dir_clean(alias_dir)
                try:
                    os.symlink(images_dir, alias_dir, target_is_directory=True)
                except Exception:
                    shutil.copytree(images_dir, alias_dir, dirs_exist_ok=True)

            # conditions: colmap, vggt_vanilla, vggt_confidence
            for condition in ("colmap", "vggt_vanilla", "vggt_confidence"):
                run_root = results_root / dataset_name / scene / setting_label / condition
                run_root.mkdir(parents=True, exist_ok=True)
                run_meta = {
                    "dataset": dataset_name,
                    "scene": scene,
                    "setting": setting_label,
                    "condition": condition,
                    "seed": seed,
                    "images_source": str(source_images_dir),
                    "selected_images": [p.name for p in selected_images] if k_val else "all",
                    "gpu": gpu_info,
                }

                stages_meta = {}
                metrics_data = None

                if condition == "colmap":
                    print("[STAGE] COLMAP baseline")
                    colmap_meta = run_colmap_paper(
                        stage_scene_dir,
                        num_threads=int(colmap_cfg.get("threads", 8)),
                        use_gpu=not colmap_cfg.get("cpu", False),
                        log_dir=run_root / "logs",
                        gpu_index=args.gpu_index,
                        colmap_bin=str(args.colmap_bin),
                    )
                    stages_meta["colmap"] = colmap_meta
                    if not colmap_meta.get("success"):
                        print(f"[SKIP] COLMAP failed for {scene} ({setting_label})")
                        run_meta["stages"] = stages_meta
                        (run_root / "run_metadata.json").write_text(json.dumps(run_meta, indent=2))
                        continue
                    dataset_dir = stage_scene_dir
                else:
                    print("[STAGE] VGGT export + COLMAP-format conversion")
                    dataset_dir, vggt_meta = run_vggt_pipeline_for_scene(
                        stage_scene_dir=stage_scene_dir,
                        output_root=run_root / "pipeline",
                        vggt_cfg=vggt_cfg,
                        python_exe=python_exe,
                        gpu_index=args.gpu_index,
                    )
                    stages_meta["vggt_export"] = vggt_meta
                    if not dataset_dir:
                        print(f"[SKIP] VGGT pipeline did not produce dataset for {scene} ({setting_label})")
                        run_meta["stages"] = stages_meta
                        (run_root / "run_metadata.json").write_text(json.dumps(run_meta, indent=2))
                        continue

                use_conf = condition_uses_point_conf(condition)
                model_dir = run_root / "model"
                train_meta, metrics_data = run_training_and_eval(
                    dataset_dir=dataset_dir,
                    model_dir=model_dir,
                    gaussian_repo=gaussian_repo,
                    python_exe=python_exe,
                    gpu_index=args.gpu_index,
                    resolution=resolution_override,
                    use_confidence=use_conf,
                )
                stages_meta.update(train_meta)

                run_meta["stages"] = stages_meta
                run_meta["metrics"] = metrics_data
                run_meta_path = run_root / "run_metadata.json"
                run_meta_path.write_text(json.dumps(run_meta, indent=2))

                if metrics_data:
                    metrics_out = run_root / "metrics.json"
                    metrics_out.write_text(json.dumps(metrics_data, indent=2))
                    results_rows.append(
                        {
                            "dataset": dataset_name,
                            "scene": scene,
                            "setting": setting_label,
                            "condition": condition,
                            "PSNR": metrics_data.get("PSNR"),
                            "SSIM": metrics_data.get("SSIM"),
                            "LPIPS": metrics_data.get("LPIPS"),
                            "train_elapsed_s": stages_meta.get("train", {}).get("elapsed_s"),
                            "render_elapsed_s": stages_meta.get("render", {}).get("elapsed_s"),
                            "metrics_elapsed_s": stages_meta.get("metrics", {}).get("elapsed_s"),
                            "colmap_elapsed_s": stages_meta.get("colmap", {}).get("elapsed_s"),
                            "vggt_elapsed_s": stages_meta.get("vggt_export", {}).get("elapsed_s"),
                        }
                    )

    if results_rows:
        csv_path = results_root / "metrics_all.csv"
        fields = [
            "dataset",
            "scene",
            "setting",
            "condition",
            "PSNR",
            "SSIM",
            "LPIPS",
            "train_elapsed_s",
            "render_elapsed_s",
            "metrics_elapsed_s",
            "colmap_elapsed_s",
            "vggt_elapsed_s",
        ]
        write_csv(csv_path, results_rows, fields)
        print(f"\n[SUMMARY] Aggregated metrics written to {csv_path}")
    else:
        print("\n[SUMMARY] No runs produced metrics.")


def resolve_mipnerf360_images_subdir(cfg: dict, scene: str) -> str:
    d = cfg.get("datasets", {}).get("mipnerf360", {}) or {}
    indoor = set(d.get("indoor_scenes", []) or [])
    outdoor = set(d.get("outdoor_scenes", []) or [])
    subdirs = d.get("images_subdir", {}) or {}
    if scene in indoor:
        return subdirs.get("indoor", "images_2")
    if scene in outdoor:
        return subdirs.get("outdoor", "images_4")
    return "images"


def run_sparse_fixed(args: argparse.Namespace) -> None:
    """Sparse train / fixed test protocol (NeRF-style)."""
    if not args.paper_config.exists():
        raise FileNotFoundError(f"Paper config not found: {args.paper_config}")

    cfg = yaml.safe_load(args.paper_config.read_text())
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    seed = int(cfg.get("seed", 0))
    set_global_seeds(seed)
    gpu_info = collect_gpu_info()

    output_root = Path(cfg.get("output_root", args.output_root)).resolve()
    work_root = output_root / "paper_eval_work_fixed"
    results_root = output_root / "paper_eval"
    gaussian_repo = Path(cfg.get("gaussian_repo", "./repos/gaussian-splatting")).resolve()
    python_exe = Path(sys.executable)
    resolution_override = cfg.get("resolution")
    colmap_cfg = cfg.get("colmap", {}) or {}
    vggt_cfg = cfg.get("vggt", {}) or {}
    mast3r_cfg = cfg.get("mast3r", {}) or {}
    if args.vggt_force_cpu:
        vggt_cfg = dict(vggt_cfg)
        vggt_cfg["force_cpu"] = True

    d = cfg.get("datasets", {}).get("mipnerf360", {}) or {}
    root = Path(d.get("root", "")) if d.get("root") else None
    if not root or not root.exists():
        raise FileNotFoundError(f"mipnerf360 root missing in config: {root}")

    scene = args.scene
    scene_root = root / scene
    if not scene_root.exists():
        raise FileNotFoundError(f"Scene folder missing: {scene_root}")

    img_subdir = args.images_subdir or resolve_mipnerf360_images_subdir(cfg, scene)
    source_images_dir = resolve_images_dir(scene_root, img_subdir)
    if not source_images_dir:
        raise FileNotFoundError(f"No images found for {scene_root} (preferred subdir {img_subdir})")

    dataset_name = "mipnerf360"
    fields = [
        "dataset",
        "scene",
        "setting",
        "condition",
        "PSNR",
        "SSIM",
        "LPIPS",
        "train_elapsed_s",
        "render_elapsed_s",
        "metrics_elapsed_s",
        "colmap_elapsed_s",
        "vggt_elapsed_s",
    ]
    metrics_csv = results_root / "metrics_all.csv"

    for n_train in args.n_train:
        setting_label = f"fixed_test_n{n_train}"
        print(f"\n=== {dataset_name} / {scene} / {setting_label} ===")
        stage_scene_dir = work_root / dataset_name / scene / setting_label
        try:
            staged = stage_sparse_fixed_test_dataset(
                source_dir=source_images_dir,
                stage_dir=stage_scene_dir,
                holdout=args.holdout,
                n_train=n_train,
            )
        except Exception as exc:  # pragma: no cover
            print(f"[SKIP] Failed to stage images for {scene} ({setting_label}): {exc}")
            continue

        # Add alias to match dataset convention
        if img_subdir and img_subdir != "images":
            alias_dir = stage_scene_dir / img_subdir
            if alias_dir.exists() or alias_dir.is_symlink():
                if alias_dir.is_symlink() or alias_dir.is_file():
                    alias_dir.unlink()
                else:
                    shutil.rmtree(alias_dir)
            try:
                os.symlink(stage_scene_dir / "images", alias_dir, target_is_directory=True)
            except Exception:
                shutil.copytree(stage_scene_dir / "images", alias_dir, dirs_exist_ok=True)

        if args.conditions:
            conditions = list(args.conditions)
        else:
            conditions = ["vggt_vanilla", "vggt_confidence"]
            if not args.skip_colmap:
                conditions.insert(0, "colmap")
        if args.skip_colmap and "colmap" in conditions:
            conditions = [c for c in conditions if c != "colmap"]

        for condition in conditions:
            run_root = results_root / dataset_name / scene / setting_label / condition
            run_root.mkdir(parents=True, exist_ok=True)
            metrics_path = run_root / "metrics.json"
            if metrics_path.exists():
                if existing_strict_trainonly_result_valid(run_root, condition):
                    print(f"[SKIP] Metrics already exist for {scene} ({setting_label}) {condition}")
                    continue
                stale_path = archive_stale_metrics(metrics_path, "strict_protocol")
                print(
                    f"[RERUN] Existing metrics for {scene} ({setting_label}) {condition} "
                    f"lack required strict protocol/confidence proof; archived {stale_path.name}"
                )
            run_meta = {
                "mode": args.mode,
                "protocol_mode": "sparse_fixed",
                "init_protocol": "train_plus_test_vggt_init",
                "dataset": dataset_name,
                "scene": scene,
                "setting": setting_label,
                "condition": condition,
                "seed": seed,
                "holdout": args.holdout,
                "n_train": n_train,
                "images_source": str(source_images_dir),
                "test_images": [p.name for p in staged["test_images"]],
                "train_subset": [p.name for p in staged["train_subset"]],
                "gpu": gpu_info,
            }
            (run_root / "run_metadata.json").write_text(json.dumps(run_meta, indent=2))

            stages_meta = {}
            metrics_data = None

            if condition == "colmap":
                print("[STAGE] COLMAP baseline")
                colmap_meta = run_colmap_paper(
                    stage_scene_dir,
                    num_threads=int(colmap_cfg.get("threads", 8)),
                    use_gpu=not (colmap_cfg.get("cpu", False) or args.colmap_cpu),
                    log_dir=run_root / "logs",
                    gpu_index=args.gpu_index,
                    colmap_bin=str(args.colmap_bin),
                )
                stages_meta["colmap"] = colmap_meta
                if not colmap_meta.get("success"):
                    print(f"[SKIP] COLMAP failed for {scene} ({setting_label})")
                    run_meta["stages"] = stages_meta
                    (run_root / "run_metadata.json").write_text(json.dumps(run_meta, indent=2))
                    continue
                dataset_dir = stage_scene_dir
                test_txt = staged["test_txt"]
                target_test = dataset_dir / "sparse" / "0" / "test.txt"
                target_test.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copy2(test_txt, target_test)
                except shutil.SameFileError:
                    # Colmap uses the staged dataset directly; avoid copying onto itself.
                    pass
            else:
                print("[STAGE] VGGT export + COLMAP-format conversion")
                dataset_dir, vggt_meta = run_vggt_pipeline_for_scene(
                    stage_scene_dir=stage_scene_dir,
                    output_root=run_root / "pipeline",
                    vggt_cfg=vggt_cfg,
                    python_exe=python_exe,
                    gpu_index=args.gpu_index,
                )
                stages_meta["vggt_export"] = vggt_meta
                if not dataset_dir:
                    print(f"[SKIP] VGGT pipeline did not produce dataset for {scene} ({setting_label})")
                    run_meta["stages"] = stages_meta
                    (run_root / "run_metadata.json").write_text(json.dumps(run_meta, indent=2))
                    continue
                # Ensure fixed test list is available in VGGT dataset
                test_txt = staged["test_txt"]
                target_test = dataset_dir / "sparse" / "0" / "test.txt"
                target_test.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copy2(test_txt, target_test)
                except shutil.SameFileError:
                    pass

            use_conf = condition_uses_point_conf(condition)
            model_dir = run_root / "model"
            train_meta, metrics_data = run_training_and_eval(
                dataset_dir=dataset_dir,
                model_dir=model_dir,
                gaussian_repo=gaussian_repo,
                python_exe=python_exe,
                gpu_index=args.gpu_index,
                resolution=resolution_override,
                use_confidence=use_conf,
                llffhold=0,
            )
            stages_meta.update(train_meta)
            if "alignment" in stages_meta:
                run_meta["train_images_used"] = stages_meta["alignment"].get("train_used", train_names)
                run_meta["train_images_missing"] = stages_meta["alignment"].get("train_missing", [])

            run_meta["stages"] = stages_meta
            run_meta["metrics"] = metrics_data
            (run_root / "run_metadata.json").write_text(json.dumps(run_meta, indent=2))

            if metrics_data:
                metrics_out = run_root / "metrics.json"
                metrics_out.write_text(json.dumps(metrics_data, indent=2))
                row = {
                    "dataset": dataset_name,
                    "scene": scene,
                    "setting": setting_label,
                    "condition": condition,
                    "PSNR": metrics_data.get("PSNR"),
                    "SSIM": metrics_data.get("SSIM"),
                    "LPIPS": metrics_data.get("LPIPS"),
                    "train_elapsed_s": stages_meta.get("train", {}).get("elapsed_s"),
                    "render_elapsed_s": stages_meta.get("render", {}).get("elapsed_s"),
                    "metrics_elapsed_s": stages_meta.get("metrics", {}).get("elapsed_s"),
                    "colmap_elapsed_s": stages_meta.get("colmap", {}).get("elapsed_s"),
                    "vggt_elapsed_s": stages_meta.get("vggt_export", {}).get("elapsed_s"),
                }
                update_metrics_csv(metrics_csv, row, fields)


def run_sparse_fixed_trainonly_init_ref(args: argparse.Namespace) -> None:
    """Sparse train / fixed test with train-only init aligned to full reference model."""
    if not args.paper_config.exists():
        raise FileNotFoundError(f"Paper config not found: {args.paper_config}")

    cfg = yaml.safe_load(args.paper_config.read_text())
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    seed = int(cfg.get("seed", 0))
    set_global_seeds(seed)
    gpu_info = collect_gpu_info()

    output_root = Path(cfg.get("output_root", args.output_root)).resolve()
    work_root = output_root / "paper_eval_work_trainonly_ref"
    results_root = output_root / "paper_eval"
    gaussian_repo = Path(cfg.get("gaussian_repo", "./repos/gaussian-splatting")).resolve()
    python_exe = Path(sys.executable)
    resolution_override = cfg.get("resolution")
    colmap_cfg = cfg.get("colmap", {}) or {}
    vggt_cfg = cfg.get("vggt", {}) or {}
    mast3r_cfg = cfg.get("mast3r", {}) or {}
    if args.vggt_force_cpu:
        vggt_cfg = dict(vggt_cfg)
        vggt_cfg["force_cpu"] = True

    dataset_name = args.dataset
    d = cfg.get("datasets", {}).get(dataset_name, {}) or {}
    root = Path(d.get("root", "")) if d.get("root") else None
    if not root or not root.exists():
        raise FileNotFoundError(f"{dataset_name} root missing in config: {root}")

    scene = args.scene
    scene_root = root / scene
    if not scene_root.exists():
        raise FileNotFoundError(f"Scene folder missing: {scene_root}")

    if dataset_name == "mipnerf360":
        default_subdir = resolve_mipnerf360_images_subdir(cfg, scene)
    else:
        default_subdir = d.get("images_subdir", "images")
        if isinstance(default_subdir, dict):
            default_subdir = "images"
    img_subdir = args.images_subdir or default_subdir
    source_images_dir = resolve_images_dir(scene_root, img_subdir)
    if not source_images_dir:
        raise FileNotFoundError(f"No images found for {scene_root} (preferred subdir {img_subdir})")

    reference_model_dir = find_reference_colmap_model(scene_root)
    fields = [
        "dataset",
        "scene",
        "setting",
        "condition",
        "PSNR",
        "SSIM",
        "LPIPS",
        "train_elapsed_s",
        "render_elapsed_s",
        "metrics_elapsed_s",
        "colmap_elapsed_s",
        "vggt_elapsed_s",
        "vggt_ba_elapsed_s",
        "mast3r_elapsed_s",
        "pseudo_gen_elapsed_s",
    ]
    metrics_csv = results_root / "metrics_all.csv"

    for n_train in args.n_train:
        setting_label = f"fixed_test_n{n_train}"
        print(f"\n=== {dataset_name} / {scene} / {setting_label} (train-only init) ===")
        stage_scene_dir = work_root / dataset_name / scene / setting_label
        split_row = None
        split_source_proof = None
        active_source_images_dir = source_images_dir
        active_reference_model_dir = reference_model_dir
        camera_preprocess_meta = None
        try:
            if args.split_manifest:
                split_dataset = args.split_dataset or dataset_name
                split_scene = args.split_scene or scene
                split_row = load_explicit_split_manifest_row(
                    args.split_manifest,
                    dataset=split_dataset,
                    scene=split_scene,
                    n_train=n_train,
                )
                split_source_proof = validate_explicit_split_sources(
                    split_row=split_row,
                    scene_root=scene_root,
                    source_images_dir=source_images_dir,
                    reference_model_dir=reference_model_dir,
                )
                active_source_images_dir, active_reference_model_dir, camera_preprocess_meta = prepare_3dgs_camera_reference(
                    raw_images_dir=source_images_dir,
                    raw_reference_model_dir=reference_model_dir,
                    work_dir=stage_scene_dir / "camera_preprocess",
                    gaussian_repo=gaussian_repo,
                    colmap_bin=str(args.colmap_bin),
                    radial_policy=args.radial_camera_policy,
                    expected_sorted_names=split_row["sorted_images"],
                    log_dir=stage_scene_dir / "logs",
                    gpu_index=args.gpu_index,
                )
                staged = stage_explicit_trainonly_datasets(
                    source_dir=active_source_images_dir,
                    stage_root=stage_scene_dir,
                    train_image_names=split_row["train_images"],
                    test_image_names=split_row["test_images"],
                )
                print(
                    f"[STAGE] explicit split from {args.split_manifest}: "
                    f"{split_dataset}/{split_scene} n={n_train}"
                )
            else:
                active_source_images_dir, active_reference_model_dir, camera_preprocess_meta = prepare_3dgs_camera_reference(
                    raw_images_dir=source_images_dir,
                    raw_reference_model_dir=reference_model_dir,
                    work_dir=stage_scene_dir / "camera_preprocess",
                    gaussian_repo=gaussian_repo,
                    colmap_bin=str(args.colmap_bin),
                    radial_policy=args.radial_camera_policy,
                    expected_sorted_names=None,
                    log_dir=stage_scene_dir / "logs",
                    gpu_index=args.gpu_index,
                )
                staged = stage_sparse_fixed_trainonly_datasets(
                    source_dir=active_source_images_dir,
                    stage_root=stage_scene_dir,
                    holdout=args.holdout,
                    n_train=n_train,
                )
        except Exception as exc:  # pragma: no cover
            if args.split_manifest:
                raise RuntimeError(
                    f"Manifest-driven camera-ready staging failed for {scene} ({setting_label}); "
                    "aborting instead of skipping."
                ) from exc
            print(f"[SKIP] Failed to stage images for {scene} ({setting_label}): {exc}")
            continue

        train_names = [p.name for p in staged["train_subset"]]
        test_names = [p.name for p in staged["test_images"]]
        init_dir = staged["init_dir"]
        final_base_dir = staged["final_dir"]
        target_image_size = get_sample_image_size(active_source_images_dir)
        staging_meta = build_staging_proof(
            staged=staged,
            train_names=train_names,
            test_names=test_names,
            source_images_dir=active_source_images_dir,
            reference_model_dir=active_reference_model_dir,
            camera_preprocess_meta=camera_preprocess_meta,
            expected_sorted_names=split_row.get("sorted_images") if split_row else None,
        )

        print(
            f"[STAGE] init images: {len(train_names)}; final images: {len(train_names) + len(test_names)}; "
            f"test size: {len(test_names)}"
        )

        conditions = list(args.conditions) if args.conditions else []
        if not conditions:
            conditions = ["vggt_vanilla", "vggt_confidence"]
            if not args.skip_colmap:
                conditions.insert(0, "colmap")
        if args.skip_colmap and "colmap" in conditions:
            conditions = [c for c in conditions if c != "colmap"]

        for condition in conditions:
            if condition.startswith("mast3r_sfm"):
                init_backend = "mast3r_sfm"
            elif condition == "colmap" or condition.startswith("colmap_trainonly"):
                init_backend = "colmap_trainonly"
            elif condition.startswith("colmap_refpose_"):
                init_backend = "reference_pose_colmap"
            else:
                init_backend = "vggt"
            run_root = results_root / dataset_name / scene / setting_label / condition
            run_root.mkdir(parents=True, exist_ok=True)
            metrics_path = run_root / "metrics.json"
            if metrics_path.exists():
                reusable = existing_strict_trainonly_result_valid(run_root, condition)
                if reusable and split_row:
                    reusable = existing_run_matches_split(run_root, split_row)
                if reusable:
                    reusable = existing_camera_preprocess_result_valid(run_root, camera_preprocess_meta)
                if reusable:
                    print(f"[SKIP] Metrics already exist for {scene} ({setting_label}) {condition}")
                    continue
                stale_path = archive_stale_metrics(metrics_path, "strict_protocol_or_split")
                print(
                    f"[RERUN] Existing metrics for {scene} ({setting_label}) {condition} "
                    f"lack required strict protocol/confidence/split proof; archived {stale_path.name}"
                )
            run_meta = {
                "mode": args.mode,
                "protocol_mode": "sparse_fixed_trainonly_init_ref",
                "init_protocol": f"train_only_{init_backend}_init_reference_aligned",
                "init_backend": init_backend,
                "dataset": dataset_name,
                "scene": scene,
                "setting": setting_label,
                "condition": condition,
                "seed": seed,
                "holdout": args.holdout,
                "n_train": n_train,
                "images_source": str(active_source_images_dir),
                "raw_images_source": str(source_images_dir),
                "train_images": train_names,
                "test_images": test_names,
                "reference_model": str(active_reference_model_dir),
                "raw_reference_model": str(reference_model_dir),
                "images_subdir": img_subdir,
                "radial_camera_policy": args.radial_camera_policy,
                "camera_preprocess": camera_preprocess_meta,
                "gpu": gpu_info,
            }
            if split_row:
                run_meta["split_manifest"] = str(args.split_manifest)
                run_meta["split_manifest_sha256"] = _file_sha256(args.split_manifest)
                run_meta["split_dataset"] = split_row.get("dataset")
                run_meta["split_scene"] = split_row.get("scene")
                run_meta["split_rule"] = split_row.get("split_rule")
                run_meta["split_source_dir"] = split_row.get("source_dir")
                run_meta["split_image_dir"] = split_row.get("image_dir")
                run_meta["sorted_images"] = split_row.get("sorted_images")
                run_meta["train_indices_0based"] = split_row.get("train_indices_0based")
                run_meta["test_indices_0based"] = split_row.get("test_indices_0based")
                run_meta["split_row_sha256"] = split_row.get("manifest_row_sha256")
                run_meta["split_source_proof"] = split_source_proof
                if init_backend == "mast3r_sfm":
                    init_usage = "MASt3R-SfM initialization runs on staged train images only"
                elif init_backend == "colmap_trainonly":
                    init_usage = "COLMAP train-only SfM initialization runs on staged train images only"
                elif init_backend == "reference_pose_colmap":
                    init_usage = "reference train poses are used for this diagnostic/reference-pose condition"
                else:
                    init_usage = "VGGT initialization runs on staged train images only"
                if camera_preprocess_meta and camera_preprocess_meta.get("required"):
                    run_meta["reference_model_usage"] = (
                        "official_24_views_sparse_0 was COLMAP-undistorted to obtain 3DGS-compatible "
                        "PINHOLE/SIMPLE_PINHOLE cameras; the undistorted reference is used for "
                        f"alignment and test camera metadata only; {init_usage.replace('staged train', 'staged undistorted train')}"
                    )
                else:
                    run_meta["reference_model_usage"] = (
                        "official_24_views_sparse_0_for_reference_alignment_and_test_camera_metadata_only; "
                        f"{init_usage}"
                    )

            stages_meta = {}
            if camera_preprocess_meta and camera_preprocess_meta.get("required"):
                stages_meta["camera_undistortion"] = camera_preprocess_meta
            stages_meta["staging"] = staging_meta
            run_meta["stages"] = stages_meta
            (run_root / "run_metadata.json").write_text(json.dumps(run_meta, indent=2))

            metrics_data = None
            vggt_dataset_dir = None  # VGGT pipeline output (has depths/ if available)

            if condition == "colmap" or condition.startswith("colmap_trainonly"):
                print(f"[STAGE] COLMAP train-only SfM baseline ({condition})")
                colmap_meta = run_colmap_paper(
                    init_dir,
                    num_threads=int(colmap_cfg.get("threads", 8)),
                    use_gpu=not (colmap_cfg.get("cpu", False) or args.colmap_cpu),
                    log_dir=run_root / "logs",
                    gpu_index=args.gpu_index,
                    gaussian_repo=gaussian_repo,
                    colmap_bin=str(args.colmap_bin),
                )
                stages_meta["colmap"] = colmap_meta
                if not colmap_meta.get("success"):
                    print(f"[SKIP] COLMAP failed for {scene} ({setting_label})")
                    run_meta["stages"] = stages_meta
                    (run_root / "run_metadata.json").write_text(json.dumps(run_meta, indent=2))
                    continue
                init_model_dir = init_dir / "sparse" / "0"
            elif condition.startswith("colmap_refpose_"):
                print(f"[STAGE] Reference-pose COLMAP ({condition})")
                # Build dataset directly from reference COLMAP poses — no SfM, no alignment
                dataset_dir = run_root / "dataset"
                ensure_dir_clean(dataset_dir)
                images_dir = dataset_dir / "images"
                try:
                    os.symlink(final_base_dir / "images", images_dir, target_is_directory=True)
                except Exception:
                    shutil.copytree(final_base_dir / "images", images_dir, dirs_exist_ok=True)
                target_test = dataset_dir / "sparse" / "0" / "test.txt"
                target_test.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(staged["test_txt"], target_test)

                # Run feature extraction + matching + point_triangulator with reference poses
                refpose_work = run_root / "refpose_work"
                refpose_work.mkdir(parents=True, exist_ok=True)
                # Symlink images into work dir for COLMAP
                refpose_images = refpose_work / "images"
                if not refpose_images.exists():
                    try:
                        os.symlink(init_dir / "images", refpose_images, target_is_directory=True)
                    except Exception:
                        shutil.copytree(init_dir / "images", refpose_images, dirs_exist_ok=True)

                refpose_meta = run_colmap_refpose(
                    scene_dir=refpose_work,
                    reference_model_dir=active_reference_model_dir,
                    train_names=train_names,
                    gaussian_repo=gaussian_repo,
                    log_dir=run_root / "logs",
                    colmap_bin=str(args.colmap_bin),
                    gpu_index=args.gpu_index,
                    num_threads=int(colmap_cfg.get("threads", 8)),
                    colmap_cpu=colmap_cfg.get("cpu", False) or args.colmap_cpu,
                )
                stages_meta["colmap_refpose"] = refpose_meta
                if not refpose_meta.get("success"):
                    print(f"[SKIP] Reference-pose COLMAP failed for {scene} ({setting_label})")
                    run_meta["stages"] = stages_meta
                    (run_root / "run_metadata.json").write_text(json.dumps(run_meta, indent=2))
                    continue

                # Read triangulated model and reference model to build final dataset
                rwm = _get_rwm(gaussian_repo)
                tri_dir = refpose_work / "sparse" / "0"
                cameras_tri, images_tri, points_tri = rwm.read_model(str(tri_dir))
                cameras_ref, images_ref, _ = rwm.read_model(str(active_reference_model_dir))

                # Scale cameras if needed (reference may be full-res, training at different res)
                if target_image_size:
                    ref_cam = next(iter(cameras_ref.values()))
                    if ref_cam.width != target_image_size[0] or ref_cam.height != target_image_size[1]:
                        cameras_ref, _ = scale_cameras_to_size(
                            cameras_ref, target_image_size[0], target_image_size[1], rwm
                        )

                # Build final images: ALL train cameras from reference model, test from reference
                # (use reference poses for all train_names, not just triangulated subset)
                ref_by_name = {img.name: img for img in images_ref.values()}
                empty_xys = np.empty((0, 2))
                empty_ids = np.empty((0,), dtype=int)
                final_images = {}
                # Train cameras: all requested train views from reference model
                for name in train_names:
                    if name in ref_by_name:
                        ref_img = ref_by_name[name]
                        final_images[ref_img.id] = rwm.Image(
                            id=ref_img.id, qvec=ref_img.qvec, tvec=ref_img.tvec,
                            camera_id=ref_img.camera_id, name=ref_img.name,
                            xys=empty_xys, point3D_ids=empty_ids,
                        )
                # Test cameras from reference model
                for name in test_names:
                    if name in ref_by_name:
                        ref_img = ref_by_name[name]
                        final_images[ref_img.id] = rwm.Image(
                            id=ref_img.id, qvec=ref_img.qvec, tvec=ref_img.tvec,
                            camera_id=ref_img.camera_id, name=ref_img.name,
                            xys=empty_xys, point3D_ids=empty_ids,
                        )

                used_cam_ids = {img.camera_id for img in final_images.values()}
                cameras_final = {cid: cam for cid, cam in cameras_ref.items() if cid in used_cam_ids}

                # Use triangulated sparse points (or dense fused.ply for dense condition)
                points_final = {}
                for pid, pt in points_tri.items():
                    points_final[pid] = rwm.Point3D(
                        id=pid, xyz=pt.xyz, rgb=pt.rgb, error=pt.error,
                        image_ids=np.empty((0,), dtype=int),
                        point2D_idxs=np.empty((0,), dtype=int),
                    )

                if split_row:
                    stages_meta["camera_model_validation"] = validate_3dgs_camera_models(cameras_final)
                write_colmap_text_model(
                    dataset_dir / "sparse" / "0", cameras_final, final_images, points_final, gaussian_repo
                )
                stages_meta["alignment"] = {
                    "shared_train": len([n for n in train_names if n in {img.name for img in images_tri.values()}]),
                    "train_missing_in_init": 0,
                    "method": "reference_poses",
                    "triangulated_points": len(points_final),
                }
                print(f"[REFPOSE] {len(final_images)} cameras, {len(points_final)} triangulated points")

                # For dense condition: run dense stereo and replace point cloud
                if "dense" in condition:
                    print("[STAGE] Running dense stereo (undistort + patch_match + fusion)")
                    fused_ply, dense_meta = run_dense_stereo(
                        scene_dir=refpose_work,
                        log_dir=run_root / "logs",
                        colmap_bin=str(args.colmap_bin),
                        gpu_index=args.gpu_index,
                    )
                    stages_meta["dense_stereo"] = dense_meta
                    if fused_ply and fused_ply.exists():
                        # Convert fused.ply to 3DGS-compatible format and replace sparse points
                        from plyfile import PlyData
                        plydata = PlyData.read(str(fused_ply))
                        vertex = plydata['vertex']
                        xyz = np.column_stack([vertex['x'], vertex['y'], vertex['z']])
                        rgb = np.column_stack([vertex['red'], vertex['green'], vertex['blue']])
                        # Write as 3DGS-compatible PLY
                        target_ply = dataset_dir / "sparse" / "0" / "points3D.ply"
                        # Use storePly from 3DGS utils
                        sys.path.insert(0, str(gaussian_repo))
                        from utils.graphics_utils import BasicPointCloud
                        from scene.dataset_readers import storePly
                        storePly(str(target_ply), xyz, rgb)
                        dense_point_count = len(xyz)
                        stages_meta["dense_stereo"]["dense_point_count"] = dense_point_count
                        print(f"[DENSE] Wrote {dense_point_count} dense points to {target_ply}")
                    else:
                        print(f"[SKIP] Dense stereo failed (no fused.ply) for {scene} ({setting_label}) — aborting, will not write invalid dense metrics")
                        run_meta["stages"] = stages_meta
                        (run_root / "run_metadata.json").write_text(json.dumps(run_meta, indent=2))
                        continue

                # Skip the normal alignment + dataset construction flow — go straight to training
                # (dataset_dir, images, test.txt, and sparse/0/ model already set up above)

                # --- Depth / confidence / GS args setup for refpose conditions ---
                depths_dir_name = None
                use_depth = False
                extra_args_for_condition = None
                # Match DropGaussian's 10K training budget for _10k conditions
                if condition.endswith("_10k"):
                    gs_args_for_condition = {
                        **PAPER_GS_ARGS,
                        "iterations": 10000,
                        "test_iterations": [10000],
                        "save_iterations": [10000],
                        "position_lr_max_steps": 10000,
                        "densify_until_iter": 10000,
                    }
                else:
                    gs_args_for_condition = None  # → defaults to PAPER_GS_ARGS (30K)

                model_dir = run_root / "model"
                train_meta, metrics_data = run_training_and_eval(
                    dataset_dir=dataset_dir,
                    model_dir=model_dir,
                    gaussian_repo=gaussian_repo,
                    python_exe=python_exe,
                    gpu_index=args.gpu_index,
                    resolution=resolution_override,
                    use_confidence=False,
                    llffhold=0,
                    use_pseudo_views=False,
                    refine_poses=False,
                    pose_params=None,
                    depths_dir=depths_dir_name,
                    gs_args=gs_args_for_condition,
                    extra_args=extra_args_for_condition,
                )
                stages_meta.update(train_meta)

                run_meta["stages"] = stages_meta
                run_meta["metrics"] = metrics_data
                (run_root / "run_metadata.json").write_text(json.dumps(run_meta, indent=2))

                if metrics_data:
                    metrics_out = run_root / "metrics.json"
                    metrics_out.write_text(json.dumps(metrics_data, indent=2))
                    print(f"[DONE] {condition} {scene} n={n_train}: PSNR={metrics_data.get('PSNR', 0):.2f}")
                continue  # Skip the rest of the per-condition loop (alignment, depth, training below)

            elif condition.startswith("vggt_ba"):
                print("[STAGE] VGGT + Bundle Adjustment (train-only)")
                vggt_repo = Path(vggt_cfg.get("repo", "./repos/vggt"))
                ba_work_dir, ba_meta = run_vggt_ba_for_scene(
                    init_dir=init_dir,
                    output_root=run_root / "pipeline",
                    gpu_index=args.gpu_index,
                    python_exe=python_exe,
                    vggt_repo=vggt_repo,
                )
                stages_meta["vggt_ba"] = ba_meta
                if not ba_work_dir:
                    print(f"[SKIP] VGGT+BA failed for {scene} ({setting_label})")
                    run_meta["stages"] = stages_meta
                    (run_root / "run_metadata.json").write_text(json.dumps(run_meta, indent=2))
                    continue
                # demo_colmap.py writes to sparse/ (not sparse/0/)
                init_model_dir = ba_work_dir / "sparse"
            elif condition.startswith("mast3r_sfm"):
                print("[STAGE] MASt3R-SfM export + COLMAP-format conversion (train-only)")
                mast3r_model_dir, mast3r_meta = run_mast3r_sfm_for_scene(
                    stage_scene_dir=init_dir,
                    output_root=run_root / "pipeline",
                    mast3r_cfg=mast3r_cfg,
                    gaussian_repo=gaussian_repo,
                    python_exe=python_exe,
                    gpu_index=args.gpu_index,
                )
                stages_meta["mast3r_sfm"] = mast3r_meta
                if not mast3r_model_dir:
                    print(f"[SKIP] MASt3R-SfM did not produce a model for {scene} ({setting_label})")
                    run_meta["stages"] = stages_meta
                    (run_root / "run_metadata.json").write_text(json.dumps(run_meta, indent=2))
                    continue
                stages_meta["mast3r_sfm"] = build_mast3r_sfm_proof(
                    init_dir=init_dir,
                    model_dir=mast3r_model_dir,
                    train_names=train_names,
                    test_names=test_names,
                    meta=mast3r_meta,
                )
                init_model_dir = mast3r_model_dir
            else:
                print("[STAGE] VGGT export + COLMAP-format conversion (train-only)")
                dataset_dir, vggt_meta = run_vggt_pipeline_for_scene(
                    stage_scene_dir=init_dir,
                    output_root=run_root / "pipeline",
                    vggt_cfg=vggt_cfg,
                    python_exe=python_exe,
                    gpu_index=args.gpu_index,
                    disable_median_norm=condition_disables_bridge_median_norm(
                        condition, args.bridge_no_median_norm
                    ),
                )
                stages_meta["vggt_export"] = vggt_meta
                if not dataset_dir:
                    print(f"[SKIP] VGGT pipeline did not produce dataset for {scene} ({setting_label})")
                    run_meta["stages"] = stages_meta
                    (run_root / "run_metadata.json").write_text(json.dumps(run_meta, indent=2))
                    continue
                stages_meta["vggt_export"] = build_vggt_export_proof(
                    init_dir=init_dir,
                    dataset_dir=dataset_dir,
                    train_names=train_names,
                    test_names=test_names,
                    meta=vggt_meta,
                )
                init_model_dir = dataset_dir / "sparse" / "0"
                vggt_dataset_dir = dataset_dir  # save ref before overwrite

            # Build per-condition final dataset dir (train+test images + aligned model)
            dataset_dir = run_root / "dataset"
            ensure_dir_clean(dataset_dir)
            images_dir = dataset_dir / "images"
            try:
                os.symlink(final_base_dir / "images", images_dir, target_is_directory=True)
            except Exception:
                shutil.copytree(final_base_dir / "images", images_dir, dirs_exist_ok=True)
            target_test = dataset_dir / "sparse" / "0" / "test.txt"
            target_test.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(staged["test_txt"], target_test)

            # Align init model to reference frame using train images only
            try:
                cameras_final, images_final, points_final, align_meta = align_init_to_reference(
                    init_model_dir=init_model_dir,
                    reference_model_dir=active_reference_model_dir,
                    train_names=train_names,
                    test_names=test_names,
                    gaussian_repo=gaussian_repo,
                    target_image_size=target_image_size,
                )
            except Exception as exc:
                print(f"[SKIP] Alignment failed for {scene} ({setting_label}, {condition}): {exc}")
                run_meta["stages"] = stages_meta
                (run_root / "run_metadata.json").write_text(json.dumps(run_meta, indent=2))
                continue

            if args.pose_source == "oracle":
                _project_root = Path(__file__).resolve().parents[1]
                if str(_project_root) not in sys.path:
                    sys.path.insert(0, str(_project_root))
                from pipelines.pose_injection import inject_oracle_train_poses
                rwm = _get_rwm(gaussian_repo)
                cameras_final, images_final, swap_meta = inject_oracle_train_poses(
                    cameras_aligned=cameras_final,
                    images_aligned=images_final,
                    train_image_names=train_names,
                    reference_model_dir=active_reference_model_dir,
                    rwm=rwm,
                )
                stages_meta["pose_injection"] = swap_meta
                print(f"[STAGE] pose_injection: swapped {len(swap_meta['train_swapped'])} train poses for oracle (missing from ref: {len(swap_meta['train_missing_from_reference'])})")

            if args.point_source == "colmap_sparse":
                _project_root = Path(__file__).resolve().parents[1]
                if str(_project_root) not in sys.path:
                    sys.path.insert(0, str(_project_root))
                from pipelines.point_injection import colmap_triangulate_train
                rwm = _get_rwm(gaussian_repo)
                try:
                    new_points = colmap_triangulate_train(
                        train_images_dir=init_dir / "images",
                        oracle_model_dir=active_reference_model_dir,
                        work_dir=run_root / "point_injection",
                        train_image_names=train_names,
                        rwm=rwm,
                        colmap_bin=args.colmap_bin,
                    )
                    points_final = new_points
                    stages_meta["point_injection"] = {
                        "source": "colmap_sparse",
                        "n_points": len(new_points),
                    }
                    print(f"[STAGE] point_injection: replaced VGGT cloud with {len(new_points)} COLMAP-triangulated points")
                except Exception as exc:
                    print(f"[WARN] point_injection failed: {exc}; keeping VGGT cloud")
                    stages_meta["point_injection"] = {"source": "colmap_sparse", "error": str(exc)}

            if args.pose_noise_rot_deg > 0 or args.pose_noise_trans_pct > 0:
                _project_root = Path(__file__).resolve().parents[1]
                if str(_project_root) not in sys.path:
                    sys.path.insert(0, str(_project_root))
                from pipelines.pose_noise import perturb_train_poses
                rwm = _get_rwm(gaussian_repo)
                cameras_final, images_final = perturb_train_poses(
                    cameras=cameras_final,
                    images=images_final,
                    train_image_names=train_names,
                    sigma_rot_deg=args.pose_noise_rot_deg,
                    sigma_trans_pct=args.pose_noise_trans_pct,
                    rwm=rwm,
                    seed=args.pose_noise_seed,
                )
                stages_meta["pose_noise"] = {
                    "sigma_rot_deg": args.pose_noise_rot_deg,
                    "sigma_trans_pct": args.pose_noise_trans_pct,
                    "seed": args.pose_noise_seed,
                }
                print(f"[STAGE] pose_noise: perturbed {len(train_names)} train poses with σR={args.pose_noise_rot_deg}° σt={args.pose_noise_trans_pct}%")

            if split_row:
                stages_meta["camera_model_validation"] = validate_3dgs_camera_models(cameras_final)
            write_colmap_text_model(dataset_dir / "sparse" / "0", cameras_final, images_final, points_final, gaussian_repo)
            stages_meta["alignment"] = align_meta

            if "denseinit" in condition:
                try:
                    denseinit_meta = write_vggt_dense_init_ply(
                        vggt_dataset_dir=vggt_dataset_dir,
                        train_names=train_names,
                        test_names=test_names,
                        align_meta=align_meta,
                        target_ply=dataset_dir / "sparse" / "0" / "points3D.ply",
                        gaussian_repo=gaussian_repo,
                    )
                    stages_meta["vggt_dense_init"] = denseinit_meta
                    print(
                        "[STAGE] vggt_dense_init: wrote "
                        f"{denseinit_meta['point_count']} train-only dense points "
                        f"to {denseinit_meta['output_ply']}"
                    )
                except Exception as exc:
                    print(f"[SKIP] VGGT dense initializer failed for {scene} ({setting_label}, {condition}): {exc}")
                    stages_meta["vggt_dense_init"] = {"error": str(exc)}
                    run_meta["stages"] = stages_meta
                    (run_root / "run_metadata.json").write_text(json.dumps(run_meta, indent=2))
                    continue

            # Preserve VGGT confidence sidecars for methods that actually consume them.
            # Without this handoff, photoconf/xyz-anchor/init-filter conditions train as
            # near-vanilla variants because the final train+test dataset lacks VGGT conf.
            needs_point_conf = condition_uses_point_conf(condition)
            if needs_point_conf and "dense" in condition:
                print(
                    f"[SKIP] {condition} requested point confidence with a dense-replaced "
                    "point cloud; no matching dense confidence sidecar is available"
                )
                stages_meta["confidence_sidecar"] = {
                    "error": "point_conf_dense_incompatible",
                    "reason": "dense replacement changes point count/order",
                }
                run_meta["stages"] = stages_meta
                (run_root / "run_metadata.json").write_text(json.dumps(run_meta, indent=2))
                continue
            if needs_point_conf:
                conf_src = init_model_dir / "points3D_conf.npy"
                if conf_src.exists():
                    conf_values = np.load(conf_src)
                    expected_points = len(points_final)
                    if conf_values.shape[0] != expected_points:
                        print(
                            f"[SKIP] {condition} requested point confidence but sidecar length "
                            f"{conf_values.shape[0]} != final point count {expected_points}"
                        )
                        stages_meta["confidence_sidecar"] = {
                            "error": "point_conf_length_mismatch",
                            "source": str(conf_src),
                            "sidecar_points": int(conf_values.shape[0]),
                            "final_points": int(expected_points),
                        }
                        run_meta["stages"] = stages_meta
                        (run_root / "run_metadata.json").write_text(json.dumps(run_meta, indent=2))
                        continue
                    conf_dst = dataset_dir / "sparse" / "0" / "points3D_conf.npy"
                    shutil.copy2(conf_src, conf_dst)
                    stages_meta["confidence_sidecar"] = {
                        "points3D_conf": str(conf_dst),
                        "points": int(conf_values.shape[0]),
                        "bytes": conf_dst.stat().st_size,
                    }
                    print(f"[STAGE] Copied point confidence sidecar for {condition}: {conf_dst}")
                else:
                    print(f"[SKIP] {condition} requested point confidence but {conf_src} is missing")
                    stages_meta["confidence_sidecar"] = {
                        "error": "point_conf_missing",
                        "source": str(conf_src),
                    }
                    run_meta["stages"] = stages_meta
                    (run_root / "run_metadata.json").write_text(json.dumps(run_meta, indent=2))
                    continue

            needs_depth_conf = (
                "photoconf" in condition
                or "confdepth" in condition
                or "depthconf" in condition
                or condition in {
                    "vggt_improved_v2_frozenpos_vggtdepth_depthgrad_10k_sh3",
                    "vggt_improved_v2_frozenpos_vggtdepth_depthgrad_smooth_10k_sh3",
                }
            )
            if needs_depth_conf and vggt_dataset_dir is not None:
                depth_conf_src = vggt_dataset_dir / "sparse" / "0" / "depth_conf"
                if depth_conf_src.exists():
                    missing_depth_conf = [
                        name for name in train_names
                        if not (depth_conf_src / f"{os.path.splitext(name)[0]}.npy").exists()
                    ]
                    if missing_depth_conf:
                        print(
                            f"[SKIP] {condition} requested depth confidence but maps are missing for "
                            f"{len(missing_depth_conf)}/{len(train_names)} train images: {missing_depth_conf[:5]}"
                        )
                        stages_meta["confidence_sidecar"] = {
                            **stages_meta.get("confidence_sidecar", {}),
                            "error": "depth_conf_missing_train_maps",
                            "missing_train_maps": missing_depth_conf,
                        }
                        run_meta["stages"] = stages_meta
                        (run_root / "run_metadata.json").write_text(json.dumps(run_meta, indent=2))
                        continue
                    invalid_depth_conf: list[str] = []
                    depth_conf_shapes: set[tuple[int, ...]] = set()
                    for name in train_names:
                        conf_path = depth_conf_src / f"{os.path.splitext(name)[0]}.npy"
                        try:
                            conf_arr = np.load(conf_path)
                            depth_conf_shapes.add(tuple(conf_arr.shape))
                            if conf_arr.ndim != 2 or conf_arr.size == 0 or not np.isfinite(conf_arr).all():
                                invalid_depth_conf.append(f"{name}: shape={conf_arr.shape} finite={np.isfinite(conf_arr).all()}")
                        except Exception as exc:
                            invalid_depth_conf.append(f"{name}: {exc}")
                    if len(depth_conf_shapes) > 1:
                        invalid_depth_conf.append(f"inconsistent_shapes={sorted(depth_conf_shapes)}")
                    if invalid_depth_conf:
                        print(
                            f"[SKIP] {condition} requested depth confidence but maps failed validation: "
                            f"{invalid_depth_conf[:5]}"
                        )
                        stages_meta["confidence_sidecar"] = {
                            **stages_meta.get("confidence_sidecar", {}),
                            "error": "depth_conf_invalid",
                            "invalid_depth_conf": invalid_depth_conf,
                        }
                        run_meta["stages"] = stages_meta
                        (run_root / "run_metadata.json").write_text(json.dumps(run_meta, indent=2))
                        continue
                    depth_conf_dst = dataset_dir / "sparse" / "0" / "depth_conf"
                    if depth_conf_dst.exists():
                        shutil.rmtree(depth_conf_dst)
                    depth_conf_dst.mkdir(parents=True, exist_ok=True)
                    for name in train_names:
                        stem = os.path.splitext(name)[0]
                        shutil.copy2(depth_conf_src / f"{stem}.npy", depth_conf_dst / f"{stem}.npy")
                    depth_conf_count = len(list(depth_conf_dst.glob("*.npy")))
                    stages_meta["confidence_sidecar"] = {
                        **stages_meta.get("confidence_sidecar", {}),
                        "depth_conf": str(depth_conf_dst),
                        "depth_conf_maps": depth_conf_count,
                    }
                    print(f"[STAGE] Copied depth_conf maps for {condition}: {depth_conf_count}")
                else:
                    print(f"[SKIP] {condition} requested depth confidence but {depth_conf_src} is missing")
                    stages_meta["confidence_sidecar"] = {
                        **stages_meta.get("confidence_sidecar", {}),
                        "error": "depth_conf_missing",
                        "source": str(depth_conf_src),
                    }
                    run_meta["stages"] = stages_meta
                    (run_root / "run_metadata.json").write_text(json.dumps(run_meta, indent=2))
                    continue
            elif needs_depth_conf:
                print(f"[SKIP] {condition} requested depth confidence but no VGGT dataset is available")
                stages_meta["confidence_sidecar"] = {
                    **stages_meta.get("confidence_sidecar", {}),
                    "error": "depth_conf_source_unavailable",
                }
                run_meta["stages"] = stages_meta
                (run_root / "run_metadata.json").write_text(json.dumps(run_meta, indent=2))
                continue

            # Copy depth maps for depth regularization if requested
            use_depth = "depth" in condition and not condition.startswith("vggt_improved")
            depths_dir_name = None
            if use_depth and vggt_dataset_dir is not None:
                src_depths = vggt_dataset_dir / "depths"
                src_params = vggt_dataset_dir / "sparse" / "0" / "depth_params.json"
                if src_depths.exists() and src_params.exists():
                    dst_depths = dataset_dir / "depths"
                    if dst_depths.exists():
                        shutil.rmtree(dst_depths)
                    shutil.copytree(src_depths, dst_depths)
                    # Adjust depth_params for Sim(3) alignment scale:
                    # After alignment, depth_aligned = sim3_scale * depth_vggt
                    # So invdepth_aligned = invdepth_vggt / sim3_scale
                    # depth_params applies: invdepth_final = invdepth_stored * dp_scale + dp_offset
                    # We need: invdepth_final = invdepth_stored / sim3_scale
                    # So dp_scale = 1/sim3_scale, dp_offset = 0
                    sim3_scale = align_meta.get("scale", 1.0)
                    dp_scale = 1.0 / sim3_scale if sim3_scale > 0 else 1.0
                    with open(src_params, "r") as f:
                        orig_params = json.load(f)
                    adjusted_params = {}
                    for stem in orig_params:
                        adjusted_params[stem] = {"scale": dp_scale, "offset": 0.0}

                    # Create zero-filled depth maps for images without VGGT depth
                    # (test images don't have VGGT depth since inference is train-only).
                    # scale=0 triggers depth_reliable=False in Camera.__init__
                    import cv2 as _cv2
                    all_image_names = train_names + test_names
                    sample_depth = next(dst_depths.glob("*.png"), None)
                    if sample_depth is not None:
                        sample = _cv2.imread(str(sample_depth), _cv2.IMREAD_UNCHANGED)
                        dh, dw = sample.shape[:2]
                    else:
                        dh, dw = 480, 640
                    n_created = 0
                    for img_name in all_image_names:
                        n_remove = len(img_name.split(".")[-1]) + 1
                        stem = img_name[:-n_remove]
                        depth_file = dst_depths / f"{stem}.png"
                        if not depth_file.exists():
                            _cv2.imwrite(str(depth_file), np.zeros((dh, dw), dtype=np.uint16))
                            adjusted_params[stem] = {"scale": 0.0, "offset": 0.0}
                            n_created += 1

                    dst_params = dataset_dir / "sparse" / "0" / "depth_params.json"
                    with open(dst_params, "w") as f:
                        json.dump(adjusted_params, f, indent=2)
                    depths_dir_name = "depths"
                    n_real = len(list(dst_depths.glob("*.png"))) - n_created
                    print(f"[STAGE] Depth maps: {n_real} real (VGGT) + {n_created} placeholder (test images)")
                    print(f"[STAGE] Depth params: scale={dp_scale:.6f} (sim3_scale={sim3_scale:.6f})")
                else:
                    print(f"[WARN] Depth regularization requested but depth maps not found in VGGT output")
                    print(f"  depths dir: {src_depths} (exists={src_depths.exists()})")
                    print(f"  params: {src_params} (exists={src_params.exists()})")

            # Improved/MCMC conditions: sparse-tuned args + extra train args
            IMPROVED_EXTRA_ARGS = {
                "vggt_improved": {
                    "drop_prob_init": 0.1,
                    "drop_prob_peak": 0.3,
                    "drop_prob_peak_iter": 5000,
                    "drop_prob_end_iter": 9000,
                    "use_pearson_depth": "FLAG",
                    "wavelet_weight": 0.05,
                    "wavelet_sparse_lambda": 0.01,
                },
                # Ablation: no DropGaussian (Pearson + wavelet only)
                "vggt_improved_nodrop": {
                    "use_pearson_depth": "FLAG",
                    "wavelet_weight": 0.05,
                    "wavelet_sparse_lambda": 0.01,
                },
                # Ablation: no Pearson depth (DropGaussian + wavelet, default L1 depth)
                "vggt_improved_nopearson": {
                    "drop_prob_init": 0.1,
                    "drop_prob_peak": 0.3,
                    "drop_prob_peak_iter": 5000,
                    "drop_prob_end_iter": 9000,
                    "wavelet_weight": 0.05,
                    "wavelet_sparse_lambda": 0.01,
                },
                # Ablation: no wavelet (DropGaussian + Pearson only)
                "vggt_improved_nowavelet": {
                    "drop_prob_init": 0.1,
                    "drop_prob_peak": 0.3,
                    "drop_prob_peak_iter": 5000,
                    "drop_prob_end_iter": 9000,
                    "use_pearson_depth": "FLAG",
                },
                "vggt_mcmc": {
                    "use_mcmc": "FLAG",
                    "mcmc_cap_max": 200000,
                    "mcmc_noise_lr": 5e5,
                    "mcmc_dead_threshold": 0.005,
                    "use_pearson_depth": "FLAG",
                },
                "vggt_improved_highdrop": {
                    "drop_prob_init": 0.15,
                    "drop_prob_peak": 0.4,
                    "drop_prob_peak_iter": 4000,
                    "drop_prob_end_iter": 7000,
                    "use_pearson_depth": "FLAG",
                    "wavelet_weight": 0.05,
                    "wavelet_sparse_lambda": 0.01,
                },
                "vggt_improved_strongdepth": {
                    "drop_prob_init": 0.1,
                    "drop_prob_peak": 0.3,
                    "drop_prob_peak_iter": 5000,
                    "drop_prob_end_iter": 9000,
                    "use_pearson_depth": "FLAG",
                    "wavelet_weight": 0.05,
                    "wavelet_sparse_lambda": 0.01,
                    "depth_l1_weight_init": 2.0,
                    "depth_l1_weight_final": 0.05,
                },
                # v2: Paper-faithful monotonic schedule (0→γ), no wavelet, Pearson depth only
                "vggt_improved_v2": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.2,
                    "drop_prob_peak_iter": -1,  # sentinel: monotonic (peak at final iter)
                    "drop_prob_end_iter": -1,   # sentinel: no ramp-down
                    "use_pearson_depth": "FLAG",
                },
                # v2 matched for COLMAP comparison: same method, sh3 + 10K budget
                "vggt_improved_v2_10k_sh3": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.2,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                },
                # 30K iter variant — same regularization, longer training. For
                # very-sparse cases (LLFF n=3) where 10K is undertrained.
                "vggt_improved_v2_30k_sh3": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.2,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                },
                # 30K + frozen-pos — does longer training amplify the +0.10 signal?
                "vggt_improved_v2_frozenpos_30k_sh3": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.2,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                    "position_lr_init": 0.0,
                    "position_lr_final": 0.0,
                    "scaling_lr": 0.0,
                    "rotation_lr": 0.0,
                    "densify_until_iter": 0,
                },
                # 30K + frozen-pos + exposure compensation
                "vggt_improved_v2_frozenpos_exp_30k_sh3": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.2,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                    "position_lr_init": 0.0,
                    "position_lr_final": 0.0,
                    "scaling_lr": 0.0,
                    "rotation_lr": 0.0,
                    "densify_until_iter": 0,
                    "train_test_exp": "FLAG",
                },
                #
                # Motivated by diagnostic finding that bonsai-like scenes have +1.95 dB
                # oracle-pose headroom at n=12 — see EXPERIMENT_DIAGNOSIS_PLAN.md.
                "vggt_improved_v2_poserefine_10k_sh3": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.2,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                },
                # AGGRESSIVE pose refinement: 10x LR, 100x lower reg, full duration.
                # Conservative settings made delta_xi move only ~0.1deg vs VGGT's ~2.5deg
                # actual error on bonsai n=6 — gradient signal exists but optimizer too slow.
                "vggt_improved_v2_poserefine_aggro_10k_sh3": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.2,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                },
                # CONFIDENCE-WEIGHTED depth (Path B1, alternative to pose refine):
                # weights Pearson depth loss by VGGT confidence per-pixel. Tests whether
                # foundation-model-aware loss design closes any of the headroom.
                "vggt_improved_v2_confdepth_10k_sh3": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.2,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                    "use_depth_conf": "FLAG",
                },
                # COMBINED: aggressive pose refine + confidence-weighted depth.
                "vggt_improved_v2_poserefine_aggro_confdepth_10k_sh3": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.2,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                    "use_depth_conf": "FLAG",
                },
                # DD-Drop only (port D2GS's depth-density dropout; replaces DropGaussian).
                "vggt_improved_v2_dddrop_10k_sh3": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.0,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                    "use_dd_drop": "FLAG",
                    "dd_depth_weight": 0.5,
                    "dd_density_weight": 0.5,
                    "dd_drop_min": 0.05,
                    "dd_drop_max": 0.3,
                },
                "vggt_improved_v2_dafe_10k_sh3": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.2,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                    "lambda_far": 0.5,
                    "far_mask_quantile": 0.33,
                },
                "vggt_improved_v2_dd_dafe_10k_sh3": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.0,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                    "use_dd_drop": "FLAG",
                    "dd_depth_weight": 0.5,
                    "dd_density_weight": 0.5,
                    "dd_drop_min": 0.05,
                    "dd_drop_max": 0.3,
                    "lambda_far": 0.5,
                    "far_mask_quantile": 0.33,
                },
                #
                # Drops bottom 20% of init points by confidence BEFORE training.
                # Architecturally different from training-time interventions.
                "vggt_improved_v2_initfilter20_10k_sh3": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.2,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                    "conf_percentile_filter": 0.2,
                },
                "vggt_improved_v2_initfilter40_10k_sh3": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.2,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                    "conf_percentile_filter": 0.4,
                },
                #
                # trainer. Tests whether init-time pose correction via BA captures the
                # +1.96 dB bonsai n=12 headroom that training-time methods cannot. Name
                # MUST start with "vggt_ba" to trigger the BA init branch at line ~2541.
                "vggt_ba_improved_v2_10k_sh3": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.2,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                },
                # More aggressive init-filter (drop bottom 60%, bottom 80%) — for the
                # hypothesis that confidence-driven init pruning has a sweet spot.
                "vggt_improved_v2_initfilter60_10k_sh3": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.2,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                    "conf_percentile_filter": 0.6,
                },
                "vggt_improved_v2_initfilter80_10k_sh3": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.2,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                    "conf_percentile_filter": 0.8,
                },
                #
                # rotation LRs zeroed and densification disabled — Gaussian geometry is
                # locked at VGGT init. Only SH features + opacity + pose adjust during
                # training. Tests whether pose can move when geometry CANNOT compensate
                # (the diagnostic-implied failure mode for joint refinement).
                "vggt_improved_v2_frozenpos_10k_sh3": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.2,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                    "position_lr_init": 0.0,
                    "position_lr_final": 0.0,
                    "scaling_lr": 0.0,
                    "rotation_lr": 0.0,
                    "densify_until_iter": 0,
                },
                # Geometry-constrained pseudo-view test: same frozenpos/pose-refine
                # guardrails as frozenpos, but adds IP-Adapter + CLIP-filtered pseudo
                # views. This isolates whether generated views help once geometry drift
                # is constrained.
                "vggt_improved_v2_frozenpos_pseudo_v2_10k_sh3": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.2,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                    "position_lr_init": 0.0,
                    "position_lr_final": 0.0,
                    "scaling_lr": 0.0,
                    "rotation_lr": 0.0,
                    "densify_until_iter": 0,
                },
                # GS-GS/G4Splat-lite pivot: trainable VGGT geometry with VGGT
                # train-depth anchoring, then bootstrap renders refined by SD
                # img2img and consumed as low-weight pseudo views.
                "vggt_improved_v2_vggtdepth_gsgs_lite_bootstrap_sd_10k_sh3": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.2,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                },
                "vggt_improved_v2_vggtdepth_gsgs_lite_bootstrap_10k_sh3": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.2,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                },
                # Existing-code G4Splat-light gate: frozen VGGT geometry plus
                # VGGT train-view depth supervision and DAFE far-field emphasis.
                # The "vggtdepth" token intentionally selects VGGT depth maps
                # instead of DA V2 below.
                "vggt_improved_v2_frozenpos_vggtdepth_dafe_10k_sh3": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.2,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                    "position_lr_init": 0.0,
                    "position_lr_final": 0.0,
                    "scaling_lr": 0.0,
                    "rotation_lr": 0.0,
                    "densify_until_iter": 0,
                    "lambda_far": 0.25,
                    "far_mask_quantile": 0.33,
                },
                "vggt_improved_v2_frozenpos_vggtdepth_10k_sh3": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.2,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                    "position_lr_init": 0.0,
                    "position_lr_final": 0.0,
                    "scaling_lr": 0.0,
                    "rotation_lr": 0.0,
                    "densify_until_iter": 0,
                },
                # Train-only G4Splat-light depth geometry guard: same claim-safe
                # frozenpos + VGGT-depth path, with local depth-gradient alignment.
                "vggt_improved_v2_frozenpos_vggtdepth_depthgrad_10k_sh3": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.2,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                    "position_lr_init": 0.0,
                    "position_lr_final": 0.0,
                    "scaling_lr": 0.0,
                    "rotation_lr": 0.0,
                    "densify_until_iter": 0,
                    "depth_grad_weight": 0.15,
                    "depth_grad_use_conf": "FLAG",
                    "depth_geom_warmup_iter": 500,
                    "depth_geom_max_iter": 8000,
                },
                "vggt_improved_v2_frozenpos_vggtdepth_depthgrad_smooth_10k_sh3": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.2,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                    "position_lr_init": 0.0,
                    "position_lr_final": 0.0,
                    "scaling_lr": 0.0,
                    "rotation_lr": 0.0,
                    "densify_until_iter": 0,
                    "depth_grad_weight": 0.12,
                    "depth_grad_use_conf": "FLAG",
                    "depth_smooth_weight": 0.02,
                    "depth_smooth_edge_weight": 10.0,
                    "depth_geom_warmup_iter": 500,
                    "depth_geom_max_iter": 8000,
                },
                "vggt_improved_v2_frozenpos_vggtdepth_dafe50_10k_sh3": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.2,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                    "position_lr_init": 0.0,
                    "position_lr_final": 0.0,
                    "scaling_lr": 0.0,
                    "rotation_lr": 0.0,
                    "densify_until_iter": 0,
                    "lambda_far": 0.5,
                    "far_mask_quantile": 0.33,
                },
                # Pure "no densification" baseline (no pose refine, no frozen positions).
                # Tests whether densification spawning wrong Gaussians is the issue.
                "vggt_improved_v2_nodensify_10k_sh3": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.2,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                    "densify_until_iter": 0,
                },
                # Frozen geometry WITHOUT pose refine. Only SH features + opacity
                # are trainable. Tests whether the frozenpos result (+0.10 dB
                # on bonsai n=12) comes from frozen geometry or pose refinement.
                "vggt_improved_v2_frozenonly_10k_sh3": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.2,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                    "position_lr_init": 0.0,
                    "position_lr_final": 0.0,
                    "scaling_lr": 0.0,
                    "rotation_lr": 0.0,
                    "densify_until_iter": 0,
                },
                # Frozen geometry + init-filter combinations. Tests whether
                # dropping low-conf init points helps WHEN geometry is frozen
                # (since we can't grow back the dropped points). Three filter
                # levels with frozen-geometry + pose refine.
                "vggt_improved_v2_frozenpos_filter20_10k_sh3": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.2,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                    "position_lr_init": 0.0,
                    "position_lr_final": 0.0,
                    "scaling_lr": 0.0,
                    "rotation_lr": 0.0,
                    "densify_until_iter": 0,
                    "conf_percentile_filter": 0.2,
                },
                "vggt_improved_v2_frozenpos_filter40_10k_sh3": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.2,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                    "position_lr_init": 0.0,
                    "position_lr_final": 0.0,
                    "scaling_lr": 0.0,
                    "rotation_lr": 0.0,
                    "densify_until_iter": 0,
                    "conf_percentile_filter": 0.4,
                },
                # Frozen geometry + per-camera EXPOSURE COMPENSATION. Sparse-view
                # 360 captures have varying exposure between train views — without
                # compensation, the model must absorb variation in opacity/color,
                # fighting geometry. train_test_exp enables learnable per-camera exposure.
                "vggt_improved_v2_frozenpos_exp_10k_sh3": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.2,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                    "position_lr_init": 0.0,
                    "position_lr_final": 0.0,
                    "scaling_lr": 0.0,
                    "rotation_lr": 0.0,
                    "densify_until_iter": 0,
                    "train_test_exp": "FLAG",
                },
                # Frozen geometry + exposure, WITHOUT pose refine (isolation)
                "vggt_improved_v2_frozenonly_exp_10k_sh3": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.2,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                    "position_lr_init": 0.0,
                    "position_lr_final": 0.0,
                    "scaling_lr": 0.0,
                    "rotation_lr": 0.0,
                    "densify_until_iter": 0,
                    "train_test_exp": "FLAG",
                },
                # Baseline improved_v2 + exposure compensation only (no freeze).
                "vggt_improved_v2_exp_10k_sh3": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.2,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                    "train_test_exp": "FLAG",
                },
                # Cross-view consistency variant. This was validated in earlier
                # logs but the active condition table had lost the definition.
                "vggt_improved_v2_exp_xview_10k_sh3": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.2,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                    "train_test_exp": "FLAG",
                    "use_xview": "FLAG",
                    "xview_weight": 0.2,
                    "xview_warmup_iter": 500,
                    "xview_max_iter": 8000,
                },
                # Strict held-out variants: same regularizers, no train_test_exp.
                # These are the only improved VGGT variants eligible for global
                # sparse-view leaderboard claims.
                "vggt_improved_v2_xview_10k_sh3": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.2,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                    "use_xview": "FLAG",
                    "xview_weight": 0.2,
                    "xview_warmup_iter": 500,
                    "xview_max_iter": 8000,
                },
                "vggt_improved_v2_xyzanchor_strong_10k_sh3": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.2,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                    "use_conf_xyz_anchor": "FLAG",
                    "conf_xyz_anchor_strength": 0.9,
                    "conf_xyz_anchor_floor": 0.05,
                },
                "vggt_improved_v2_exp_xview_strong_10k_sh3": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.2,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                    "train_test_exp": "FLAG",
                    "use_xview": "FLAG",
                    "xview_weight": 0.4,
                    "xview_warmup_iter": 500,
                    "xview_max_iter": 8000,
                },
                # Confidence-anchored xyz gradients. High-confidence VGGT
                # points are allowed to move less; low-confidence points can
                # still escape bad initialization.
                "vggt_improved_v2_exp_xyzanchor_10k_sh3": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.2,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                    "train_test_exp": "FLAG",
                    "use_conf_xyz_anchor": "FLAG",
                    "conf_xyz_anchor_strength": 0.7,
                    "conf_xyz_anchor_floor": 0.1,
                },
                "vggt_improved_v2_exp_xyzanchor_strong_10k_sh3": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.2,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                    "train_test_exp": "FLAG",
                    "use_conf_xyz_anchor": "FLAG",
                    "conf_xyz_anchor_strength": 0.9,
                    "conf_xyz_anchor_floor": 0.05,
                },
                "vggt_improved_v2_exp_xyzanchor_extreme_10k_sh3": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.2,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                    "train_test_exp": "FLAG",
                    "use_conf_xyz_anchor": "FLAG",
                    "conf_xyz_anchor_strength": 0.98,
                    "conf_xyz_anchor_floor": 0.01,
                },
                #
                # photometric loss with annealed temperature. Builds on the diagnostic
                # finding that train_test_exp absorbs photometric variance uniformly;
                # this method *gates* that absorption by VGGT confidence.
                "vggt_improved_v2_exp_photoconf_10k_sh3": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.2,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                    "train_test_exp": "FLAG",
                    "use_photo_conf": "FLAG",
                    "photo_conf_temp_init": 2.0,
                    "photo_conf_temp_final": 0.5,
                    "photo_conf_floor": 0.1,
                    "photo_conf_max_steps": 10000,
                },
                # Claim-safe exposure compensation: optimize per-train-image exposure
                # without adding held-out cameras to the train set.
                "vggt_improved_v2_trainexp_photoconf_vggtdepth_10k_sh3": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.2,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                    "use_train_exposure": "FLAG",
                    "use_photo_conf": "FLAG",
                    "photo_conf_temp_init": 2.0,
                    "photo_conf_temp_final": 0.5,
                    "photo_conf_floor": 0.1,
                    "photo_conf_max_steps": 10000,
                },
                "vggt_improved_v2_exp_photoconf_xview_10k_sh3": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.2,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                    "train_test_exp": "FLAG",
                    "use_photo_conf": "FLAG",
                    "photo_conf_temp_init": 2.0,
                    "photo_conf_temp_final": 0.5,
                    "photo_conf_floor": 0.1,
                    "photo_conf_max_steps": 10000,
                    "use_xview": "FLAG",
                    "xview_weight": 0.2,
                    "xview_warmup_iter": 500,
                    "xview_max_iter": 8000,
                },
                "vggt_improved_v2_photoconf_xview_10k_sh3": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.2,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                    "use_photo_conf": "FLAG",
                    "photo_conf_temp_init": 2.0,
                    "photo_conf_temp_final": 0.5,
                    "photo_conf_floor": 0.1,
                    "photo_conf_max_steps": 10000,
                    "use_xview": "FLAG",
                    "xview_weight": 0.2,
                    "xview_warmup_iter": 500,
                    "xview_max_iter": 8000,
                },
                "vggt_improved_v2_exp_photoconf_xyzanchor_10k_sh3": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.2,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                    "train_test_exp": "FLAG",
                    "use_photo_conf": "FLAG",
                    "photo_conf_temp_init": 2.0,
                    "photo_conf_temp_final": 0.5,
                    "photo_conf_floor": 0.1,
                    "photo_conf_max_steps": 10000,
                    "use_conf_xyz_anchor": "FLAG",
                    "conf_xyz_anchor_strength": 0.7,
                    "conf_xyz_anchor_floor": 0.1,
                },
                "vggt_improved_v2_exp_photoconf_xyzanchor_strong_10k_sh3": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.2,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                    "train_test_exp": "FLAG",
                    "use_photo_conf": "FLAG",
                    "photo_conf_temp_init": 2.0,
                    "photo_conf_temp_final": 0.5,
                    "photo_conf_floor": 0.1,
                    "photo_conf_max_steps": 10000,
                    "use_conf_xyz_anchor": "FLAG",
                    "conf_xyz_anchor_strength": 0.9,
                    "conf_xyz_anchor_floor": 0.05,
                },
                "vggt_improved_v2_photoconf_xyzanchor_strong_10k_sh3": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.2,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                    "use_photo_conf": "FLAG",
                    "photo_conf_temp_init": 2.0,
                    "photo_conf_temp_final": 0.5,
                    "photo_conf_floor": 0.1,
                    "photo_conf_max_steps": 10000,
                    "use_conf_xyz_anchor": "FLAG",
                    "conf_xyz_anchor_strength": 0.9,
                    "conf_xyz_anchor_floor": 0.05,
                },
                # SOTA candidate: current best VGGT+3DGS recipe plus the existing
                # VGGT-pose COLMAP triangulation + dense stereo initializer. The
                # "_dense_" substring intentionally activates the dense branch below,
                # while the args here preserve improved_v2 + exposure + photoconf.
                "vggt_improved_v2_dense_exp_photoconf_10k_sh3": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.2,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                    "train_test_exp": "FLAG",
                    "use_photo_conf": "FLAG",
                    "photo_conf_temp_init": 2.0,
                    "photo_conf_temp_final": 0.5,
                    "photo_conf_floor": 0.1,
                    "photo_conf_max_steps": 10000,
                },
                "vggt_improved_v2_dense_photoconf_10k_sh3": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.2,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                    "use_photo_conf": "FLAG",
                    "photo_conf_temp_init": 2.0,
                    "photo_conf_temp_final": 0.5,
                    "photo_conf_floor": 0.1,
                    "photo_conf_max_steps": 10000,
                },
                "vggt_improved_v2_dense_merge_photoconf_10k_sh3": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.2,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                    "use_photo_conf": "FLAG",
                    "photo_conf_temp_init": 2.0,
                    "photo_conf_temp_final": 0.5,
                    "photo_conf_floor": 0.1,
                    "photo_conf_max_steps": 10000,
                },
                # Ablation: photoconf without train_test_exp (isolates effect of confidence weighting)
                "vggt_improved_v2_photoconf_10k_sh3": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.2,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                    "use_photo_conf": "FLAG",
                    "photo_conf_temp_init": 2.0,
                    "photo_conf_temp_final": 0.5,
                    "photo_conf_floor": 0.1,
                    "photo_conf_max_steps": 10000,
                },
                # Sharper temperature variant (trust VGGT more strongly throughout)
                "vggt_improved_v2_exp_photoconf_sharp_10k_sh3": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.2,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                    "train_test_exp": "FLAG",
                    "use_photo_conf": "FLAG",
                    "photo_conf_temp_init": 3.0,
                    "photo_conf_temp_final": 1.0,
                    "photo_conf_floor": 0.05,
                    "photo_conf_max_steps": 10000,
                },
                # Softer temperature variant (closer to uniform)
                "vggt_improved_v2_exp_photoconf_soft_10k_sh3": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.2,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                    "train_test_exp": "FLAG",
                    "use_photo_conf": "FLAG",
                    "photo_conf_temp_init": 1.5,
                    "photo_conf_temp_final": 0.3,
                    "photo_conf_floor": 0.2,
                    "photo_conf_max_steps": 10000,
                },
                # Frozen geometry + AGGRESSIVE SH/opacity LRs — since geometry
                # is locked, give SH/opacity more learning capacity.
                "vggt_improved_v2_frozenpos_aggro_10k_sh3": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.2,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                    "position_lr_init": 0.0,
                    "position_lr_final": 0.0,
                    "scaling_lr": 0.0,
                    "rotation_lr": 0.0,
                    "densify_until_iter": 0,
                    "feature_lr": 0.01,  # 4x default (0.0025)
                    "opacity_lr": 0.10,  # 2x default (0.05)
                },
                # Frozen + 5x boosted Pearson depth weight
                "vggt_improved_v2_frozenpos_depth5x_10k_sh3": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.2,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                    "position_lr_init": 0.0,
                    "position_lr_final": 0.0,
                    "scaling_lr": 0.0,
                    "rotation_lr": 0.0,
                    "densify_until_iter": 0,
                    "depth_l1_weight_init": 5.0,    # 5x default (1.0)
                    "depth_l1_weight_final": 0.05,  # 5x default (0.01)
                },
                # Frozen + aggressive DropGaussian (γ=0.5)
                "vggt_improved_v2_frozenpos_drop50_10k_sh3": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.5,           # 2.5x default 0.2
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                    "position_lr_init": 0.0,
                    "position_lr_final": 0.0,
                    "scaling_lr": 0.0,
                    "rotation_lr": 0.0,
                    "densify_until_iter": 0,
                },
                # Frozen + sh_degree=1 (less SH overfitting)
                "vggt_improved_v2_frozenpos_sh1_10k_sh3": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.2,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                    "position_lr_init": 0.0,
                    "position_lr_final": 0.0,
                    "scaling_lr": 0.0,
                    "rotation_lr": 0.0,
                    "densify_until_iter": 0,
                    "sh_degree": 1,
                },
                # Frozen + droponly (no Pearson depth) — does Pearson on frozen geom hurt?
                "vggt_improved_v2_frozenpos_droponly_10k_sh3": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.2,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "position_lr_init": 0.0,
                    "position_lr_final": 0.0,
                    "scaling_lr": 0.0,
                    "rotation_lr": 0.0,
                    "densify_until_iter": 0,
                },
                # Frozen + pearsononly (no DropGaussian)
                "vggt_improved_v2_frozenpos_pearsononly_10k_sh3": {
                    "use_pearson_depth": "FLAG",
                    "position_lr_init": 0.0,
                    "position_lr_final": 0.0,
                    "scaling_lr": 0.0,
                    "rotation_lr": 0.0,
                    "densify_until_iter": 0,
                },
                # Frozen + NEITHER (no drop, no pearson) — pure freeze baseline
                "vggt_improved_v2_frozenpos_baseline_10k_sh3": {
                    "position_lr_init": 0.0,
                    "position_lr_final": 0.0,
                    "scaling_lr": 0.0,
                    "rotation_lr": 0.0,
                    "densify_until_iter": 0,
                },
                #
                # Gaussians whose VGGT confidence exceeds threshold. Prevents bad-pose-
                # induced wrong densification.
                "vggt_improved_v2_confdensify30_10k_sh3": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.2,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                    "use_conf_densify_gate": "FLAG",
                    "conf_densify_threshold": 0.3,
                },
                "vggt_improved_v2_confdensify50_10k_sh3": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.2,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                    "use_conf_densify_gate": "FLAG",
                    "conf_densify_threshold": 0.5,
                },
                "vggt_improved_v2_confdensify70_10k_sh3": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.2,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                    "use_conf_densify_gate": "FLAG",
                    "conf_densify_threshold": 0.7,
                },
                #
                # Stage 1 (iter 0..1000): pose-only optimization, Gaussians frozen.
                # Stage 2 (iter 1001..10000): Gaussian-only optimization, poses frozen.
                # Tests the hypothesis that joint pose+Gaussian training fails because
                # Gaussians absorb pose error during co-optimization. Decoupling phases
                # lets poses move against fixed geometry (real gradient signal), then
                # geometry refines against fixed poses (standard 3DGS optimization).
                # MUST be added to POSE_PRESETS to enable --refine_poses.
                "vggt_improved_v2_twostage_10k_sh3": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.2,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                    "use_two_stage_pose": "FLAG",
                    "two_stage_split_iter": 1000,
                },
                # NOVEL: DD-Drop + DAFE + VGGT confidence as 3rd dropout signal.
                "vggt_improved_v2_dd_dafe_conf_10k_sh3": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.0,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                    "use_dd_drop": "FLAG",
                    "dd_depth_weight": 0.4,
                    "dd_density_weight": 0.4,
                    "dd_conf_weight": 0.2,
                    "dd_drop_min": 0.05,
                    "dd_drop_max": 0.3,
                    "lambda_far": 0.5,
                    "far_mask_quantile": 0.33,
                },
            }
            BRIDGE_ABLATION_EXTRA_ARGS = {
                # Clean camera-ready bridge ablations. These intentionally avoid
                # Pearson depth, DropGaussian, pose refinement, pseudo views, and
                # train/test exposure, so the only changed factors are bridge scale
                # normalization and init-time confidence seed filtering.
                "vggt_bridge_gate_10k_sh3": {
                    "conf_percentile_filter": 0.4,
                },
                "vggt_bridge_full_10k_sh3": {
                    "conf_percentile_filter": 0.4,
                },
            }
            IMPROVED_EXTRA_ARGS.update({
                # TWINGS-lite initializer probe: replace sparse VGGT points with
                # a train-only dense VGGT points3d PLY transformed by the same
                # Sim(3) alignment used for the cameras, then use the existing
                # train-only VGGT depth supervision path.
                "vggt_improved_v2_denseinit_vggtdepth_10k_sh3": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.2,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                },
                # Dense VGGT initializer, but geometry is frozen during 3DGS.
                # Tests whether the dense init is useful but gets damaged by
                # sparse-view densification/geometry updates.
                "vggt_improved_v2_denseinit_frozenpos_vggtdepth_10k_sh3": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.2,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                    "position_lr_init": 0.0,
                    "position_lr_final": 0.0,
                    "scaling_lr": 0.0,
                    "rotation_lr": 0.0,
                    "densify_until_iter": 0,
                },
                # Lightweight G4Splat-inspired geometry branch: keep the strict
                # train-only VGGT+DA depth setup, but add local depth-gradient
                # alignment so sparse views cannot satisfy Pearson depth with
                # locally implausible surfaces.
                "vggt_improved_v2_depthgrad_10k_sh3": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.2,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                    "depth_grad_weight": 0.15,
                    "depth_grad_use_conf": "FLAG",
                    "depth_geom_warmup_iter": 500,
                    "depth_geom_max_iter": 8000,
                },
                # Adds weak edge-aware smoothness on top of depth-gradient
                # alignment, approximating a plane prior in textureless regions.
                "vggt_improved_v2_depthgrad_smooth_10k_sh3": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.2,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                    "depth_grad_weight": 0.12,
                    "depth_grad_use_conf": "FLAG",
                    "depth_smooth_weight": 0.02,
                    "depth_smooth_edge_weight": 10.0,
                    "depth_geom_warmup_iter": 500,
                    "depth_geom_max_iter": 8000,
                },
                # Train-only VGGT-depth plane/depth probe with trainable geometry.
                # Uses aggressive pose refine plus depth-gradient and edge-aware
                # smoothness, without the unsafe DA-V2 depth surface.
                "vggt_improved_v2_vggtdepth_depthgrad_smooth_poserefine_10k_sh3": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.2,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                    "depth_grad_weight": 0.12,
                    "depth_grad_use_conf": "FLAG",
                    "depth_smooth_weight": 0.02,
                    "depth_smooth_edge_weight": 10.0,
                    "depth_geom_warmup_iter": 500,
                    "depth_geom_max_iter": 8000,
                },
                "vggt_improved_v2_vggtdepth_depthconf_depthgrad_smooth_poserefine_10k_sh3": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.2,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                    "use_depth_conf": "FLAG",
                    "depth_grad_weight": 0.12,
                    "depth_grad_use_conf": "FLAG",
                    "depth_smooth_weight": 0.02,
                    "depth_smooth_edge_weight": 10.0,
                    "depth_geom_warmup_iter": 500,
                    "depth_geom_max_iter": 8000,
                },
                # Creative sparse-view pivot: force pose to move before Gaussian
                # geometry can absorb errors, then train with only train-view VGGT
                # depth/confidence, cross-view consistency, and confidence-anchored
                # geometry. Claim-safe: no train_test_exp and no held-out depth.
                "vggt_improved_v2_vggtdepth_depthconf_twostage_xview_xyzanchor_10k_sh3": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.2,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                    "use_depth_conf": "FLAG",
                    "depth_conf_temperature": 1.5,
                    "use_two_stage_pose": "FLAG",
                    "two_stage_split_iter": 2000,
                    "depth_grad_weight": 0.10,
                    "depth_grad_use_conf": "FLAG",
                    "depth_smooth_weight": 0.015,
                    "depth_smooth_edge_weight": 10.0,
                    "depth_geom_warmup_iter": 250,
                    "depth_geom_max_iter": 7000,
                    "use_xview": "FLAG",
                    "xview_weight": 0.15,
                    "xview_warmup_iter": 750,
                    "xview_max_iter": 7000,
                    "use_conf_xyz_anchor": "FLAG",
                    "conf_xyz_anchor_strength": 0.75,
                    "conf_xyz_anchor_floor": 0.10,
                },
                # Clean ablation: DropGaussian only (no Pearson depth), matched settings
                "vggt_droponly_10k_sh3": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.2,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                },
                # Clean ablation: Pearson depth only (no DropGaussian), matched settings
                "vggt_pearsononly_10k_sh3": {
                    "use_pearson_depth": "FLAG",
                },
                # v3: v2 + slower depth decay + higher SSIM weight + lower densify threshold
                "vggt_improved_v3": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.2,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                    "depth_l1_weight_final": 0.1,   # 10x higher floor (0.01→0.1)
                    "lambda_dssim": 0.3,             # more structural similarity (0.2→0.3)
                    "densify_grad_threshold": 0.00015,  # lower threshold for sparse gradients
                },
                # Ablation: use VGGT depth maps instead of DA V2 for Pearson depth supervision
                "vggt_improved_v2_vggtdepth": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.2,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                },
                # Profiling variants: identical to base conditions, separate output dirs
                "vggt_improved_v2_profile": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.2,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                },
                "vggt_improved_v2_vggtdepth_profile": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.2,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                },
                # Profiling v2 (fixed DropGaussian schedule scaling at 30K)
                "vggt_improved_v2_profile2": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.2,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                },
                "vggt_improved_v2_vggtdepth_profile2": {
                    "drop_prob_init": 0.0,
                    "drop_prob_peak": 0.2,
                    "drop_prob_peak_iter": -1,
                    "drop_prob_end_iter": -1,
                    "use_pearson_depth": "FLAG",
                },
            })

            # Depth supervision for improved/mcmc conditions
            if condition in IMPROVED_EXTRA_ARGS and "vggtdepth" in condition:
                # Use VGGT depth maps instead of DA V2
                src_depths = vggt_dataset_dir / "depths" if vggt_dataset_dir else None
                if src_depths and src_depths.exists() and any(src_depths.glob("*.png")):
                    print("[STAGE] Using VGGT depth maps for Pearson depth supervision")
                    dst_depths = dataset_dir / "depths"
                    if dst_depths.exists():
                        shutil.rmtree(dst_depths)
                    shutil.copytree(src_depths, dst_depths)

                    # Build depth_params: train=1.0 (reliable), test=0.0 (unreliable)
                    depth_params = {}
                    for img_name in train_names:
                        stem = os.path.splitext(img_name)[0]
                        depth_params[stem] = {"scale": 1.0, "offset": 0.0}

                    # Create zero-filled placeholders for test images
                    import cv2 as _cv2
                    sample = next(dst_depths.glob("*.png"))
                    _h, _w = _cv2.imread(str(sample), _cv2.IMREAD_UNCHANGED).shape[:2]
                    for img_name in test_names:
                        stem = os.path.splitext(img_name)[0]
                        depth_file = dst_depths / f"{stem}.png"
                        if not depth_file.exists():
                            _cv2.imwrite(str(depth_file), np.zeros((_h, _w), dtype=np.uint16))
                        depth_params[stem] = {"scale": 0.0, "offset": 0.0}

                    dst_params = dataset_dir / "sparse" / "0" / "depth_params.json"
                    with open(dst_params, "w") as f:
                        json.dump(depth_params, f, indent=2)
                    depths_dir_name = "depths"
                    print(f"[STAGE] VGGT depth maps: {len(train_names)} train views")
                else:
                    print(f"[WARN] VGGT depth maps not found for vggtdepth condition, falling back to no depth supervision")
                    print(f"  vggt_dataset_dir: {vggt_dataset_dir}")

            elif condition in IMPROVED_EXTRA_ARGS:
                print("[STAGE] Running Depth Anything V2 on all training images")
                try:
                    sys.path.insert(0, str(gaussian_repo))
                    from utils.depth_anything import DepthAnythingV2Wrapper
                    da_wrapper = DepthAnythingV2Wrapper()

                    # Get the images directory used by 3DGS
                    images_subdir = dataset_dir / "images"
                    if resolution_override and resolution_override > 1:
                        images_subdir_res = dataset_dir / f"images_{resolution_override}"
                        if images_subdir_res.exists():
                            images_subdir = images_subdir_res

                    dst_depths = dataset_dir / "depths"
                    if dst_depths.exists():
                        shutil.rmtree(dst_depths)

                    depth_params = da_wrapper.process_scene(str(images_subdir), str(dst_depths))

                    # Mark test images as unreliable (scale=0) so they don't get depth supervision
                    for img_name in test_names:
                        stem = os.path.splitext(img_name)[0]
                        depth_params[stem] = {"scale": 0.0, "offset": 0.0}

                    dst_params = dataset_dir / "sparse" / "0" / "depth_params.json"
                    with open(dst_params, "w") as f:
                        json.dump(depth_params, f, indent=2)
                    depths_dir_name = "depths"
                    n_train_depth = len([n for n in train_names if os.path.splitext(n)[0] in depth_params and depth_params[os.path.splitext(n)[0]].get("scale", 0) != 0])
                    print(f"[STAGE] DA V2 depth maps: {n_train_depth} train views with depth supervision")
                except Exception as e:
                    print(f"[WARN] Depth Anything V2 failed: {e}")
                    print("[WARN] Falling back to no depth supervision")
                finally:
                    if str(gaussian_repo) in sys.path:
                        sys.path.remove(str(gaussian_repo))

            # Generate pseudo-views if requested
            use_pseudo = condition in PSEUDO_VIEW_CONDITIONS
            if use_pseudo:
                is_v2 = "pseudo_v2" in condition
                print(f"[STAGE] Generating pseudo-views {'(v2: IP-Adapter + CLIP)' if is_v2 else '(original)'}")
                pseudo_meta = run_pseudo_view_generation(
                    dataset_dir=dataset_dir,
                    gaussian_repo=gaussian_repo,
                    python_exe=python_exe,
                    gpu_index=args.gpu_index,
                    num_views=args.pseudo_num_views,
                    seed=seed,
                    use_ip_adapter=is_v2,
                    ip_adapter_scale=0.6,
                    clip_filter=is_v2,
                    clip_threshold=0.7,
                )
                stages_meta["pseudo_gen"] = pseudo_meta
                if pseudo_meta["code"] != 0:
                    print(f"[SKIP] Pseudo-view generation failed for {scene} ({setting_label})")
                    run_meta["stages"] = stages_meta
                    (run_root / "run_metadata.json").write_text(json.dumps(run_meta, indent=2))
                    continue

            # 3-loop self-training bootstrap (disabled — degrades results, costs 8K extra iters)
            if False and condition == "vggt_improved":
                loop_iters = [3000, 5000]  # coarse loops; final loop uses SPARSE_GS_ARGS (10K)
                # Coarse loops: strip DropGaussian args (schedule targets 5K-9K, won't activate properly)
                coarse_extra_args = {k: v for k, v in IMPROVED_EXTRA_ARGS.get(condition, {}).items()
                                     if not k.startswith("drop_prob")}
                prev_pseudo_dir = None
                for loop_idx, loop_iter in enumerate(loop_iters):
                    loop_label = f"loop_{loop_idx}"
                    print(f"[STAGE] Bootstrap loop {loop_idx+1}/{len(loop_iters)+1}: training ({loop_iter} iters)")
                    loop_model_dir = run_root / "pipeline" / loop_label / "model"

                    loop_meta, _ = run_training_and_eval(
                        dataset_dir=dataset_dir,
                        model_dir=loop_model_dir,
                        gaussian_repo=gaussian_repo,
                        python_exe=python_exe,
                        gpu_index=args.gpu_index,
                        resolution=resolution_override,
                        use_confidence=False,
                        llffhold=0,
                        depths_dir=depths_dir_name,
                        iterations=loop_iter,
                        gs_args=SPARSE_GS_ARGS,
                        extra_args=coarse_extra_args,
                        use_pseudo_views=(prev_pseudo_dir is not None),
                    )
                    stages_meta[f"{loop_label}_train"] = loop_meta

                    if loop_meta.get("train", {}).get("code", 1) != 0:
                        print(f"[SKIP] Bootstrap loop {loop_idx+1} failed")
                        break

                    print(f"[STAGE] Bootstrap loop {loop_idx+1}: rendering pseudo-views")
                    bootstrap_dir = run_root / "pipeline" / loop_label / "bootstrap_renders"
                    bootstrap_meta = run_bootstrap_render(
                        model_dir=loop_model_dir,
                        dataset_dir=dataset_dir,
                        output_dir=bootstrap_dir,
                        gaussian_repo=gaussian_repo,
                        python_exe=python_exe,
                        gpu_index=args.gpu_index,
                        num_views=args.pseudo_num_views,
                        iteration=loop_iter,
                        seed=seed,
                    )
                    stages_meta[f"{loop_label}_render"] = bootstrap_meta

                    if bootstrap_meta["code"] != 0:
                        print(f"[SKIP] Bootstrap render failed in loop {loop_idx+1}")
                        break

                    prev_pseudo_dir = bootstrap_dir

                use_pseudo = (prev_pseudo_dir is not None)
                print(f"[STAGE] Final training loop (10K iters) with pseudo-views={use_pseudo}")

            # Self-training bootstrap: coarse train → render → [SD refine] → retrain
            use_bootstrap = "bootstrap" in condition
            if use_bootstrap:
                print("[STAGE] Self-training bootstrap: coarse training (7K iters)")
                coarse_dir = run_root / "pipeline" / "coarse_model"
                coarse_meta, _ = run_training_and_eval(
                    dataset_dir=dataset_dir,
                    model_dir=coarse_dir,
                    gaussian_repo=gaussian_repo,
                    python_exe=python_exe,
                    gpu_index=args.gpu_index,
                    resolution=resolution_override,
                    use_confidence=False,
                    llffhold=0,
                    depths_dir=depths_dir_name,
                    iterations=7000,
                )
                stages_meta["coarse_train"] = coarse_meta

                print("[STAGE] Rendering novel views from coarse model")
                bootstrap_dir = run_root / "pipeline" / "bootstrap_renders"
                bootstrap_meta = run_bootstrap_render(
                    model_dir=coarse_dir,
                    dataset_dir=dataset_dir,
                    output_dir=bootstrap_dir,
                    gaussian_repo=gaussian_repo,
                    python_exe=python_exe,
                    gpu_index=args.gpu_index,
                    num_views=args.pseudo_num_views,
                    iteration=7000,
                    seed=seed,
                )
                stages_meta["bootstrap_render"] = bootstrap_meta

                if bootstrap_meta["code"] != 0:
                    print(f"[SKIP] Bootstrap render failed for {scene} ({setting_label})")
                    run_meta["stages"] = stages_meta
                    (run_root / "run_metadata.json").write_text(json.dumps(run_meta, indent=2))
                    continue

                # Optional SD refinement
                use_sd_refine = condition.endswith("_sd") or "_bootstrap_sd" in condition
                if use_sd_refine:
                    print("[STAGE] SD img2img refinement of bootstrap renders")
                    sd_meta = run_sd_refinement(
                        bootstrap_dir=bootstrap_dir,
                        dataset_dir=dataset_dir,
                        gaussian_repo=gaussian_repo,
                        python_exe=python_exe,
                        gpu_index=args.gpu_index,
                    )
                    stages_meta["sd_refinement"] = sd_meta

                use_pseudo = True  # bootstrap renders are packaged as pseudo-views

            use_conf = condition_uses_point_conf(condition)
            # Pose refinement conditions and hyperparameter presets
            POSE_PRESETS = {
                "vggt_ba_poserefine": {},  # use defaults from arguments/__init__.py
                "vggt_ba_poserefine_conservative": {
                    "pose_reg_lambda": 0.001,
                    "pose_grad_clip": 0.1,
                    "pose_lr_init": 5e-5,
                    "pose_start_iter": 2000,
                    "pose_end_iter": 20000,
                },
                "vggt_ba_poserefine_nocamp": {
                    "pose_no_camp": "FLAG",  # bare flag (store_true)
                },
                "vggt_poserefine": {},  # pose refinement on vanilla VGGT (no BA)
                "vggt_poserefine_conservative": {
                    "pose_reg_lambda": 0.001,
                    "pose_grad_clip": 0.1,
                    "pose_lr_init": 5e-5,
                    "pose_start_iter": 2000,
                    "pose_end_iter": 20000,
                },
                #
                # pose_start=1000 (after densification warmup), pose_end=8000 (final 2K
                # for Gaussians to settle into refined poses). Conservative reg & lr to
                # avoid the bicycle-style failure where photometric-only signal drives
                # delta_xi to noise.
                "vggt_improved_v2_poserefine_10k_sh3": {
                    "pose_reg_lambda": 0.001,
                    "pose_grad_clip": 0.1,
                    "pose_lr_init": 5e-5,
                    "pose_start_iter": 1000,
                    "pose_end_iter": 8000,
                },
                # AGGRESSIVE: 10x lr, 100x lower reg, looser grad clip, full duration.
                # Goal: actually move poses to correct VGGT's ~2.5deg mean rotation error.
                "vggt_improved_v2_poserefine_aggro_10k_sh3": {
                    "pose_reg_lambda": 1e-5,
                    "pose_grad_clip": 1.0,
                    "pose_lr_init": 5e-4,
                    "pose_lr_final": 1e-6,
                    "pose_lr_max_steps": 10000,
                    "pose_start_iter": 500,
                    "pose_end_iter": 10000,
                },
                # COMBINED: aggressive pose refine + confidence-weighted depth.
                "vggt_improved_v2_poserefine_aggro_confdepth_10k_sh3": {
                    "pose_reg_lambda": 1e-5,
                    "pose_grad_clip": 1.0,
                    "pose_lr_init": 5e-4,
                    "pose_lr_final": 1e-6,
                    "pose_lr_max_steps": 10000,
                    "pose_start_iter": 500,
                    "pose_end_iter": 10000,
                },
                #
                # locked, pose can move using aggressive LR since geometry can't absorb
                # the error gradient.
                "vggt_improved_v2_frozenpos_10k_sh3": {
                    "pose_reg_lambda": 1e-5,
                    "pose_grad_clip": 1.0,
                    "pose_lr_init": 5e-4,
                    "pose_lr_final": 1e-6,
                    "pose_lr_max_steps": 10000,
                    "pose_start_iter": 500,
                    "pose_end_iter": 10000,
                },
                "vggt_improved_v2_frozenpos_pseudo_v2_10k_sh3": {
                    "pose_reg_lambda": 1e-5,
                    "pose_grad_clip": 1.0,
                    "pose_lr_init": 5e-4,
                    "pose_lr_final": 1e-6,
                    "pose_lr_max_steps": 10000,
                    "pose_start_iter": 500,
                    "pose_end_iter": 10000,
                },
                "vggt_improved_v2_frozenpos_vggtdepth_dafe_10k_sh3": {
                    "pose_reg_lambda": 1e-5,
                    "pose_grad_clip": 1.0,
                    "pose_lr_init": 5e-4,
                    "pose_lr_final": 1e-6,
                    "pose_lr_max_steps": 10000,
                    "pose_start_iter": 500,
                    "pose_end_iter": 10000,
                },
                "vggt_improved_v2_frozenpos_vggtdepth_10k_sh3": {
                    "pose_reg_lambda": 1e-5,
                    "pose_grad_clip": 1.0,
                    "pose_lr_init": 5e-4,
                    "pose_lr_final": 1e-6,
                    "pose_lr_max_steps": 10000,
                    "pose_start_iter": 500,
                    "pose_end_iter": 10000,
                },
                "vggt_improved_v2_frozenpos_vggtdepth_depthgrad_10k_sh3": {
                    "pose_reg_lambda": 1e-5,
                    "pose_grad_clip": 1.0,
                    "pose_lr_init": 5e-4,
                    "pose_lr_final": 1e-6,
                    "pose_lr_max_steps": 10000,
                    "pose_start_iter": 500,
                    "pose_end_iter": 10000,
                },
                "vggt_improved_v2_frozenpos_vggtdepth_depthgrad_smooth_10k_sh3": {
                    "pose_reg_lambda": 1e-5,
                    "pose_grad_clip": 1.0,
                    "pose_lr_init": 5e-4,
                    "pose_lr_final": 1e-6,
                    "pose_lr_max_steps": 10000,
                    "pose_start_iter": 500,
                    "pose_end_iter": 10000,
                },
                "vggt_improved_v2_frozenpos_vggtdepth_dafe50_10k_sh3": {
                    "pose_reg_lambda": 1e-5,
                    "pose_grad_clip": 1.0,
                    "pose_lr_init": 5e-4,
                    "pose_lr_final": 1e-6,
                    "pose_lr_max_steps": 10000,
                    "pose_start_iter": 500,
                    "pose_end_iter": 10000,
                },
                "vggt_improved_v2_denseinit_vggtdepth_10k_sh3": {
                    "pose_reg_lambda": 1e-5,
                    "pose_grad_clip": 1.0,
                    "pose_lr_init": 5e-4,
                    "pose_lr_final": 1e-6,
                    "pose_lr_max_steps": 10000,
                    "pose_start_iter": 500,
                    "pose_end_iter": 10000,
                },
                "vggt_improved_v2_denseinit_frozenpos_vggtdepth_10k_sh3": {
                    "pose_reg_lambda": 1e-5,
                    "pose_grad_clip": 1.0,
                    "pose_lr_init": 5e-4,
                    "pose_lr_final": 1e-6,
                    "pose_lr_max_steps": 10000,
                    "pose_start_iter": 500,
                    "pose_end_iter": 10000,
                },
                "vggt_improved_v2_vggtdepth_depthgrad_smooth_poserefine_10k_sh3": {
                    "pose_reg_lambda": 1e-5,
                    "pose_grad_clip": 1.0,
                    "pose_lr_init": 5e-4,
                    "pose_lr_final": 1e-6,
                    "pose_lr_max_steps": 10000,
                    "pose_start_iter": 500,
                    "pose_end_iter": 10000,
                },
                "vggt_improved_v2_vggtdepth_depthconf_depthgrad_smooth_poserefine_10k_sh3": {
                    "pose_reg_lambda": 1e-5,
                    "pose_grad_clip": 1.0,
                    "pose_lr_init": 5e-4,
                    "pose_lr_final": 1e-6,
                    "pose_lr_max_steps": 10000,
                    "pose_start_iter": 500,
                    "pose_end_iter": 10000,
                },
                "vggt_improved_v2_vggtdepth_depthconf_twostage_xview_xyzanchor_10k_sh3": {
                    "pose_reg_lambda": 1e-5,
                    "pose_grad_clip": 1.0,
                    "pose_lr_init": 7.5e-4,
                    "pose_lr_final": 1e-6,
                    "pose_lr_max_steps": 2000,
                    "pose_start_iter": 0,
                    "pose_end_iter": 2000,
                },
                # Frozen + init-filter combos — same pose params as base frozenpos
                "vggt_improved_v2_frozenpos_filter20_10k_sh3": {
                    "pose_reg_lambda": 1e-5,
                    "pose_grad_clip": 1.0,
                    "pose_lr_init": 5e-4,
                    "pose_lr_final": 1e-6,
                    "pose_lr_max_steps": 10000,
                    "pose_start_iter": 500,
                    "pose_end_iter": 10000,
                },
                "vggt_improved_v2_frozenpos_filter40_10k_sh3": {
                    "pose_reg_lambda": 1e-5,
                    "pose_grad_clip": 1.0,
                    "pose_lr_init": 5e-4,
                    "pose_lr_final": 1e-6,
                    "pose_lr_max_steps": 10000,
                    "pose_start_iter": 500,
                    "pose_end_iter": 10000,
                },
                # Frozen + aggressive SH/opacity LRs — same pose params
                "vggt_improved_v2_frozenpos_aggro_10k_sh3": {
                    "pose_reg_lambda": 1e-5,
                    "pose_grad_clip": 1.0,
                    "pose_lr_init": 5e-4,
                    "pose_lr_final": 1e-6,
                    "pose_lr_max_steps": 10000,
                    "pose_start_iter": 500,
                    "pose_end_iter": 10000,
                },
                # Frozen + exposure compensation — same pose params
                "vggt_improved_v2_frozenpos_exp_10k_sh3": {
                    "pose_reg_lambda": 1e-5,
                    "pose_grad_clip": 1.0,
                    "pose_lr_init": 5e-4,
                    "pose_lr_final": 1e-6,
                    "pose_lr_max_steps": 10000,
                    "pose_start_iter": 500,
                    "pose_end_iter": 10000,
                },
                # 30K iter frozen variants — pose params scaled to 30K
                "vggt_improved_v2_frozenpos_30k_sh3": {
                    "pose_reg_lambda": 1e-5,
                    "pose_grad_clip": 1.0,
                    "pose_lr_init": 5e-4,
                    "pose_lr_final": 1e-6,
                    "pose_lr_max_steps": 30000,
                    "pose_start_iter": 500,
                    "pose_end_iter": 30000,
                },
                "vggt_improved_v2_frozenpos_exp_30k_sh3": {
                    "pose_reg_lambda": 1e-5,
                    "pose_grad_clip": 1.0,
                    "pose_lr_init": 5e-4,
                    "pose_lr_final": 1e-6,
                    "pose_lr_max_steps": 30000,
                    "pose_start_iter": 500,
                    "pose_end_iter": 30000,
                },
                # Frozen+depth5x: same pose params, boosted depth weight
                "vggt_improved_v2_frozenpos_depth5x_10k_sh3": {
                    "pose_reg_lambda": 1e-5,
                    "pose_grad_clip": 1.0,
                    "pose_lr_init": 5e-4,
                    "pose_lr_final": 1e-6,
                    "pose_lr_max_steps": 10000,
                    "pose_start_iter": 500,
                    "pose_end_iter": 10000,
                },
                # Frozen+drop50: same pose params, aggressive dropgaussian
                "vggt_improved_v2_frozenpos_drop50_10k_sh3": {
                    "pose_reg_lambda": 1e-5,
                    "pose_grad_clip": 1.0,
                    "pose_lr_init": 5e-4,
                    "pose_lr_final": 1e-6,
                    "pose_lr_max_steps": 10000,
                    "pose_start_iter": 500,
                    "pose_end_iter": 10000,
                },
                # Frozen+sh1: same pose params, lower SH degree
                "vggt_improved_v2_frozenpos_sh1_10k_sh3": {
                    "pose_reg_lambda": 1e-5,
                    "pose_grad_clip": 1.0,
                    "pose_lr_init": 5e-4,
                    "pose_lr_final": 1e-6,
                    "pose_lr_max_steps": 10000,
                    "pose_start_iter": 500,
                    "pose_end_iter": 10000,
                },
                # Ablation variants — same pose params
                "vggt_improved_v2_frozenpos_droponly_10k_sh3": {
                    "pose_reg_lambda": 1e-5,
                    "pose_grad_clip": 1.0,
                    "pose_lr_init": 5e-4,
                    "pose_lr_final": 1e-6,
                    "pose_lr_max_steps": 10000,
                    "pose_start_iter": 500,
                    "pose_end_iter": 10000,
                },
                "vggt_improved_v2_frozenpos_pearsononly_10k_sh3": {
                    "pose_reg_lambda": 1e-5,
                    "pose_grad_clip": 1.0,
                    "pose_lr_init": 5e-4,
                    "pose_lr_final": 1e-6,
                    "pose_lr_max_steps": 10000,
                    "pose_start_iter": 500,
                    "pose_end_iter": 10000,
                },
                "vggt_improved_v2_frozenpos_baseline_10k_sh3": {
                    "pose_reg_lambda": 1e-5,
                    "pose_grad_clip": 1.0,
                    "pose_lr_init": 5e-4,
                    "pose_lr_final": 1e-6,
                    "pose_lr_max_steps": 10000,
                    "pose_start_iter": 500,
                    "pose_end_iter": 10000,
                },
                #
                # 1000 iters (Gaussians frozen), then Gaussian-only remaining 9000 iters
                # (poses frozen). Aggressive LR for pose-only phase since geometry can't
                # compensate. Weak regularization to actually let pose move.
                "vggt_improved_v2_twostage_10k_sh3": {
                    "pose_reg_lambda": 1e-5,
                    "pose_grad_clip": 1.0,
                    "pose_lr_init": 5e-4,
                    "pose_lr_final": 1e-6,
                    "pose_lr_max_steps": 1000,  # match split iter
                    "pose_start_iter": 0,       # start immediately
                    "pose_end_iter": 1000,      # stop at split iter
                },
            }
            use_pose_refine = condition in POSE_PRESETS
            pose_params = POSE_PRESETS.get(condition)

            # Matched 10K sh3 conditions: use PAPER_GS_ARGS with 10K iters
            MATCHED_10K_SH3_ARGS = {
                **PAPER_GS_ARGS,
                "iterations": 10000,
                "test_iterations": [10000],
                "save_iterations": [10000],
                "position_lr_max_steps": 10000,
                "densify_until_iter": 10000,
            }
            # 30K iter variant (for LLFF where 10K is undertrained)
            MATCHED_30K_SH3_ARGS = {
                **PAPER_GS_ARGS,
                "iterations": 30000,
                "test_iterations": [30000],
                "save_iterations": [30000],
                "position_lr_max_steps": 30000,
                "densify_until_iter": 15000,
            }

            if condition.endswith("_30k_sh3"):
                gs_args_for_condition = MATCHED_30K_SH3_ARGS
            elif condition.endswith("_10k_sh3"):
                gs_args_for_condition = MATCHED_10K_SH3_ARGS
            elif condition in IMPROVED_EXTRA_ARGS:
                if args.profile_iters:
                    gs_args_for_condition = PROFILING_GS_ARGS
                else:
                    opt_iters = get_optimal_iterations(n_train)
                    gs_args_for_condition = {
                        **SPARSE_GS_ARGS,
                        "iterations": opt_iters,
                        "test_iterations": [opt_iters],
                        "save_iterations": [opt_iters],
                        "position_lr_max_steps": opt_iters,
                    }
            else:
                gs_args_for_condition = None
            extra_args_for_condition = BRIDGE_ABLATION_EXTRA_ARGS.get(condition)
            if extra_args_for_condition is None:
                extra_args_for_condition = IMPROVED_EXTRA_ARGS.get(condition)
            if extra_args_for_condition is not None:
                opt_iters = gs_args_for_condition["iterations"] if gs_args_for_condition else SPARSE_GS_ARGS["iterations"]
                extra_args_for_condition = scale_drop_schedule(extra_args_for_condition, opt_iters)

            if "_dense_" in condition and use_conf:
                print(
                    f"[SKIP] {condition} requested point confidence with a dense-replaced "
                    "point cloud; no matching dense confidence sidecar is available"
                )
                stages_meta["confidence_sidecar"] = {
                    **stages_meta.get("confidence_sidecar", {}),
                    "error": "point_conf_dense_incompatible",
                    "reason": "dense replacement changes point count/order",
                }
                run_meta["stages"] = stages_meta
                (run_root / "run_metadata.json").write_text(json.dumps(run_meta, indent=2))
                continue

            # VGGT + dense MVS diagnostic: feature extraction + matching + point_triangulator
            # with VGGT poses, then dense stereo. This gives COLMAP the sparse tracks it needs.
            if "_dense_" in condition and not condition.startswith("colmap_refpose"):
                print("[STAGE] VGGT poses + COLMAP triangulation + dense stereo (train cameras only)")
                dense_work = run_root / "dense_work"
                ensure_dir_clean(dense_work)
                # Symlink train-only images
                dense_images = dense_work / "images"
                dense_images.mkdir(parents=True, exist_ok=True)
                for tname in train_names:
                    src = final_base_dir / "images" / tname
                    dst = dense_images / tname
                    if src.exists():
                        os.symlink(src, dst)

                # Run feature extraction + matching + point_triangulator using VGGT poses
                # This creates proper sparse tracks that dense stereo needs for source images
                rwm = _get_rwm(gaussian_repo)
                full_cameras, full_images, full_points = rwm.read_model(
                    str(dataset_dir / "sparse" / "0")
                )
                # Extract VGGT poses for train cameras only
                refpose_meta = run_colmap_refpose(
                    scene_dir=dense_work,
                    reference_model_dir=dataset_dir / "sparse" / "0",
                    train_names=train_names,
                    gaussian_repo=gaussian_repo,
                    log_dir=run_root / "logs",
                    colmap_bin=str(args.colmap_bin),
                    gpu_index=args.gpu_index,
                    num_threads=int(colmap_cfg.get("threads", 8)),
                    colmap_cpu=colmap_cfg.get("cpu", False) or args.colmap_cpu,
                )
                stages_meta["vggt_triangulation"] = refpose_meta
                tri_steps = refpose_meta.get("steps", [])
                tri_steps_ok = isinstance(tri_steps, list) and tri_steps and all(
                    isinstance(step, dict) and step.get("code") == 0 for step in tri_steps
                )
                if not refpose_meta.get("success") or not tri_steps_ok:
                    print(f"[SKIP] VGGT pose triangulation failed for {scene} — aborting dense")
                    run_meta["stages"] = stages_meta
                    (run_root / "run_metadata.json").write_text(json.dumps(run_meta, indent=2))
                    continue

                # Run dense stereo on the triangulated model (has proper sparse tracks)
                fused_ply, dense_meta = run_dense_stereo(
                    scene_dir=dense_work,
                    log_dir=run_root / "logs",
                    colmap_bin=str(args.colmap_bin),
                    gpu_index=args.gpu_index,
                )
                stages_meta["dense_stereo"] = dense_meta
                if fused_ply and fused_ply.exists() and dense_meta.get("success"):
                    from plyfile import PlyData
                    plydata = PlyData.read(str(fused_ply))
                    vertex = plydata['vertex']
                    dense_xyz = np.column_stack([vertex['x'], vertex['y'], vertex['z']])
                    dense_rgb = np.column_stack([vertex['red'], vertex['green'], vertex['blue']])
                    dense_point_count = len(dense_xyz)
                    if "_dense_merge_" in condition:
                        vggt_xyz = np.asarray([pt.xyz for pt in full_points.values()], dtype=np.float32)
                        vggt_rgb = np.asarray([pt.rgb for pt in full_points.values()], dtype=np.uint8)
                        vggt_point_count = len(vggt_xyz)
                        if vggt_point_count > 0:
                            xyz = np.concatenate([vggt_xyz, dense_xyz], axis=0)
                            rgb = np.concatenate([vggt_rgb, dense_rgb], axis=0)
                        else:
                            xyz, rgb = dense_xyz, dense_rgb
                        stages_meta["dense_stereo"]["mode"] = "merge_vggt_and_dense"
                        stages_meta["dense_stereo"]["vggt_point_count"] = int(vggt_point_count)
                        stages_meta["dense_stereo"]["merged_point_count"] = int(len(xyz))
                        print(
                            f"[DENSE] Merging {vggt_point_count} VGGT points with "
                            f"{dense_point_count} dense points"
                        )
                    else:
                        xyz, rgb = dense_xyz, dense_rgb
                        stages_meta["dense_stereo"]["mode"] = "replace_with_dense"
                        stages_meta["dense_stereo"]["merged_point_count"] = int(len(xyz))
                    target_ply = dataset_dir / "sparse" / "0" / "points3D.ply"
                    sys.path.insert(0, str(gaussian_repo))
                    from scene.dataset_readers import storePly
                    storePly(str(target_ply), xyz, rgb)
                    stages_meta["dense_stereo"]["dense_point_count"] = dense_point_count
                    print(f"[DENSE] Wrote {len(xyz)} initializer points from VGGT poses")
                else:
                    print(f"[SKIP] Dense stereo failed on VGGT poses for {scene} — aborting")
                    run_meta["stages"] = stages_meta
                    (run_root / "run_metadata.json").write_text(json.dumps(run_meta, indent=2))
                    continue

            model_dir = run_root / "model"
            train_meta, metrics_data = run_training_and_eval(
                dataset_dir=dataset_dir,
                model_dir=model_dir,
                gaussian_repo=gaussian_repo,
                python_exe=python_exe,
                gpu_index=args.gpu_index,
                resolution=resolution_override,
                use_confidence=use_conf,
                llffhold=0,
                use_pseudo_views=use_pseudo,
                refine_poses=use_pose_refine,
                pose_params=pose_params,
                depths_dir=depths_dir_name,
                gs_args=gs_args_for_condition,
                extra_args=extra_args_for_condition,
            )
            stages_meta.update(train_meta)

            run_meta["stages"] = stages_meta
            run_meta["metrics"] = metrics_data
            (run_root / "run_metadata.json").write_text(json.dumps(run_meta, indent=2))

            if metrics_data:
                metrics_out = run_root / "metrics.json"
                metrics_out.write_text(json.dumps(metrics_data, indent=2))
                row = {
                    "dataset": dataset_name,
                    "scene": scene,
                    "setting": setting_label,
                    "condition": condition,
                    "PSNR": metrics_data.get("PSNR"),
                    "SSIM": metrics_data.get("SSIM"),
                    "LPIPS": metrics_data.get("LPIPS"),
                    "train_elapsed_s": stages_meta.get("train", {}).get("elapsed_s"),
                    "render_elapsed_s": stages_meta.get("render", {}).get("elapsed_s"),
                    "metrics_elapsed_s": stages_meta.get("metrics", {}).get("elapsed_s"),
                    "colmap_elapsed_s": stages_meta.get("colmap", {}).get("elapsed_s"),
                    "vggt_elapsed_s": stages_meta.get("vggt_export", {}).get("elapsed_s"),
                    "vggt_ba_elapsed_s": stages_meta.get("vggt_ba", {}).get("elapsed_s"),
                    "mast3r_elapsed_s": stages_meta.get("mast3r_sfm", {}).get("elapsed_s"),
                    "pseudo_gen_elapsed_s": stages_meta.get("pseudo_gen", {}).get("elapsed_s"),
                }
                update_metrics_csv(metrics_csv, row, fields)


def main():
    parser = argparse.ArgumentParser(description="Run view count sweep")
    parser.add_argument(
        "--mode",
        choices=["default", "paper_eval", "sparse_fixed", "sparse_fixed_trainonly_init_ref"],
        default="default",
        help=(
            "default = legacy sweep; paper_eval = ICPR-ready protocol; "
            "sparse_fixed = fixed test split; sparse_fixed_trainonly_init_ref = "
            "train-only init aligned to full reference model"
        ),
    )
    parser.add_argument("--paper_config", type=Path, default=Path("configs/paper_setup.yaml"),
                       help="Path to paper setup YAML (used when --mode paper_eval)")
    parser.add_argument("--gpu_index", type=int, default=0, help="GPU index for VRAM polling")
    parser.add_argument("--view_counts", type=int, nargs="+",
                       default=[3, 6, 12, 15, 18, 21],
                       help="View counts to test")
    parser.add_argument("--data_root", type=Path,
                       default=Path("./data"),
                       help="Data root directory")
    parser.add_argument("--output_root", type=Path,
                       default=Path("./outputs"),
                       help="Output root directory")
    parser.add_argument("--scene", type=str, default="counter",
                        help="Scene name")
    parser.add_argument("--dataset", type=str, default="mipnerf360",
                        help="Dataset key under cfg['datasets'] (e.g. mipnerf360, llff, tanksandtemples)")
    parser.add_argument("--pose_source", type=str, default="vggt",
                        choices=["vggt", "oracle"],
                        help="Pose source for train cameras after alignment. "
                             "'vggt' = Sim(3)-aligned VGGT poses (default). "
                             "'oracle' = swap train poses for reference COLMAP poses (E1.2 diagnostic).")
    parser.add_argument("--pose_noise_rot_deg", type=float, default=0.0,
                        help="E3 sensitivity: stddev of rotation noise to add to train poses, in degrees.")
    parser.add_argument("--pose_noise_trans_pct", type=float, default=0.0,
                        help="E3 sensitivity: stddev of translation noise as %% of scene radius.")
    parser.add_argument("--pose_noise_seed", type=int, default=0,
                        help="Seed for pose noise perturbation (E3).")
    parser.add_argument("--bridge_no_median_norm", action="store_true",
                        help="E5 bridge ablation: skip the median-radial point/pose normalization in the VGGT pipeline.")
    parser.add_argument("--point_source", type=str, default="vggt",
                        choices=["vggt", "colmap_sparse"],
                        help="Seed point cloud source. 'vggt' = aligned VGGT cloud (default). "
                             "'colmap_sparse' = COLMAP point_triangulator on train images with oracle cameras (E2.2).")
    parser.add_argument("--colmap_bin", type=str,
                        default="colmap",
                        help="Path to COLMAP binary used by C2 (point_injection).")
    parser.add_argument("--skip_colmap", action="store_true",
                        help="Skip COLMAP baselines (expect failures on sparse views)")
    parser.add_argument("--iterations", type=int, default=30000,
                        help="Training iterations for 3DGS runs")
    parser.add_argument("--test_iterations", type=int, default=None,
                        help="Additional final test iteration (optional)")
    parser.add_argument("--psnr_interval", type=int, default=100,
                        help="Evaluate PSNR every N iterations (<=0 to disable interval checks)")
    parser.add_argument("--save_iterations", type=int, default=None,
                        help="Iterations at which to save checkpoints (None to disable)")
    parser.add_argument("--selection_strategy", choices=["pose", "consecutive", "hybrid"], default="pose",
                        help="View selection strategy")
    parser.add_argument("--selection_seed", type=int, default=42,
                        help="Random seed for selection strategies")
    parser.add_argument("--n_train", type=int, nargs="+", default=[12, 24],
                        help="Sparse train counts for fixed-test runs")
    parser.add_argument("--holdout", type=int, default=8,
                        help="Holdout interval for fixed-test runs (every Nth image)")
    parser.add_argument("--conditions", nargs="+", default=None,
                        help="Conditions to run (e.g., vggt_vanilla vggt_confidence colmap)")
    parser.add_argument("--images_subdir", type=str, default=None,
                        help="Override images subdir (e.g., images_4) for sparse_fixed runs")
    parser.add_argument("--split_manifest", type=Path, default=None,
                        help="CSV manifest with exact train/test filenames for fixed-test train-only runs")
    parser.add_argument("--split_dataset", type=str, default=None,
                        help="Dataset label to match inside --split_manifest when it differs from --dataset")
    parser.add_argument("--split_scene", type=str, default=None,
                        help="Scene label to match inside --split_manifest when it differs from --scene")
    parser.add_argument("--radial_camera_policy", choices=["fail", "undistort"], default="fail",
                        help="How sparse_fixed_trainonly_init_ref handles SIMPLE_RADIAL reference cameras. "
                             "Default fail preserves the existing no-silent-distortion-drop behavior; "
                             "undistort runs COLMAP image_undistorter before staging images.")
    parser.add_argument("--vggt_force_cpu", action="store_true",
                        help="Force VGGT inference on CPU (avoid GPU OOM)")
    parser.add_argument("--colmap_threads", type=int, default=8,
                        help="Number of CPU threads for COLMAP feature extraction/matching")
    parser.add_argument("--colmap_cpu", action="store_true",
                        help="Force COLMAP feature extraction/matching on CPU (disable CUDA)")
    parser.add_argument("--pseudo_num_views", type=int, default=16,
                        help="Number of pseudo-views to generate for vggt_vanilla_pseudo condition (default: 16)")
    parser.add_argument("--profile_iters", action="store_true",
                        help="Use 30K iters with frequent eval for iteration profiling")

    args = parser.parse_args()

    if args.mode == "paper_eval":
        run_paper_eval(args)
        return
    if args.mode == "sparse_fixed":
        run_sparse_fixed(args)
        return
    if args.mode == "sparse_fixed_trainonly_init_ref":
        run_sparse_fixed_trainonly_init_ref(args)
        return

    # Get python executable from current environment
    python_exe = Path(sys.executable)
    print(f"Using Python: {python_exe}")

    # Ensure headless COLMAP runs
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    # Build evaluation iteration schedule
    eval_iterations: list[int] = []
    if args.psnr_interval and args.psnr_interval > 0:
        eval_iterations.extend(range(args.psnr_interval, args.iterations + 1, args.psnr_interval))
    if args.test_iterations:
        eval_iterations.append(args.test_iterations)
    eval_iterations = sorted(set(eval_iterations))

    # Prepare save iteration list (final checkpoint only if not provided)
    save_iterations: list[int] = []
    if args.save_iterations:
        save_iterations.append(args.save_iterations)
    elif args.iterations:
        save_iterations.append(args.iterations)

    source_images = args.data_root / args.scene / "images_4"
    results_summary = []

    for n_views in args.view_counts:
        print(f"\n{'#'*80}")
        print(f"# VIEW COUNT: {n_views}")
        print(f"{'#'*80}\n")

        # Create subset directory
        subset_dir = args.output_root / f"view_sweep_{args.scene}_{n_views}views"
        subset_dir.mkdir(parents=True, exist_ok=True)

        # Select views using intelligent pose-based sampling
        subset_images_dir = subset_dir / "images"
        if not subset_images_dir.exists():
            selected_images, selected_indices = select_views(
                source_images,
                subset_images_dir,
                n_views,
                args.data_root,
                args.scene,
                strategy=args.selection_strategy,
                seed=args.selection_seed,
            )
        else:
            selected_images = list(subset_images_dir.iterdir())
            metadata_path = subset_dir / "selection_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r", encoding="utf-8") as fh:
                    metadata = json.load(fh)
                selected_indices = metadata.get("indices")
                strategy_logged = metadata.get("strategy")
                if strategy_logged and strategy_logged != args.selection_strategy:
                    print(
                        f"[INFO] Existing selection uses strategy '{strategy_logged}'. "
                        f"Override requested strategy '{args.selection_strategy}' will be ignored."
                    )
            else:
                selected_indices = None

        if selected_indices is None:
            raise RuntimeError("Selection indices not found; ensure selection_metadata.json is present")

        # Ensure indices are integers (JSON may parse as list of floats)
        selected_indices = [int(i) for i in selected_indices]

        # COLMAP baseline
        colmap_success = False
        if not args.skip_colmap:
            colmap_dataset_dir = args.output_root / "colmap_runs" / f"{args.scene}_{n_views}views"
            colmap_dataset_dir.mkdir(parents=True, exist_ok=True)

            # Copy images
            colmap_images_dir = colmap_dataset_dir / "images"
            if not colmap_images_dir.exists():
                shutil.copytree(subset_images_dir, colmap_images_dir)

            colmap_success = run_colmap_baseline(
                colmap_dataset_dir,
                f"{n_views}views",
                num_threads=args.colmap_threads,
                use_gpu=not args.colmap_cpu,
            )

            if colmap_success:
                colmap_model_dir = args.output_root / "trained_models" / f"{args.scene}_colmap_{n_views}views"
                train_success, curves = run_training(
                    colmap_dataset_dir,
                    colmap_model_dir,
                    use_confidence=False,
                    python_exe=python_exe,
                    iterations=args.iterations,
                    test_iterations=eval_iterations,
                    save_iterations=save_iterations,
                )

                if train_success:
                    metrics = extract_metrics(colmap_model_dir / "test")
                    if not metrics:
                        metrics = {
                            "PSNR": None,
                            "SSIM": None,
                            "LPIPS": None,
                            "curve": None,
                        }
                    metrics["psnr_curve"] = curves
                    if metrics:
                        results_summary.append({
                            "scene": args.scene,
                            "method": "COLMAP",
                            "n_views": n_views,
                            **metrics
                        })
                    else:
                        print(f"[WARNING] Metrics not found for COLMAP {n_views} views")

        # VGGT inference - create dedicated scene directory
        vggt_scene_dir = args.data_root / f"{args.scene}_{n_views}views"
        vggt_scene_dir.mkdir(parents=True, exist_ok=True)

        # Prepare images directories (images and images_4 for compatibility)
        scene_images_dir = vggt_scene_dir / "images"
        scene_images_down_dir = vggt_scene_dir / "images_4"

        for target_dir in (scene_images_dir, scene_images_down_dir):
            if target_dir.exists():
                shutil.rmtree(target_dir)
            target_dir.mkdir(parents=True, exist_ok=True)
            for img in subset_images_dir.iterdir():
                shutil.copy2(img, target_dir / img.name)

        # Copy filtered poses_bounds.npy to scene directory for reference
        poses_bounds_src = args.data_root / args.scene / "poses_bounds.npy"
        if poses_bounds_src.exists():
            poses, bounds, _ = load_poses_bounds(poses_bounds_src)
            filtered = np.concatenate([
                poses[selected_indices].reshape(len(selected_indices), -1),
                bounds[selected_indices]
            ], axis=1)
            np.save(vggt_scene_dir / "poses_bounds.npy", filtered)

        vggt_config = args.output_root / f"config_{args.scene}_vggt_{n_views}views.yaml"
        if n_views >= 48:
            vggt_batch = 2
        elif n_views >= 32:
            vggt_batch = 4
        else:
            vggt_batch = 6
        with open(vggt_config, 'w', encoding="utf-8") as f:
            yaml_text = textwrap.dedent(
                f"""
                scene: {args.scene}_{n_views}views
                data_root: {args.data_root}
                output_root: {args.output_root}
                pipeline: vggt_3dgs
                vggt:
                  repo: ./repos/vggt
                  checkpoint: facebook/VGGT-1B
                  batch_size: {vggt_batch}
                  max_images: {n_views}
                  point_subsample: 50
                training:
                  skip: true
                """
            ).strip() + "\n"
            f.write(yaml_text)

        force_cpu = n_views >= 48
        vggt_success = run_vggt_inference(vggt_config, f"{n_views}views", python_exe, force_cpu=force_cpu)

        if vggt_success:
            vggt_output_dir = args.output_root / f"{args.scene}_{n_views}views"
            vggt_dataset_dir = vggt_output_dir / "dataset"

            # VGGT vanilla
            vggt_vanilla_model = args.output_root / "trained_models" / f"{args.scene}_vggt_vanilla_{n_views}views"
            train_success, curves = run_training(
                vggt_dataset_dir,
                vggt_vanilla_model,
                use_confidence=False,
                python_exe=python_exe,
                iterations=args.iterations,
                test_iterations=eval_iterations,
                save_iterations=save_iterations,
            )

            if train_success:
                metrics = extract_metrics(vggt_vanilla_model / "test")
                if not metrics:
                    metrics = {
                        "PSNR": None,
                        "SSIM": None,
                        "LPIPS": None,
                        "curve": None,
                    }
                metrics["psnr_curve"] = curves
                if metrics:
                    results_summary.append({
                        "scene": args.scene,
                        "method": "VGGT_vanilla",
                        "n_views": n_views,
                        **metrics
                    })
                else:
                    print(f"[WARNING] Metrics not found for VGGT_vanilla {n_views} views")

            # VGGT confidence-aware
            vggt_conf_model = args.output_root / "trained_models" / f"{args.scene}_vggt_confidence_{n_views}views"
            train_success, curves = run_training(
                vggt_dataset_dir,
                vggt_conf_model,
                use_confidence=True,
                python_exe=python_exe,
                iterations=args.iterations,
                test_iterations=eval_iterations,
                save_iterations=save_iterations,
            )

            if train_success:
                metrics = extract_metrics(vggt_conf_model / "test")
                if not metrics:
                    metrics = {
                        "PSNR": None,
                        "SSIM": None,
                        "LPIPS": None,
                        "curve": None,
                    }
                metrics["psnr_curve"] = curves
                if metrics:
                    results_summary.append({
                        "scene": args.scene,
                        "method": "VGGT_confidence",
                        "n_views": n_views,
                        **metrics
                    })
                else:
                    print(f"[WARNING] Metrics not found for VGGT_confidence {n_views} views")

    # Save summary
    summary_path = args.output_root / "view_sweep_results.json"
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)

    print(f"\n{'='*80}")
    print("SWEEP COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {summary_path}")

    # Print summary table
    print("\n" + "="*80)
    print(f"{'Scene':<12} {'Method':<20} {'Views':<8} {'PSNR':<10} {'SSIM':<10} {'LPIPS':<10}")
    print("="*80)
    for result in results_summary:
        psnr = result.get("PSNR")
        ssim = result.get("SSIM")
        lpips = result.get("LPIPS")
        psnr_str = f"{psnr:.2f}" if isinstance(psnr, (int, float)) and psnr is not None else "NA"
        ssim_str = f"{ssim:.4f}" if isinstance(ssim, (int, float)) and ssim is not None else "NA"
        lpips_str = f"{lpips:.4f}" if isinstance(lpips, (int, float)) and lpips is not None else "NA"

        print(f"{result['scene']:<12} {result['method']:<20} {result['n_views']:<8} "
              f"{psnr_str:<10} {ssim_str:<10} {lpips_str:<10}")
    print("="*80)


if __name__ == "__main__":
    main()
