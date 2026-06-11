#!/usr/bin/env python3
"""
Smart view selection based on camera pose diversity.

Uses farthest-point sampling in SE(3) pose space to maximize coverage
and avoid redundant nearby views.
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple


def load_poses_bounds(poses_bounds_path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load camera poses from poses_bounds.npy file.

    Returns:
        poses: (N, 3, 5) array - [R|t] matrices plus hwf
        bounds: (N, 2) array - near/far bounds
        image_names: List of image filenames (sorted)
    """
    data = np.load(poses_bounds_path)  # Shape: (N, 17)
    N = data.shape[0]

    # poses_bounds format: [15 pose params, 2 bounds]
    # Pose: [3x4 matrix (12 values) + hwf (3 values)] = 15 values
    poses = data[:, :15].reshape(N, 3, 5)  # [R|t|hwf]
    bounds = data[:, 15:17]  # near, far

    # Image names: assume they're in sorted order matching poses
    # (This is a convention in NeRF datasets)
    parent_dir = poses_bounds_path.parent
    images_dir = parent_dir / "images_4"

    if images_dir.exists():
        image_names = sorted([p.name for p in images_dir.iterdir()])
    else:
        # Fallback: generate placeholder names
        image_names = [f"image_{i:03d}.jpg" for i in range(N)]

    return poses, bounds, image_names


def pose_distance(pose1: np.ndarray, pose2: np.ndarray,
                 translation_weight: float = 1.0,
                 rotation_weight: float = 0.5) -> float:
    """
    Compute distance between two camera poses in SE(3).

    Args:
        pose1, pose2: (3, 5) arrays containing [R|t|hwf]
        translation_weight: Weight for translation component
        rotation_weight: Weight for rotation component

    Returns:
        Distance metric combining translation and rotation differences
    """
    # Extract rotation (first 3 columns) and translation (4th column)
    R1, t1 = pose1[:, :3], pose1[:, 3]
    R2, t2 = pose2[:, :3], pose2[:, 3]

    # Translation distance (Euclidean)
    trans_dist = np.linalg.norm(t1 - t2)

    # Rotation distance (geodesic on SO(3))
    # Use Frobenius norm of R1^T @ R2 - I as a proxy
    R_diff = R1.T @ R2
    rot_dist = np.linalg.norm(R_diff - np.eye(3), 'fro')

    # Alternative rotation metric: trace of (I - R1^T @ R2)
    # rot_dist = np.sqrt(2 * (3 - np.trace(R_diff)))

    return translation_weight * trans_dist + rotation_weight * rot_dist


def farthest_point_sampling(poses: np.ndarray, n_samples: int,
                           translation_weight: float = 1.0,
                           rotation_weight: float = 0.5,
                           seed: int = 42) -> np.ndarray:
    """
    Select n_samples views using farthest-point sampling for maximum diversity.

    Algorithm:
    1. Start with random seed view
    2. Iteratively select view farthest from all previously selected views
    3. Ensures maximum coverage of pose space

    Args:
        poses: (N, 3, 5) array of camera poses
        n_samples: Number of views to select
        translation_weight: Weight for translation distance
        rotation_weight: Weight for rotation distance
        seed: Random seed for initial view

    Returns:
        indices: (n_samples,) array of selected view indices
    """
    N = len(poses)

    if n_samples > N:
        raise ValueError(f"Cannot select {n_samples} views from {N} total")

    if n_samples == N:
        return np.arange(N)

    np.random.seed(seed)

    # Initialize with random view
    selected_indices = [np.random.randint(N)]
    remaining_indices = set(range(N)) - set(selected_indices)

    # Iteratively select farthest view
    for _ in range(n_samples - 1):
        max_min_distance = -1
        farthest_idx = None

        for idx in remaining_indices:
            # Compute minimum distance to any selected view
            min_distance = min(
                pose_distance(poses[idx], poses[sel_idx],
                            translation_weight, rotation_weight)
                for sel_idx in selected_indices
            )

            # Track view with maximum minimum distance
            if min_distance > max_min_distance:
                max_min_distance = min_distance
                farthest_idx = idx

        selected_indices.append(farthest_idx)
        remaining_indices.remove(farthest_idx)

    return np.array(selected_indices)


def select_views_smart(data_root: Path, scene: str, n_views: int,
                      translation_weight: float = 1.0,
                      rotation_weight: float = 0.5,
                      seed: int = 42) -> Tuple[np.ndarray, List[str]]:
    """
    Select n_views from scene using intelligent pose-based sampling.

    Args:
        data_root: Root data directory
        scene: Scene name (e.g., "bicycle")
        n_views: Number of views to select
        translation_weight: Weight for translation diversity
        rotation_weight: Weight for rotation diversity
        seed: Random seed

    Returns:
        selected_indices: (n_views,) array of indices
        selected_images: List of selected image filenames
    """
    poses_bounds_path = data_root / scene / "poses_bounds.npy"

    if not poses_bounds_path.exists():
        raise FileNotFoundError(f"poses_bounds.npy not found: {poses_bounds_path}")

    poses, bounds, image_names = load_poses_bounds(poses_bounds_path)

    print(f"Loaded {len(poses)} camera poses from {scene}")
    print(f"Selecting {n_views} views using farthest-point sampling...")

    indices = farthest_point_sampling(
        poses, n_views,
        translation_weight=translation_weight,
        rotation_weight=rotation_weight,
        seed=seed
    )

    # Sort indices for reproducibility
    indices = np.sort(indices)

    selected_images = [image_names[i] for i in indices]

    print(f"Selected views: {indices}")
    print(f"Image names: {selected_images[:5]}... ({len(selected_images)} total)")

    return indices, selected_images


def main():
    """Test smart view selection."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, default=Path("./data"))
    parser.add_argument("--scene", type=str, default="bicycle")
    parser.add_argument("--n_views", type=int, default=24)
    parser.add_argument("--translation_weight", type=float, default=1.0)
    parser.add_argument("--rotation_weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    indices, images = select_views_smart(
        args.data_root, args.scene, args.n_views,
        args.translation_weight, args.rotation_weight, args.seed
    )

    print(f"\nSelected {len(indices)} views with maximum diversity")
    print(f"Indices: {indices.tolist()}")


if __name__ == "__main__":
    main()
