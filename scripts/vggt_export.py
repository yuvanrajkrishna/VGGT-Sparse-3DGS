import argparse
import json
from pathlib import Path

import numpy as np
import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def run(
    images_dir: Path,
    output_dir: Path,
    checkpoint: str | None,
    batch_size: int,
    device: str,
    dtype: torch.dtype,
    max_images: int | None,
) -> Path:
    image_paths = sorted(
        [p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    if max_images is not None:
        image_paths = image_paths[:max_images]

    output_dir.mkdir(parents=True, exist_ok=True)
    output_npz = output_dir / "predictions.npz"

    model = VGGT.from_pretrained(checkpoint or "facebook/VGGT-1B")
    model = model.to(device)
    model.eval()

    Ks = []
    poses = []
    depth_maps = []
    point_maps = []

    images_tensor = load_and_preprocess_images([str(p) for p in image_paths])
    images_tensor = images_tensor.to(device)

    with torch.no_grad():
        kwargs = {}
        if device == "cuda":
            autocast = torch.cuda.amp.autocast(dtype=dtype)
        else:
            autocast = torch.autocast(device_type="cpu", dtype=dtype)
        with autocast:
            predictions = model(images_tensor)

    pose_enc = predictions["pose_enc"]  # shape [1, N, 9]
    extrinsics, intrinsics = pose_encoding_to_extri_intri(pose_enc, images_tensor.shape[-2:])
    extrinsics = extrinsics.squeeze(0).cpu().numpy()
    intrinsics = intrinsics.squeeze(0).cpu().numpy()
    depth_maps = predictions.get("depth", torch.empty(0)).squeeze(0).cpu().numpy()
    point_maps = predictions.get("world_points", predictions.get("points", torch.empty(0))).squeeze(0).cpu().numpy()

    # Extract confidence values (critical for confidence-aware initialization)
    depth_conf = predictions.get("depth_conf", torch.empty(0)).squeeze(0).cpu().numpy()
    points_conf = predictions.get("world_points_conf", torch.empty(0)).squeeze(0).cpu().numpy()

    np.savez(
        output_npz,
        Ks=intrinsics,
        extrinsics=extrinsics,
        depths=depth_maps,
        points3d=point_maps,
        depth_conf=depth_conf,
        points_conf=points_conf,
    )
    meta = {"images": [p.name for p in image_paths]}
    (output_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    return output_npz


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VGGT inference and export predictions")
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_images", type=int, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        major, _ = torch.cuda.get_device_capability()
        dtype = torch.bfloat16 if major >= 8 else torch.float16
    else:
        dtype = torch.float32
    run(args.images, args.output, args.checkpoint, args.batch_size, device, dtype, args.max_images)


if __name__ == "__main__":
    main()
