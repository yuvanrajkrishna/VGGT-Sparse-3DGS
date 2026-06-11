#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

import yaml

from pipelines.base import PipelineConfig
from pipelines.colmap_3dgs import Colmap3DGSPipeline
from pipelines.vggt_3dgs import VGGTThreeDGSPipeline
from pipelines.evaluation import evaluate_scene


PIPELINES = {
    "colmap_3dgs": Colmap3DGSPipeline,
    "vggt_3dgs": VGGTThreeDGSPipeline,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 3DGS pipelines.")
    parser.add_argument("--config", required=True, help="Path to pipeline config (YAML)")
    parser.add_argument("--pipeline", help="Override pipeline key from config")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation after training")
    parser.add_argument(
        "--ground-truth",
        dest="ground_truth",
        help="Folder with ground-truth views (used when --evaluate)",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def instantiate_pipeline(cfg_dict: dict, override: str | None) -> Colmap3DGSPipeline:
    scene = cfg_dict["scene"]
    data_root = Path(cfg_dict.get("data_root", "./data")).resolve()
    output_root = Path(cfg_dict.get("output_root", "./outputs")).resolve()

    config = PipelineConfig(
        scene=scene,
        data_root=data_root,
        output_root=output_root,
        training=cfg_dict.get("training", {}),
        colmap=cfg_dict.get("colmap", {}),
        vggt=cfg_dict.get("vggt", {}),
    )

    pipeline_key = override or cfg_dict.get("pipeline")
    if pipeline_key not in PIPELINES:
        raise KeyError(f"Unknown pipeline '{pipeline_key}'. Options: {', '.join(PIPELINES)}")
    pipeline_cls = PIPELINES[pipeline_key]
    return pipeline_cls(config)


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config).resolve()
    cfg_dict = load_config(cfg_path)
    pipeline = instantiate_pipeline(cfg_dict, args.pipeline)

    print(f"[CLI] Running pipeline: {pipeline.__class__.__name__}")
    pipeline.run()

    if args.evaluate:
        renders = pipeline.output_dir / "renders"
        if not renders.exists():
            raise FileNotFoundError("Renders folder not found; ensure 3DGS exports renders under 'renders'.")
        if not args.ground_truth:
            raise ValueError("--ground-truth folder required for evaluation")
        metrics = evaluate_scene(renders, Path(args.ground_truth).resolve())
        print("[CLI] Evaluation metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.3f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"[CLI] Error: {exc}")
        sys.exit(1)
