import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


def run_command(cmd: List[str], workdir: Optional[Path] = None, env: Optional[Dict[str, str]] = None) -> None:
    """Execute a shell command with logging."""
    workdir = workdir or Path.cwd()
    print(f"[PIPELINE] Running: {' '.join(cmd)} (cwd={workdir})")
    try:
        subprocess.run(cmd, cwd=str(workdir), env=env, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Command failed with exit code {exc.returncode}: {' '.join(cmd)}") from exc


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_binary(name: str, env_var: str, default: str) -> str:
    """Resolve the path to an external binary, following env var overrides first."""
    if env_var in os.environ:
        return os.environ[env_var]
    return default


def scene_root(data_root: Path, scene: str) -> Path:
    root = data_root / scene
    if not root.exists():
        raise FileNotFoundError(f"Scene '{scene}' not found under {data_root}")
    return root


@dataclass
class PipelineConfig:
    scene: str
    data_root: Path
    output_root: Path
    training: Dict[str, Any] = field(default_factory=dict)
    colmap: Dict[str, Any] = field(default_factory=dict)
    vggt: Dict[str, Any] = field(default_factory=dict)


class PipelineBase:
    """Abstract base class for all pipelines."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.scene_dir = scene_root(config.data_root, config.scene)
        self.output_dir = ensure_dir(config.output_root / config.scene)
        self.debug_dir = ensure_dir(self.output_dir / "debug")
        self.images_dir = self.scene_dir / "images"
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Expected images under {self.images_dir}")

    def run(self) -> None:
        """Execute pipeline stages in order."""
        for step in self.steps():
            print(f"[PIPELINE] >>> {step.__name__}")
            step()

    # Individual pipelines override this to return a list of bound methods.
    def steps(self) -> List[Any]:  # pragma: no cover - to be overridden
        raise NotImplementedError

