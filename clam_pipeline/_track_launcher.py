from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    src = Path(__file__).parent.parent / "tracking_from_video" / "src"
    script = src / "tracking_algae.py"
    # Pass the default config path as hint if user doesn't specify --config
    default_config = src.parent / "configs" / "config_tracking.json"
    cmd = [sys.executable, str(script)] + sys.argv[1:]
    # Inject src dir so 'from tracking_core import ...' resolves correctly
    env_patch = {"PYTHONPATH": str(src)}
    import os
    env = {**os.environ, "PYTHONPATH": str(src) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    subprocess.run(cmd, env=env)
