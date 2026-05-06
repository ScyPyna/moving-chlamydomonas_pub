from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    app = Path(__file__).parent.parent / "app.py"
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app)] + sys.argv[1:]
    )
