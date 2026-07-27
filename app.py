"""Launcher for the Streamlit GUI: `streamlit run app.py`.

Streamlit executes this file as a script on every rerun, so the package import
sits behind a sys.path bootstrap and the app is invoked explicitly rather than at
import time (an imported module is cached and would not re-render).
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from smart_analytics.app.main import main  # noqa: E402  (needs the path bootstrap above)

main()
