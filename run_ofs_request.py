"""Started by the OFS extension: python run_ofs_request.py <request.json> (works from any working directory)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from funscript_ai import ofs_request  # noqa: E402

ofs_request.main()
