from __future__ import annotations

import sys
from pathlib import Path


FALLBACK_SITE_PACKAGES = [
    "/linyuanping/miniconda3/envs/lf/lib/python3.10/site-packages",
    "/linyuanping/miniconda3/envs/gridguard/lib/python3.10/site-packages",
]

for candidate in FALLBACK_SITE_PACKAGES:
    if Path(candidate).is_dir() and candidate not in sys.path:
        sys.path.append(candidate)
