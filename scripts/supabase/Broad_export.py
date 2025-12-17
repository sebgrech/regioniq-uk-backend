#!/usr/bin/env python3
"""
Compatibility shim.

The Supabase export/sync entrypoint lives at `scripts/export/Broad_export.py`,
but older pipeline configs referenced `scripts/supabase/Broad_export.py`.
This wrapper preserves that path without duplicating logic.
"""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    target = Path(__file__).resolve().parents[1] / "export" / "Broad_export.py"
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()


