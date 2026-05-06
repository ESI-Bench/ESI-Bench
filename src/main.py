from __future__ import annotations

import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    active_explore_dir = Path(__file__).resolve().parent / "active_explore"
    sys.path.insert(0, str(active_explore_dir))

    import pipeline

    return pipeline.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
