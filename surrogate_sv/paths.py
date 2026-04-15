from pathlib import Path
from typing import Iterable, Optional

from .config import DEFAULT_OUTPUT_ROOT_CANDIDATES


def resolve_output_root(candidates: Optional[Iterable[Path]] = None) -> Path:
    """Resolve an artifact root that works in Modal and locally."""
    path_candidates = tuple(candidates) if candidates is not None else DEFAULT_OUTPUT_ROOT_CANDIDATES
    for path in path_candidates:
        try:
            path.mkdir(parents=True, exist_ok=True)
            return path
        except Exception:
            continue
    raise RuntimeError("Could not create any output directory for artifacts.")


def resolve_classifier_bundle_root() -> Path:
    root = resolve_output_root()
    bundle_root = root / "classifier_bundles"
    bundle_root.mkdir(parents=True, exist_ok=True)
    return bundle_root
