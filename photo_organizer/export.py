"""Export clustered images to folders."""

import logging
import shutil
from pathlib import Path

import numpy as np

log = logging.getLogger("cluster")


def export_clusters(
    image_paths: list[Path],
    labels: np.ndarray,
    output_dir: Path,
    dry_run: bool = False,
) -> dict[int, list[tuple[int, Path]]]:
    """Organize images into cluster folders."""
    groups: dict[int, list[tuple[int, Path]]] = {}
    for idx, (label, path) in enumerate(zip(labels, image_paths, strict=False)):
        groups.setdefault(int(label), []).append((idx, path))

    if dry_run:
        log.info("=== DRY RUN — no files will be written ===")
        for label in sorted(groups):
            items = groups[label]
            log.info("  cluster_%d/  → %d images", label, len(items))
        return groups

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for label, items in groups.items():
        dir_name = output_dir / f"cluster_{label}"
        dir_name.mkdir(exist_ok=True)

        for idx, src_path in items:
            dest = dir_name / src_path.name
            if dest.exists():
                dest = dir_name / f"{src_path.stem}_{idx}{src_path.suffix}"
            shutil.copy2(str(src_path), str(dest))

    log.info("Exported %d images to %s", len(image_paths), output_dir)
    return groups
