"""Export clustered images to folders."""

import logging
import shutil
from pathlib import Path

import numpy as np

log = logging.getLogger("cluster")


def _label_to_folder(label: int) -> str:
    """Map label to folder name. Label -1 → 'unclustered'."""
    if label == -1:
        return "unclustered"
    return f"cluster_{label}"


def export_clusters(
    image_paths: list[Path],
    labels: np.ndarray,
    output_dir: Path,
    dry_run: bool = False,
    export_unclustered: bool = True,
) -> dict[int, list[tuple[int, Path]]]:
    """Organize images into cluster folders.

    Parameters
    ----------
    export_unclustered : bool
        If True (default), label=-1 (noise) images are exported to unclustered/.
        If False, noise images are skipped entirely.
    """
    groups: dict[int, list[tuple[int, Path]]] = {}
    for idx, (label, path) in enumerate(zip(labels, image_paths, strict=False)):
        if not export_unclustered and label == -1:
            continue
        groups.setdefault(int(label), []).append((idx, path))

    if dry_run:
        log.info("=== DRY RUN — no files will be written ===")
        for label in sorted(groups):
            items = groups[label]
            folder = _label_to_folder(label)
            log.info("  %s/  → %d images", folder, len(items))
        return groups

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for label, items in groups.items():
        folder = _label_to_folder(label)
        dir_name = output_dir / folder
        dir_name.mkdir(exist_ok=True)

        for idx, src_path in items:
            dest = dir_name / src_path.name
            if dest.exists():
                dest = dir_name / f"{src_path.stem}_{idx}{src_path.suffix}"
            shutil.copy2(str(src_path), str(dest))

    n_outliers = len(groups.get(-1, []))
    if n_outliers:
        log.info("Unclustered images: %d → unclustered/", n_outliers)

    if not dry_run:
        log.info("Exported %d images to %s", sum(len(v) for v in groups.values()), output_dir)
    else:
        log.info(
            "Dry run: would export %d images to %s",
            sum(len(v) for v in groups.values()),
            output_dir,
        )
    return groups
