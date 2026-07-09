"""Export clustered images to folders."""

import logging
import os
import shutil
import time
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


def folders_export(
    image_paths: list[Path],
    prefixes: dict[int, str],
    output_dir: Path,
    dry_run: bool = False,
) -> int:
    """Create prefix/dir hierarchy and place images with original names.

    Prefix "c1_c2_" → output/c1/c2/IMG.jpg
    Prefix "" → output/unclustered/IMG.jpg
    Handles collisions by appending _N to stem.
    Returns count of moved files.
    """
    if dry_run:
        log.info("=== DRY RUN — no files will be written ===")
        for idx, src_path in enumerate(image_paths):
            prefix = prefixes.get(idx, "")
            if prefix:
                parts = [p for p in prefix.rstrip("_").split("_") if p]
                dest_dir = output_dir / Path(*parts)
            else:
                dest_dir = output_dir / "unclustered"
            log.info("  would move %s → %s", src_path, dest_dir / src_path.name)
        return 0

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    failures: list[tuple[Path, Exception]] = []
    moved_count = 0

    for idx, src_path in enumerate(image_paths):
        try:
            prefix = prefixes.get(idx, "")
            if prefix:
                parts = [p for p in prefix.rstrip("_").split("_") if p]
                dest_dir = output_dir / Path(*parts)
            else:
                dest_dir = output_dir / "unclustered"

            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / src_path.name

            # Collision resolution: check if dest exists on disk
            if dest.exists():
                dest = dest_dir / f"{src_path.stem}_{idx}{src_path.suffix}"

            try:
                shutil.move(str(src_path), str(dest))
            except OSError:
                # Cross-device link: fallback to verified copy + remove
                _verified_copy(src_path, dest)
                os.remove(str(src_path))

            moved_count += 1
        except Exception as exc:
            log.error("Failed to process %s: %s", src_path, exc)
            failures.append((src_path, exc))

    log.info("Moved %d files to folder hierarchy", moved_count)
    if failures:
        log.warning("%d file(s) failed: %s", len(failures), ", ".join(str(p) for p, _ in failures))
    return moved_count


def _verified_copy(src: Path, dest: Path, retries: int = 2) -> None:
    """Copy src to dest with size verification and retry on transient errors.

    Raises the last OSError if all attempts fail or size mismatch is detected.
    """
    last_exc: OSError | None = None
    for attempt in range(retries):
        try:
            shutil.copy2(str(src), str(dest))
            if dest.stat().st_size != src.stat().st_size:
                raise OSError(
                    f"Size mismatch after copy: src={src.stat().st_size} dest={dest.stat().st_size}"
                )
            return
        except OSError as exc:
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(1)
    raise last_exc  # type: ignore[misc]


def flat_export(
    image_paths: list[Path],
    prefixes: dict[int, str],
    output_dir: Path,
    dry_run: bool = False,
) -> int:
    """Move files to flat dir with prefix prepended. Returns count of moved files.

    Uses shutil.move with cross-device fallback (copy2 + remove).
    Handles collisions by appending _N to stem.
    Logs "Moved N files" on completion.
    """
    if dry_run:
        log.info("=== DRY RUN — no files will be written ===")
        for idx, src_path in enumerate(image_paths):
            prefix = prefixes.get(idx, "")
            dest_name = f"{prefix}{src_path.name}"
            log.info("  would move %s → %s", src_path, output_dir / dest_name)
        return 0

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    failures: list[tuple[Path, Exception]] = []
    moved_count = 0

    for idx, src_path in enumerate(image_paths):
        try:
            prefix = prefixes.get(idx, "")
            dest_name = f"{prefix}{src_path.name}"
            dest = output_dir / dest_name

            # Collision resolution: check if dest exists on disk
            if dest.exists():
                stem = f"{prefix}{src_path.stem}"
                dest = output_dir / f"{stem}_{idx}{src_path.suffix}"

            try:
                shutil.move(str(src_path), str(dest))
            except OSError:
                # Cross-device link: fallback to verified copy + remove
                _verified_copy(src_path, dest)
                os.remove(str(src_path))

            moved_count += 1
        except Exception as exc:
            log.error("Failed to process %s: %s", src_path, exc)
            failures.append((src_path, exc))

    log.info("Moved %d files", moved_count)
    if failures:
        log.warning("%d file(s) failed: %s", len(failures), ", ".join(str(p) for p, _ in failures))
    return moved_count
