"""HTML gallery generation."""

import logging
from importlib.resources import files
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

log = logging.getLogger("cluster")


def generate_gallery(output_dir: Path, groups: dict[int, list[tuple[int, Path]]]):
    """Generate a self-contained index.html with inline CSS/JS via Jinja2."""
    output_dir = Path(output_dir)

    clusters = []
    for label in sorted(groups):
        items = groups[label]
        if label == -1:
            cluster_name = "unclustered"
            title = f"Unclustered ({len(items)} images)"
        else:
            cluster_name = f"cluster_{label}"
            title = f"Cluster {label} ({len(items)} images)"

        thumbnails = []
        for idx, src_path in items:
            dest = f"{cluster_name}/{src_path.stem}{src_path.suffix}"
            if Path(output_dir / dest).exists():
                thumbnails.append((dest, src_path.name))
            elif Path(output_dir / f"{src_path.stem}_{idx}{src_path.suffix}").exists():
                dest = f"{cluster_name}/{src_path.stem}_{idx}{src_path.suffix}"
                thumbnails.append((dest, src_path.name))

        clusters.append(
            {
                "name": cluster_name,
                "title": title,
                "images": thumbnails,
            }
        )

    # Sort by size descending
    clusters.sort(key=lambda c: -len(c["images"]))

    template_dir = str(files("photo_organizer") / "templates")
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template("gallery.html")
    output = template.render(
        clusters=clusters,
        total_images=sum(len(c["images"]) for c in clusters),
        total_clusters=len(clusters),
    )

    index_path = output_dir / "index.html"
    with open(index_path, "w") as f:
        f.write(output)
    log.info("Gallery: %s", index_path)
