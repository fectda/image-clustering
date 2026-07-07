"""HTML gallery generation."""

import html
import logging
from pathlib import Path

log = logging.getLogger("cluster")

HTML_HEADER = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Photo Clusters</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         background: #111; color: #eee; min-height: 100vh; }
  .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
  h1 { font-size: 1.8rem; margin-bottom: 8px; }
  .subtitle { color: #888; margin-bottom: 24px; font-size: 0.9rem; }
  .cluster { margin-bottom: 40px; }
  .cluster h2 { font-size: 1.2rem; margin-bottom: 12px; color: #ccc;
                border-bottom: 1px solid #333; padding-bottom: 6px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
          gap: 8px; }
  .thumb { overflow: hidden; border-radius: 6px; background: #222;
           aspect-ratio: 1; position: relative; }
  .thumb img { width: 100%; height: 100%; object-fit: cover;
               transition: transform 0.2s; cursor: pointer; }
  .thumb img:hover { transform: scale(1.05); }
  .lb-overlay { display: none; position: fixed; inset: 0; z-index: 9999;
                background: rgba(0,0,0,0.92); justify-content: center;
                align-items: center; }
  .lb-overlay.active { display: flex; }
  .lb-overlay img { max-width: 90vw; max-height: 90vh; object-fit: contain;
                    border-radius: 4px; }
  .lb-close { position: absolute; top: 16px; right: 24px; color: #fff;
              font-size: 2rem; cursor: pointer; opacity: 0.7; }
  .lb-close:hover { opacity: 1; }
  .lb-prev, .lb-next { position: absolute; top: 50%; transform: translateY(-50%);
                        color: #fff; font-size: 3rem; cursor: pointer; opacity: 0.5;
                        padding: 16px; user-select: none; }
  .lb-prev:hover, .lb-next:hover { opacity: 1; }
  .lb-prev { left: 16px; }
  .lb-next { right: 16px; }
  .lb-counter { position: absolute; bottom: 16px; left: 50%; transform: translateX(-50%);
                color: #888; font-size: 0.85rem; }
  @media (max-width: 600px) {
    .grid { grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)); }
  }
</style>
</head>
<body>
<div class="container">
<h1>Photo Clusters</h1>
<p class="subtitle">Clustered by visual similarity</p>
"""

HTML_FOOTER = """</div>

<div class="lb-overlay" id="lightbox">
  <span class="lb-close" onclick="closeLightbox()">&times;</span>
  <span class="lb-prev" onclick="navLightbox(-1)">&#10094;</span>
  <img id="lb-img" src="" alt="">
  <span class="lb-next" onclick="navLightbox(1)">&#10095;</span>
  <span class="lb-counter" id="lb-counter"></span>
</div>

<script>
let lbImages = [];
let lbIndex = 0;

document.querySelectorAll('[data-lightbox]').forEach(function(el, i) {
  el.addEventListener('click', function(e) {
    e.preventDefault();
    lbImages = Array.from(document.querySelectorAll('[data-lightbox]'));
    lbIndex = i;
    showLightbox();
  });
});

function showLightbox() {
  const el = lbImages[lbIndex];
  document.getElementById('lb-img').src = el.getAttribute('href');
  document.getElementById('lb-counter').textContent =
    (lbIndex + 1) + ' / ' + lbImages.length;
  document.getElementById('lightbox').classList.add('active');
}

function closeLightbox() {
  document.getElementById('lightbox').classList.remove('active');
}

function navLightbox(dir) {
  lbIndex = (lbIndex + dir + lbImages.length) % lbImages.length;
  showLightbox();
}

document.addEventListener('keydown', function(e) {
  if (!document.getElementById('lightbox').classList.contains('active')) return;
  if (e.key === 'Escape') closeLightbox();
  if (e.key === 'ArrowLeft') navLightbox(-1);
  if (e.key === 'ArrowRight') navLightbox(1);
});

document.getElementById('lightbox').addEventListener('click', function(e) {
  if (e.target === this) closeLightbox();
});
</script>
</body>
</html>"""


def generate_gallery(output_dir: Path, groups: dict[int, list[tuple[int, Path]]]):
    """Generate a self-contained index.html with inline CSS/JS."""
    output_dir = Path(output_dir)

    clusters = []
    for label in sorted(groups):
        items = groups[label]
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

    html_parts = [HTML_HEADER]
    for cluster in clusters:
        html_parts.append(f'<div class="cluster" id="{html.escape(cluster["name"])}">')
        html_parts.append(f"  <h2>{html.escape(cluster['title'])}</h2>")
        html_parts.append('  <div class="grid">')
        for img_rel, img_name in cluster["images"]:
            escaped_rel = html.escape(img_rel, quote=True)
            escaped_name = html.escape(img_name, quote=True)
            html_parts.append(
                f'    <div class="thumb">'
                f'<a href="{escaped_rel}" data-lightbox="gallery" data-title="{escaped_name}">'
                f'<img src="{escaped_rel}" alt="{escaped_name}" loading="lazy">'
                f"</a></div>"
            )
        html_parts.append("  </div>")
        html_parts.append("</div>")

    html_parts.append(HTML_FOOTER)

    index_path = output_dir / "index.html"
    with open(index_path, "w") as f:
        f.write("\n".join(html_parts))
    log.info("Gallery: %s", index_path)
