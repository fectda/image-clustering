# Photo Organizer — Internal Project Documentation

> AI-facing reference. Read this first before answering any question about this project.

## Identity

**Name:** photo-organizer
**Repo:** `/home/fectda/temporal/photo-organizer`
**Purpose:** Containerized photo clustering tool. Takes a folder of images, encodes them with a vision transformer (DINOv2 / CLIP / SigLIP), reduces dimensionality with UMAP, clusters with K-Means (auto K via silhouette score), and outputs organized folders + a self-contained HTML gallery with lightbox viewer.

**Key constraint:** GPU is REQUIRED. No CPU fallback. If CUDA is unavailable, the tool exits immediately.

## Pipeline (6 phases)

```
Scan → LoadModel → Embed → Reduce+Cluster → Export → Gallery
```

Each phase is a separate module in `photo_organizer/`:

| Module | Phase | Responsibility |
|--------|-------|---------------|
| `cli.py` | — | argparse CLI definition |
| `scanner.py` | Scan | Recursive file scan for jpg/jpeg/png/webp |
| `models.py` | LoadModel | Load DINOv2/CLIP/SigLIP on GPU |
| `embeddings.py` | Embed | Extract L2-normalized embeddings per image |
| `cluster.py` | Reduce+Cluster | UMAP (5 dims) → K-Means with silhouette auto-K |
| `export.py` | Export | Copy images into `cluster_N/` folders |
| `gallery.py` | Gallery | Generate `index.html` with inline CSS/JS lightbox |
| `main.py` | Orchestrator | Wires all phases in sequence |

## Module Details

### cli.py
- Standard argparse.
- Args: `--input/-i` (required), `--output/-o` (required), `--preview` (10 random images), `--dry-run`, `--model` (dinov2-large/basic, clip, siglip), `--umap-components` (default 5), `--max-clusters` (default 20), `--batch-size` (default 32).
- All extra args in `organize.sh` get passed through as `DOCKER_ARGS`.

### scanner.py
- `scan_images()` — rglob for `.jpg`, `.jpeg`, `.png`, `.webp` (case-insensitive). Exits if no images found.
- `sample_preview()` — random subset of N (default 10).

### models.py
- GPU check FIRST: `torch.cuda.is_available()` or die.
- Three backends:
  - **dinov2** — HuggingFace `AutoModel.from_pretrained("facebook/dinov2-large"|"facebook/dinov2-base")`. Default model.
  - **clip** — OpenAI CLIP ViT-B/32 via `clip.load()`. Uses `processor` = preprocess fn.
  - **siglip** — HF `google/siglip-base-patch16-224`. Uses pooler_output (mean pool).
- Always returns `(model, processor, backend, device)`.

### embeddings.py
- `extract_embeddings()` — batches images, opens with PIL, skips corrupt ones.
- **Backend dispatch:**
  - `clip`: `model.encode_image(batch_input)` after `processor(img)` per image.
  - `dinov2`: HF processor dict → `model(**inputs)`, takes `last_hidden_state[:, 0, :]` (CLS token).
  - `siglip`: Same as dinov2 but uses `outputs.pooler_output`.
- All features are L2-normalized.
- Returns `np.ndarray` of shape `(N, D)`.

### cluster.py
- `reduce_and_cluster()`:
  - UMAP: 5 components (default), `random_state=42`, `n_neighbors=min(15, N-1)`.
  - K-Means: iterates k=2..max_k, picks best silhouette score. Fallback to k=2.
- K-Means assigns every point; no noise cluster.

### export.py
- `export_clusters()`:
  - `dry_run=True` — only logs cluster sizes.
  - `dry_run=False` — copies images to `cluster_N/` dirs. Handles name collisions by appending index.

### gallery.py
- `generate_gallery()` — produces a single `index.html` with:
  - Dark theme CSS.
  - Grid layout (auto-fill, min 180px / 120px mobile).
  - JavaScript lightbox overlay with keyboard nav (Escape, arrows).
  - Lightbox navigation (prev/next).
- HTML is self-contained (no external deps).

### main.py
- Entry point, wired in `__init__.py` with public API.
- Phases execute sequentially with logging.

## Docker Infrastructure

### Multi-stage build (`Dockerfile`)

**Stage 1 — builder:**
- `python:3.11-slim` base.
- Creates `/venv` virtual environment.
- Installs torch + torchvision from a configurable index (CUDA URL via `--build-arg TORCH_INDEX_URL`).
- Install remaining deps from `requirements.txt`.
- Pre-downloads only `facebook/dinov2-large` as HF cache at `/venv/hf-cache`.

**Stage 2 — runtime:**
- Fresh `python:3.11-slim`.
- Installs `gosu` for UID switching.
- Copies `/venv` from builder (portable venv).
- Copies `photo_organizer/` and `scripts/entrypoint.sh`.
- ENTRYPOINT: `/entrypoint.sh`, CMD: `python -m photo_organizer.main`.

### organize.sh (Docker wrapper)
- Located at project root.
- Validates `--input` and `--output` exist.
- Detects GPU via `nvidia-smi` + Docker NVIDIA toolkit check. Dies if not found.
- Builds image if missing (use `--force-rebuild` to force).
- Runs container with:
  - `--gpus all`
  - `HOST_UID` / `HOST_GID` for file ownership.
  - Input mounted ro at `/data/input`, output at `/data/output`.
  - Passes extra CLI args through to the Python script.
- Shows gallery path on success.

### entrypoint.sh
- If `HOST_UID` and `HOST_GID` are set: creates `hostgroup` + `hostuser`, sets `TORCHINDUCTOR_CACHE_DIR` to `/tmp/torch-cache-{UID}`, drops privileges via `gosu hostuser python -m photo_organizer.main "$@"`.
- Otherwise: runs as root.

### docker-compose / CI
- `.github/workflows/docker-publish.yml`:
  - Trigger: tag push `v*` or manual `workflow_dispatch`.
  - Builds with `TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124`.
  - Pushes to `ghcr.io/${{ github.repository }}` as `latest` + semver tags.
  - Uses GHCR + GHA cache.

### .dockerignore
- Excludes: `.git`, `__pycache__`, `*.pyc`, `.DS_Store`, `README.md`, `organize.sh`, `tests/`, `pytest.ini`, `scripts/**` (except `!scripts/entrypoint.sh`), `*.md`.

## Testing

### E2E test — `tests/test_e2e_preview.py`
- **REAL Docker test** — not mocked.
- Fixture `output_dir`: creates `/home/fectda/temporal/photo-organizer/temporal/test-run-{PID}` as output.
- Test `test_e2e_preview_gallery`:
  1. Runs Docker: `docker run --rm --gpus all -e HOST_UID -e HOST_GID -v /home/fectda/temporal/images:/data/input:ro -v {output}:/data/output vision-photo-organizer:latest --input /data/input --output /data/output --preview`
  2. Expects exit code 0.
  3. Verifies `index.html` exists and is readable.
  4. Parse all `src="..."` from HTML, verify each image file exists.
  5. Checks `data-lightbox` count matches image count.
  6. Checks CSS has no "mincast" typo.
  7. Verifies cluster directories exist.
- Images source: `/home/fectda/temporal/images` (symlink or real dir — 325 JPG files from WhatsApp).
- Uses `subprocess.run` with 300s timeout.

## Images Location

- **Absolute path:** `/home/fectda/temporal/images`
- **Relative to project:** `../images`
- **Contents:** 325 JPEG files (IMG-YYYYMMDD-WA*.jpg + some null-*.jpg)
- **Source:** WhatsApp exported photos

## Commands

```bash
# Build image
docker build -t vision-photo-organizer:latest .

# Build with CUDA
docker build -t vision-photo-organizer:latest \
  --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 .

# Run via wrapper
./organize.sh -i ../images -o /tmp/salida --preview

# Full run
./organize.sh -i ../images -o /tmp/salida

# Test
pytest tests/test_e2e_preview.py -v

# Direct Docker (no wrapper)
docker run --rm --gpus all \
  -e HOST_UID=$(id -u) -e HOST_GID=$(id -g) \
  -v /home/fectda/temporal/images:/data/input:ro \
  -v /tmp/salida:/data/output \
  vision-photo-organizer:latest \
  --input /data/input --output /data/output --preview
```

## Git State

**Remote:** `origin/main` — up to date.

**Unstaged changes (not yet committed):**
- `.dockerignore` — adds `!scripts/entrypoint.sh` exception so the entrypoint is included in Docker build context.
- `.gitignore` — adds `temporal/` (test output directory).
- `Dockerfile` — refactored to multi-stage build with venv, pre-downloads `dinov2-large` in builder, more efficient layers.
- `tests/test_e2e_preview.py` — changed from `tempfile.mkdtemp` to project-local `temporal/` output dir.

## Key Gotchas

1. **GPU is mandatory.** `models.py` calls `sys.exit(1)` if CUDA unavailable. `organize.sh` also checks.
2. **Docker image is ~6+ GB** (torch + torchvision + DINOv2 model weights).
3. **Images must exist** at `/home/fectda/temporal/images` for the E2E test.
4. **Model cache** is at `/venv/hf-cache` inside the container — pre-downloaded at build time.
5. **CLIP backend** has a different preprocessing path than DINOv2/SigLIP in `embeddings.py`.
6. **DINOv2** uses CLS token for embedding; **SigLIP** uses `pooler_output` (mean pool).
7. **Only dinov2-large is pre-cached** in the Docker image. Other models download on first use.
8. **No local Python dev environment** — everything runs through Docker. No venv/setup.py/pyproject.toml.
9. **File ownership** is handled via gosu in entrypoint.sh — files written to host volume will match host UID/GID.
10. **K-Means fallback:** if all points map to one cluster, falls back to k=2.

## Scripts

| Script | Location | Purpose |
|--------|----------|---------|
| `organize.sh` | `/home/fectda/temporal/photo-organizer/` | Docker wrapper with GPU detection and build |
| `scripts/entrypoint.sh` | `/home/fectda/temporal/photo-organizer/scripts/` | Container entrypoint — UID switching |
| `scripts/download_models.py` | `/home/fectda/temporal/photo-organizer/scripts/` | Pre-download all 4 models during build (replaced by inline build-time download in current Dockerfile) |

## External Data Files (Not in Repo)

- `/home/fectda/temporal/get_user_messages.sh` — extracts WhatsApp-like user messages from the OpenCode SQLite DB. Not related to photo-organizer.
- `/home/fectda/temporal/mensajes_usuario_*.txt` — extracted message output from the above script. Not related to photo-organizer.
