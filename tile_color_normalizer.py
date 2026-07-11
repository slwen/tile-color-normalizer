#!/usr/bin/env python3
"""
Tile Color Normalizer
=====================

Put tiles in ./input, then run:

    python tile_color_normalizer.py

That reduces colors and opens a browser UI to remap them.
Exports land in ./input/reduced and ./input/mapped.

Options:
    -m / --max-colors   Max colors after reduce (default: 32)
    -i / --input-dir    Source tiles folder (default: input)
    --map-only          Skip reduce; just reopen the mapping UI
    --reduce-only       Only reduce; do not open the map UI
    --port / --host     Map UI bind address (default: 127.0.0.1:8765)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import threading
import webbrowser
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tga"}


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

def rgb_to_hex(rgb) -> str:
    return "#{:02x}{:02x}{:02x}".format(*[int(max(0, min(255, c))) for c in rgb])


def hex_to_rgb(hex_str: str) -> tuple[int, int, int]:
    hex_str = hex_str.strip().lstrip("#")
    if len(hex_str) != 6:
        raise ValueError(f"Invalid hex color: {hex_str!r} (need 6 hex digits)")
    return tuple(int(hex_str[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def get_image_files(input_dir: str | Path) -> list[Path]:
    """Sorted image files in the directory (non-recursive). Skips palette preview."""
    p = Path(input_dir)
    if not p.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    files = [
        f
        for f in p.iterdir()
        if f.is_file()
        and f.suffix.lower() in IMAGE_EXTS
        and f.name not in {"palette_preview.png"}
    ]
    return sorted(files)


def load_rgba(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        return np.array(img, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Color analysis / quantize
# ---------------------------------------------------------------------------

def collect_color_counter(image_files: list[Path], alpha_threshold: int = 0):
    """Count RGB of pixels with alpha > threshold across all images."""
    counter: Counter = Counter()
    total_visible = 0
    for img_path in image_files:
        try:
            arr = load_rgba(img_path)
            mask = arr[:, :, 3] > alpha_threshold
            if not np.any(mask):
                continue
            rgb = arr[mask][:, :3]
            total_visible += int(rgb.shape[0])
            packed = (
                rgb[:, 0].astype(np.int32) << 16
                | rgb[:, 1].astype(np.int32) << 8
                | rgb[:, 2].astype(np.int32)
            )
            uniq, counts = np.unique(packed, return_counts=True)
            for val, cnt in zip(uniq.tolist(), counts.tolist()):
                r = (val >> 16) & 255
                g = (val >> 8) & 255
                b = val & 255
                counter[(r, g, b)] += int(cnt)
        except Exception as e:
            print(f"  [Warning] Could not read {img_path.name}: {e}")
    return counter, total_visible


def create_palette_entries(counter: Counter, max_colors: int | None, total_visible: int) -> list[dict]:
    """Build ordered palette entries (most common first). max_colors=None keeps all."""
    if not counter:
        return []
    items = counter.most_common(max_colors)
    palette = []
    for idx, (rgb, count) in enumerate(items):
        pct = (count / total_visible * 100.0) if total_visible > 0 else 0.0
        hex_code = rgb_to_hex(rgb)
        palette.append(
            {
                "index": idx,
                "rgb": list(rgb),
                "hex": hex_code,
                "count": int(count),
                "percentage": round(pct, 2),
                "new_hex": hex_code,
            }
        )
    return palette


def unique_colors_in_arr(arr: np.ndarray, alpha_threshold: int = 0) -> list[tuple[int, int, int]]:
    """Return unique visible RGB colors in an RGBA array."""
    mask = arr[:, :, 3] > alpha_threshold
    if not np.any(mask):
        return []
    rgb = arr[mask][:, :3]
    packed = (
        rgb[:, 0].astype(np.int32) << 16
        | rgb[:, 1].astype(np.int32) << 8
        | rgb[:, 2].astype(np.int32)
    )
    uniq = np.unique(packed)
    colors = []
    for val in uniq.tolist():
        colors.append(((val >> 16) & 255, (val >> 8) & 255, val & 255))
    return colors


def snap_image_to_palette(arr: np.ndarray, palette_rgb: list[tuple[int, int, int]]) -> np.ndarray:
    """Snap visible pixels to nearest palette color. Transparent RGB preserved."""
    if not palette_rgb:
        return arr.copy()

    out = arr.copy()
    rgb = out[:, :, :3]
    alpha = out[:, :, 3]
    h, w = alpha.shape

    flat_rgb = rgb.reshape(-1, 3)
    flat_alpha = alpha.reshape(-1)
    visible_idx = np.flatnonzero(flat_alpha > 0)
    if len(visible_idx) == 0:
        return out

    visible_rgb = flat_rgb[visible_idx].astype(np.int32)
    pal_arr = np.array(palette_rgb, dtype=np.int32)
    # Chunk if huge to avoid massive memory on (N, K, 3)
    nearest = np.empty(len(visible_idx), dtype=np.int32)
    chunk = 200_000
    for start in range(0, len(visible_idx), chunk):
        end = min(start + chunk, len(visible_idx))
        block = visible_rgb[start:end]
        diffs = block[:, None, :] - pal_arr[None, :, :]
        dists = np.sum(diffs * diffs, axis=2)
        nearest[start:end] = np.argmin(dists, axis=1)
    flat_rgb[visible_idx] = pal_arr[nearest].astype(np.uint8)
    out[:, :, :3] = flat_rgb.reshape(h, w, 3)
    return out


def _luminance(rgb: np.ndarray) -> np.ndarray:
    """Relative luminance for (N, 3) RGB."""
    r = rgb[:, 0].astype(np.float64)
    g = rgb[:, 1].astype(np.float64)
    b = rgb[:, 2].astype(np.float64)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def stratified_color_sample(visible: np.ndarray, max_samples: int = 40_000, bins: int = 8, seed: int = 0) -> np.ndarray:
    """
    Sample visible pixels with equal budget per luminance band.

    Prevents huge dark regions from starving highlight / midtone colors during
    palette fitting (the usual failure mode of pure frequency or median-cut).
    """
    n = int(visible.shape[0])
    if n <= max_samples:
        return visible

    rng = np.random.default_rng(seed)
    lum = _luminance(visible)
    # Adaptive bin edges from data percentiles so sparse highlights still get a bin
    edges = np.unique(np.quantile(lum, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        idx = rng.choice(n, size=max_samples, replace=False)
        return visible[idx]

    per_bin = max(1, max_samples // (len(edges) - 1))
    picked = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i == len(edges) - 2:
            m = (lum >= lo) & (lum <= hi)
        else:
            m = (lum >= lo) & (lum < hi)
        idxs = np.flatnonzero(m)
        if len(idxs) == 0:
            continue
        take = min(per_bin, len(idxs))
        picked.append(idxs if take == len(idxs) else rng.choice(idxs, size=take, replace=False))

    if not picked:
        idx = rng.choice(n, size=max_samples, replace=False)
        return visible[idx]

    sel = np.concatenate(picked)
    if len(sel) > max_samples:
        sel = rng.choice(sel, size=max_samples, replace=False)
    return visible[sel]


def quantize_image_local(arr: np.ndarray, max_colors: int) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """
    Reduce one image to at most max_colors.

    Uses luminance-stratified sampling + k-means++ so each tile keeps a mix of
    darks, mids, and lights — not only the densest (often darkest) clusters.
    Returns (quantized RGBA, list of palette colors used).
    """
    max_colors = max(1, max_colors)
    alpha = arr[:, :, 3]
    mask = alpha > 0
    if not np.any(mask):
        return arr.copy(), []

    visible = arr[mask][:, :3]
    packed = (
        visible[:, 0].astype(np.int32) << 16
        | visible[:, 1].astype(np.int32) << 8
        | visible[:, 2].astype(np.int32)
    )
    n_unique = int(np.unique(packed).size)
    if n_unique <= max_colors:
        return arr.copy(), unique_colors_in_arr(arr)

    sample = stratified_color_sample(visible, max_samples=40_000, bins=max(8, max_colors * 2))
    # Equal weight on stratified sample (already de-biased by luminance bins)
    palette_rgb = kmeans_palette(sample.astype(np.float64), max_colors, weights=None)
    if not palette_rgb:
        return arr.copy(), unique_colors_in_arr(arr)

    quantized = snap_image_to_palette(arr, palette_rgb)
    used = unique_colors_in_arr(quantized)
    return quantized, used


def kmeans_palette(
    points: np.ndarray,
    k: int,
    weights: np.ndarray | None = None,
    max_iter: int = 30,
    seed: int = 42,
) -> list[tuple[int, int, int]]:
    """
    Weighted k-means++ on RGB points → k representative colors.
    points: (N, 3) float/int, weights: (N,) non-negative (optional).
    """
    if points.size == 0:
        return []
    pts = points.astype(np.float64)
    n = pts.shape[0]
    if weights is None:
        w = np.ones(n, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        w = np.maximum(w, 0.0)
        if w.sum() <= 0:
            w = np.ones(n, dtype=np.float64)

    # Unique rows first (keeps heaviest weight per unique RGB)
    packed = (
        np.clip(pts[:, 0], 0, 255).astype(np.int32) << 16
        | np.clip(pts[:, 1], 0, 255).astype(np.int32) << 8
        | np.clip(pts[:, 2], 0, 255).astype(np.int32)
    )
    uniq, inv = np.unique(packed, return_inverse=True)
    if len(uniq) <= k:
        colors = []
        for val in uniq.tolist():
            colors.append(((val >> 16) & 255, (val >> 8) & 255, val & 255))
        return colors

    # Aggregate weights onto unique colors
    agg_w = np.zeros(len(uniq), dtype=np.float64)
    np.add.at(agg_w, inv, w)
    uniq_pts = np.stack(
        [(uniq >> 16) & 255, (uniq >> 8) & 255, uniq & 255],
        axis=1,
    ).astype(np.float64)
    n = len(uniq)
    w = agg_w
    pts = uniq_pts
    k = min(k, n)

    rng = np.random.default_rng(seed)

    # k-means++ init (weight-aware)
    centers = np.empty((k, 3), dtype=np.float64)
    probs = w / w.sum()
    centers[0] = pts[rng.choice(n, p=probs)]
    closest = np.full(n, np.inf)
    for c in range(1, k):
        d = np.sum((pts - centers[c - 1]) ** 2, axis=1)
        closest = np.minimum(closest, d)
        scores = closest * w
        total = scores.sum()
        if total <= 0:
            centers[c] = pts[rng.integers(0, n)]
        else:
            centers[c] = pts[rng.choice(n, p=scores / total)]

    for _ in range(max_iter):
        dists = np.sum((pts[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = np.argmin(dists, axis=1)
        residuals = dists[np.arange(n), labels].copy()
        new_centers = centers.copy()
        reseeded: set[int] = set()

        for ci in range(k):
            m = labels == ci
            if np.any(m):
                ww = w[m]
                s = ww.sum()
                if s <= 0:
                    new_centers[ci] = pts[m].mean(axis=0)
                else:
                    new_centers[ci] = (pts[m] * ww[:, None]).sum(axis=0) / s
                continue

            # Empty cluster: reseed to a distinct high-residual point
            order = np.argsort(-residuals)
            for idx in order:
                ii = int(idx)
                if ii in reseeded or residuals[ii] < 0:
                    continue
                new_centers[ci] = pts[ii].copy()
                reseeded.add(ii)
                residuals[ii] = -1.0
                break

        if np.allclose(new_centers, centers, atol=0.25):
            centers = new_centers
            break
        centers = new_centers

    out = []
    for c in centers:
        rgb = tuple(int(max(0, min(255, round(v)))) for v in c)
        out.append(rgb)  # type: ignore[arg-type]
    # Deduplicate after rounding (near-identical centers can still collapse)
    seen = set()
    unique = []
    for c in out:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


def build_global_palette(
    per_image_colors: list[list[tuple[int, int, int]]],
    max_colors: int,
) -> list[tuple[int, int, int]]:
    """
    Build a shared palette from per-image local palettes.

    Each image votes equally (so large dark areas in one tile cannot drown out
    highlight colors from others). Then weighted k-means picks a representative
    mix of max_colors centers — not the globally most frequent pixels.
    """
    points = []
    weights = []
    for colors in per_image_colors:
        if not colors:
            continue
        # Equal share per image, split across that image's local colors
        share = 1.0 / len(colors)
        for rgb in colors:
            points.append(rgb)
            weights.append(share)

    if not points:
        return []

    pts = np.array(points, dtype=np.float64)
    w = np.array(weights, dtype=np.float64)
    return kmeans_palette(pts, max_colors, weights=w)


def apply_color_map(arr: np.ndarray, old_to_new: dict[tuple[int, int, int], tuple[int, int, int]]) -> np.ndarray:
    """Exact remap of visible pixels using old RGB → new RGB. No nearest-neighbor."""
    if not old_to_new:
        return arr.copy()

    out = arr.copy()
    rgb = out[:, :, :3]
    alpha = out[:, :, 3]
    h, w = alpha.shape

    flat_rgb = rgb.reshape(-1, 3)
    flat_alpha = alpha.reshape(-1)
    visible_idx = np.flatnonzero(flat_alpha > 0)
    if len(visible_idx) == 0:
        return out

    vis = flat_rgb[visible_idx].astype(np.int32)
    packed = (vis[:, 0] << 16) | (vis[:, 1] << 8) | vis[:, 2]

    # searchsorted on sorted old keys for vectorized remap
    olds = np.array([(o[0] << 16) | (o[1] << 8) | o[2] for o in old_to_new.keys()], dtype=np.int32)
    news = np.array([old_to_new[o] for o in old_to_new.keys()], dtype=np.uint8)
    order = np.argsort(olds)
    olds_s = olds[order]
    news_s = news[order]

    idx = np.searchsorted(olds_s, packed)
    idx = np.clip(idx, 0, len(olds_s) - 1)
    match = olds_s[idx] == packed
    result = flat_rgb[visible_idx].copy()
    result[match] = news_s[idx[match]]
    flat_rgb[visible_idx] = result
    out[:, :, :3] = flat_rgb.reshape(h, w, 3)
    return out


def save_palette_json(palette: list[dict], total_unique: int, total_visible: int, max_colors, output_path: Path):
    data = {
        "max_colors_requested": max_colors,
        "total_unique_colors_found": total_unique,
        "total_visible_pixels_analyzed": total_visible,
        "palette": palette,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"✓ Saved palette → {output_path}")


def create_palette_preview(palette: list[dict], output_path: Path, swatch_size: int = 56, cols: int = 8):
    if not palette:
        return
    n = len(palette)
    rows = (n + cols - 1) // cols
    width = cols * swatch_size
    height = rows * (swatch_size + 24) + 10
    img = Image.new("RGB", (width, height), (30, 30, 30))
    draw = ImageDraw.Draw(img)

    font = ImageFont.load_default()
    for fp in (
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf",
    ):
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, 11)
                break
            except Exception:
                pass

    for i, entry in enumerate(palette):
        row, col = divmod(i, cols)
        x = col * swatch_size + 2
        y = row * (swatch_size + 24) + 2
        rgb = tuple(entry["rgb"])
        draw.rectangle([x, y, x + swatch_size - 3, y + swatch_size - 3], fill=rgb, outline=(180, 180, 180))
        text = entry["hex"]
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
        except Exception:
            tw = len(text) * 6
        draw.text((x + (swatch_size - 3 - tw) // 2, y + swatch_size + 3), text, fill=(220, 220, 220), font=font)

    img.save(output_path)
    print(f"✓ Saved palette preview → {output_path}")


# ---------------------------------------------------------------------------
# Step 1: REDUCE
# ---------------------------------------------------------------------------

def check_stem_collisions(image_files: list[Path]) -> str | None:
    """Return an error message if two sources would write the same stem.png."""
    by_stem: dict[str, list[str]] = {}
    for p in image_files:
        by_stem.setdefault(p.stem, []).append(p.name)
    collisions = {s: names for s, names in by_stem.items() if len(names) > 1}
    if not collisions:
        return None
    parts = [f"{s!r} ← {names}" for s, names in sorted(collisions.items())]
    return (
        "Export stem collision: multiple sources share the same filename stem "
        "(would overwrite as .png):\n  " + "\n  ".join(parts)
    )


def run_reduce(input_dir: Path, output_dir: Path, max_colors: int) -> int:
    """
    Two-pass color reduce:

    1. Per image — luminance-stratified k-means++ down to max_colors so each
       tile keeps lights / midtones / darks (not just dark-pixel majority).
    2. Across images — each tile votes equally with its local palette; weighted
       k-means builds a representative shared palette of max_colors, then every
       tile is snapped to that shared set.
    """
    max_colors = max(1, max_colors)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== REDUCE ===")
    print(f"Input:  {input_dir.resolve()}")
    print(f"Output: {output_dir.resolve()}")
    print(f"Max colors: {max_colors} (per image, then shared)")

    image_files = get_image_files(input_dir)
    if not image_files:
        print("No supported image files found.")
        return 1

    collision = check_stem_collisions(image_files)
    if collision:
        print(f"ERROR: {collision}")
        return 1

    print(f"Found {len(image_files)} image(s).")

    # --- Pass 1: per-image local palettes only (no full images kept in RAM) ---
    print(f"\nPass 1/2: reduce each image to ≤{max_colors} colors (stratified k-means)...")
    per_image_colors: list[list[tuple[int, int, int]]] = []
    ok_paths: list[Path] = []
    total_unique_before = 0

    for i, img_path in enumerate(image_files, 1):
        try:
            arr = load_rgba(img_path)
            before = len(unique_colors_in_arr(arr))
            total_unique_before += before
            _quantized, local_pal = quantize_image_local(arr, max_colors)
            del arr, _quantized
            per_image_colors.append(local_pal)
            ok_paths.append(img_path)
            if i % 10 == 0 or i == len(image_files):
                print(f"  [{i}/{len(image_files)}] {img_path.name}: {before} → {len(local_pal)} colors")
        except Exception as e:
            print(f"  [Error] {img_path.name}: {e}")

    if not ok_paths:
        print("No images could be processed.")
        return 1

    # --- Pass 2: representative shared palette, then snap originals ---
    print(f"\nPass 2/2: build shared palette of {max_colors} representative colors...")
    pooled = [c for colors in per_image_colors for c in colors]
    print(f"  Local palette colors pooled: {len(pooled)} ({len(set(pooled))} unique)")
    global_palette = build_global_palette(per_image_colors, max_colors)
    print(f"  Shared palette size: {len(global_palette)}")
    print("  Shared colors:")
    for rgb in global_palette:
        print(f"    {rgb_to_hex(rgb)}  {list(rgb)}")

    print(f"\nWriting normalized tiles → {output_dir}")
    processed = 0
    for i, img_path in enumerate(ok_paths, 1):
        try:
            arr = load_rgba(img_path)
            final = snap_image_to_palette(arr, global_palette)
            out_path = output_dir / (img_path.stem + ".png")
            Image.fromarray(final, "RGBA").save(out_path, "PNG")
            processed += 1
            if i % 10 == 0 or i == len(ok_paths):
                print(f"  [{processed}/{len(ok_paths)}] {img_path.name}")
        except Exception as e:
            print(f"  [Error] {img_path.name}: {e}")

    reduced_files = get_image_files(output_dir)
    post_counter, post_visible = collect_color_counter(reduced_files)
    final_palette = create_palette_entries(post_counter, None, post_visible)

    save_palette_json(
        final_palette,
        total_unique_before,
        post_visible,
        max_colors,
        output_dir / "palette.json",
    )
    create_palette_preview(final_palette, output_dir / "palette_preview.png")

    print(
        f"\n✓ Reduced {processed} image(s). "
        f"Remaining unique colors: {len(final_palette)} "
        f"(from ~{total_unique_before} pre-reduce uniques across tiles)\n"
    )
    return 0


# ---------------------------------------------------------------------------
# Step 2: MAP (browser UI)
# ---------------------------------------------------------------------------

MAP_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Tile Color Mapper</title>
<style>
  :root {
    --bg: #12141a;
    --panel: #1c1f28;
    --border: #2e3340;
    --text: #e8eaef;
    --muted: #8b93a7;
    --accent: #6c9eff;
    --accent-dim: #3d5a99;
    --ok: #3ecf8e;
    --danger: #e85d5d;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
    background: var(--bg); color: var(--text); min-height: 100vh;
  }
  header {
    display: flex; flex-wrap: wrap; align-items: center; gap: 12px;
    padding: 16px 20px; border-bottom: 1px solid var(--border);
    background: var(--panel); position: sticky; top: 0; z-index: 10;
  }
  header h1 { font-size: 1.1rem; margin: 0; font-weight: 600; letter-spacing: 0.02em; }
  header .meta { color: var(--muted); font-size: 0.85rem; margin-left: 4px; }
  .spacer { flex: 1; }
  button, .btn {
    appearance: none; border: 1px solid var(--border); background: #252a36;
    color: var(--text); padding: 8px 14px; border-radius: 8px; cursor: pointer;
    font-size: 0.9rem; font-weight: 500;
  }
  button:hover { border-color: var(--accent-dim); background: #2c3240; }
  button.primary { background: var(--accent); border-color: var(--accent); color: #0b1220; }
  button.primary:hover { filter: brightness(1.08); }
  button:disabled { opacity: 0.5; cursor: not-allowed; }
  main { display: grid; grid-template-columns: 1fr 340px; gap: 0; min-height: calc(100vh - 64px); }
  @media (max-width: 960px) { main { grid-template-columns: 1fr; } }
  #colors {
    padding: 16px 20px; display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 12px; align-content: start;
  }
  .card {
    background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
    padding: 12px; display: grid; grid-template-columns: 56px 1fr; gap: 12px; align-items: center;
  }
  .swatch-wrap { display: flex; flex-direction: column; gap: 6px; align-items: center; }
  .swatch {
    width: 56px; height: 56px; border-radius: 10px; border: 1px solid #555;
    box-shadow: inset 0 0 0 1px rgba(255,255,255,0.06);
  }
  .arrow { color: var(--muted); font-size: 0.75rem; }
  .fields { display: flex; flex-direction: column; gap: 6px; min-width: 0; }
  .row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
  .hex {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.85rem; color: var(--muted);
  }
  .hex.old { color: var(--text); }
  label.small { font-size: 0.7rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }
  input[type="color"] {
    width: 40px; height: 32px; border: 1px solid var(--border); border-radius: 6px;
    padding: 0; background: transparent; cursor: pointer;
  }
  input[type="text"].hex-input {
    width: 7.5em; font-family: ui-monospace, Menlo, monospace; font-size: 0.9rem;
    background: #0f1218; border: 1px solid var(--border); color: var(--text);
    border-radius: 6px; padding: 6px 8px;
  }
  input[type="text"].hex-input:focus { outline: 1px solid var(--accent); border-color: var(--accent); }
  .stats { font-size: 0.8rem; color: var(--muted); }
  .changed .swatch.target { outline: 2px solid var(--ok); outline-offset: 1px; }
  aside {
    border-left: 1px solid var(--border); background: var(--panel);
    padding: 16px; display: flex; flex-direction: column; gap: 12px;
    max-height: calc(100vh - 64px); position: sticky; top: 64px; overflow: auto;
  }
  @media (max-width: 960px) { aside { border-left: none; border-top: 1px solid var(--border); position: static; max-height: none; } }
  aside h2 { margin: 0; font-size: 0.95rem; }
  .thumb-grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(72px, 1fr));
    gap: 8px;
  }
  .thumb-grid img {
    width: 100%; aspect-ratio: 1; object-fit: contain; image-rendering: pixelated;
    background: repeating-conic-gradient(#2a2a2a 0% 25%, #1a1a1a 0% 50%) 50% / 12px 12px;
    border-radius: 6px; border: 1px solid var(--border);
  }
  #status { font-size: 0.85rem; color: var(--muted); min-height: 1.2em; }
  #status.ok { color: var(--ok); }
  #status.err { color: var(--danger); }
  .hint { font-size: 0.8rem; color: var(--muted); line-height: 1.4; }
  .tools { display: flex; flex-wrap: wrap; gap: 8px; }
</style>
</head>
<body>
<header>
  <h1>Tile Color Mapper</h1>
  <span class="meta" id="meta"></span>
  <div class="spacer"></div>
  <div class="tools">
    <button type="button" id="btn-reset">Reset mappings</button>
    <button type="button" id="btn-identity">Identity (no change)</button>
    <button type="button" class="primary" id="btn-apply">Apply &amp; export</button>
  </div>
</header>
<main>
  <section id="colors"></section>
  <aside>
    <h2>Tiles</h2>
    <p class="hint">Reduced tiles in this folder. After export, remapped images are written to the output directory.</p>
    <div class="thumb-grid" id="thumbs"></div>
    <div id="status"></div>
    <p class="hint" id="out-hint"></p>
  </aside>
</main>
<script>
const state = { colors: [], mappings: {}, baseline: {}, images: [], outputDir: "" };

function normalizeHex(v) {
  let s = (v || "").trim().replace(/^#/, "").toLowerCase();
  if (/^[0-9a-f]{3}$/.test(s)) s = s.split("").map(c => c + c).join("");
  if (!/^[0-9a-f]{6}$/.test(s)) return null;
  return "#" + s;
}

function setStatus(msg, cls) {
  const el = document.getElementById("status");
  el.textContent = msg || "";
  el.className = cls || "";
}

function render() {
  const root = document.getElementById("colors");
  root.innerHTML = "";
  state.colors.forEach((c, i) => {
    const newHex = state.mappings[c.hex] || c.hex;
    const changed = newHex.toLowerCase() !== c.hex.toLowerCase();
    const card = document.createElement("div");
    card.className = "card" + (changed ? " changed" : "");
    card.innerHTML = `
      <div class="swatch-wrap">
        <div class="swatch source" style="background:${c.hex}" title="source ${c.hex}"></div>
        <span class="arrow">↓</span>
        <div class="swatch target" style="background:${newHex}" title="target ${newHex}"></div>
      </div>
      <div class="fields">
        <div class="row">
          <span class="hex old">${c.hex}</span>
          <span class="stats">${c.count.toLocaleString()} px · ${c.percentage}%</span>
        </div>
        <div class="row">
          <label class="small">Map to</label>
          <input type="color" data-i="${i}" value="${newHex}" class="picker"/>
          <input type="text" class="hex-input" data-i="${i}" value="${newHex}" spellcheck="false"/>
        </div>
      </div>`;
    root.appendChild(card);
  });

  root.querySelectorAll(".picker").forEach(el => {
    el.addEventListener("input", e => {
      const i = +e.target.dataset.i;
      const hex = e.target.value;
      state.mappings[state.colors[i].hex] = hex;
      const input = root.querySelector(`.hex-input[data-i="${i}"]`);
      if (input) input.value = hex;
      e.target.closest(".card").classList.toggle("changed", hex.toLowerCase() !== state.colors[i].hex.toLowerCase());
      e.target.closest(".card").querySelector(".swatch.target").style.background = hex;
    });
  });
  root.querySelectorAll(".hex-input").forEach(el => {
    el.addEventListener("change", e => {
      const i = +e.target.dataset.i;
      const hex = normalizeHex(e.target.value);
      if (!hex) { setStatus("Invalid hex: " + e.target.value, "err"); return; }
      state.mappings[state.colors[i].hex] = hex;
      e.target.value = hex;
      const picker = root.querySelector(`.picker[data-i="${i}"]`);
      if (picker) picker.value = hex;
      e.target.closest(".card").classList.toggle("changed", hex.toLowerCase() !== state.colors[i].hex.toLowerCase());
      e.target.closest(".card").querySelector(".swatch.target").style.background = hex;
      setStatus("");
    });
  });
}

async function load() {
  try {
    const res = await fetch("/api/state");
    if (!res.ok) throw new Error("Failed to load state (" + res.status + ")");
    const data = await res.json();
    state.colors = data.colors || [];
    state.images = data.images || [];
    state.outputDir = data.output_dir || "";
    state.mappings = {};
    state.baseline = {};
    state.colors.forEach(c => {
      const target = c.new_hex || c.hex;
      state.mappings[c.hex] = target;
      state.baseline[c.hex] = target;
    });
    document.getElementById("meta").textContent =
      `${state.colors.length} colors · ${state.images.length} tiles · ${(data.total_visible || 0).toLocaleString()} visible px`;
    document.getElementById("out-hint").textContent = "Export → " + state.outputDir;
    const thumbs = document.getElementById("thumbs");
    thumbs.replaceChildren();
    const bust = Date.now();
    state.images.forEach(name => {
      const img = document.createElement("img");
      img.src = "/tile/" + encodeURIComponent(name) + "?t=" + bust;
      img.alt = name;
      img.title = name;
      thumbs.appendChild(img);
    });
    render();
    setStatus("");
  } catch (err) {
    setStatus(String(err.message || err), "err");
  }
}

document.getElementById("btn-reset").onclick = () => {
  // Restore targets loaded at session start (includes palette.json new_hex)
  state.colors.forEach(c => {
    state.mappings[c.hex] = state.baseline[c.hex] || c.hex;
  });
  render();
  setStatus("Mappings reset to loaded baselines.");
};
document.getElementById("btn-identity").onclick = () => {
  // No remap: each color maps to itself
  state.colors.forEach(c => { state.mappings[c.hex] = c.hex; });
  render();
  setStatus("Identity mapping (no color changes).");
};
document.getElementById("btn-apply").onclick = async () => {
  setStatus("Applying…");
  const btn = document.getElementById("btn-apply");
  btn.disabled = true;
  try {
    const mappings = Object.entries(state.mappings).map(([old_hex, new_hex]) => ({ old_hex, new_hex }));
    const res = await fetch("/api/apply", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mappings }),
    });
    let data = {};
    try { data = await res.json(); } catch (_) {}
    if (!res.ok) throw new Error(data.error || "Apply failed");
    setStatus(`Exported ${data.processed} image(s) → ${data.output_dir}`, "ok");
    // Update baseline to last successful export so Reset returns here
    state.colors.forEach(c => {
      state.baseline[c.hex] = state.mappings[c.hex] || c.hex;
    });
  } catch (err) {
    setStatus(String(err.message || err), "err");
  } finally {
    btn.disabled = false;
  }
};

load();
</script>
</body>
</html>
"""


def build_map_app(input_dir: Path, output_dir: Path):
    try:
        from flask import Flask, jsonify, request, send_from_directory, Response
    except ImportError:
        print("ERROR: flask is required for the map UI. Install with: pip install flask")
        sys.exit(1)

    app = Flask(__name__)
    app.config["INPUT_DIR"] = input_dir.resolve()
    app.config["OUTPUT_DIR"] = output_dir.resolve()

    def scan_state():
        files = get_image_files(input_dir)
        counter, total_visible = collect_color_counter(files)
        colors = create_palette_entries(counter, None, total_visible)
        # Load saved new_hex from palette.json if present
        palette_path = input_dir / "palette.json"
        saved = {}
        if palette_path.exists():
            try:
                with open(palette_path, encoding="utf-8") as f:
                    data = json.load(f)
                for e in data.get("palette", []):
                    if "hex" in e and "new_hex" in e:
                        saved[e["hex"].lower()] = e["new_hex"]
            except Exception:
                pass
        for c in colors:
            if c["hex"].lower() in saved:
                c["new_hex"] = saved[c["hex"].lower()]
        return {
            "colors": colors,
            "images": [f.name for f in files],
            "total_visible": total_visible,
            "output_dir": str(output_dir.resolve()),
            "input_dir": str(input_dir.resolve()),
        }

    @app.get("/")
    def index():
        return Response(MAP_HTML, mimetype="text/html")

    @app.get("/api/state")
    def api_state():
        return jsonify(scan_state())

    @app.get("/tile/<path:name>")
    def tile(name):
        # Only serve files that are real images in the input dir
        safe = Path(name).name
        path = input_dir / safe
        if not path.exists() or path.suffix.lower() not in IMAGE_EXTS:
            return "Not found", 404
        return send_from_directory(input_dir, safe)

    @app.post("/api/apply")
    def api_apply():
        body = request.get_json(force=True, silent=True) or {}
        mappings = body.get("mappings") or []
        old_to_new: dict[tuple[int, int, int], tuple[int, int, int]] = {}
        for m in mappings:
            try:
                old = hex_to_rgb(m["old_hex"])
                new = hex_to_rgb(m["new_hex"])
            except Exception as e:
                return jsonify({"error": str(e)}), 400
            old_to_new[old] = new

        files = get_image_files(input_dir)
        if not files:
            return jsonify({"error": "No images to process"}), 400
        collision = check_stem_collisions(files)
        if collision:
            return jsonify({"error": collision}), 400

        # Write everything to a temp dir, then swap into place on full success
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=".mapped_tmp_", dir=str(output_dir.parent)))
        try:
            processed = 0
            for img_path in files:
                arr = load_rgba(img_path)
                remapped = apply_color_map(arr, old_to_new)
                out_path = tmp / (img_path.stem + ".png")
                Image.fromarray(remapped, "RGBA").save(out_path, "PNG")
                processed += 1

            # Palette of final mapped colors (from temp outputs)
            out_files = get_image_files(tmp)
            post_c, post_v = collect_color_counter(out_files)
            final_pal = create_palette_entries(post_c, None, post_v)
            save_palette_json(final_pal, len(post_c), post_v, len(final_pal), tmp / "palette.json")
            create_palette_preview(final_pal, tmp / "palette_preview.png")

            # Persist mapping into palette.json on the reduced input folder
            counter, total_visible = collect_color_counter(files)
            palette = create_palette_entries(counter, None, total_visible)
            for e in palette:
                key = tuple(e["rgb"])
                if key in old_to_new:
                    e["new_hex"] = rgb_to_hex(old_to_new[key])
            save_palette_json(
                palette,
                len(counter),
                total_visible,
                len(palette),
                input_dir / "palette.json",
            )

            # Publish by directory swap so a mid-move failure cannot leave a
            # half-updated mapped/ folder.
            backup = None
            if output_dir.exists():
                backup = output_dir.with_name(output_dir.name + ".bak_swap")
                if backup.exists():
                    shutil.rmtree(backup)
                output_dir.replace(backup)
            try:
                tmp.replace(output_dir)
            except Exception:
                # Roll back previous output if swap failed
                if backup is not None and backup.exists() and not output_dir.exists():
                    backup.replace(output_dir)
                raise
            if backup is not None and backup.exists():
                shutil.rmtree(backup, ignore_errors=True)
            tmp = None  # successfully moved; don't rmtree in finally
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            if tmp is not None and tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)

        return jsonify(
            {
                "processed": processed,
                "output_dir": str(output_dir.resolve()),
                "remaining_colors": len(final_pal),
            }
        )

    return app


def _is_loopback_host(host: str) -> bool:
    h = (host or "").strip().lower()
    return h in {"127.0.0.1", "localhost", "::1", "0"}


def run_map(
    input_dir: Path,
    output_dir: Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
    allow_remote: bool = False,
) -> int:
    if not input_dir.exists():
        print(f"ERROR: Input directory not found: {input_dir}")
        return 1

    if not _is_loopback_host(host) and not allow_remote:
        print(
            f"ERROR: Refusing to bind map UI to non-loopback host {host!r}.\n"
            "  The map API has no authentication and can write files.\n"
            "  Re-run with --allow-remote if you really intend this."
        )
        return 1
    if not _is_loopback_host(host) and allow_remote:
        print(
            f"WARNING: Binding map UI to {host!r} with no authentication. "
            "Anyone who can reach it can export remapped tiles."
        )

    files = get_image_files(input_dir)
    if not files:
        print("No images found in", input_dir)
        print("Put tiles in ./input and run without --map-only first.")
        return 1

    print("\n=== MAP UI ===")
    print(f"Input (reduced tiles): {input_dir.resolve()}")
    print(f"Output (mapped tiles): {output_dir.resolve()}")

    counter, total_visible = collect_color_counter(files)
    print(f"Found {len(files)} image(s), {len(counter)} remaining colors, {total_visible:,} visible px")

    app = build_map_app(input_dir, output_dir)
    url = f"http://{host}:{port}/"
    print(f"\nOpen {url} in your browser to map colors.")
    print("Press Ctrl+C to stop the server.\n")

    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    try:
        app.run(host=host, port=port, debug=False, use_reloader=False)
    except OSError as e:
        print(f"ERROR: Could not start map UI on {host}:{port}: {e}")
        print(f"  Try a different port, e.g.  python tile_color_normalizer.py --map-only --port {port + 1}")
        return 1
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Put tiles in ./input, then:  python tile_color_normalizer.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        default="input",
        help="Source tiles folder (default: input)",
    )
    parser.add_argument(
        "-m",
        "--max-colors",
        type=int,
        default=32,
        help="Max colors after reduce (default: 32)",
    )
    parser.add_argument(
        "--map-only",
        action="store_true",
        help="Skip reduce; only open the mapping UI on input/reduced",
    )
    parser.add_argument(
        "--reduce-only",
        action="store_true",
        help="Only run color reduce; do not open the map UI",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not auto-open a browser for the map UI",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Map UI bind host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Map UI port (default: 8765)",
    )
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        help="Allow binding the map UI to a non-loopback host (no auth)",
    )

    args = parser.parse_args()
    if args.map_only and args.reduce_only:
        parser.error("Use only one of --map-only or --reduce-only")

    source = Path(args.input_dir)
    reduced = source / "reduced"
    mapped = source / "mapped"

    if not args.map_only:
        rc = run_reduce(source, reduced, args.max_colors)
        if rc != 0:
            sys.exit(rc)

    if args.reduce_only:
        return

    rc = run_map(
        reduced,
        mapped,
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser,
        allow_remote=args.allow_remote,
    )
    if rc != 0:
        sys.exit(rc)


if __name__ == "__main__":
    main()
