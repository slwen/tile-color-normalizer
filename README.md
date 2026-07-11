# Tile Color Normalizer

Normalize color palettes across a set of game tiles (pixel art / AI-generated tilesets).

1. **Reduce** — two-pass color reduction across every tile in a folder  
2. **Map** — open a browser UI to remap those remaining colors (recolor / merge)

## Requirements

```bash
pip install -r requirements.txt
```

## Quick start

1. Put your tile images in `./input` (PNG recommended).
2. Run:

```bash
python tile_color_normalizer.py
```

That will:

1. Reduce colors for all images in `./input`
2. Write reduced tiles to `./input/reduced`
3. Open a browser UI at [http://127.0.0.1:8765/](http://127.0.0.1:8765/) so you can map colors
4. When you click **Apply & export** in the UI, write final tiles to `./input/mapped`

Stop the UI server with `Ctrl+C` in the terminal.

## Commands

### Default (reduce + map UI)

```bash
python tile_color_normalizer.py
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `-i`, `--input-dir` | Folder with source tiles | `input` |
| `-m`, `--max-colors` | Max colors kept after reduce | `32` |
| `--map-only` | Skip reduce; only open the mapping UI on `<input>/reduced` | off |
| `--reduce-only` | Only reduce colors; do not open the map UI | off |
| `--no-browser` | Don’t auto-open a browser | off |
| `--port` | Map UI port | `8765` |
| `--host` | Map UI bind host | `127.0.0.1` |
| `--allow-remote` | Allow non-loopback `--host` (no auth — careful) | off |
| `-h`, `--help` | Show help | — |

### Examples

```bash
# Full pipeline on ./input (default)
python tile_color_normalizer.py

# Keep fewer colors
python tile_color_normalizer.py -m 16

# Different source folder
python tile_color_normalizer.py -i ./tiles

# Headless reduce only (no browser UI)
python tile_color_normalizer.py --reduce-only -m 16

# Reopen the mapping UI without re-running reduce
python tile_color_normalizer.py --map-only

# Run UI without opening a browser (open the URL yourself)
python tile_color_normalizer.py --map-only --no-browser

# Use a different port if 8765 is busy
python tile_color_normalizer.py --map-only --port 8766
```

## Output layout

Assuming the default input folder:

```
input/
  *.png                 # your originals
  reduced/
    *.png               # color-reduced tiles
    palette.json        # remaining colors (and last mapping)
    palette_preview.png # swatch sheet
  mapped/
    *.png               # after Apply & export in the UI
    palette.json
    palette_preview.png
```

## Mapping UI

In the browser you can:

- See every remaining color across all reduced tiles (swatch, hex, usage %)
- Pick a new color with the color picker or hex field
- Merge colors by mapping several sources to the same target
- Preview tile thumbnails (hover to enlarge)
- **Apply & export** to write remapped PNGs into `mapped/`

## How reduce works

`-m` / `--max-colors` is applied in **two passes** (not “top N pixels globally”):

1. **Per image** — each tile is reduced to ≤ N colors using luminance-stratified sampling + k-means++, so lights/midtones aren’t drowned out by large dark regions.
2. **Across images** — each tile votes equally with its local palette; weighted k-means builds a **representative shared palette** of N colors, then every tile is snapped to that set.

This avoids the common failure mode where a global frequency ranking (or unweighted median-cut) keeps only the darkest pixels.

## Notes

- Only top-level images in the input directory are processed (not recursive).
- Supported formats: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.gif`, `.webp`, `.tga`
- Transparent pixels keep their original RGB; only visible pixels (`alpha > 0`) are quantized/remapped.
