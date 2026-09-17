# Magic Mirror

Scans a directory of photos, groups near-duplicate/burst shots of the same
scene together, and picks the best face photo out of each group — so you
don't have to manually sort through ten near-identical shots to find the one
where nobody blinked.

## How it works

1. **Scan** — reads every image in the input directory (optionally
   recursive), pulling the EXIF capture timestamp (falls back to file
   modified time) and a perceptual hash of each one.
2. **Group** — chains images into a "burst" when consecutive shots (by
   timestamp) are both close in time (`--time-gap`, default 3s) *and* close
   in perceptual hash (`--hash-threshold`, default 10). Images that don't
   match anything nearby become their own singleton group, so unrelated
   photos are never lumped together.
3. **Score** — detects every face in every image and scores it on:
   | Criterion | How it's measured |
   |---|---|
   | Sharpness | Laplacian variance over the face region, normalized against the other shots in its group (so it's relative to that burst, not an absolute threshold) |
   | Eyes open | Eye-aspect-ratio from face-mesh landmarks — catches full blinks and half-closed eyes |
   | Not occluded | Face-detector confidence, which drops when part of the face is covered or out of frame |
   | Natural expression | Penalizes a wide-open jaw (mid-word/yawn) or lopsided mouth corners (grimace/smirk); small reward for lifted corners (smile) |
   | Facing the camera | Yaw/pitch estimated from landmark geometry (nose position relative to eyes/forehead/chin) |
   | Centered in frame | Distance from the face's center to the image's center |
   | Reasonable size | Penalizes tiny/background faces |

   Each face's subscores are combined into a weighted total (see `WEIGHTS`
   near the top of `main.py` if you want to rebalance them). For
   images with more than one face, the image's score is the area-weighted
   average across faces, so the main subject matters more than someone in
   the background.
4. **Pick & arrange** — the highest-scoring image in each group is
   "shortlisted"; every other image in the group is "eliminated." If a
   group has no detected faces at all, it falls back to picking the
   sharpest frame instead. Shortlisted and eliminated photos are then
   arranged according to `--layout` (see below), and a `score.csv` report
   is written with every image's subscores, so you can see why a
   particular shot won.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Why `mediapipe==0.10.14`, not the latest?** The current mediapipe 1.0.x
> "Tasks" API hard-crashes on some macOS versions with a Metal-service bug
> inside Google's own compiled binary (an uncatchable native abort, not
> fixable from application code). 0.10.14 uses mediapipe's older
> `solutions` API instead — CPU-only, bundled models, no external downloads,
> no GPU dependency, and it's what this script is built against. It requires
> Python ≤ 3.13.

## Usage

```bash
python3 main.py /path/to/photos
```

By default (`--layout ugly`), shortlisted photos stay exactly where they
are and eliminated photos are moved into `/path/to/photos/ugly/`, alongside
`score.csv`.

### File arrangement (`--layout`)

| Layout | Shortlisted photos | Eliminated photos | `score.csv` |
|---|---|---|---|
| `ugly` (default) | stay in place | moved to `ugly/` | in `ugly/` |
| `beauty` | moved to `beauty/` | stay in place | in `beauty/` |
| `split` | moved to `beauty/` | moved to `ugly/` | in the input directory |

Photos are **moved**, not copied — pick `--dry-run` first if you want to
check the results before anything is relocated. Pass `--link` to symlink
into `ugly`/`beauty` instead of moving, if you'd rather leave your library
untouched. `--ugly-dir` and `--beauty-dir` override the default subfolder
locations, and `--report` overrides where `score.csv` is written.

With `-r/--recursive`, any file already inside the `ugly`/`beauty` folder is
skipped from scanning (so a second run doesn't re-ingest its own output) —
if your library happens to have its own subfolder named `ugly` or `beauty`
(e.g. an "ugly sweater party" album), point `--ugly-dir`/`--beauty-dir`
somewhere else to avoid the collision. The tool prints how many files it
skipped for this reason.

### Options

| Flag | Default | Description |
|---|---|---|
| `-r, --recursive` | off | Also scan subdirectories |
| `--time-gap` | `3.0` | Max seconds between shots to count as the same burst |
| `--hash-threshold` | `10` | Max perceptual-hash distance to count as the same scene |
| `--min-detection-confidence` | `0.5` | Min face-detector confidence |
| `--max-faces` | `5` | Max faces to analyze per image |
| `--detector-model` | `1` | `0` = short-range (<2m), `1` = full-range |
| `--layout` | `ugly` | File arrangement mode: `ugly`, `beauty`, or `split` (see above) |
| `--ugly-dir` | `<input_dir>/ugly` | Where eliminated photos go |
| `--beauty-dir` | `<input_dir>/beauty` | Where shortlisted photos go |
| `--link` | off | Symlink into place instead of moving the originals |
| `--report` | depends on `--layout` | Where to write `score.csv` |
| `--skip-singletons` | off | Leave images with no similar neighbors untouched instead of shortlisting them |
| `--top-n` | off | Only shortlist the N highest-scoring winners across all groups |
| `--top-percent` | off | Only shortlist the top X% highest-scoring winners across all groups (0-100, rounded up, minimum 1) |
| `--dry-run` | off | Score and write `score.csv` only; don't move or symlink any photos |

`--top-n` and `--top-percent` are mutually exclusive and apply *after* the
one-winner-per-group selection — they trim the shortlist down further, they
don't shortlist more than one photo per group. `score.csv` gets a
`Shortlisted` column showing exactly which files made the final cut vs.
which were just the best of their group but got trimmed.

Start with `--dry-run` on a new photo set and check `score.csv` before
letting it move anything — it's the quickest way to see if `--time-gap` or
`--hash-threshold` need adjusting for your camera's burst behavior.

## score.csv

Every image gets a row, with subscores reported as percentages (0-100, not
0-1):

| Column | Meaning |
|---|---|
| `Group ID` | Which burst/singleton group the image belongs to |
| `File` | Path to the image |
| `Group Winner` | Whether this was the highest-scoring image in its group |
| `Shortlisted` | Whether this image actually made the final cut (after `--skip-singletons`/`--top-n`/`--top-percent`) |
| `Group Size` | Number of images in the group |
| `Faces Detected` | Number of faces found in the image |
| `Error` | Set if the image couldn't be read/scored |
| `Image Score (%)` | The image's overall score (area-weighted average across faces) |
| `Sharpness (%)`, `Eyes Open (%)`, `Unoccluded (%)`, `Expression (%)`, `Facing Camera (%)`, `Centered (%)`, `Face Size (%)` | Subscores for the image's top-scoring face |

## Tuning

- **Bursts not grouping?** Raise `--hash-threshold` (perceptual hash is not
  shift-invariant, so a subject that moves noticeably between frames, or a
  camera that reframes slightly, can push the distance above the default).
- **Unrelated photos grouped together?** Lower `--hash-threshold` or
  `--time-gap`.
- **Scoring weights** live in the `WEIGHTS` dict near the top of
  `main.py` — e.g. bump `eyes_open` if blinking is your main
  complaint, or drop `centered` to near-zero if your subject isn't expected
  to be centered.

## License

MIT — see [LICENSE](LICENSE).
