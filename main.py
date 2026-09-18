#!/usr/bin/env python3
"""
main.py — pick the best face photo out of each burst of similar images.

Pipeline:
  1. Scan a directory for images, read EXIF timestamp (fallback: file mtime)
     and a perceptual hash of each image.
  2. Group images into "bursts": consecutive-in-time images whose capture
     gap is small AND whose perceptual hashes are close (same scene).
  3. For every face found in every image, score it on:
       - sharpness         (Laplacian variance over the face region,
                             normalized against the rest of its group)
       - eyes open         (eye-aspect-ratio from face-mesh landmarks)
       - not occluded      (face-detector confidence as a proxy)
       - natural expression (penalizes a wide-open jaw / lopsided mouth,
                             small reward for lifted mouth corners = smile)
       - facing the camera (yaw/pitch estimated from landmark geometry)
       - centered in frame (distance of face center from image center)
       - reasonable size   (face not tiny/background)
  4. Picks the highest-scoring image in each group (the "shortlisted" one)
     and arranges the photos according to --layout:
       - ugly   (default): move eliminated photos into an 'ugly' subfolder;
                 shortlisted photos stay where they are
       - beauty: move shortlisted photos into a 'beauty' subfolder;
                 eliminated photos stay where they are
       - split:  move shortlisted photos into 'beauty' and eliminated
                 photos into 'ugly'
     A score.csv report is written alongside with every subscore as a
     percentage.

Usage:
  python main.py /path/to/photos [options]

Uses mediapipe's legacy `solutions` face_detection/face_mesh models, which
are bundled in the package (no external downloads, no GPU dependency).
NOTE: this requires mediapipe<1.0 (e.g. 0.10.14) on Python <=3.13 — the 1.0.x
"Tasks" API hard-crashes on some macOS versions due to an unrelated Metal
service bug in Google's compiled binary.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageOps

try:
    import imagehash
except ImportError:
    print("Missing dependency 'imagehash'. Install with: pip install imagehash", file=sys.stderr)
    raise

import mediapipe as mp

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Face-mesh landmark indices (468/478-point topology) used for geometry-based
# scoring. See https://github.com/google-ai-edge/mediapipe for the diagram.
RIGHT_EYE_EAR = [33, 160, 158, 133, 153, 144]   # p1,p2,p3,p4,p5,p6
LEFT_EYE_EAR = [362, 385, 387, 263, 373, 380]
MOUTH_LEFT_CORNER, MOUTH_RIGHT_CORNER = 61, 291
MOUTH_TOP, MOUTH_BOTTOM = 13, 14
NOSE_TIP = 1
FOREHEAD, CHIN = 10, 152
EYE_OUTER_L, EYE_OUTER_R = 33, 263

EAR_CLOSED, EAR_OPEN = 0.15, 0.30  # calibration for eyes-open score ramp

# Final per-face score weights (must sum to ~1.0; tweak to taste).
WEIGHTS = {
    "sharpness": 0.25,
    "eyes_open": 0.20,
    "unoccluded": 0.15,
    "expression": 0.15,
    "facing_camera": 0.10,
    "centered": 0.10,
    "face_size": 0.05,
}

# A face taking up this fraction of the frame area (or more) gets full
# marks on the "face_size" subscore.
TARGET_FACE_AREA_FRAC = 0.04


def iter_images(root: Path, recursive: bool):
    pattern = "**/*" if recursive else "*"
    for p in sorted(root.glob(pattern)):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and not p.name.startswith("."):
            yield p


def is_within(path: Path, folder: Path) -> bool:
    try:
        path.resolve().relative_to(folder.resolve())
        return True
    except ValueError:
        return False


def read_capture_time(pil_img: Image.Image, fallback_path: Path) -> float:
    try:
        exif = pil_img.getexif()
        dt_str = exif.get(36867) or exif.get(306)
        if not dt_str:
            exif_ifd = exif.get_ifd(0x8769)
            dt_str = exif_ifd.get(36867)
        if dt_str:
            return datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S").timestamp()
    except Exception:
        pass
    return fallback_path.stat().st_mtime


def load_image(path: Path):
    """Returns (rgb_uint8_array, gray_uint8_array, capture_ts), or None if
    the file can't be read as an image. Applies EXIF orientation."""
    try:
        pil_img = Image.open(path)
        pil_img.load()
    except Exception:
        return None
    capture_ts = read_capture_time(pil_img, path)
    pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
    rgb = np.array(pil_img)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return rgb, gray, capture_ts


def bbox_from_landmarks(landmarks, img_w: int, img_h: int):
    xs = [lm.x for lm in landmarks.landmark]
    ys = [lm.y for lm in landmarks.landmark]
    x0, x1 = min(xs) * img_w, max(xs) * img_w
    y0, y1 = min(ys) * img_h, max(ys) * img_h
    return x0, y0, x1 - x0, y1 - y0


def bbox_iou(a, b) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def pt(landmarks, idx, img_w, img_h):
    lm = landmarks.landmark[idx]
    return lm.x * img_w, lm.y * img_h


def dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def eye_aspect_ratio(landmarks, idxs, img_w, img_h) -> float:
    p = [pt(landmarks, i, img_w, img_h) for i in idxs]
    horiz = dist(p[0], p[3]) or 1.0
    return (dist(p[1], p[5]) + dist(p[2], p[4])) / (2 * horiz)


@dataclass
class FaceScore:
    bbox: tuple
    area_frac: float
    sharpness_raw: float
    sharpness: float = 0.0  # filled in after group-relative normalization
    eyes_open: float = 0.0
    unoccluded: float = 0.0
    expression: float = 0.0
    facing_camera: float = 0.0
    centered: float = 0.0
    face_size: float = 0.0
    total: float = 0.0

    def finalize(self):
        total_w = sum(WEIGHTS.values())
        self.total = (
            self.sharpness * WEIGHTS["sharpness"]
            + self.eyes_open * WEIGHTS["eyes_open"]
            + self.unoccluded * WEIGHTS["unoccluded"]
            + self.expression * WEIGHTS["expression"]
            + self.facing_camera * WEIGHTS["facing_camera"]
            + self.centered * WEIGHTS["centered"]
            + self.face_size * WEIGHTS["face_size"]
        ) / total_w


@dataclass
class ImageRecord:
    path: Path
    timestamp: float
    phash: Optional["imagehash.ImageHash"] = None
    faces: list = field(default_factory=list)
    error: Optional[str] = None
    group_id: Optional[int] = None
    image_score: float = 0.0


class FaceScorer:
    def __init__(self, min_detection_confidence: float, max_faces: int, model_selection: int):
        self.detector = mp.solutions.face_detection.FaceDetection(
            model_selection=model_selection, min_detection_confidence=min_detection_confidence
        )
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
        )

    def score_image(self, rgb: np.ndarray, gray: np.ndarray) -> list:
        img_h, img_w = gray.shape[:2]

        det_result = self.detector.process(rgb)
        det_boxes = []
        if det_result.detections:
            for det in det_result.detections:
                bb = det.location_data.relative_bounding_box
                det_boxes.append((
                    (bb.xmin * img_w, bb.ymin * img_h, bb.width * img_w, bb.height * img_h),
                    det.score[0] if det.score else 0.5,
                ))

        mesh_result = self.mesh.process(rgb)
        faces = []
        if mesh_result.multi_face_landmarks:
            for landmarks in mesh_result.multi_face_landmarks:
                bbox = bbox_from_landmarks(landmarks, img_w, img_h)
                faces.append(self._score_face(bbox, landmarks, gray, img_w, img_h, det_boxes))
        return faces

    def _score_face(self, bbox, landmarks, gray, img_w, img_h, det_boxes) -> FaceScore:
        x, y, w, h = bbox
        area_frac = (w * h) / (img_w * img_h)

        # --- occlusion / confidence proxy: match against detector boxes ---
        best_iou, best_score = 0.0, 0.5
        for det_bbox, det_score in det_boxes:
            iou = bbox_iou(bbox, det_bbox)
            if iou > best_iou:
                best_iou, best_score = iou, det_score
        unoccluded = best_score if best_iou > 0.3 else 0.5

        # --- sharpness: Laplacian variance over an expanded face crop ---
        pad_w, pad_h = w * 0.2, h * 0.2
        cx0 = max(0, int(x - pad_w))
        cy0 = max(0, int(y - pad_h))
        cx1 = min(img_w, int(x + w + pad_w))
        cy1 = min(img_h, int(y + h + pad_h))
        crop = gray[cy0:cy1, cx0:cx1]
        sharpness_raw = float(cv2.Laplacian(crop, cv2.CV_64F).var()) if crop.size else 0.0

        # --- eyes open: eye-aspect-ratio, averaged over both eyes ---
        ear_r = eye_aspect_ratio(landmarks, RIGHT_EYE_EAR, img_w, img_h)
        ear_l = eye_aspect_ratio(landmarks, LEFT_EYE_EAR, img_w, img_h)
        ear = (ear_r + ear_l) / 2
        eyes_open = clamp01((ear - EAR_CLOSED) / (EAR_OPEN - EAR_CLOSED))

        # --- expression: mouth geometry relative to face scale ---
        face_h = dist(pt(landmarks, FOREHEAD, img_w, img_h), pt(landmarks, CHIN, img_w, img_h)) or 1.0
        mouth_top = pt(landmarks, MOUTH_TOP, img_w, img_h)
        mouth_bottom = pt(landmarks, MOUTH_BOTTOM, img_w, img_h)
        corner_l = pt(landmarks, MOUTH_LEFT_CORNER, img_w, img_h)
        corner_r = pt(landmarks, MOUTH_RIGHT_CORNER, img_w, img_h)

        mouth_open_ratio = dist(mouth_top, mouth_bottom) / face_h
        lip_center_y = (mouth_top[1] + mouth_bottom[1]) / 2
        corner_avg_y = (corner_l[1] + corner_r[1]) / 2
        smile_lift = (lip_center_y - corner_avg_y) / face_h  # positive = corners lifted (smile)
        corner_asymmetry = abs(corner_l[1] - corner_r[1]) / face_h  # smirk/grimace

        weird_penalty = clamp01(max(0.0, mouth_open_ratio - 0.06) * 4.0) + clamp01(corner_asymmetry * 8.0)
        smile_bonus = clamp01(max(0.0, smile_lift) * 6.0)
        expression = clamp01(1.0 - weird_penalty + 0.25 * smile_bonus)

        # --- head pose from landmark geometry (heuristic, not a real solvePnP) ---
        lx, _ = pt(landmarks, EYE_OUTER_L, img_w, img_h)
        rx, _ = pt(landmarks, EYE_OUTER_R, img_w, img_h)
        nx, ny = pt(landmarks, NOSE_TIP, img_w, img_h)
        _, fy = pt(landmarks, FOREHEAD, img_w, img_h)
        _, cy = pt(landmarks, CHIN, img_w, img_h)
        eye_span = abs(rx - lx) / 2 or 1.0
        yaw_ratio = (nx - (lx + rx) / 2) / eye_span
        v_span = abs(cy - fy) / 2 or 1.0
        pitch_ratio = (ny - (fy + cy) / 2) / v_span
        facing_camera = clamp01(1.0 - min(1.0, math.hypot(yaw_ratio, pitch_ratio)))

        # --- centering: distance of face center from image center ---
        face_cx, face_cy = x + w / 2, y + h / 2
        img_cx, img_cy = img_w / 2, img_h / 2
        d = math.hypot(face_cx - img_cx, face_cy - img_cy)
        diag_half = math.hypot(img_w / 2, img_h / 2)
        centered = clamp01(1.0 - d / diag_half)

        # --- size: reward faces that are a reasonable fraction of the frame ---
        face_size = clamp01(area_frac / TARGET_FACE_AREA_FRAC)

        return FaceScore(
            bbox=bbox,
            area_frac=area_frac,
            sharpness_raw=sharpness_raw,
            eyes_open=eyes_open,
            unoccluded=unoccluded,
            expression=expression,
            facing_camera=facing_camera,
            centered=centered,
            face_size=face_size,
        )

    def close(self):
        self.detector.close()
        self.mesh.close()


def group_images(records: list, time_gap: float, hash_threshold: int) -> list:
    n = len(records)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    order = sorted(range(n), key=lambda i: records[i].timestamp)
    for k in range(len(order) - 1):
        i, j = order[k], order[k + 1]
        dt = abs(records[j].timestamp - records[i].timestamp)
        if dt > time_gap:
            continue
        if records[i].phash is not None and records[j].phash is not None:
            if (records[i].phash - records[j].phash) <= hash_threshold:
                union(i, j)
        else:
            union(i, j)

    groups: dict = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=lambda idxs: min(records[i].timestamp for i in idxs))


def normalize_sharpness_in_group(records: list, group: list):
    raws = [f.sharpness_raw for i in group for f in records[i].faces]
    if not raws:
        return
    lo, hi = min(raws), max(raws)
    span = hi - lo
    for i in group:
        for f in records[i].faces:
            f.sharpness = 1.0 if span == 0 else clamp01((f.sharpness_raw - lo) / span)
            f.finalize()


def score_image_from_faces(rec: ImageRecord):
    if not rec.faces:
        rec.image_score = 0.0
        return
    total_area = sum(f.area_frac for f in rec.faces) or 1.0
    rec.image_score = sum(f.total * f.area_frac for f in rec.faces) / total_area


def overall_sharpness(path: Path) -> float:
    gray = cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_dir", type=Path, help="Directory of photos to scan")
    parser.add_argument("-r", "--recursive", action="store_true", help="Scan subdirectories too")
    parser.add_argument("--time-gap", type=float, default=3.0, help="Max seconds between shots to be considered the same burst (default: 3.0)")
    parser.add_argument("--hash-threshold", type=int, default=10, help="Max perceptual-hash distance to be considered the same scene (default: 10)")
    parser.add_argument("--min-detection-confidence", type=float, default=0.5, help="Min face-detector confidence (default: 0.5)")
    parser.add_argument("--max-faces", type=int, default=5, help="Max faces to analyze per image (default: 5)")
    parser.add_argument("--detector-model", type=int, choices=[0, 1], default=1, help="mediapipe face-detector model: 0=short-range (<2m), 1=full-range (default: 1)")
    parser.add_argument(
        "--layout", choices=["ugly", "beauty", "split"], default="ugly",
        help=(
            "How to arrange photos after scoring (default: ugly). score.csv always stays in the input directory. "
            "ugly = move eliminated photos into an 'ugly' subfolder, shortlisted photos stay put; "
            "beauty = move shortlisted photos into a 'beauty' subfolder, eliminated photos stay put; "
            "split = move shortlisted into 'beauty' and eliminated into 'ugly'"
        ),
    )
    parser.add_argument("--ugly-dir", type=Path, default=None, help="Where eliminated photos go (default: <input_dir>/ugly)")
    parser.add_argument("--beauty-dir", type=Path, default=None, help="Where shortlisted photos go (default: <input_dir>/beauty)")
    parser.add_argument("--link", action="store_true", help="Symlink photos into place instead of moving the originals")
    parser.add_argument("--report", type=Path, default=None, help="score.csv path (default depends on --layout)")
    parser.add_argument("--skip-singletons", action="store_true", help="Leave images with no similar neighbors untouched instead of shortlisting them")
    parser.add_argument("--dry-run", action="store_true", help="Score and write score.csv only; don't move or symlink any photos")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--top-n", type=int, default=None, help="Only shortlist the N highest-scoring winners across all groups")
    selection.add_argument("--top-percent", type=float, default=None, help="Only shortlist the top X%% highest-scoring winners across all groups (0-100)")
    args = parser.parse_args()

    input_dir = args.input_dir
    if not input_dir.is_dir():
        parser.error(f"{input_dir} is not a directory")

    ugly_dir = args.ugly_dir or (input_dir / "ugly")
    beauty_dir = args.beauty_dir or (input_dir / "beauty")
    report_path = args.report if args.report is not None else input_dir / "score.csv"

    paths = list(iter_images(input_dir, args.recursive))
    # Don't re-ingest photos a previous run already relocated.
    kept, skipped = [], 0
    for p in paths:
        if is_within(p, ugly_dir) or is_within(p, beauty_dir):
            skipped += 1
        else:
            kept.append(p)
    paths = kept
    if skipped:
        print(
            f"Skipping {skipped} image(s) already inside {ugly_dir} or {beauty_dir} "
            "(previous run output, or a pre-existing folder with that name)."
        )
    if not paths:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(paths)} images.")
    scorer = FaceScorer(args.min_detection_confidence, args.max_faces, args.detector_model)

    records = []
    for idx, path in enumerate(paths, 1):
        loaded = load_image(path)
        if loaded is None:
            records.append(ImageRecord(path=path, timestamp=path.stat().st_mtime, error="unreadable"))
            continue
        rgb, gray, ts = loaded
        rec = ImageRecord(path=path, timestamp=ts)
        try:
            rec.phash = imagehash.phash(Image.fromarray(rgb))
            rec.faces = scorer.score_image(rgb, gray)
        except Exception as e:  # keep going on a bad file
            rec.error = str(e)
        records.append(rec)
        if idx % 10 == 0 or idx == len(paths):
            print(f"  scored {idx}/{len(paths)}")

    scorer.close()

    groups = group_images(records, args.time_gap, args.hash_threshold)
    print(f"Grouped into {len(groups)} burst(s).")

    for gid, group in enumerate(groups):
        for i in group:
            records[i].group_id = gid
        normalize_sharpness_in_group(records, group)
        for i in group:
            score_image_from_faces(records[i])

    winners = {}
    for gid, group in enumerate(groups):
        candidates = [i for i in group if records[i].error is None]
        if not candidates:
            continue
        with_faces = [i for i in candidates if records[i].faces]
        if with_faces:
            best = max(with_faces, key=lambda i: records[i].image_score)
        else:
            # No face detected anywhere in this group: fall back to the
            # sharpest overall frame instead of a face-based score.
            best = max(candidates, key=lambda i: overall_sharpness(records[i].path))
        winners[gid] = best

    # Candidate pool for shortlisting: winners, minus singletons if
    # requested, ranked best-first so --top-n / --top-percent can trim it.
    candidate_gids = [gid for gid in winners if not (args.skip_singletons and len(groups[gid]) == 1)]
    candidate_gids.sort(key=lambda gid: records[winners[gid]].image_score, reverse=True)

    if args.top_n is not None:
        selected_gids = set(candidate_gids[: max(0, args.top_n)])
    elif args.top_percent is not None:
        n_keep = math.ceil(len(candidate_gids) * args.top_percent / 100) if candidate_gids else 0
        n_keep = max(1, n_keep) if candidate_gids else 0
        selected_gids = set(candidate_gids[:n_keep])
    else:
        selected_gids = set(candidate_gids)

    # Every image ends up in exactly one bucket: shortlisted (the winner of
    # its group, after --top-n/--top-percent trimming), untouched (unreadable
    # files, or singletons skipped via --skip-singletons — left exactly where
    # they are), or eliminated (everything else).
    shortlisted = {winners[gid] for gid in selected_gids}
    untouched = {i for i, rec in enumerate(records) if rec.error is not None}
    if args.skip_singletons:
        for gid, group in enumerate(groups):
            if len(group) == 1:
                untouched.update(group)
    eliminated = {i for i in range(len(records)) if i not in shortlisted and i not in untouched}

    def relocate(indices, dest_root):
        for i in sorted(indices):
            src = records[i].path
            dest = dest_root / src.relative_to(input_dir)
            dest.parent.mkdir(parents=True, exist_ok=True)
            paths = [(src, dest)]
            sidecar_src = src.with_suffix(".txt")
            if sidecar_src.exists():
                paths.append((sidecar_src, dest.with_suffix(".txt")))
            for s, d in paths:
                if args.link:
                    if d.exists() or d.is_symlink():
                        d.unlink()
                    d.symlink_to(s.resolve())
                else:
                    shutil.move(str(s), str(d))

    if not args.dry_run:
        if args.layout in ("ugly", "split"):
            relocate(eliminated, ugly_dir)
        if args.layout in ("beauty", "split"):
            relocate(shortlisted, beauty_dir)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "Group ID", "File", "Group Winner", "Shortlisted", "Group Size", "Faces Detected", "Error",
            "Image Score (%)", "Sharpness (%)", "Eyes Open (%)", "Unoccluded (%)",
            "Expression (%)", "Facing Camera (%)", "Centered (%)", "Face Size (%)",
        ])
        for gid, group in enumerate(groups):
            for i in group:
                rec = records[i]
                is_winner = winners.get(gid) == i
                is_shortlisted = i in shortlisted
                if rec.faces:
                    top = max(rec.faces, key=lambda f: f.total)
                    face_cols = [
                        f"{top.sharpness * 100:.2f}", f"{top.eyes_open * 100:.2f}", f"{top.unoccluded * 100:.2f}",
                        f"{top.expression * 100:.2f}", f"{top.facing_camera * 100:.2f}", f"{top.centered * 100:.2f}",
                        f"{top.face_size * 100:.2f}",
                    ]
                else:
                    face_cols = ["", "", "", "", "", "", ""]
                writer.writerow([
                    gid, rec.path, is_winner, is_shortlisted, len(group), len(rec.faces), rec.error or "",
                    f"{rec.image_score * 100:.2f}", *face_cols,
                ])

    n_singletons = sum(1 for g in groups if len(g) == 1)
    print(f"Done. {len(groups)} groups ({n_singletons} singleton, {len(groups) - n_singletons} bursts).")
    if args.top_n is not None or args.top_percent is not None:
        print(f"Shortlisted {len(selected_gids)} of {len(candidate_gids)} candidate winners.")
    verb = "Symlinked" if args.link else "Moved"
    if not args.dry_run:
        if args.layout in ("ugly", "split"):
            print(f"{verb} {len(eliminated)} eliminated photo(s) to: {ugly_dir}")
        if args.layout in ("beauty", "split"):
            print(f"{verb} {len(shortlisted)} shortlisted photo(s) to: {beauty_dir}")
    else:
        print("Dry run: no photos were moved or symlinked.")
    print(f"score.csv: {report_path}")


if __name__ == "__main__":
    main()
