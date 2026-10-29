"""Create a vertically stabilized time-space image from video frames.

The vertical displacement of each frame is estimated from stable horizontal
texture in a user-selected reference band, then applied to the extracted strip.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from make_spacetime_image import add_axes, find_frames


def vertical_profile(path: Path, x0: int, x1: int, y0: int, y1: int) -> np.ndarray:
    with Image.open(path) as opened:
        # Downsampling only across x retains the vertical bands while making
        # thousands of-frame processing practical.
        gray = opened.convert("L").crop((x0, y0, x1, y1)).resize((96, y1 - y0), Image.Resampling.BILINEAR)
        gray = np.asarray(gray, dtype=np.float32)
    profile = np.mean(gray, axis=1)
    profile = np.diff(profile, prepend=profile[0])
    profile -= profile.mean()
    scale = profile.std()
    return profile / scale if scale > 1e-6 else profile


def estimate_shifts(frames: list[Path], x0: int, x1: int, y0: int, y1: int, maximum_shift: int) -> np.ndarray:
    reference = vertical_profile(frames[0], x0, x1, y0, y1)
    shifts: list[float] = []
    offsets = range(-maximum_shift, maximum_shift + 1)
    for frame in frames:
        profile = vertical_profile(frame, x0, x1, y0, y1)
        scores = []
        for offset in offsets:
            if offset < 0:
                score = np.dot(reference[-offset:], profile[:offset])
            elif offset > 0:
                score = np.dot(reference[:-offset], profile[offset:])
            else:
                score = np.dot(reference, profile)
            scores.append(float(score))
        best_index = int(np.argmax(scores))
        best_offset = list(offsets)[best_index]
        # Quadratic peak fitting provides a small sub-pixel refinement.
        correction = 0.0
        if 0 < best_index < len(scores) - 1:
            left, middle, right = scores[best_index - 1:best_index + 2]
            denominator = left - 2 * middle + right
            if abs(denominator) > 1e-9:
                correction = 0.5 * (left - right) / denominator
        shifts.append(best_offset + correction)
    return np.asarray(shifts, dtype=np.float32)


def shifted_strip(strip: np.ndarray, shift: float) -> Image.Image:
    rows = np.arange(strip.shape[0], dtype=np.float32) - shift
    rows = np.clip(np.rint(rows).astype(int), 0, strip.shape[0] - 1)
    return Image.fromarray(strip[rows], "RGB")


def make_stabilized_spacetime(
    folder: Path, output: Path, x: int, strip_width: int, reference_x0: int, reference_x1: int,
    reference_y0: int, reference_y1: int, maximum_shift: int,
) -> tuple[int, np.ndarray]:
    frames = find_frames(folder)
    with Image.open(frames[0]) as first:
        width, height = first.size
    if not (0 <= reference_x0 < reference_x1 <= width and 0 <= reference_y0 < reference_y1 <= height):
        raise ValueError("reference area must be inside the source frames")
    left = max(0, min(width - strip_width, x - strip_width // 2))
    right = left + strip_width
    shifts = estimate_shifts(frames, reference_x0, reference_x1, reference_y0, reference_y1, maximum_shift)
    canvas = Image.new("RGB", (len(frames) * strip_width, height))
    for index, (frame, shift) in enumerate(zip(frames, shifts)):
        with Image.open(frame) as opened:
            rgb = np.asarray(opened.convert("RGB").crop((left, 0, right, height)))
        # A positive estimated shift means that the frame content moved downward
        # relative to the reference, so apply the opposite translation.
        canvas.paste(shifted_strip(rgb, float(-shift)), (index * strip_width, 0))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, compress_level=6)
    return len(frames), shifts


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a vertically stabilized time-space image")
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shifts-csv", type=Path, required=True)
    parser.add_argument("--annotated-output", type=Path)
    parser.add_argument("--x", type=int, default=1352)
    parser.add_argument("--strip-width", type=int, default=3)
    parser.add_argument("--reference-x0", type=int, default=250)
    parser.add_argument("--reference-x1", type=int, default=2200)
    parser.add_argument("--reference-y0", type=int, default=650)
    parser.add_argument("--reference-y1", type=int, default=1450)
    parser.add_argument("--maximum-shift", type=int, default=24)
    parser.add_argument("--fps", type=float, default=240.0)
    parser.add_argument("--pixels-per-cm", type=float, default=93.4)
    args = parser.parse_args()
    count, shifts = make_stabilized_spacetime(
        args.folder, args.output, args.x, args.strip_width, args.reference_x0, args.reference_x1,
        args.reference_y0, args.reference_y1, args.maximum_shift,
    )
    with args.shifts_csv.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame_index", "time_s", "vertical_shift_px"])
        for index, shift in enumerate(shifts):
            writer.writerow([index, index / args.fps, float(shift)])
    if args.annotated_output:
        add_axes(args.output, args.annotated_output, count, args.fps, args.strip_width, args.pixels_per_cm)
    print(f"Created {args.output} ({count} frames); shift range {shifts.min():.2f} to {shifts.max():.2f} px")


if __name__ == "__main__":
    main()
