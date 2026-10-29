"""Batch stabilize time-space images from lower-island texture.

Low-contrast (dark) frames are rejected and their displacement is obtained by
smooth interpolation between reliable neighbouring frames.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

from make_spacetime_image import add_axes


def island_profile(strip: np.ndarray, y0: int, y1: int) -> np.ndarray:
    gray = np.mean(strip[y0:y1].astype(np.float32), axis=(1, 2))
    # Remove illumination level and retain the stable horizontal texture.
    gray = np.convolve(gray, np.ones(5, dtype=np.float32) / 5.0, mode="same")
    profile = np.diff(gray, prepend=gray[0])
    profile -= profile.mean()
    return profile


def match_shift(reference: np.ndarray, profile: np.ndarray, maximum: int) -> tuple[float, float, bool]:
    reference_norm = float(np.linalg.norm(reference))
    profile_norm = float(np.linalg.norm(profile))
    if reference_norm < 1e-6 or profile_norm < 1e-6:
        return 0.0, 0.0, False
    offsets = range(-maximum, maximum + 1)
    scores: list[float] = []
    for offset in offsets:
        if offset < 0:
            first, second = reference[-offset:], profile[:offset]
        elif offset > 0:
            first, second = reference[:-offset], profile[offset:]
        else:
            first, second = reference, profile
        denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
        scores.append(float(np.dot(first, second) / denominator) if denominator > 1e-6 else 0.0)
    index = int(np.argmax(scores))
    shift = float(list(offsets)[index])
    if 0 < index < len(scores) - 1:
        left, middle, right = scores[index - 1:index + 2]
        denominator = left - 2 * middle + right
        if abs(denominator) > 1e-9:
            shift += 0.5 * (left - right) / denominator
    return shift, scores[index], index not in (0, len(scores) - 1)


def smoothed_interpolation(shifts: np.ndarray, reliable: np.ndarray) -> np.ndarray:
    positions = np.arange(len(shifts), dtype=float)
    if reliable.sum() < 2:
        return np.zeros_like(shifts)
    interpolated = np.interp(positions, positions[reliable], shifts[reliable])
    # A short moving average suppresses pixel-level tracking noise while
    # preserving camera motion at the 240 fps time scale.
    kernel = np.ones(9, dtype=float) / 9.0
    padded = np.pad(interpolated, (4, 4), mode="edge")
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def process(input_path: Path, output_dir: Path, strip_width: int, fps: float, pixels_per_cm: float) -> str:
    base = input_path.name.removesuffix("_x1352_w3_spacetime.png")
    raw_output = output_dir / f"{base}_raw.png"
    axes_output = output_dir / f"{base}.png"
    shifts_output = output_dir / f"{base}_shifts.csv"
    image = np.asarray(Image.open(input_path).convert("RGB"))
    height, width, _ = image.shape
    count = width // strip_width
    y0, y1, maximum = 650, 1320, 30
    profiles = [island_profile(image[:, index * strip_width:(index + 1) * strip_width], y0, y1) for index in range(count)]
    reference = profiles[0]
    raw_shifts = np.empty(count, dtype=np.float32)
    correlations = np.empty(count, dtype=np.float32)
    within_range = np.zeros(count, dtype=bool)
    contrast = np.asarray([float(np.std(profile)) for profile in profiles], dtype=np.float32)
    for index, profile in enumerate(profiles):
        raw_shifts[index], correlations[index], within_range[index] = match_shift(reference, profile, maximum)
    contrast_floor = max(float(np.median(contrast)) * 0.20, 0.75)
    reliable = (correlations >= 0.32) & within_range & (contrast >= contrast_floor)
    reliable[0] = True
    applied_shifts = smoothed_interpolation(raw_shifts, reliable)
    result = np.zeros((height, width, 4), dtype=np.uint8)
    rows = np.arange(height, dtype=np.float32)
    for index, shift in enumerate(applied_shifts):
        source_rows = np.rint(rows + shift).astype(int)
        valid = (source_rows >= 0) & (source_rows < height)
        x0, x1 = index * strip_width, (index + 1) * strip_width
        result[valid, x0:x1, :3] = image[source_rows[valid], x0:x1]
        result[valid, x0:x1, 3] = 255
    Image.fromarray(result, "RGBA").save(raw_output, compress_level=6)
    with shifts_output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame_index", "time_s", "raw_island_shift_px", "applied_shift_px", "match_correlation", "contrast", "reliable"])
        for index in range(count):
            writer.writerow([index, index / fps, float(raw_shifts[index]), float(applied_shifts[index]), float(correlations[index]), float(contrast[index]), bool(reliable[index])])
    add_axes(raw_output, axes_output, count, fps, strip_width, pixels_per_cm, 2.0)
    return f"{base}: reliable {int(reliable.sum())}/{count}, applied {applied_shifts.min():.1f} to {applied_shifts.max():.1f} px"


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch stabilize time-space images by lower-island texture")
    parser.add_argument("--input", type=Path, default=Path("spacetime_output/all"))
    parser.add_argument("--output", type=Path, default=Path("spacetime_output/all2"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--fps", type=float, default=240.0)
    parser.add_argument("--pixels-per-cm", type=float, default=93.4)
    args = parser.parse_args()
    inputs = sorted(path for path in args.input.glob("*_x1352_w3_spacetime.png") if "stabilized" not in path.name)
    if not inputs:
        raise SystemExit("No input time-space images found")
    args.output.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process, path, args.output, 3, args.fps, args.pixels_per_cm): path for path in inputs}
        for index, future in enumerate(as_completed(futures), start=1):
            print(f"[{index}/{len(inputs)}] {future.result()}", flush=True)


if __name__ == "__main__":
    main()
