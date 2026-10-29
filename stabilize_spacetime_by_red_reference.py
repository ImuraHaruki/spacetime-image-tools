"""Stabilize an existing time-space image using its continuous red ruler mark."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from make_spacetime_image import add_axes


def red_profile(strip: np.ndarray, y0: int, y1: int) -> np.ndarray:
    values = strip[y0:y1].astype(np.float32)
    profile = np.mean(values[:, :, 0] - (values[:, :, 1] + values[:, :, 2]) / 2.0, axis=1)
    profile = np.diff(profile, prepend=profile[0])
    profile -= profile.mean()
    scale = profile.std()
    return profile / scale if scale > 1e-6 else profile


def estimate_shift(reference: np.ndarray, profile: np.ndarray, maximum: int) -> float:
    offsets = range(-maximum, maximum + 1)
    scores = []
    for offset in offsets:
        if offset < 0:
            score = np.dot(reference[-offset:], profile[:offset])
        elif offset > 0:
            score = np.dot(reference[:-offset], profile[offset:])
        else:
            score = np.dot(reference, profile)
        scores.append(float(score))
    index = int(np.argmax(scores))
    correction = 0.0
    if 0 < index < len(scores) - 1:
        left, middle, right = scores[index - 1:index + 2]
        denominator = left - 2 * middle + right
        if abs(denominator) > 1e-9:
            correction = 0.5 * (left - right) / denominator
    return list(offsets)[index] + correction


def main() -> None:
    parser = argparse.ArgumentParser(description="Stabilize a time-space image from its red reference mark")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shifts-csv", type=Path, required=True)
    parser.add_argument("--annotated-output", type=Path)
    parser.add_argument("--strip-width", type=int, default=3)
    parser.add_argument("--reference-y0", type=int, default=1350)
    parser.add_argument("--reference-y1", type=int, default=1515)
    parser.add_argument("--maximum-shift", type=int, default=24)
    parser.add_argument("--fps", type=float, default=240.0)
    parser.add_argument("--pixels-per-cm", type=float, default=93.4)
    args = parser.parse_args()
    image = np.asarray(Image.open(args.input).convert("RGB"))
    height, width, _ = image.shape
    count = width // args.strip_width
    if width % args.strip_width or not (0 <= args.reference_y0 < args.reference_y1 <= height):
        raise ValueError("input width or reference y range is invalid")
    reference = red_profile(image[:, :args.strip_width], args.reference_y0, args.reference_y1)
    shifts = np.asarray([
        estimate_shift(reference, red_profile(image[:, index * args.strip_width:(index + 1) * args.strip_width], args.reference_y0, args.reference_y1), args.maximum_shift)
        for index in range(count)
    ], dtype=np.float32)
    result = np.zeros((height, width, 4), dtype=np.uint8)
    rows = np.arange(height, dtype=np.float32)
    for index, shift in enumerate(shifts):
        source_rows_float = np.rint(rows + shift).astype(int)
        valid_rows = (source_rows_float >= 0) & (source_rows_float < height)
        x0, x1 = index * args.strip_width, (index + 1) * args.strip_width
        result[valid_rows, x0:x1, :3] = image[source_rows_float[valid_rows], x0:x1]
        result[valid_rows, x0:x1, 3] = 255
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result).save(args.output, compress_level=6)
    with args.shifts_csv.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame_index", "time_s", "red_reference_shift_px"])
        writer.writerows((index, index / args.fps, float(shift)) for index, shift in enumerate(shifts))
    if args.annotated_output:
        add_axes(args.output, args.annotated_output, count, args.fps, args.strip_width, args.pixels_per_cm, 2.0)
    print(f"Created {args.output}; shift range {shifts.min():.2f} to {shifts.max():.2f} px")


if __name__ == "__main__":
    main()
