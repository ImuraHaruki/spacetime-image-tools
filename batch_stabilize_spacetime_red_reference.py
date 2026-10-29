"""Batch-create transparent, red-reference-stabilized time-space images."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

from make_spacetime_image import add_axes
from stabilize_spacetime_by_red_reference import estimate_shift, red_profile


def process(input_path: Path, output_dir: Path, strip_width: int, fps: float, pixels_per_cm: float) -> str:
    base = input_path.name.removesuffix("_x1352_w3_spacetime.png")
    raw_output = output_dir / f"{base}_raw.png"
    axes_output = output_dir / f"{base}.png"
    shifts_output = output_dir / f"{base}_shifts.csv"
    image = np.asarray(Image.open(input_path).convert("RGB"))
    height, width, _ = image.shape
    count = width // strip_width
    y0, y1, maximum = 1350, 1515, 24
    reference = red_profile(image[:, :strip_width], y0, y1)
    shifts = np.asarray([
        estimate_shift(reference, red_profile(image[:, index * strip_width:(index + 1) * strip_width], y0, y1), maximum)
        for index in range(count)
    ], dtype=np.float32)
    result = np.zeros((height, width, 4), dtype=np.uint8)
    rows = np.arange(height, dtype=np.float32)
    for index, shift in enumerate(shifts):
        source_rows = np.rint(rows + shift).astype(int)
        valid = (source_rows >= 0) & (source_rows < height)
        x0, x1 = index * strip_width, (index + 1) * strip_width
        result[valid, x0:x1, :3] = image[source_rows[valid], x0:x1]
        result[valid, x0:x1, 3] = 255
    Image.fromarray(result, "RGBA").save(raw_output, compress_level=6)
    with shifts_output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame_index", "time_s", "red_reference_shift_px"])
        writer.writerows((index, index / fps, float(shift)) for index, shift in enumerate(shifts))
    add_axes(raw_output, axes_output, count, fps, strip_width, pixels_per_cm, 2.0)
    return f"{base}: {shifts.min():.1f} to {shifts.max():.1f} px"


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch stabilize time-space images with a red reference")
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
