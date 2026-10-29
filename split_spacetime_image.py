"""Split a time-space image at an elapsed time while preserving its time axis."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from make_spacetime_image import add_axes


def save_part(image: Image.Image, output: Path, start_frame: int, end_frame: int, strip_width: int, fps: float, pixels_per_cm: float) -> None:
    raw_output = output.with_name(f"{output.stem}_raw.png")
    crop = image.crop((start_frame * strip_width, 0, end_frame * strip_width, image.height))
    crop.save(raw_output, compress_level=6)
    add_axes(raw_output, output, end_frame - start_frame, fps, strip_width, pixels_per_cm, 2.0, start_frame / fps)


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a corrected time-space image")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--split-time", type=float, required=True)
    parser.add_argument("--strip-width", type=int, default=3)
    parser.add_argument("--fps", type=float, default=240.0)
    parser.add_argument("--pixels-per-cm", type=float, default=93.4)
    args = parser.parse_args()
    with Image.open(args.input) as opened:
        image = opened.copy()
    count = image.width // args.strip_width
    split_frame = round(args.split_time * args.fps)
    if image.width % args.strip_width or not 0 < split_frame < count:
        raise ValueError("split time must be inside the image time range")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_part(image, args.output_dir / f"{args.name}_0-{args.split_time:g}s.png", 0, split_frame, args.strip_width, args.fps, args.pixels_per_cm)
    save_part(image, args.output_dir / f"{args.name}_{args.split_time:g}s-end.png", split_frame, count, args.strip_width, args.fps, args.pixels_per_cm)
    print(f"Split {count} frames at frame {split_frame} ({args.split_time:g} s)")


if __name__ == "__main__":
    main()
