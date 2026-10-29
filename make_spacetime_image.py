"""Create a time-space image from a directory of video frames.

Each output column is a narrow vertical strip sampled from one source frame.
Time runs from left (earliest) to right (latest), matching the final orientation
described in the supplied Photoshop procedure.
"""

from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
EXCLUDED_WORDS = {"preview", "reference", "spacetime", "location"}


def natural_key(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def find_frames(folder: Path) -> list[Path]:
    frames = []
    with os.scandir(folder) as entries:
        for entry in entries:
            if not entry.is_file():
                continue
            path = folder / entry.name
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if any(word in path.stem.lower() for word in EXCLUDED_WORDS):
                continue
            if any(char.isdigit() for char in path.stem):
                frames.append(path)
    frames.sort(key=natural_key)
    if not frames:
        raise ValueError(f"No frame images found in {folder}")
    return frames


def make_location_preview(source: Path, x: int, output: Path) -> None:
    with Image.open(source) as opened:
        image = opened.convert("RGB")
    draw = ImageDraw.Draw(image)
    line_width = max(3, image.width // 500)
    draw.line((x, 0, x, image.height - 1), fill=(255, 0, 0), width=line_width)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, quality=95)


def make_spacetime(
    folder: Path,
    output: Path,
    x: int,
    strip_width: int,
    y_start: int,
    y_end: int | None,
    step: int,
) -> tuple[int, int, int]:
    frames = find_frames(folder)[::step]
    with Image.open(frames[0]) as first:
        frame_width, frame_height = first.size

    if y_end is None:
        y_end = frame_height
    if not (0 <= x < frame_width):
        raise ValueError(f"x must be between 0 and {frame_width - 1}")
    if strip_width < 1:
        raise ValueError("strip-width must be at least 1")
    if strip_width > frame_width:
        raise ValueError(f"strip-width must not exceed the frame width ({frame_width})")
    if not (0 <= y_start < y_end <= frame_height):
        raise ValueError(f"y range must satisfy 0 <= y-start < y-end <= {frame_height}")

    left = max(0, x - strip_width // 2)
    right = min(frame_width, left + strip_width)
    left = right - strip_width
    output_image = Image.new("RGB", (len(frames) * strip_width, y_end - y_start))

    for index, frame_path in enumerate(frames):
        with Image.open(frame_path) as opened:
            if opened.size != (frame_width, frame_height):
                raise ValueError(
                    f"Frame size changed at {frame_path.name}: "
                    f"{opened.size} != {(frame_width, frame_height)}"
                )
            strip = opened.convert("RGB").crop((left, y_start, right, y_end))
        output_image.paste(strip, (index * strip_width, 0))

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.stem}.tmp{output.suffix}")
    output_image.save(temporary, compress_level=6)
    temporary.replace(output)
    return len(frames), output_image.width, output_image.height


def choose_time_tick(duration: float) -> float:
    candidates = (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300)
    target = max(duration / 8, candidates[0])
    return min(candidates, key=lambda value: abs(math.log(value / target)))


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = ["meiryob.ttc", "YuGothB.ttc"] if bold else ["meiryo.ttc", "YuGothM.ttc"]
    for name in names:
        path = Path("C:/Windows/Fonts") / name
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def add_axes(
    spacetime_path: Path,
    output: Path,
    frame_count: int,
    fps: float,
    strip_width: int,
    pixels_per_cm: float,
    vertical_scale: float = 1.0,
    time_start_s: float = 0.0,
) -> None:
    if fps <= 0:
        raise ValueError("fps must be greater than zero")
    if pixels_per_cm <= 0:
        raise ValueError("pixels-per-cm must be greater than zero")
    if vertical_scale <= 0:
        raise ValueError("vertical-scale must be greater than zero")

    with Image.open(spacetime_path) as opened:
        image = opened.convert("RGBA")
    if not math.isclose(vertical_scale, 1.0):
        image = image.resize(
            (image.width, round(image.height * vertical_scale)),
            Image.Resampling.BICUBIC,
        )
    display_pixels_per_cm = pixels_per_cm * vertical_scale

    left_margin, right_margin, top_margin, bottom_margin = 210, 35, 30, 170
    canvas = Image.new(
        "RGB",
        (left_margin + image.width + right_margin, top_margin + image.height + bottom_margin),
        "white",
    )
    canvas.paste(image, (left_margin, top_margin), image)
    draw = ImageDraw.Draw(canvas)
    tick_font = load_font(27)
    label_font = load_font(36, bold=True)
    axis_color = (20, 20, 20)
    x0, y0 = left_margin, top_margin + image.height
    x1 = left_margin + image.width
    draw.line((x0, top_margin, x0, y0), fill=axis_color, width=3)
    draw.line((x0, y0, x1, y0), fill=axis_color, width=3)

    duration = (frame_count - 1) / fps if frame_count > 1 else 0.0
    time_end_s = time_start_s + duration
    time_tick = choose_time_tick(max(duration, 0.1))
    minor_tick = 0.25
    tick_index = max(0, math.ceil((time_start_s - 1e-9) / minor_tick))
    while True:
        tick = tick_index * minor_tick
        if tick > time_end_s + 1e-9:
            break
        frame_position = (tick - time_start_s) * fps
        x = left_margin + frame_position * strip_width + (strip_width - 1) / 2
        is_major = math.isclose(tick / time_tick, round(tick / time_tick), abs_tol=1e-8)
        is_half = math.isclose(tick * 2, round(tick * 2), abs_tol=1e-8)
        tick_length = 13 if is_major else 9 if is_half else 5
        tick_width = 3 if is_major else 2
        draw.line((x, y0, x, y0 + tick_length), fill=axis_color, width=tick_width)
        if is_major:
            label = f"{tick:g}"
            draw.text((x, y0 + 42), label, fill=axis_color, font=tick_font, anchor="mm")
        tick_index += 1

    max_cm = int((image.height - 1) // display_pixels_per_cm)
    for cm in range(max_cm + 1):
        y = y0 - cm * display_pixels_per_cm
        draw.line((x0 - 13, y, x0, y), fill=axis_color, width=3)
        draw.text((x0 - 20, y), str(cm), fill=axis_color, font=tick_font, anchor="rm")

    draw.text(
        ((x0 + x1) / 2, canvas.height - 43),
        "経過時間 (s)",
        fill=axis_color,
        font=label_font,
        anchor="mm",
    )
    label_box = Image.new("RGBA", (250, 70), (255, 255, 255, 0))
    label_draw = ImageDraw.Draw(label_box)
    label_draw.text((125, 35), "長さ (cm)", fill=axis_color, font=label_font, anchor="mm")
    label_box = label_box.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)
    canvas.paste(label_box, (22, top_margin + (image.height - label_box.height) // 2), label_box)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.stem}.tmp{output.suffix}")
    canvas.save(temporary, compress_level=6)
    temporary.replace(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a time-space image by joining vertical strips from video frames."
    )
    parser.add_argument("folder", type=Path, help="Folder containing source frame images")
    parser.add_argument("--x", type=int, required=True, help="Horizontal analysis location in pixels")
    parser.add_argument("--strip-width", type=int, default=3, help="Strip width per frame (default: 3)")
    parser.add_argument("--y-start", type=int, default=0, help="Top of retained vertical range")
    parser.add_argument("--y-end", type=int, help="Bottom of retained vertical range; default: full height")
    parser.add_argument("--step", type=int, default=1, help="Use every Nth frame (default: 1)")
    parser.add_argument("--output", type=Path, required=True, help="Output PNG path")
    parser.add_argument("--preview", type=Path, help="Optional first-frame preview with analysis line")
    parser.add_argument("--annotated-output", type=Path, help="Optional PNG with time and length axes")
    parser.add_argument("--fps", type=float, default=240.0, help="Frame rate for the time axis (default: 240)")
    parser.add_argument(
        "--pixels-per-cm",
        type=float,
        default=93.4,
        help="Vertical calibration from the photographed ruler (default: 93.4 px/cm)",
    )
    parser.add_argument(
        "--vertical-scale",
        type=float,
        default=1.0,
        help="Display-only vertical enlargement factor (default: 1)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.step < 1:
        raise ValueError("step must be at least 1")
    frames = find_frames(args.folder)
    if args.preview:
        make_location_preview(frames[0], args.x, args.preview)
    count, width, height = make_spacetime(
        folder=args.folder,
        output=args.output,
        x=args.x,
        strip_width=args.strip_width,
        y_start=args.y_start,
        y_end=args.y_end,
        step=args.step,
    )
    print(f"Created {args.output} from {count} frames ({width}x{height})")
    if args.annotated_output:
        add_axes(
            spacetime_path=args.output,
            output=args.annotated_output,
            frame_count=count,
            fps=args.fps,
            strip_width=args.strip_width,
            pixels_per_cm=args.pixels_per_cm,
            vertical_scale=args.vertical_scale,
        )
        print(
            f"Created {args.annotated_output} with {args.fps:g} fps time axis "
            f"and {args.pixels_per_cm:g} px/cm length axis"
        )


if __name__ == "__main__":
    main()
