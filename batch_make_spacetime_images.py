"""Batch-create calibrated time-space images for all GX*_timestamp_JST folders."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from make_spacetime_image import add_axes, find_frames, make_spacetime


def process_folder(
    folder_string: str,
    output_string: str,
    x: int,
    strip_width: int,
    fps: float,
    pixels_per_cm: float,
    vertical_scale: float,
    overwrite: bool,
) -> dict[str, object]:
    started = time.monotonic()
    folder = Path(folder_string)
    output_dir = Path(output_string)
    base_name = folder.name.removesuffix("_timestamp_JST")
    raw_output = output_dir / f"{base_name}_x{x}_w{strip_width}_spacetime.png"
    scale_suffix = "" if vertical_scale == 1 else f"_y{vertical_scale:g}"
    axes_output = output_dir / f"{base_name}_x{x}_w{strip_width}_spacetime_axes{scale_suffix}.png"

    if axes_output.exists() and not overwrite:
        return {
            "folder": folder.name,
            "status": "skipped",
            "frames": len(find_frames(folder)),
            "output": str(axes_output),
            "seconds": round(time.monotonic() - started, 2),
        }

    frames = find_frames(folder)
    if overwrite or not raw_output.exists():
        frame_count, _, _ = make_spacetime(
            folder=folder,
            output=raw_output,
            x=x,
            strip_width=strip_width,
            y_start=0,
            y_end=None,
            step=1,
        )
    else:
        frame_count = len(frames)

    add_axes(
        spacetime_path=raw_output,
        output=axes_output,
        frame_count=frame_count,
        fps=fps,
        strip_width=strip_width,
        pixels_per_cm=pixels_per_cm,
        vertical_scale=vertical_scale,
    )
    return {
        "folder": folder.name,
        "status": "complete",
        "frames": frame_count,
        "output": str(axes_output),
        "seconds": round(time.monotonic() - started, 2),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-create calibrated time-space images")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("spacetime_output/all"))
    parser.add_argument("--x", type=int, default=1352)
    parser.add_argument("--strip-width", type=int, default=3)
    parser.add_argument("--fps", type=float, default=240.0)
    parser.add_argument("--pixels-per-cm", type=float, default=93.4)
    parser.add_argument("--vertical-scale", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only", help="Process one exact folder name")
    parser.add_argument("--from-folder", help="First folder name to process (inclusive)")
    parser.add_argument("--to-folder", help="Last folder name to process (inclusive)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    folders = sorted(args.root.glob("GX*_timestamp_JST"), key=lambda path: path.name)
    if args.only:
        folders = [path for path in folders if path.name == args.only]
    if args.from_folder:
        folders = [path for path in folders if path.name >= args.from_folder]
    if args.to_folder:
        folders = [path for path in folders if path.name <= args.to_folder]
    if not folders:
        raise SystemExit("No matching frame folders found")
    if args.workers < 1:
        raise SystemExit("workers must be at least 1")

    args.output.mkdir(parents=True, exist_ok=True)
    total_frames = sum(len(find_frames(folder)) for folder in folders)
    print(
        f"Batch plan: {len(folders)} folders, {total_frames} frames, "
        f"x={args.x}, width={args.strip_width}, fps={args.fps:g}, "
        f"scale={args.pixels_per_cm:g} px/cm, vertical={args.vertical_scale:g}x, "
        f"workers={args.workers}",
        flush=True,
    )
    if args.dry_run:
        for folder in folders:
            print(f"  {folder.name}: {len(find_frames(folder))} frames", flush=True)
        return

    log_path = args.output / "batch_results.jsonl"
    completed = 0
    failed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_folder,
                str(folder),
                str(args.output),
                args.x,
                args.strip_width,
                args.fps,
                args.pixels_per_cm,
                args.vertical_scale,
                args.overwrite,
            ): folder
            for folder in folders
        }
        for future in as_completed(futures):
            folder = futures[future]
            try:
                result = future.result()
                completed += 1
            except Exception as error:
                failed += 1
                result = {
                    "folder": folder.name,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
            record = {"time": datetime.now().isoformat(timespec="seconds"), **result}
            with log_path.open("a", encoding="utf-8") as log:
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                f"[{completed + failed}/{len(folders)}] {result['folder']}: "
                f"{result['status']} ({result.get('seconds', '-')} s)",
                flush=True,
            )

    print(f"Finished: {completed} succeeded/skipped, {failed} failed", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
