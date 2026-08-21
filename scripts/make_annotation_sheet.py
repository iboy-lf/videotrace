from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(prog="videotrace-annotation-sheet")
    parser.add_argument("video")
    parser.add_argument("--output", default=str(ROOT / "outputs_annotation"))
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--page-frames", type=int, default=16)
    parser.add_argument("--tile-width", type=int, default=360)
    args = parser.parse_args()

    paths = make_annotation_sheets(
        video_path=args.video,
        output_dir=args.output,
        interval_sec=args.interval,
        columns=args.columns,
        page_frames=args.page_frames,
        tile_width=args.tile_width,
    )
    print(json.dumps({"video": str(Path(args.video).resolve()), "sheets": [str(path) for path in paths]}, indent=2))


def make_annotation_sheets(
    video_path: str,
    output_dir: str,
    interval_sec: float = 5.0,
    columns: int = 4,
    page_frames: int = 16,
    tile_width: int = 360,
) -> list[Path]:
    path = Path(video_path).resolve()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if fps > 0 else 0.0
    if duration <= 0:
        cap.release()
        raise ValueError(f"Video has no readable duration: {path}")

    interval_sec = max(0.25, float(interval_sec))
    columns = max(1, int(columns))
    page_frames = max(columns, int(page_frames))
    tile_width = max(160, int(tile_width))
    times = np.arange(interval_sec / 2.0, duration, interval_sec).tolist()
    if not times:
        times = [duration / 2.0]

    output = Path(output_dir).resolve() / path.stem
    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    records: list[dict] = []
    for page_index in range(math.ceil(len(times) / page_frames)):
        page_times = times[page_index * page_frames : (page_index + 1) * page_frames]
        tiles = [_read_tile(cap, sec, tile_width) for sec in page_times]
        rows = math.ceil(len(tiles) / columns)
        blank = np.zeros_like(tiles[0])
        tiles.extend([blank] * (rows * columns - len(tiles)))
        sheet = np.vstack(
            [np.hstack(tiles[row * columns : (row + 1) * columns]) for row in range(rows)]
        )
        sheet_path = output / f"sheet_{page_index + 1:03d}.jpg"
        if not cv2.imwrite(str(sheet_path), sheet):
            raise OSError(f"Failed to write annotation sheet: {sheet_path}")
        paths.append(sheet_path)
        records.append(
            {
                "sheet": sheet_path.name,
                "timestamps_sec": [round(float(sec), 3) for sec in page_times],
            }
        )
    cap.release()
    (output / "index.json").write_text(
        json.dumps(
            {
                "video_path": str(path),
                "duration_sec": duration,
                "fps": fps,
                "interval_sec": interval_sec,
                "pages": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return paths


def _read_tile(cap, sec: float, tile_width: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_MSEC, float(sec) * 1000.0)
    ok, frame = cap.read()
    if not ok:
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
    height = max(90, int(frame.shape[0] * tile_width / max(1, frame.shape[1])))
    tile = cv2.resize(frame, (tile_width, height), interpolation=cv2.INTER_AREA)
    header_height = max(28, int(height * 0.13))
    cv2.rectangle(tile, (0, 0), (tile_width, header_height), (0, 0, 0), -1)
    cv2.putText(
        tile,
        f"{sec:07.2f}s",
        (10, int(header_height * 0.75)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.55, tile_width / 560.0),
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return tile


if __name__ == "__main__":
    main()
