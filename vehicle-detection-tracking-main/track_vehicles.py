import argparse
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


REFERENCE_WIDTH = 1920
REFERENCE_HEIGHT = 1080

ZONE_IN_POLYGONS = [
    np.array([[592, 282], [900, 282], [900, 82], [592, 82]]),
    np.array([[958, 868], [1250, 860], [1250, 1060], [950, 1060]]),
    np.array([[592, 582], [592, 860], [392, 860], [392, 582]]),
    np.array([[1250, 282], [1250, 530], [1450, 530], [1450, 282]]),
]

ZONE_OUT_POLYGONS = [
    np.array([[950, 282], [1250, 282], [1250, 82], [950, 82]]),
    np.array([[592, 860], [900, 860], [900, 1060], [592, 1060]]),
    np.array([[592, 282], [592, 550], [392, 550], [392, 282]]),
    np.array([[1250, 860], [1250, 560], [1450, 560], [1450, 860]]),
]

ZONE_COLORS = [
    (75, 25, 230),
    (75, 180, 60),
    (25, 225, 255),
    (210, 118, 60),
]

UNASSIGNED_COLOR = (180, 180, 180)


def scale_polygon(polygon: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = np.array([width / REFERENCE_WIDTH, height / REFERENCE_HEIGHT])
    return np.rint(polygon * scale).astype(np.int32)


def draw_zone(
    frame: np.ndarray,
    polygon: np.ndarray,
    color: tuple[int, int, int],
    label: str,
) -> None:
    overlay = frame.copy()
    cv2.fillPoly(overlay, [polygon], color)
    cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
    cv2.polylines(frame, [polygon], True, color, 3, cv2.LINE_AA)

    x, y = polygon.min(axis=0)
    cv2.putText(
        frame,
        label,
        (int(x) + 8, max(int(y) + 26, 26)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )


def find_entry_zone(point: tuple[int, int], zones: list[np.ndarray]) -> int | None:
    for index, polygon in enumerate(zones):
        if cv2.pointPolygonTest(polygon, point, False) >= 0:
            return index
    return None


def draw_tracks(
    frame: np.ndarray,
    result,
    zones_in: list[np.ndarray],
    track_zones: dict[int, int],
    track_history: dict[int, deque[tuple[int, int]]],
    last_seen: dict[int, int],
    frame_number: int,
    trail_length: int,
) -> None:
    boxes = result.boxes
    if boxes is None or boxes.id is None:
        return

    xyxy = boxes.xyxy.int().cpu().numpy()
    track_ids = boxes.id.int().cpu().tolist()
    class_ids = boxes.cls.int().cpu().tolist()

    for bounds, track_id, class_id in zip(xyxy, track_ids, class_ids):
        x1, y1, x2, y2 = (int(value) for value in bounds)
        center = ((x1 + x2) // 2, (y1 + y2) // 2)

        if track_id not in track_zones:
            zone_index = find_entry_zone(center, zones_in)
            if zone_index is not None:
                track_zones[track_id] = zone_index

        zone_index = track_zones.get(track_id)
        color = ZONE_COLORS[zone_index] if zone_index is not None else UNASSIGNED_COLOR

        if zone_index is None:
            continue

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        history = track_history.setdefault(track_id, deque(maxlen=trail_length))
        history.append(center)
        last_seen[track_id] = frame_number

        if len(history) > 1:
            points = np.array(history, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], False, color, 3, cv2.LINE_AA)

        class_name = result.names.get(class_id, str(class_id))
        label = f"{class_name} #{track_id}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
        )
        label_top = max(y1 - text_height - baseline - 6, 0)
        cv2.rectangle(
            frame,
            (x1, label_top),
            (x1 + text_width + 8, label_top + text_height + baseline + 6),
            color,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (x1 + 4, label_top + text_height + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    expired_ids = [
        track_id
        for track_id, seen_at in last_seen.items()
        if frame_number - seen_at > trail_length
    ]
    for track_id in expired_ids:
        track_history.pop(track_id, None)
        last_seen.pop(track_id, None)


def process_video(
    weights: Path,
    source: Path,
    output: Path,
    confidence: float,
    show: bool,
    trail_length: int,
) -> None:
    model = YOLO(str(weights))
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {source}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video: {output}")

    zones_in = [scale_polygon(p, width, height) for p in ZONE_IN_POLYGONS]
    zones_out = [scale_polygon(p, width, height) for p in ZONE_OUT_POLYGONS]
    track_zones: dict[int, int] = {}
    track_history: dict[int, deque[tuple[int, int]]] = {}
    last_seen: dict[int, int] = {}

    frame_number = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            result = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=confidence,
                verbose=False,
            )[0]
            annotated = frame.copy()

            for index, (zone_in, zone_out, color) in enumerate(
                zip(zones_in, zones_out, ZONE_COLORS), start=1
            ):
                draw_zone(annotated, zone_in, color, f"IN {index}")
                draw_zone(annotated, zone_out, color, f"OUT {index}")

            draw_tracks(
                annotated,
                result,
                zones_in,
                track_zones,
                track_history,
                last_seen,
                frame_number,
                trail_length,
            )

            writer.write(annotated)
            frame_number += 1
            print(f"\rProcessed {frame_number}/{total_frames} frames", end="", flush=True)

            if show:
                cv2.imshow("Vehicle tracking with zones", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        print()
        capture.release()
        writer.release()
        cv2.destroyAllWindows()

    print(f"Saved annotated video to: {output.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track vehicles and draw traffic zones.")
    parser.add_argument("--weights", type=Path, default=Path("data/traffic_analysis.pt"))
    parser.add_argument("--source", type=Path, default=Path("data/traffic_analysis.mov"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/track/zones/traffic_analysis_zones.mp4"),
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--trail-length", type=int, default=60)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_video(
        args.weights,
        args.source,
        args.output,
        args.conf,
        args.show,
        args.trail_length,
    )


if __name__ == "__main__":
    main()
