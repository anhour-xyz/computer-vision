# Computer Vision Learning & Projects

A collection of Python computer vision projects, learning notebooks, and coursework. The projects explore real-time card recognition, vehicle tracking, image classification, and image processing with OpenCV, PyTorch, and Ultralytics YOLO.

## Learning



## Projects

### 1. Vehicle detection and direction tracking

Detect and track vehicles in a traffic video using YOLO and ByteTrack. Four pairs of colored `IN` and `OUT` zones mark traffic directions. Once a tracked object's center enters an `IN` zone, its bounding box, ID label, and movement trail use that zone's color.

![Vehicle direction tracking demo](projects/vehicle-detection-tracking/assets/tracking-demo.gif)

The zone coordinates are designed for the sample intersection and scale with the video resolution. For other camera views, edit the polygons in `track_vehicles.py`. The `OUT` zones are visual guides; the script does not calculate entry-to-exit counts.

[Project files](projects/vehicle-detection-tracking/)

### 2. Poker hand recognition

Recognize playing cards from a webcam feed using a custom YOLO model. The app draws card labels and confidence scores, then evaluates the poker hand when exactly five distinct cards meet the confidence threshold of 0.5 in a frame.

The hand evaluator covers categories from high card through royal flush. Ace-low straights (`A–2–3–4–5`) are not currently handled.



