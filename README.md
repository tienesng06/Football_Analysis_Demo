# Player Tracking with YOLOv8 and ByteTrack

This project performs **player and referee detection and tracking** in football broadcast videos using **YOLOv8** for object detection and **ByteTrack** for multi-object tracking.

The system processes raw football match videos and outputs annotated videos with tracked players and referees across frames.

---

## Features

- Football player and referee detection using YOLOv8
- Multi-object tracking with ByteTrack
- Batch inference for efficient video processing
- Stable tracking IDs across video frames
- Video input/output handling using OpenCV
- Custom-trained YOLOv8 model support

---

## Project Status

**Current Status:** Active (Prototype / Research Phase)

The core detection and tracking pipeline is implemented and functional.

### Completed
- YOLOv8-based detection for players and referees
- Batch inference on video frames
- ByteTrack multi-object tracking integration
- Video reading and writing with OpenCV
- End-to-end pipeline from input video to tracked output
- Custom trained model integration

### In Progress
- Tracking ID stability improvements
- False positive reduction in crowded scenes
- Code refactoring and modularization
- Visualization improvements (labels, colors, track IDs)

### Planned
- Perspective normalization and pitch calibration
- Performance optimization for long videos
- Real-time inference support
- Evaluation metrics (mAP, IDF1)

### Known Limitations
- Detection accuracy varies across different broadcast angles
- Tracking may fragment under heavy occlusion
- Model generalization depends on training data diversity

---

## Project Structure

```text
FOOTBALL_ANALYSIS/
│
├── input_videos/           # Raw input videos
├── output_videos/          # Processed output videos
│
├── trackers/
│   ├── __init__.py
│   └── tracker.py          # YOLOv8 + ByteTrack logic
│
├── utils/
│   ├── __init__.py
│   ├── video_utils.py      # Video read/write utilities
│   └── bbox_utils.py       # Bounding box utilities
│
├── training/               # Training notebooks and datasets
│
├── main.py                 # Main execution script
├── yolov8.pt               # Trained YOLOv8 model
├── requirements.txt
└── README.md

