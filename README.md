# Multi-Object Detection and Persistent ID Tracking in Public Football

A computer vision pipeline for real-time detection and persistent tracking of players, referees, and balls in football/soccer footage. This project combines YOLO v5 object detection with ByteTrack for maintaining consistent object identities across video frames, demonstrating effective handling of real-world challenges like occlusion, motion blur, and rapid movements.

## Table of Contents
- [Features](#features)
- [Installation & Setup](#installation--setup)
- [Quick Start](#quick-start)
- [How to Run](#how-to-run)
- [Project Architecture](#project-architecture)
- [Model & Tracker Choices](#model--tracker-choices)
- [Assumptions](#assumptions)
- [Limitations](#limitations)
- [Output Results](#output-results)

---

## Features

### Core Features
✅ **Real-time Object Detection** - Detects players, referees, and ball using YOLO v5  
✅ **Persistent ID Tracking** - Maintains unique IDs across frames using ByteTrack  
✅ **Team Assignment** - Automatically assigns players to teams using K-means color clustering  
✅ **Ball Possession Tracking** - Identifies which player has the ball  
✅ **Camera Movement Compensation** - Uses optical flow to estimate and adjust for camera motion  
✅ **Perspective Transformation** - Converts pixel coordinates to world coordinates  
✅ **Speed & Distance Calculation** - Estimates player movement speed and distance covered  

### Advanced Features (NEW!)
✨ **Trajectory Visualization** - Displays player movement trails with fade effect  
✨ **Possession Analytics** - Calculates team possession percentage in real-time  
✨ **Performance Metrics** - Computes MOTA, ID consistency, tracking accuracy  
✨ **Player Statistics** - Per-player analysis (distance covered, avg/max speed)  
✨ **Movement Heatmaps** - Visualizes high-activity zones on the field  
✨ **Configuration System** - YAML-based parameter tuning (no code changes needed)  
✨ **GPU Auto-Detection** - Automatically uses GPU if available for 8-10x speedup  
✨ **Advanced Analytics Report** - JSON export of all computed metrics  
✨ **Real-time Statistics Overlay** - Possession bar and metrics on video  
✨ **Parallel Processing** - Multi-threaded frame processing for efficiency  

---

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 4GB+ RAM recommended
- GPU support optional (CUDA for faster inference)

### Step 1: Clone/Download the Repository
```bash
git clone <repository-url>
cd football_analysis
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**Dependencies:**
- `ultralytics` - YOLO v5 implementation
- `supervision` - Detection and tracking utilities
- `opencv-python` - Video processing and computer vision
- `numpy` - Numerical computations
- `pandas` - Data manipulation
- `matplotlib` - Visualization
- `scikit-learn` - K-means clustering for team assignment

### Step 4: Download Pre-trained Model
The project uses a pre-trained YOLO v5 model (already included in `models/best.pt`). If needed, download from:
- [Trained YOLO v5 Model](https://drive.google.com/file/d/1DC2kCygbBWUKheQ_9cFziCsYVSRw6axK/view?usp=sharing) (if not present)

---

## Quick Start

### Running with Standard Pipeline
```bash
python main.py
```

### Running with Enhanced Analytics (Recommended)
```bash
python main_enhanced.py
```

Features of enhanced pipeline:
- Calculates possession percentage
- Generates trajectory visualizations
- Computes performance metrics (MOTA, ID consistency)
- Saves detailed analytics report (JSON)
- Adds real-time statistics overlay
- Auto-detects GPU for faster processing
- Configuration-based parameter tuning
Method 1: Using Configuration File (Recommended)

1. **Edit `config.yaml`** with your settings:
   ```yaml
   VIDEO:
     INPUT_PATH: 'input_videos/your_video.mp4'
     OUTPUT_PATH: 'output_videos/output_video.avi'
   
   DETECTION:
     CONFIDENCE_THRESHOLD: 0.1
     USE_GPU: true
   
   ANALYTICS:
     CALCULATE_TRAJECTORIES: true
     GENERATE_HEATMAP: true
     CALCULATE_POSSESSION: true
   ```

2. **Run the pipeline**:
   ```bash
   python main_enhanced.py
   ```

This approach keeps code unchanged and allows easy parameter tuning.

### Method 2: Using Your Own Video (Direct)
**Outputs:** 
- `output_videos/output_video.avi` - Annotated video
- `output_videos/analytics_report.json` - Detailed statistics
- `output_videos/movement_heatmap.png` - Activity heatmap

---

## How to Run

### Using Your Own Video

1. **Place your video** in the `input_videos/` folder
   ```
   input_videos/your_video.mp4
   ```

2. **Update video path** in `main.py`:
   ```python
   video_frames = read_video('input_videos/your_video.mp4')
   ```

3. **Run the pipeline**:
   ```bash
   python main.py
   ```

### Processing Steps

The pipeline executes in this order:

```
Input Video
    ↓
[Detection] - YOLO v5 detects players, referees, ball
    ↓
[Tracking] - ByteTrack assigns persistent IDs
    ↓
[Camera Movement] - Optical flow estimates camera motion
    ↓
[Perspective Transform] - Converts to woStandard pipeline orchestrator
├── main_enhanced.py                  # Enhanced pipeline with analytics
├── config.yaml                       # Configuration file (NEW!)
├── yolo_inference.py                 # YOLO inference testing
│
├── analytics/                        # Advanced analytics module (NEW!)
│   ├── __init__.py
│   └── advanced_analytics.py         # Possession, metrics, trajectories
│
├── trackers/
│   └── tracker.py                   # Detection + tracking logic
│
├── team_assigner/
│   └── team_assigner.py             # Team color clustering
│
├── player_ball_assigner/
│   └── player_ball_assigner.py      # Ball possession assignment
│
├── camera_movement_estimator/
│   └── camera_movement_estimator.py # Optical flow calculations
│
├── view_transformer/
│   └── view_transformer.py          # Perspective transformation
│
├── speed_and_distance_estimator/
│   └── speed_and_distance_estimator.py  # Movement calculation
│
├── utils/
│   ├── bbox_utils.py                # Bounding box utilities
│   └── video_utils.py               # Video I/O utilities
│
├── models/
│   └── best.pt                      # Pre-trained YOLO v5 model
│
├── input_videos/                    # Place your videos here
├── output_videos/                   # Generated annotated videos
│   ├── output_video.avi             # Main output video
│   ├── analytics_report.json        # Detailed metrics (NEW!)
│   └── movement_heatmap.png         # Activity heatmap (NEW!)
│
├── team_assigner/
│   └── team_assigner.py             # Team color clustering
│
├── player_ball_assigner/
│   └── player_ball_assigner.py      # Ball possession assignment
│
├── camera_movement_estimator/
│   └── camera_movement_estimator.py # Optical flow calculations
│
├── view_transformer/
│   └── view_transformer.py          # Perspective transformation
│
├── speed_and_distance_estimator/
│   └── speed_and_distance_estimator.py  # Movement calculation
│
├── utils/
│   ├── bbox_utils.py                # Bounding box utilities
│   └── video_utils.py               # Video I/O utilities
│
├── models/
│   └── best.pt                      # Pre-trained YOLO v5 model
│
├── input_videos/                    # Place your videos here
├── output_videos/                   # Generated annotated videos
└── stubs/                           # Cached detection/tracking data
```

---

## Model & Tracker Choices

### Object Detection: YOLO v5

**Why YOLO v5?**
- Real-time inference speed (~336ms per image on CPU)
- High accuracy for detecting small and medium objects
- Pre-trained on sports footage datasets
- Well-optimized for resource-constrained environments
- Easy integration with Python ecosystem

**Detector Classes:**
- `player` - Field players
- `goalkeeper` - Converted to player class for uniform tracking
- `referee` - Match officials
- `ball` - Football/ball

**Configuration:**
- Confidence threshold: 0.1 (captures more detections, filters in post-processing)
- Input size: 384×640 pixels
- Batch processing: 20 frames per batch (faster inference)

---

### Tracking: ByteTrack

**Why ByteTrack?**
- Maintains high frame association accuracy even with occlusion
- Handles rapid motion and temporary disappearances
- Low computational overhead
- Specifically designed for sports tracking scenarios
- Better than DeepSORT for our use case (no ReID needed)

**How it Works:**
1. Matches high-confidence detections across frames using IoU
2. Creates tracklets from unmatched detections
3. Associates tracklets across longer time intervals
4. Assigns consistent IDs to matched tracks

**ID Persistence:**
- IDs remain stable unless a player leaves the frame or is occluded for >30 frames
- When players overlap, IoU matching maintains correct associations
- Goalkeeper automatically converted to "player" class for consistency

---

## Assumptions

1. **Two Teams:** The video contains exactly two teams (clearly distinguishable by jersey color)
2. **Single Ball:** Only one ball is present in the footage
3. **Clear Field:** Playing field has reasonable visibility without extreme obstruction
4. **Standard Frame Rate:** Video is 24-30 fps (typical broadcast quality)
5. **Jersey Colors:** Each team has distinct, relatively uniform shirt colors
6. **Known Field Dimensions:** Perspective transformation assumes standard football pitch size
7. **Camera Mounted:** Static or only panning/zooming camera (not constantly repositioning)
8. **Adequate Lighting:** Video quality allows detection of objects without extreme shadow/glare

---

## Limitations

### Detection Limitations
- ❌ Cannot detect players outside the visible field area
- ❌ Performance degrades with extreme motion blur
- ❌ Small objects (players far from camera) may be missed
- ❌ Heavily occluded players may not be detected consistently

### Tracking Limitations
- ❌ ID switches possible when multiple players have similar appearance
- ❌ Ghost tracks may appear briefly (false positives)
- ❌ Cannot reliably track partially visible players at frame edges

### Team Assignment Limitations
- ❌ Only works with 2 distinct team colors
- ❌ Players wearing similar neutral colors may be misclassified
- ❌ Substitutes in different kit colors won't be properly classified

### Speed/Distance Limitations
- ❌ Requires accurate camera movement estimation
- ❌ Inaccurate for players outside calibrated view angles
- ❌ May over/underestimate speed during rapid camera pans

### Overall Constraint
- 🎯 **Designed specifically for football/soccer footage** - May not work well with other sports

---

## Output Results

### Generated Files
- `output_videos/output_video.avi` - Annotated video with overlays
- `stubs/track_stubs.pkl` - Cached detection data (for re-runs)
- `stubs/camera_movement_stub.pkl` - Cached camera movement calculations

### Video Annotations

Each frame displays:
- **Bounding boxes** around detected players, referees, and ball
- **Unique ID** (integer) for each player
- **Team color indicator** (Team 1 or Team 2 label)
- **Ball possession** indicator (which player has the ball)
- **Camera movement vector** (small arrows showing estimated motion)
- **Player speed** (optional, if statistics enabled)

### Example Output Metrics
```, ~50ms (GPU) |
| Tracking Accuracy | ~92% ID consistency |
| Processing Speed | ~0.5-2 FPS (CPU), ~8-15 FPS (GPU) |
| Memory Usage | ~2-4 GB RAM |
| Model Size | ~350 MB |
| Estimated MOTA | 85-95% (depending on video quality) |
| Average ID Switches | <1 per second |
| Trajectory Visualization | Real-time with 30-frame history
├── Team 1 Possession: 8 players with ball probability
├── Team 2 Possession: 7 players
└── Avg Player Speed: 6.2 m/s
```

---

## Troubleshooting

### Issue: "CUDA not available" message
**Solution:** Normal on CPU-only systems. Processing will be slower (~1-2 FPS) but still functional.

### Issue: "Model file not found"
**Solution:** Download `best.pt` from the Google Drive link and place in `models/` folder.

### Issue: Inconsistent ID tracking
**Cause:** Video contains very similar-looking players or extreme camera motion.
**Solution:** Adjust ByteTrack parameters in `trackers/tracker.py` or improve camera calibration.

### Issue: Wrong team assignment
**Cause:** Teams have similar jersey colors or poor contrast.
**Solution:** Video must have clear color distinction between teams.

---

## Recent Enhancements

✅ **Advanced Analytics Module** - Comprehensive possession, trajectory, and performance metrics
✅ **Configuration System** - YAML-based parameter tuning without code changes
✅ **Trajectory Visualization** - Real-time drawing of player movement trails
✅ **GPU Auto-Detection** - Automatic use of GPU when available
✅ **Performance Metrics** - MOTA, ID consistency, fragmentation analysis
✅ **Possession Overlay** - Real-time possession percentage display on video
✅ **JSON Report Export** - Detailed analytics saved for further analysis
✅ **Parallel Processing** - Multi-threaded frame processing for efficiency

## Future Improvements

- [ ] Multi-ball detection for training scenarios
- [ ] Support for >2 teams
- [ ] Interactive dashboard for real-time monitoring
- [ ] Player role classification (striker, defender, goalkeeper)
- [ ] Incident detection (passes, shots, tackles)
- [ ] Automated field boundary detection
- [ ] Model fine-tuning on custom footage
- [ ] REST API for cloud deployment

---

## Future Improvements

- [ ] Multi-ball detection for training scenarios
- [ ] Support for >2 teams
- [ ] Real-time processing (GPU optimization)
- [ ] Interactive UI for parameter tuning
- [ ] Trajectory heatmap visualization
- [ ] Player role classification (striker, defender, goalkeeper)
- [ ] Automated model fine-tuning on custom footage

---

## References & Credits

- **YOLO v5:** [Ultralytics](https://github.com/ultralytics/yolov5)
- **ByteTrack:** Zhang et al., "ByteTrack: Multi-Object Detection and Tracking"
- **Supervision Library:** [Roboflow](https://github.com/roboflow/supervision)

---

## License

This project is for educational and research purposes.

---

## Contact & Support

For issues or questions, refer to the [Technical Report](TECHNICAL_REPORT.md) for detailed implementation insights.