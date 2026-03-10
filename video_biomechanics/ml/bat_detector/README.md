# Bat Endpoint Detector

Neural network for detecting baseball bat endpoints (knob and tip) in video frames.

## Training Pipeline

### 1. Extract Frames
```bash
python extract_frames.py --frames-per-video 8
```
Selects diverse frames across swing phases using motion analysis. Creates `frame_selection.json`.

### 2. Annotate Frames
```bash
cd ../plate_detector
python annotate.py --mode bat
```
Uses the shared annotator in bat mode:
- **Left click**: Add point (knob first, then tip)
- **Right click**: Remove last point
- **Space**: Save and advance to next frame
- **S**: Save current annotation
- **X**: Skip frame (bat obscured/not visible)
- **N/P**: Next/previous frame
- **Z**: Zoom toggle
- **Q**: Quit

Creates `bat_annotations.json` in bat_detector folder.

### 3. Generate Synthetic Data
```bash
python augment_annotations.py
```
Expands training set via:
- Affine transforms (scale, rotation, translation)
- Image augmentations (blur, noise, brightness)
- Synthetic bat overlays on background frames

Creates `synthetic_images/` and `synthetic_bat_annotations.json`.

### 4. Train Model
```bash
python train.py --epochs 100 --batch-size 16
```
Trains ResNet18 backbone with keypoint regression head. Saves to `checkpoints/best_bat_model.pth`.

## Usage

```python
from ml.bat_detector import BatDetector

detector = BatDetector("checkpoints/best_bat_model.pth")

# Detect in single frame
endpoints, confidence = detector.detect(frame)
print(f"Knob: {endpoints[0]}, Tip: {endpoints[1]}")

# Compute bat length
length_pixels = detector.compute_bat_length(endpoints)

# Estimate scale from known bat length
scale, conf = detector.estimate_scale(video_path, bat_length_m=0.84)
```

## Architecture

- **Backbone**: ResNet18 (pretrained on ImageNet)
- **Head**: FC layers with dropout
- **Output**: 4 values (knob_x, knob_y, tip_x, tip_y) normalized to [0, 1]
- **Loss**: Wing loss for keypoint localization
- **Parameters**: ~11M
