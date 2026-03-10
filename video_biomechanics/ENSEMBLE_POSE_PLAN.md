# Ensemble Pose Estimation Plan

**Goal**: Combine multiple 3D pose estimation methods with intelligent fusion to maximize accuracy, using UPLIFT data as ground truth for training.

---

## Architecture Overview

```
                    ┌─────────────────────────────────────────────────────┐
                    │                   INPUT VIDEOS                       │
                    │              (Side View + Back View)                 │
                    └─────────────────────────────────────────────────────┘
                                           │
                    ┌──────────────────────┼──────────────────────┐
                    │                      │                      │
                    ▼                      ▼                      ▼
        ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐
        │   Method 1:       │  │   Method 2:       │  │   Method 3:       │
        │   YOLOv8 + Lift   │  │   MotionBERT      │  │   Multi-View      │
        │   (current)       │  │   (transformer)   │  │   Triangulation   │
        └───────────────────┘  └───────────────────┘  └───────────────────┘
                    │                      │                      │
                    │   3D Pose +          │   3D Pose +          │   3D Pose +
                    │   Confidence         │   Confidence         │   Reproj Error
                    │                      │                      │
                    └──────────────────────┼──────────────────────┘
                                           │
                                           ▼
                    ┌─────────────────────────────────────────────────────┐
                    │                 FUSION MODULE                        │
                    │  ┌─────────────────────────────────────────────┐    │
                    │  │  Stage 1: Outlier Rejection                 │    │
                    │  │  - Remove joints where methods disagree     │    │
                    │  │    by > threshold (e.g., 10cm)              │    │
                    │  └─────────────────────────────────────────────┘    │
                    │                      │                              │
                    │                      ▼                              │
                    │  ┌─────────────────────────────────────────────┐    │
                    │  │  Stage 2: Confidence-Weighted Average       │    │
                    │  │  - Per-joint weighted combination           │    │
                    │  │  - Weights from method confidence scores    │    │
                    │  └─────────────────────────────────────────────┘    │
                    │                      │                              │
                    │                      ▼                              │
                    │  ┌─────────────────────────────────────────────┐    │
                    │  │  Stage 3: Biomechanical Constraints         │    │
                    │  │  - Enforce bone length consistency          │    │
                    │  │  - Joint angle limits                       │    │
                    │  │  - Temporal smoothness                      │    │
                    │  └─────────────────────────────────────────────┘    │
                    │                      │                              │
                    │                      ▼                              │
                    │  ┌─────────────────────────────────────────────┐    │
                    │  │  Stage 4: Learned Fusion (Neural Network)   │    │
                    │  │  - Trained on UPLIFT ground truth           │    │
                    │  │  - Learns optimal per-joint weighting       │    │
                    │  │  - Context-aware (pose-dependent)           │    │
                    │  └─────────────────────────────────────────────┘    │
                    │                      │                              │
                    │                      ▼                              │
                    │  ┌─────────────────────────────────────────────┐    │
                    │  │  Stage 5: Temporal Consistency              │    │
                    │  │  - Kalman filter for smooth trajectories    │    │
                    │  │  - Velocity/acceleration limits             │    │
                    │  └─────────────────────────────────────────────┘    │
                    └─────────────────────────────────────────────────────┘
                                           │
                                           ▼
                    ┌─────────────────────────────────────────────────────┐
                    │               FINAL 3D POSE OUTPUT                   │
                    │         (17 joints × 3 coordinates × N frames)       │
                    └─────────────────────────────────────────────────────┘
```

---

## Phase 1: Additional Pose Estimation Methods

### 1.1 MotionBERT Integration
**Why**: State-of-the-art single-view 3D pose, uses temporal context
**Output**: 17-joint H36M skeleton with confidence per joint

```python
# New file: pose_estimators/motionbert.py
class MotionBERTEstimator:
    def __init__(self, model_path: str = None):
        # Load pretrained MotionBERT
        pass

    def estimate_3d(self, video_path: str) -> List[Pose3D]:
        # Returns 3D poses with confidence scores
        pass
```

**Installation**:
```bash
pip install motion-bert  # or clone from GitHub
# Download pretrained weights (~200MB)
```

### 1.2 Improved Multi-View Triangulation
**Why**: Actual geometric measurement, most accurate when calibrated
**Enhancement**: Add automatic camera calibration

```python
# Enhanced: multiview.py
class CalibratedTriangulator:
    def calibrate_from_checkerboard(self, images: List[np.ndarray]):
        # OpenCV checkerboard calibration
        pass

    def triangulate_with_ransac(self, points_2d: List[np.ndarray]) -> np.ndarray:
        # Robust triangulation with outlier rejection
        pass
```

### 1.3 Optional: HMR 2.0 (SMPL-based)
**Why**: Provides body mesh, handles occlusion well
**Output**: SMPL parameters + regressed joints

```python
# New file: pose_estimators/hmr2.py
class HMR2Estimator:
    def __init__(self):
        # Load 4D-Humans / HMR2 model
        pass

    def estimate_3d(self, frames: List[np.ndarray]) -> List[Pose3D]:
        # Returns SMPL joints
        pass
```

---

## Phase 2: Fusion Strategies

### 2.1 Outlier Rejection Module
```python
# New file: fusion/outlier_rejection.py

class OutlierRejector:
    def __init__(self, threshold_cm: float = 10.0):
        self.threshold = threshold_cm / 100  # Convert to meters

    def reject_outliers(self, poses: List[np.ndarray],
                        confidences: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        For each joint, if one method disagrees with majority by > threshold,
        exclude it from averaging.

        Args:
            poses: List of (17, 3) arrays from each method
            confidences: List of (17,) confidence arrays

        Returns:
            filtered_poses: Poses with outliers masked
            valid_mask: (N_methods, 17) bool array
        """
        n_methods = len(poses)
        n_joints = poses[0].shape[0]

        valid_mask = np.ones((n_methods, n_joints), dtype=bool)

        for j in range(n_joints):
            joint_positions = np.array([p[j] for p in poses])
            median_pos = np.median(joint_positions, axis=0)

            for m in range(n_methods):
                dist = np.linalg.norm(joint_positions[m] - median_pos)
                if dist > self.threshold:
                    valid_mask[m, j] = False

        return poses, valid_mask
```

### 2.2 Confidence-Weighted Average
```python
# New file: fusion/weighted_average.py

class WeightedAverageFusion:
    def fuse(self, poses: List[np.ndarray],
             confidences: List[np.ndarray],
             valid_mask: np.ndarray) -> np.ndarray:
        """
        Weighted average of poses based on confidence scores.

        Args:
            poses: List of (17, 3) arrays
            confidences: List of (17,) arrays
            valid_mask: (N_methods, 17) bool array

        Returns:
            fused_pose: (17, 3) array
        """
        n_joints = poses[0].shape[0]
        fused = np.zeros((n_joints, 3))

        for j in range(n_joints):
            weights = []
            joint_pos = []

            for m, (pose, conf) in enumerate(zip(poses, confidences)):
                if valid_mask[m, j]:
                    weights.append(conf[j])
                    joint_pos.append(pose[j])

            if weights:
                weights = np.array(weights)
                weights = weights / weights.sum()  # Normalize
                fused[j] = np.average(joint_pos, weights=weights, axis=0)
            else:
                # Fallback: use median
                fused[j] = np.median([p[j] for p in poses], axis=0)

        return fused
```

### 2.3 Biomechanical Constraints
```python
# New file: fusion/biomechanical_constraints.py

class BiomechanicalConstraints:
    # Average bone lengths (meters) - H36M skeleton
    BONE_LENGTHS = {
        ('pelvis', 'right_hip'): 0.11,
        ('right_hip', 'right_knee'): 0.42,
        ('right_knee', 'right_ankle'): 0.40,
        ('pelvis', 'left_hip'): 0.11,
        ('left_hip', 'left_knee'): 0.42,
        ('left_knee', 'left_ankle'): 0.40,
        ('pelvis', 'spine'): 0.22,
        ('spine', 'neck'): 0.25,
        ('neck', 'head'): 0.12,
        ('neck', 'left_shoulder'): 0.15,
        ('left_shoulder', 'left_elbow'): 0.28,
        ('left_elbow', 'left_wrist'): 0.25,
        ('neck', 'right_shoulder'): 0.15,
        ('right_shoulder', 'right_elbow'): 0.28,
        ('right_elbow', 'right_wrist'): 0.25,
    }

    # Joint angle limits (degrees)
    JOINT_LIMITS = {
        'knee_flexion': (0, 150),
        'elbow_flexion': (0, 145),
        'shoulder_flexion': (-60, 180),
        'hip_flexion': (-30, 120),
    }

    def enforce_bone_lengths(self, pose: np.ndarray,
                             reference_lengths: dict = None) -> np.ndarray:
        """Scale/adjust joints to maintain consistent bone lengths."""
        # Use iterative adjustment to preserve bone ratios
        pass

    def enforce_joint_limits(self, pose: np.ndarray) -> np.ndarray:
        """Clamp joints to anatomically possible angles."""
        pass

    def apply_constraints(self, pose: np.ndarray) -> np.ndarray:
        """Apply all biomechanical constraints."""
        pose = self.enforce_bone_lengths(pose)
        pose = self.enforce_joint_limits(pose)
        return pose
```

### 2.4 Temporal Consistency (Kalman Filter)
```python
# New file: fusion/temporal_filter.py

class TemporalKalmanFilter:
    def __init__(self, n_joints: int = 17, fps: float = 240):
        self.n_joints = n_joints
        self.dt = 1.0 / fps
        self.filters = [self._create_filter() for _ in range(n_joints)]

    def _create_filter(self):
        """Create Kalman filter for single joint (position + velocity)."""
        from filterpy.kalman import KalmanFilter

        kf = KalmanFilter(dim_x=6, dim_z=3)  # [x,y,z,vx,vy,vz]
        # ... configure matrices
        return kf

    def filter_sequence(self, poses: List[np.ndarray]) -> List[np.ndarray]:
        """Apply Kalman filtering to smooth trajectory."""
        filtered = []
        for pose in poses:
            filtered_pose = np.zeros_like(pose)
            for j in range(self.n_joints):
                self.filters[j].predict()
                self.filters[j].update(pose[j])
                filtered_pose[j] = self.filters[j].x[:3]
            filtered.append(filtered_pose)
        return filtered
```

---

## Phase 3: Learned Fusion Network

### 3.1 Data Preparation
```python
# New file: fusion/data_preparation.py

class UPLIFTDataLoader:
    def __init__(self, uplift_csv_path: str, our_output_dir: str):
        self.uplift_data = pd.read_csv(uplift_csv_path)
        self.our_data = self._load_our_outputs(our_output_dir)

    def create_training_pairs(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create (input, target) pairs for training.

        Input: Concatenated outputs from all methods (N_methods * 17 * 3)
        Target: UPLIFT ground truth (17 * 3)

        Returns:
            X: (N_frames, N_methods * 17 * 3 + N_methods * 17)  # poses + confidences
            y: (N_frames, 17 * 3)  # ground truth positions
        """
        pass

    def temporal_align(self, our_poses, uplift_poses):
        """Align sequences using DTW or interpolation."""
        pass
```

### 3.2 Fusion Network Architecture
```python
# New file: fusion/learned_fusion.py

import torch
import torch.nn as nn

class FusionNetwork(nn.Module):
    """
    Learns optimal per-joint weighting based on:
    - Input poses from each method
    - Confidence scores
    - Pose context (what position is the body in)
    """

    def __init__(self, n_methods: int = 3, n_joints: int = 17):
        super().__init__()

        self.n_methods = n_methods
        self.n_joints = n_joints

        # Input: all method outputs + confidences
        input_dim = n_methods * n_joints * 3 + n_methods * n_joints

        # Context encoder - understands pose configuration
        self.context_encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # Per-joint weight predictors
        self.joint_weight_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128 + n_methods * 3 + n_methods, 64),
                nn.ReLU(),
                nn.Linear(64, n_methods),
                nn.Softmax(dim=-1)
            )
            for _ in range(n_joints)
        ])

        # Residual correction (optional refinement)
        self.residual_predictor = nn.Sequential(
            nn.Linear(128 + n_joints * 3, 256),
            nn.ReLU(),
            nn.Linear(256, n_joints * 3)
        )

    def forward(self, method_poses, confidences):
        """
        Args:
            method_poses: (B, N_methods, 17, 3)
            confidences: (B, N_methods, 17)

        Returns:
            fused_pose: (B, 17, 3)
        """
        B = method_poses.shape[0]

        # Flatten inputs
        poses_flat = method_poses.reshape(B, -1)
        conf_flat = confidences.reshape(B, -1)
        combined = torch.cat([poses_flat, conf_flat], dim=-1)

        # Get pose context
        context = self.context_encoder(combined)

        # Predict weights for each joint
        fused_pose = torch.zeros(B, self.n_joints, 3, device=method_poses.device)

        for j in range(self.n_joints):
            # Joint-specific input
            joint_poses = method_poses[:, :, j, :]  # (B, N_methods, 3)
            joint_confs = confidences[:, :, j]       # (B, N_methods)

            joint_input = torch.cat([
                context,
                joint_poses.reshape(B, -1),
                joint_confs
            ], dim=-1)

            weights = self.joint_weight_heads[j](joint_input)  # (B, N_methods)

            # Weighted combination
            weighted = (joint_poses * weights.unsqueeze(-1)).sum(dim=1)
            fused_pose[:, j] = weighted

        # Optional residual correction
        residual_input = torch.cat([context, fused_pose.reshape(B, -1)], dim=-1)
        residual = self.residual_predictor(residual_input).reshape(B, self.n_joints, 3)

        fused_pose = fused_pose + 0.1 * residual  # Small residual

        return fused_pose


class FusionTrainer:
    def __init__(self, model: FusionNetwork, learning_rate: float = 1e-4):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0

        for batch in dataloader:
            method_poses, confidences, ground_truth = batch

            self.optimizer.zero_grad()

            predicted = self.model(method_poses, confidences)
            loss = self.loss_fn(predicted, ground_truth)

            # Add bone length consistency loss
            bone_loss = self._bone_length_loss(predicted)
            total_loss_batch = loss + 0.1 * bone_loss

            total_loss_batch.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def _bone_length_loss(self, poses):
        """Encourage consistent bone lengths."""
        # Calculate bone lengths and penalize variance
        pass
```

### 3.3 Training Pipeline
```python
# New file: fusion/train_fusion.py

def train_fusion_model(
    uplift_data_path: str,
    video_paths: List[str],
    output_model_path: str,
    epochs: int = 100
):
    """
    Main training script for learned fusion.

    1. Process videos with all methods
    2. Align with UPLIFT ground truth
    3. Train fusion network
    4. Save model weights
    """

    # Step 1: Run all pose estimation methods
    print("Running pose estimation methods...")
    method_outputs = run_all_methods(video_paths)

    # Step 2: Load and align UPLIFT data
    print("Aligning with UPLIFT ground truth...")
    data_loader = UPLIFTDataLoader(uplift_data_path, method_outputs)
    X, y = data_loader.create_training_pairs()

    # Step 3: Create DataLoader
    dataset = TensorDataset(
        torch.tensor(X['poses']),
        torch.tensor(X['confidences']),
        torch.tensor(y)
    )
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Step 4: Train model
    print("Training fusion network...")
    model = FusionNetwork(n_methods=3)
    trainer = FusionTrainer(model)

    for epoch in range(epochs):
        loss = trainer.train_epoch(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.6f}")

    # Step 5: Save model
    torch.save(model.state_dict(), output_model_path)
    print(f"Model saved to {output_model_path}")
```

---

## Phase 4: Integration

### 4.1 Unified Ensemble Pipeline
```python
# New file: ensemble_pipeline.py

class EnsemblePosePipeline:
    """Main pipeline combining all methods and fusion."""

    def __init__(self,
                 fusion_model_path: str = None,
                 use_methods: List[str] = ['lifting', 'motionbert', 'triangulation']):

        # Initialize pose estimators
        self.estimators = {}
        if 'lifting' in use_methods:
            self.estimators['lifting'] = VideoPose3DLifter()
        if 'motionbert' in use_methods:
            self.estimators['motionbert'] = MotionBERTEstimator()
        if 'triangulation' in use_methods:
            self.estimators['triangulation'] = CalibratedTriangulator()

        # Initialize fusion modules
        self.outlier_rejector = OutlierRejector(threshold_cm=10.0)
        self.weighted_fusion = WeightedAverageFusion()
        self.biomech_constraints = BiomechanicalConstraints()
        self.temporal_filter = TemporalKalmanFilter()

        # Load learned fusion if available
        if fusion_model_path and os.path.exists(fusion_model_path):
            self.learned_fusion = FusionNetwork(n_methods=len(self.estimators))
            self.learned_fusion.load_state_dict(torch.load(fusion_model_path))
            self.learned_fusion.eval()
        else:
            self.learned_fusion = None

    def process_videos(self, video_paths: List[str]) -> List[np.ndarray]:
        """
        Process videos through ensemble pipeline.

        Returns:
            List of (17, 3) pose arrays, one per frame
        """
        # Step 1: Run all estimators
        all_poses = {}
        all_confidences = {}

        for name, estimator in self.estimators.items():
            print(f"Running {name}...")
            poses, confs = estimator.estimate_3d(video_paths)
            all_poses[name] = poses
            all_confidences[name] = confs

        # Step 2: Align frame counts
        n_frames = min(len(p) for p in all_poses.values())

        # Step 3: Fuse frame by frame
        fused_poses = []

        for i in range(n_frames):
            frame_poses = [all_poses[n][i] for n in self.estimators.keys()]
            frame_confs = [all_confidences[n][i] for n in self.estimators.keys()]

            # Outlier rejection
            poses, valid_mask = self.outlier_rejector.reject_outliers(
                frame_poses, frame_confs
            )

            # Weighted average
            fused = self.weighted_fusion.fuse(poses, frame_confs, valid_mask)

            # Biomechanical constraints
            fused = self.biomech_constraints.apply_constraints(fused)

            # Learned fusion refinement (if available)
            if self.learned_fusion:
                with torch.no_grad():
                    poses_tensor = torch.tensor(frame_poses).unsqueeze(0)
                    confs_tensor = torch.tensor(frame_confs).unsqueeze(0)
                    fused = self.learned_fusion(poses_tensor, confs_tensor)
                    fused = fused.squeeze(0).numpy()

            fused_poses.append(fused)

        # Step 4: Temporal smoothing
        fused_poses = self.temporal_filter.filter_sequence(fused_poses)

        return fused_poses
```

### 4.2 App Integration
```python
# Update app.py to use ensemble pipeline

def process_with_ensemble(video_paths, settings):
    pipeline = EnsemblePosePipeline(
        fusion_model_path='models/fusion_model.pt',
        use_methods=['lifting', 'motionbert', 'triangulation']
    )

    poses_3d = pipeline.process_videos(video_paths)

    # Continue with existing angle calculation...
    angle_calc = JointAngleCalculator3D()
    joint_angles = [angle_calc.calculate(pose) for pose in poses_3d]

    return joint_angles
```

---

## File Structure

```
video_biomechanics/
├── pose_estimators/
│   ├── __init__.py
│   ├── base.py              # Abstract base class
│   ├── yolo_lifting.py      # Current YOLOv8 + VideoPose3D
│   ├── motionbert.py        # MotionBERT estimator
│   ├── triangulation.py     # Multi-view triangulation
│   └── hmr2.py              # Optional SMPL-based
│
├── fusion/
│   ├── __init__.py
│   ├── outlier_rejection.py
│   ├── weighted_average.py
│   ├── biomechanical_constraints.py
│   ├── temporal_filter.py
│   ├── learned_fusion.py    # Neural network fusion
│   ├── data_preparation.py  # UPLIFT data loading
│   └── train_fusion.py      # Training script
│
├── ensemble_pipeline.py     # Main unified pipeline
├── calibration.py           # Camera calibration tools
│
└── models/
    └── fusion_model.pt      # Trained fusion weights
```

---

## Implementation Order

### Week 1: Foundation
- [ ] Create pose_estimators/ module structure
- [ ] Refactor current lifting code into yolo_lifting.py
- [ ] Add MotionBERT integration
- [ ] Improve triangulation with calibration

### Week 2: Fusion Strategies
- [ ] Implement outlier rejection
- [ ] Implement weighted average fusion
- [ ] Add biomechanical constraints
- [ ] Add Kalman temporal filtering

### Week 3: Learned Fusion
- [ ] Create UPLIFT data loader and alignment
- [ ] Build FusionNetwork architecture
- [ ] Train on UPLIFT ground truth
- [ ] Validate fusion model

### Week 4: Integration & Testing
- [ ] Create EnsemblePosePipeline
- [ ] Integrate into web app
- [ ] Run comparison vs UPLIFT
- [ ] Performance optimization

---

## Dependencies to Add

```bash
# MotionBERT
pip install motion-bert-pose  # or clone from GitHub

# Kalman filtering
pip install filterpy

# SMPL (optional)
pip install smplx

# Additional torch utilities
pip install torch-geometric  # if using graph networks
```

---

## Success Metrics

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Joint position MAE | < 3cm | vs UPLIFT ground truth |
| Angle correlation | > 0.95 | Pearson correlation |
| Angle MAE | < 5° | Mean absolute error |
| Temporal smoothness | < 50°/s jitter | Max frame-to-frame noise |
| Processing speed | < 2x current | Frames per second |

---

## Notes

- Start with 2 methods (lifting + MotionBERT) before adding triangulation
- The learned fusion is the key differentiator - UPLIFT data is valuable
- Consider data augmentation for training (add noise, temporal shifts)
- May need multiple UPLIFT sessions for robust training
