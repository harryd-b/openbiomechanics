"""
MotionBERT 3D pose estimation.

MotionBERT is a transformer-based model that directly estimates
3D pose from 2D detections with strong temporal modeling.

Paper: "MotionBERT: A Unified Perspective on Learning Human Motion Representations"
GitHub: https://github.com/Walter0807/MotionBERT
"""

import numpy as np
from typing import List, Optional
from pathlib import Path
import cv2

from .base import PoseEstimator3D, Pose3DResult, H36M_JOINTS


class MotionBERTEstimator(PoseEstimator3D):
    """
    MotionBERT transformer-based 3D pose estimation.

    Uses temporal context for more accurate depth estimation.
    Requires 2D detections as input (uses YOLO internally).
    """

    # Hugging Face model info for automatic download
    HF_REPO_ID = "walterzhu/MotionBERT"
    HF_FILENAME = "checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin"

    # Fallback: Manual download instructions
    MANUAL_DOWNLOAD_URL = "https://onedrive.live.com/?authkey=%21AMkCK4HtMtrI4LA&id=A5438CD242871DF%21209&cid=0A5438CD242871DF"
    MANUAL_DOWNLOAD_INSTRUCTIONS = """
    MotionBERT weights not found. To download manually:

    1. Visit: https://github.com/Walter0807/MotionBERT#model-zoo
    2. Download "3D Pose (H36M-SH, ft)" checkpoint from OneDrive
    3. Save as: video_biomechanics/models/motionbert_h36m.pth

    Or install huggingface_hub and run this script again for automatic download.
    """

    def __init__(self,
                 model_path: Optional[str] = None,
                 yolo_model: str = 'yolov8m-pose.pt',
                 temporal_window: int = 243,
                 device: str = 'auto'):
        """
        Initialize MotionBERT estimator.

        Args:
            model_path: Path to MotionBERT weights (None = auto-download)
            yolo_model: YOLOv8 model for 2D detection
            temporal_window: Number of frames for temporal context
            device: 'cuda', 'cpu', or 'auto'
        """
        super().__init__(name='motionbert')
        self.model_path = model_path
        self.yolo_model_name = yolo_model
        self.temporal_window = temporal_window
        self.device_str = device

        self.model = None
        self.pose_estimator = None
        self.device = None

    def initialize(self) -> None:
        """Load MotionBERT model and 2D detector."""
        import torch

        # Determine device
        if self.device_str == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(self.device_str)

        # Load 2D pose detector
        from pose_estimation import PoseEstimator
        self.pose_estimator = PoseEstimator(model_name=self.yolo_model_name)

        # Load MotionBERT model
        self.model = self._load_model()
        self._is_initialized = True

        print(f"MotionBERT initialized on {self.device}")

    def _load_model(self):
        """Load or download MotionBERT model."""
        import torch
        import torch.nn as nn

        # Try to import MotionBERT
        try:
            from lib.model.DSTformer import DSTformer
            model_class = DSTformer
        except ImportError:
            # Use our simplified implementation
            model_class = self._create_motionbert_model()

        # Get model path
        if self.model_path is None:
            self.model_path = self._download_model()

        # Create model
        model = model_class(
            dim_in=3,
            dim_out=3,
            dim_feat=256,
            dim_rep=512,
            depth=5,
            num_heads=8,
            maxlen=self.temporal_window,
        )

        # Load weights if available
        if self.model_path and Path(self.model_path).exists():
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            if 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            print(f"Loaded MotionBERT weights from {self.model_path}")
        else:
            print("Warning: MotionBERT weights not found, using random initialization")

        model = model.to(self.device)
        model.eval()

        return model

    def _create_motionbert_model(self):
        """Create a simplified MotionBERT-like model."""
        import torch
        import torch.nn as nn

        class SimplifiedMotionBERT(nn.Module):
            """
            Simplified transformer for 2D-to-3D lifting with temporal context.
            """

            def __init__(self, dim_in=3, dim_out=3, dim_feat=256, dim_rep=512,
                         depth=5, num_heads=8, maxlen=243):
                super().__init__()

                self.maxlen = maxlen
                self.dim_feat = dim_feat

                # Input projection
                self.input_proj = nn.Linear(dim_in * 17, dim_feat)

                # Positional encoding
                self.pos_embed = nn.Parameter(torch.randn(1, maxlen, dim_feat) * 0.02)

                # Transformer encoder
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=dim_feat,
                    nhead=num_heads,
                    dim_feedforward=dim_rep,
                    dropout=0.1,
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

                # Output projection
                self.output_proj = nn.Linear(dim_feat, dim_out * 17)

            def forward(self, x):
                """
                Args:
                    x: (B, T, 17, 3) - 2D keypoints with confidence

                Returns:
                    (B, T, 17, 3) - 3D joint positions
                """
                B, T, J, C = x.shape

                # Flatten joints
                x = x.reshape(B, T, J * C)

                # Project
                x = self.input_proj(x)

                # Add positional encoding
                x = x + self.pos_embed[:, :T, :]

                # Transformer
                x = self.transformer(x)

                # Output
                x = self.output_proj(x)
                x = x.reshape(B, T, J, 3)

                return x

        return SimplifiedMotionBERT

    def _download_model(self) -> Optional[str]:
        """Download MotionBERT pretrained weights from Hugging Face Hub."""
        models_dir = Path(__file__).parent.parent / 'models'
        models_dir.mkdir(exist_ok=True)

        model_path = models_dir / 'motionbert_h36m.pth'

        # Check if already downloaded
        if model_path.exists():
            return str(model_path)

        # Also check for nested path from previous download
        nested_path = models_dir / 'checkpoint' / 'pose3d' / 'FT_MB_lite_MB_ft_h36m_global_lite' / 'best_epoch.bin'
        if nested_path.exists():
            import shutil
            shutil.copy2(nested_path, model_path)
            print(f"Using existing weights from {nested_path}")
            return str(model_path)

        # Try Hugging Face Hub
        try:
            from huggingface_hub import hf_hub_download

            print("Downloading MotionBERT weights from Hugging Face Hub...")
            downloaded_path = hf_hub_download(
                repo_id=self.HF_REPO_ID,
                filename=self.HF_FILENAME,
                local_dir=str(models_dir),
            )

            # Copy to standard location
            downloaded = Path(downloaded_path)
            if downloaded.exists():
                import shutil
                shutil.copy2(downloaded, model_path)
                print(f"Downloaded to {model_path}")
                return str(model_path)

        except ImportError:
            print("huggingface_hub not installed. Install with: pip install huggingface_hub")
        except Exception as e:
            print(f"Hugging Face download failed: {e}")

        # Provide manual download instructions
        print(self.MANUAL_DOWNLOAD_INSTRUCTIONS)
        return None

    def estimate_frame(self,
                       frame: np.ndarray,
                       frame_number: int,
                       timestamp: float) -> Optional[Pose3DResult]:
        """
        Estimate 3D pose from single frame.

        Note: For best results, use estimate_video() which uses temporal context.
        """
        if not self._is_initialized:
            self.initialize()

        # Get 2D keypoints
        pose_frame = self.pose_estimator.process_frame(frame, frame_number, timestamp)

        if pose_frame is None:
            return None

        keypoints_2d = pose_frame.keypoints

        # For single frame, create a batch of 1 and use center frame
        import torch

        kp_tensor = torch.tensor(keypoints_2d, dtype=torch.float32)
        kp_tensor = kp_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, 17, 3)

        # Pad to minimum temporal window
        if kp_tensor.shape[1] < self.temporal_window:
            pad_size = self.temporal_window - kp_tensor.shape[1]
            kp_tensor = torch.nn.functional.pad(
                kp_tensor, (0, 0, 0, 0, 0, pad_size), mode='replicate'
            )

        kp_tensor = kp_tensor.to(self.device)

        with torch.no_grad():
            output_3d = self.model(kp_tensor)

        # Get center frame
        joints_3d = output_3d[0, 0].cpu().numpy()

        # Normalize
        joints_3d = self.normalize_skeleton(joints_3d)

        confidences = keypoints_2d[:, 2] if keypoints_2d.shape[1] > 2 else np.ones(17)

        return Pose3DResult(
            joints_3d=joints_3d,
            confidences=confidences,
            frame_number=frame_number,
            timestamp=timestamp,
            joints_2d=keypoints_2d[:, :2],
            metadata={'method': 'motionbert'}
        )

    def estimate_video(self,
                       video_path: str,
                       max_frames: Optional[int] = None) -> List[Pose3DResult]:
        """
        Estimate 3D poses from video with full temporal context.

        This is the preferred method as MotionBERT benefits from
        seeing the full sequence.
        """
        if not self._is_initialized:
            self.initialize()

        import torch

        # Get 2D poses for entire video
        poses_2d = self.pose_estimator.process_video(video_path, max_frames=max_frames)

        if not poses_2d:
            return []

        # Stack keypoints
        keypoints = np.stack([p.keypoints for p in poses_2d])  # (T, 17, 3)
        timestamps = [p.timestamp for p in poses_2d]

        T = len(keypoints)

        # Process in windows with overlap
        all_outputs = np.zeros((T, 17, 3))
        all_counts = np.zeros(T)

        # Convert to tensor
        kp_tensor = torch.tensor(keypoints, dtype=torch.float32).to(self.device)

        # Process full sequence at once if possible
        if T <= self.temporal_window:
            # Pad to temporal window
            padded = torch.nn.functional.pad(
                kp_tensor.unsqueeze(0),
                (0, 0, 0, 0, 0, self.temporal_window - T),
                mode='replicate'
            )

            with torch.no_grad():
                output = self.model(padded)

            all_outputs[:T] = output[0, :T].cpu().numpy()
            all_counts[:T] = 1

        else:
            # Sliding window with overlap
            stride = self.temporal_window // 2

            for start in range(0, T - self.temporal_window + 1, stride):
                end = start + self.temporal_window
                window = kp_tensor[start:end].unsqueeze(0)

                with torch.no_grad():
                    output = self.model(window)

                all_outputs[start:end] += output[0].cpu().numpy()
                all_counts[start:end] += 1

            # Handle last window if needed
            if T > self.temporal_window and (T - self.temporal_window) % stride != 0:
                start = T - self.temporal_window
                window = kp_tensor[start:].unsqueeze(0)

                with torch.no_grad():
                    output = self.model(window)

                all_outputs[start:] += output[0].cpu().numpy()
                all_counts[start:] += 1

        # Average overlapping predictions
        all_outputs = all_outputs / np.maximum(all_counts[:, None, None], 1)

        # Convert to results
        results = []
        for i in range(T):
            joints_3d = self.normalize_skeleton(all_outputs[i])
            confidences = keypoints[i, :, 2] if keypoints.shape[2] > 2 else np.ones(17)

            results.append(Pose3DResult(
                joints_3d=joints_3d,
                confidences=confidences,
                frame_number=i,
                timestamp=timestamps[i],
                joints_2d=keypoints[i, :, :2],
                metadata={'method': 'motionbert'}
            ))

        return results

    def get_confidence_weights(self) -> np.ndarray:
        """
        Get per-joint reliability weights.

        MotionBERT has better temporal consistency and depth estimation
        than simple lifting, especially for dynamic movements.
        """
        weights = np.ones(17)

        # MotionBERT is strong for temporal consistency
        # Good for all joints due to attention mechanism
        weights[H36M_JOINTS['pelvis']] = 1.2
        weights[H36M_JOINTS['spine']] = 1.1
        weights[H36M_JOINTS['neck']] = 1.1

        # Better depth estimation for extremities than lifting
        weights[H36M_JOINTS['left_wrist']] = 0.95
        weights[H36M_JOINTS['right_wrist']] = 0.95

        return weights / weights.sum()
