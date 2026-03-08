"""Training script for plate corner detection model."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
from datetime import datetime
from typing import Tuple, Optional
import json

from model import create_model, KeypointLoss
from dataset import PlateDataset, create_dataloaders


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    img_size: Tuple[int, int]
) -> Tuple[float, float]:
    """Validate the model.

    Returns:
        loss: Average loss
        pixel_error: Average pixel error per keypoint
    """
    model.eval()
    total_loss = 0.0
    total_pixel_error = 0.0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * images.size(0)

            # Calculate pixel error
            pred = outputs.view(-1, 5, 2)
            target = targets.view(-1, 5, 2)

            # Scale to pixel coordinates
            h, w = img_size
            pred_pixels = pred.clone()
            pred_pixels[:, :, 0] *= w
            pred_pixels[:, :, 1] *= h

            target_pixels = target.clone()
            target_pixels[:, :, 0] *= w
            target_pixels[:, :, 1] *= h

            # Euclidean distance per keypoint
            dist = torch.sqrt(((pred_pixels - target_pixels) ** 2).sum(dim=-1))
            total_pixel_error += dist.mean().item() * images.size(0)

    n = len(loader.dataset)
    return total_loss / n, total_pixel_error / n


def train(
    annotations_file: Path,
    data_dir: Path,
    output_dir: Path,
    model_type: str = "resnet18",
    img_size: Tuple[int, int] = (224, 224),
    batch_size: int = 16,
    epochs: int = 100,
    lr: float = 1e-4,
    patience: int = 15,
    device: Optional[str] = None,
    synthetic_annotations_file: Optional[Path] = None,
    synthetic_images_dir: Optional[Path] = None
) -> Path:
    """Train the plate corner detection model.

    Args:
        annotations_file: Path to annotations JSON
        data_dir: Base directory containing videos
        output_dir: Directory for saving checkpoints
        model_type: "resnet18" or "mobilenet"
        img_size: Input image size
        batch_size: Batch size
        epochs: Maximum epochs
        lr: Learning rate
        patience: Early stopping patience
        device: Device to use (auto-detect if None)
        synthetic_annotations_file: Optional path to synthetic annotations
        synthetic_images_dir: Optional path to synthetic images directory

    Returns:
        Path to best model checkpoint
    """
    # Setup device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create dataloaders
    print("Loading dataset...")
    train_loader, val_loader = create_dataloaders(
        annotations_file, data_dir, batch_size, img_size,
        synthetic_annotations_file=synthetic_annotations_file,
        synthetic_images_dir=synthetic_images_dir
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    # Create model
    print(f"Creating {model_type} model...")
    model = create_model(model_type, pretrained=True)
    model = model.to(device)

    # Loss and optimizer
    criterion = KeypointLoss(use_wing_loss=True)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Training loop
    best_val_loss = float('inf')
    best_pixel_error = float('inf')
    epochs_without_improvement = 0
    best_model_path = output_dir / "best_model.pth"

    print(f"\nTraining for up to {epochs} epochs...")
    print("-" * 60)

    for epoch in range(epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, pixel_error = validate(model, val_loader, criterion, device, img_size)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Logging
        print(f"Epoch {epoch+1:3d} | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"Pixel Error: {pixel_error:.2f}px")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_pixel_error = pixel_error
            epochs_without_improvement = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'pixel_error': pixel_error,
                'model_type': model_type,
                'img_size': img_size
            }, best_model_path)
            print(f"  -> Saved best model (pixel error: {pixel_error:.2f}px)")
        else:
            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping after {patience} epochs without improvement")
            break

    print("-" * 60)
    print(f"Training complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Best pixel error: {best_pixel_error:.2f}px")
    print(f"Model saved to: {best_model_path}")

    # Save training config
    config = {
        'model_type': model_type,
        'img_size': img_size,
        'batch_size': batch_size,
        'epochs_trained': epoch + 1,
        'best_val_loss': best_val_loss,
        'best_pixel_error': best_pixel_error,
        'trained_at': datetime.now().isoformat()
    }
    with open(output_dir / "training_config.json", 'w') as f:
        json.dump(config, f, indent=2)

    return best_model_path


def main():
    parser = argparse.ArgumentParser(description="Train plate corner detection model")
    parser.add_argument("--annotations", type=Path, default=Path("plate_annotations.json"),
                       help="Path to annotations file")
    parser.add_argument("--data-dir", type=Path, default=Path("../../training_data"),
                       help="Path to training data directory")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"),
                       help="Output directory for model checkpoints")
    parser.add_argument("--model", type=str, default="resnet18",
                       choices=["resnet18", "mobilenet"],
                       help="Model architecture")
    parser.add_argument("--img-size", type=int, default=224,
                       help="Input image size")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-synthetic", action="store_true",
                       help="Disable synthetic data augmentation")

    args = parser.parse_args()

    # Resolve paths
    script_dir = Path(__file__).parent
    annotations_file = args.annotations
    if not annotations_file.is_absolute():
        annotations_file = (script_dir / annotations_file).resolve()

    data_dir = args.data_dir
    if not data_dir.is_absolute():
        data_dir = (script_dir / data_dir).resolve()

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = (script_dir / output_dir).resolve()

    if not annotations_file.exists():
        print(f"Error: Annotations file not found: {annotations_file}")
        print("Run 'python annotate.py' first to create annotations")
        return

    # Check for synthetic data
    synthetic_annotations_file = None
    synthetic_images_dir = None
    if not args.no_synthetic:
        synthetic_annotations_file = script_dir / "synthetic_annotations.json"
        synthetic_images_dir = script_dir / "synthetic_images"
        if not synthetic_annotations_file.exists() or not synthetic_images_dir.exists():
            print("No synthetic data found. Run 'python augment_annotations.py' to generate.")
            synthetic_annotations_file = None
            synthetic_images_dir = None

    train(
        annotations_file=annotations_file,
        data_dir=data_dir,
        output_dir=output_dir,
        model_type=args.model,
        img_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        device=args.device,
        synthetic_annotations_file=synthetic_annotations_file,
        synthetic_images_dir=synthetic_images_dir
    )


if __name__ == "__main__":
    main()
