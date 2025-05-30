# Voxel Occupancy Network - Standalone Module

A standalone implementation of voxel-conditioned occupancy networks extracted from the original [Occupancy Networks](https://github.com/autonomousvision/occupancy_networks) codebase by Mescheder et al.

## Overview

This module provides a complete, self-contained implementation of voxel-to-mesh conversion using occupancy networks. It preserves the exact neural network architectures and mathematical computations from the original codebase while providing a clean, modern interface.

## Key Features

- **Exact Architecture Preservation**: Maintains identical neural network structures and mathematical computations
- **Zero External Dependencies**: Completely independent from the original `im2mesh` codebase
- **Backward Compatibility**: Compatible with pre-trained weights from the original implementation
- **Modern Interface**: Clean, well-documented API with type hints
- **Multiple Voxel Formats**: Support for `.binvox`, `.npy`, and `.npz` files
- **Efficient Training**: Optimized training loops with progress monitoring
- **High-Quality Mesh Generation**: Marching cubes-based mesh extraction

## Installation

### Requirements

```bash
pip install torch torchvision numpy scikit-image trimesh matplotlib tqdm
```

### Optional Dependencies

For `.binvox` file support:
```bash
pip install binvox-rw
```

## Quick Start

### Basic Usage

```python
import torch
from voxelOccupancyNetwork import VoxelEncoder, DecoderCBatchNorm, OccupancyNetwork
from voxelOccupancyNetwork import VoxelToMeshGenerator, load_voxel_data

# Create model (exact same architecture as original)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder = VoxelEncoder(c_dim=256)
decoder = DecoderCBatchNorm(dim=3, z_dim=0, c_dim=256, hidden_size=256)
model = OccupancyNetwork(decoder, encoder, device=device)

# Load voxel data
voxels = load_voxel_data('input.npy')

# Generate mesh
generator = VoxelToMeshGenerator(model, device)
mesh = generator.generate_mesh(voxels, resolution=64, threshold=0.5)

# Save result
mesh.export('output.ply')
```

### Training a Model

```python
from voxelOccupancyNetwork import train_voxel_model, VoxelDataset
from torch.utils.data import DataLoader

# Create dataset
voxel_files = ['voxel1.npy', 'voxel2.npy', ...]
point_files = ['points1.npz', 'points2.npz', ...]

train_dataset = VoxelDataset(voxel_files, point_files)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Train model
train_losses, val_losses = train_voxel_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    n_epochs=100,
    lr=1e-4,
    device=device
)
```

## Architecture Details

### VoxelEncoder
- **Input**: 3D voxel grids (batch_size, depth, height, width)
- **Architecture**: 3D CNN with progressive downsampling (/2, /4, /8, /16)
- **Output**: Feature vectors (batch_size, c_dim)

### DecoderCBatchNorm
- **Input**: 3D points + conditioning features
- **Architecture**: 5 conditional ResNet blocks with conditional batch normalization
- **Output**: Occupancy logits

### OccupancyNetwork
- **Components**: VoxelEncoder + DecoderCBatchNorm
- **Training**: Binary cross-entropy loss with optional KL divergence
- **Inference**: Deterministic (z_dim=0) or probabilistic (z_dim>0)

## Configuration Compatibility

This module replicates the exact functionality of the original `configs/voxels/onet.yaml` configuration:

```yaml
# Original configuration
model:
  encoder: voxel_simple      # -> VoxelEncoder
  decoder: cbatchnorm        # -> DecoderCBatchNorm
  c_dim: 256
  z_dim: 0                   # Deterministic model
```

## Data Format

### Voxel Data
- **Format**: Binary 3D grids (typically 32³ or 64³)
- **Files**: `.npy`, `.npz`, `.binvox`
- **Values**: 0.0 (empty) or 1.0 (occupied)

### Point Data (for training)
- **Format**: `.npz` files with keys:
  - `points`: (N, 3) 3D coordinates
  - `occupancies`: (N,) binary occupancy values

## Advanced Usage

### Custom Training Loop

```python
from voxelOccupancyNetwork import VoxelOccupancyTrainer

trainer = VoxelOccupancyTrainer(model, optimizer, device)

for epoch in range(num_epochs):
    for batch in train_loader:
        loss = trainer.train_step(batch)

    # Evaluation
    eval_dict = trainer.evaluate(val_loader)
    print(f"Epoch {epoch}: Loss={eval_dict['loss']:.4f}, IoU={eval_dict['iou']:.4f}")
```

### Batch Processing

```python
from voxelOccupancyNetwork import batch_process_voxels

# Process multiple voxel files
voxel_files = ['input1.npy', 'input2.npy', 'input3.npy']
output_files = batch_process_voxels(
    voxel_files, model, 'output_meshes/',
    resolution=128, threshold=0.4
)
```

### Adaptive Threshold Selection

```python
# Try multiple thresholds automatically
mesh = generator.generate_mesh_adaptive_threshold(
    voxels,
    thresholds=(0.3, 0.4, 0.5, 0.6, 0.7)
)
```

## Visualization

```python
from voxelOccupancyNetwork import visualize_voxels, visualize_mesh, compare_voxel_and_mesh

# Visualize input voxels
visualize_voxels(voxels, "Input Voxels")

# Visualize generated mesh
visualize_mesh(mesh, "Generated Mesh")

# Side-by-side comparison
compare_voxel_and_mesh(voxels, mesh, "Voxel vs Mesh")
```

## Model Checkpoints

### Saving
```python
from voxelOccupancyNetwork import save_model_checkpoint

save_model_checkpoint(
    model, optimizer, epoch, train_loss, val_loss,
    'checkpoint.pth'
)
```

### Loading
```python
from voxelOccupancyNetwork import load_model_checkpoint

model, info = load_model_checkpoint(model, 'checkpoint.pth', device)
print(f"Loaded model from epoch {info['epoch']}")
```

## Performance Tips

1. **Memory Management**: Use smaller batch sizes for inference if GPU memory is limited
2. **Quality vs Speed**: Higher resolution (128³) gives better quality but slower generation
3. **Threshold Tuning**: Try different thresholds (0.3-0.7) for optimal results
4. **Batch Processing**: Process multiple voxels together for efficiency

## Troubleshooting

### Empty Meshes
```python
# Try different thresholds
for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
    mesh = generator.generate_mesh(voxels, threshold=threshold)
    if len(mesh.vertices) > 0:
        break
```

### Memory Issues
```python
# Reduce batch size or resolution
mesh = generator.generate_mesh(
    voxels, resolution=32, batch_size=50000
)
```

## Citation

If you use this module, please cite the original Occupancy Networks paper:

```bibtex
@inproceedings{Mescheder2019CVPR,
    title = {Occupancy Networks: Learning 3D Reconstruction in Function Space},
    author = {Mescheder, Lars and Oechsle, Michael and Niemeyer, Michael and Nowozin, Sebastian and Geiger, Andreas},
    booktitle = {Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
    year = {2019}
}
```

## License

This module preserves the original license terms from the Occupancy Networks codebase.
