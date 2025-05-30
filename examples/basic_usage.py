#!/usr/bin/env python3
"""
Basic usage example for the Voxel Occupancy Network standalone module.

This script demonstrates how to:
1. Create a voxel-conditioned occupancy network
2. Generate synthetic voxel data
3. Train the model
4. Extract meshes from voxel inputs
5. Visualize results

Run with: python basic_usage.py
"""

import torch
import numpy as np
import os
import sys

# Add parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voxelOccupancyNetwork import (
    VoxelEncoder, DecoderCBatchNorm, OccupancyNetwork,
    VoxelToMeshGenerator, train_voxel_model, VoxelDataset,
    create_synthetic_voxel_data, visualize_voxels, visualize_mesh,
    compare_voxel_and_mesh, save_mesh
)
from torch.utils.data import DataLoader


def create_model(device):
    """Create voxel-conditioned occupancy network with exact original architecture."""
    print("Creating model...")
    
    # Exact same architecture as configs/voxels/onet.yaml
    encoder = VoxelEncoder(c_dim=256)
    decoder = DecoderCBatchNorm(
        dim=3, z_dim=0, c_dim=256, hidden_size=256, legacy=False
    )
    model = OccupancyNetwork(decoder, encoder, device=device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    return model


def create_synthetic_data():
    """Create synthetic voxel and point data for demonstration."""
    print("Creating synthetic data...")
    
    # Create different voxel shapes
    voxel_sphere = create_synthetic_voxel_data('sphere', resolution=32)
    voxel_cube = create_synthetic_voxel_data('cube', resolution=32)
    voxel_torus = create_synthetic_voxel_data('torus', resolution=32)
    
    voxel_data = [voxel_sphere, voxel_cube, voxel_torus]
    
    # Create corresponding point data (simplified for demo)
    point_data = []
    for voxels in voxel_data:
        # Sample points from voxel grid
        occupied_coords = np.where(voxels > 0.5)
        if len(occupied_coords[0]) > 0:
            # Sample random points
            n_points = 1024
            points = np.random.uniform(-0.6, 0.6, (n_points, 3)).astype(np.float32)
            
            # Create occupancy labels (simplified)
            occupancies = np.random.choice([0, 1], n_points).astype(np.float32)
            
            point_data.append({
                'points': points,
                'occupancies': occupancies
            })
        else:
            # Fallback if no occupied voxels
            points = np.random.uniform(-0.6, 0.6, (1024, 3)).astype(np.float32)
            occupancies = np.zeros(1024, dtype=np.float32)
            point_data.append({
                'points': points,
                'occupancies': occupancies
            })
    
    return voxel_data, point_data


def create_dataset(voxel_data, point_data):
    """Create dataset from synthetic data."""
    print("Creating dataset...")
    
    class SyntheticVoxelDataset(torch.utils.data.Dataset):
        def __init__(self, voxel_data, point_data):
            self.voxel_data = voxel_data
            self.point_data = point_data
        
        def __len__(self):
            return len(self.voxel_data)
        
        def __getitem__(self, idx):
            return {
                'inputs': torch.from_numpy(self.voxel_data[idx]).float(),
                'points': torch.from_numpy(self.point_data[idx]['points']).float(),
                'points.occ': torch.from_numpy(self.point_data[idx]['occupancies']).float()
            }
    
    dataset = SyntheticVoxelDataset(voxel_data, point_data)
    return dataset


def train_model_demo(model, dataset, device):
    """Demonstrate model training."""
    print("Training model (demo with few epochs)...")
    
    # Create data loaders
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    
    # Train for a few epochs (demo purposes)
    train_losses, val_losses = train_voxel_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=5,  # Small number for demo
        lr=1e-4,
        device=device,
        save_checkpoints=True,
        checkpoint_dir='demo_checkpoints',
        eval_every=2,
        print_every=1
    )
    
    print(f"Training completed. Final train loss: {train_losses[-1]:.6f}")
    if val_losses:
        print(f"Final validation loss: {val_losses[-1]:.6f}")
    
    return model


def generate_meshes_demo(model, voxel_data, device):
    """Demonstrate mesh generation."""
    print("Generating meshes...")
    
    generator = VoxelToMeshGenerator(model, device)
    
    meshes = []
    shape_names = ['sphere', 'cube', 'torus']
    
    for i, (voxels, name) in enumerate(zip(voxel_data, shape_names)):
        print(f"Generating mesh for {name}...")
        
        # Try adaptive threshold selection
        mesh = generator.generate_mesh_adaptive_threshold(
            voxels, 
            resolution=32,  # Lower resolution for demo speed
            thresholds=(0.3, 0.4, 0.5, 0.6, 0.7)
        )
        
        meshes.append(mesh)
        
        # Save mesh
        output_file = f"demo_output_{name}.ply"
        save_mesh(mesh, output_file)
        
        print(f"Mesh for {name}: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    return meshes


def visualize_results(voxel_data, meshes):
    """Visualize input voxels and generated meshes."""
    print("Visualizing results...")
    
    shape_names = ['sphere', 'cube', 'torus']
    
    for voxels, mesh, name in zip(voxel_data, meshes, shape_names):
        print(f"Visualizing {name}...")
        
        # Visualize voxels
        visualize_voxels(voxels, f"Input Voxels - {name.title()}")
        
        # Visualize mesh
        if len(mesh.vertices) > 0:
            visualize_mesh(mesh, f"Generated Mesh - {name.title()}")
            
            # Side-by-side comparison
            compare_voxel_and_mesh(voxels, mesh, f"Comparison - {name.title()}")
        else:
            print(f"Warning: Empty mesh generated for {name}")


def main():
    """Main demonstration function."""
    print("=== Voxel Occupancy Network - Basic Usage Demo ===\n")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = create_model(device)
    
    # Create synthetic data
    voxel_data, point_data = create_synthetic_data()
    dataset = create_dataset(voxel_data, point_data)
    
    print(f"Created dataset with {len(dataset)} samples")
    
    # Train model (demo)
    model = train_model_demo(model, dataset, device)
    
    # Generate meshes
    meshes = generate_meshes_demo(model, voxel_data, device)
    
    # Visualize results
    try:
        visualize_results(voxel_data, meshes)
    except Exception as e:
        print(f"Visualization skipped (display not available): {e}")
    
    print("\n=== Demo completed successfully! ===")
    print("Generated files:")
    print("- demo_output_sphere.ply")
    print("- demo_output_cube.ply") 
    print("- demo_output_torus.ply")
    print("- demo_checkpoints/best_model.pth")


if __name__ == "__main__":
    main()
