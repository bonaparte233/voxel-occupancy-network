#!/usr/bin/env python3
"""
Example of loading and using a pre-trained voxel occupancy network.

This script demonstrates how to:
1. Load a pre-trained model from the original Occupancy Networks
2. Convert voxel data to meshes using the pre-trained weights
3. Batch process multiple voxel files

The model architecture is exactly compatible with the original implementation,
so you can use checkpoints trained with the original codebase.

Run with: python load_pretrained.py --model_path path/to/model.pth --input_dir path/to/voxels/
"""

import torch
import numpy as np
import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voxelOccupancyNetwork import (
    VoxelEncoder, DecoderCBatchNorm, OccupancyNetwork,
    VoxelToMeshGenerator, load_voxel_data, save_mesh,
    load_model_checkpoint, batch_process_voxels,
    visualize_voxels, visualize_mesh, compare_voxel_and_mesh
)


def create_compatible_model(device):
    """Create model with exact same architecture as original configs/voxels/onet.yaml"""
    print("Creating model compatible with original Occupancy Networks...")
    
    # Exact architecture from configs/voxels/onet.yaml
    encoder = VoxelEncoder(dim=3, c_dim=256)  # voxel_simple
    decoder = DecoderCBatchNorm(              # cbatchnorm
        dim=3, 
        z_dim=0,           # Deterministic model (no latent code)
        c_dim=256, 
        hidden_size=256,
        leaky=False,
        legacy=False
    )
    
    # Create occupancy network
    model = OccupancyNetwork(
        decoder=decoder,
        encoder=encoder,
        encoder_latent=None,  # No latent encoder for z_dim=0
        p0_z=None,          # No prior for z_dim=0
        device=device
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    return model


def load_pretrained_model(model_path, device):
    """Load pre-trained model from checkpoint."""
    print(f"Loading pre-trained model from {model_path}...")
    
    # Create model
    model = create_compatible_model(device)
    
    # Load checkpoint
    try:
        model, checkpoint_info = load_model_checkpoint(model, model_path, device)
        print(f"Successfully loaded model from epoch {checkpoint_info['epoch']}")
        if checkpoint_info['train_loss']:
            print(f"Training loss: {checkpoint_info['train_loss']:.6f}")
        if checkpoint_info['val_loss']:
            print(f"Validation loss: {checkpoint_info['val_loss']:.6f}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Trying to load state dict directly...")
        
        # Try loading state dict directly (for models saved with torch.save(model.state_dict(), ...))
        try:
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            print("Successfully loaded state dict")
        except Exception as e2:
            print(f"Error loading state dict: {e2}")
            raise
    
    model.eval()
    return model


def process_single_voxel(model, voxel_path, output_path, resolution=64, threshold=0.5, device=None):
    """Process a single voxel file to generate mesh."""
    print(f"Processing {voxel_path}...")
    
    # Load voxel data
    try:
        voxels = load_voxel_data(voxel_path)
        print(f"Loaded voxel grid with shape: {voxels.shape}")
    except Exception as e:
        print(f"Error loading voxel data: {e}")
        return None
    
    # Create generator
    generator = VoxelToMeshGenerator(model, device)
    
    # Generate mesh with adaptive threshold
    mesh = generator.generate_mesh_adaptive_threshold(
        voxels,
        resolution=resolution,
        thresholds=(0.3, 0.4, 0.5, 0.6, 0.7)
    )
    
    if len(mesh.vertices) == 0:
        print(f"Warning: Generated empty mesh for {voxel_path}")
        return None
    
    # Save mesh
    save_mesh(mesh, output_path)
    print(f"Generated mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    return mesh


def process_voxel_directory(model, input_dir, output_dir, resolution=64, device=None):
    """Process all voxel files in a directory."""
    print(f"Processing voxel files in {input_dir}...")
    
    # Find voxel files
    input_path = Path(input_dir)
    voxel_extensions = ['.npy', '.npz', '.binvox']
    voxel_files = []
    
    for ext in voxel_extensions:
        voxel_files.extend(list(input_path.glob(f'*{ext}')))
    
    if not voxel_files:
        print(f"No voxel files found in {input_dir}")
        return []
    
    print(f"Found {len(voxel_files)} voxel files")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process files
    results = []
    for voxel_file in voxel_files:
        output_file = output_path / f"{voxel_file.stem}_mesh.ply"
        
        mesh = process_single_voxel(
            model, str(voxel_file), str(output_file), 
            resolution=resolution, device=device
        )
        
        if mesh is not None:
            results.append((str(voxel_file), str(output_file), mesh))
    
    print(f"Successfully processed {len(results)} files")
    return results


def visualize_results(results, max_visualizations=3):
    """Visualize some of the results."""
    print("Visualizing results...")
    
    for i, (voxel_path, mesh_path, mesh) in enumerate(results[:max_visualizations]):
        print(f"Visualizing result {i+1}: {Path(voxel_path).name}")
        
        try:
            # Load original voxel data
            voxels = load_voxel_data(voxel_path)
            
            # Visualize comparison
            compare_voxel_and_mesh(
                voxels, mesh, 
                f"Result {i+1}: {Path(voxel_path).stem}"
            )
            
        except Exception as e:
            print(f"Visualization error for {voxel_path}: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Load pre-trained voxel occupancy network and generate meshes')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to pre-trained model checkpoint')
    parser.add_argument('--input_dir', type=str, 
                       help='Directory containing voxel files')
    parser.add_argument('--input_file', type=str,
                       help='Single voxel file to process')
    parser.add_argument('--output_dir', type=str, default='output_meshes',
                       help='Output directory for generated meshes')
    parser.add_argument('--resolution', type=int, default=64,
                       help='Marching cubes resolution')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize results (requires display)')
    
    args = parser.parse_args()
    
    if not args.input_dir and not args.input_file:
        print("Error: Must specify either --input_dir or --input_file")
        return
    
    print("=== Pre-trained Voxel Occupancy Network Demo ===\n")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load pre-trained model
    try:
        model = load_pretrained_model(args.model_path, device)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    # Process voxel data
    results = []
    
    if args.input_file:
        # Process single file
        output_file = Path(args.output_dir) / f"{Path(args.input_file).stem}_mesh.ply"
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
        mesh = process_single_voxel(
            model, args.input_file, str(output_file), 
            args.resolution, device=device
        )
        
        if mesh is not None:
            results.append((args.input_file, str(output_file), mesh))
    
    elif args.input_dir:
        # Process directory
        results = process_voxel_directory(
            model, args.input_dir, args.output_dir, 
            args.resolution, device=device
        )
    
    # Visualize results if requested
    if args.visualize and results:
        try:
            visualize_results(results)
        except Exception as e:
            print(f"Visualization skipped: {e}")
    
    print(f"\n=== Processing completed! ===")
    print(f"Generated {len(results)} meshes in {args.output_dir}")


if __name__ == "__main__":
    main()
