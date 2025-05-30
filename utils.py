"""
Utility functions for voxel-conditioned occupancy networks.
Extracted from the original Occupancy Networks codebase.

This module contains visualization, data loading, and other utility functions.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh
from typing import Union, Tuple, Optional, List
import os


def load_voxel_data(filepath: str) -> np.ndarray:
    """Load voxel data from various file formats.

    Args:
        filepath: Path to voxel file

    Returns:
        Voxel data as numpy array
    """
    if filepath.endswith('.npy'):
        return np.load(filepath).astype(np.float32)
    elif filepath.endswith('.npz'):
        data = np.load(filepath)
        if 'voxels' in data:
            return data['voxels'].astype(np.float32)
        elif 'data' in data:
            return data['data'].astype(np.float32)
        else:
            return list(data.values())[0].astype(np.float32)
    elif filepath.endswith('.binvox'):
        try:
            import binvox_rw
            with open(filepath, 'rb') as f:
                voxels = binvox_rw.read_as_3d_array(f)
            return voxels.data.astype(np.float32)
        except ImportError:
            raise ImportError("binvox_rw package required for .binvox files. "
                            "Install with: pip install binvox-rw")
    else:
        raise ValueError(f"Unsupported file format: {filepath}")


def save_voxel_data(voxels: np.ndarray, filepath: str) -> None:
    """Save voxel data to file.

    Args:
        voxels: Voxel data
        filepath: Output file path
    """
    if filepath.endswith('.npy'):
        np.save(filepath, voxels)
    elif filepath.endswith('.npz'):
        np.savez_compressed(filepath, voxels=voxels)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")

    print(f"Voxel data saved to {filepath}")


def save_mesh(mesh: trimesh.Trimesh, filepath: str) -> None:
    """Save mesh to file.

    Args:
        mesh: Trimesh object
        filepath: Output file path
    """
    mesh.export(filepath)
    print(f"Mesh saved to {filepath}")


def visualize_voxels(voxels: np.ndarray, title: str = "Voxel Grid",
                    threshold: float = 0.5, figsize: Tuple[int, int] = (10, 8)) -> None:
    """Visualize 3D voxel grid.

    Args:
        voxels: 3D voxel grid
        title: Plot title
        threshold: Threshold for binary visualization
        figsize: Figure size
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Get occupied voxel coordinates
    occupied = voxels > threshold
    x, y, z = np.where(occupied)

    # Create scatter plot
    ax.scatter(x, y, z, c='blue', alpha=0.6, s=20)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    # Set equal aspect ratio
    max_range = max(voxels.shape)
    ax.set_xlim([0, max_range])
    ax.set_ylim([0, max_range])
    ax.set_zlim([0, max_range])

    plt.tight_layout()
    plt.show()


def visualize_mesh(mesh: trimesh.Trimesh, title: str = "Generated Mesh",
                  figsize: Tuple[int, int] = (10, 8)) -> None:
    """Visualize 3D mesh.

    Args:
        mesh: Trimesh object
        title: Plot title
        figsize: Figure size
    """
    if len(mesh.vertices) == 0:
        print("Warning: Empty mesh, cannot visualize")
        return

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Plot mesh vertices
    vertices = mesh.vertices
    faces = mesh.faces

    # Create triangular mesh plot
    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                   triangles=faces, alpha=0.7, cmap='viridis')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    # Set equal aspect ratio
    ax.set_box_aspect([1,1,1])

    plt.tight_layout()
    plt.show()


def compare_voxel_and_mesh(voxels: np.ndarray, mesh: trimesh.Trimesh,
                          title: str = "Voxel vs Mesh Comparison",
                          threshold: float = 0.5,
                          figsize: Tuple[int, int] = (15, 6)) -> None:
    """Compare voxel input and generated mesh side by side.

    Args:
        voxels: Input voxel grid
        mesh: Generated mesh
        title: Plot title
        threshold: Voxel threshold
        figsize: Figure size
    """
    fig = plt.figure(figsize=figsize)

    # Plot voxels
    ax1 = fig.add_subplot(121, projection='3d')
    occupied = voxels > threshold
    x, y, z = np.where(occupied)
    ax1.scatter(x, y, z, c='blue', alpha=0.6, s=20)
    ax1.set_title('Input Voxels')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # Plot mesh
    ax2 = fig.add_subplot(122, projection='3d')
    if len(mesh.vertices) > 0:
        vertices = mesh.vertices
        faces = mesh.faces
        ax2.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                        triangles=faces, alpha=0.7, cmap='viridis')
    ax2.set_title('Generated Mesh')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def create_synthetic_voxel_data(shape: str = 'sphere', resolution: int = 32,
                               noise_level: float = 0.0) -> np.ndarray:
    """Create synthetic voxel data for testing.

    Args:
        shape: Shape type ('sphere', 'cube', 'torus')
        resolution: Voxel grid resolution
        noise_level: Amount of noise to add

    Returns:
        Synthetic voxel grid
    """
    voxels = np.zeros((resolution, resolution, resolution), dtype=np.float32)
    center = resolution // 2

    if shape == 'sphere':
        for i in range(resolution):
            for j in range(resolution):
                for k in range(resolution):
                    dist = np.sqrt((i - center)**2 + (j - center)**2 + (k - center)**2)
                    if dist <= resolution * 0.3:
                        voxels[i, j, k] = 1.0

    elif shape == 'cube':
        start = resolution // 4
        end = 3 * resolution // 4
        voxels[start:end, start:end, start:end] = 1.0

    elif shape == 'torus':
        R = resolution * 0.25  # Major radius
        r = resolution * 0.1   # Minor radius
        for i in range(resolution):
            for j in range(resolution):
                for k in range(resolution):
                    x, y, z = i - center, j - center, k - center
                    dist_to_center = np.sqrt(x**2 + y**2)
                    torus_dist = np.sqrt((dist_to_center - R)**2 + z**2)
                    if torus_dist <= r:
                        voxels[i, j, k] = 1.0

    # Add noise if requested
    if noise_level > 0:
        noise = np.random.random(voxels.shape) < noise_level
        voxels = np.logical_xor(voxels, noise).astype(np.float32)

    return voxels


def compute_mesh_metrics(mesh: trimesh.Trimesh) -> dict:
    """Compute basic metrics for a mesh.

    Args:
        mesh: Trimesh object

    Returns:
        Dictionary of mesh metrics
    """
    if len(mesh.vertices) == 0:
        return {
            'num_vertices': 0,
            'num_faces': 0,
            'volume': 0.0,
            'surface_area': 0.0,
            'is_watertight': False
        }

    return {
        'num_vertices': len(mesh.vertices),
        'num_faces': len(mesh.faces),
        'volume': mesh.volume if mesh.is_volume else 0.0,
        'surface_area': mesh.area,
        'is_watertight': mesh.is_watertight,
        'bounds': mesh.bounds
    }


def batch_process_voxels(voxel_files: List[str], model, output_dir: str,
                        resolution: int = 64, threshold: float = 0.5,
                        device=None) -> List[str]:
    """Process a batch of voxel files to generate meshes.

    Args:
        voxel_files: List of voxel file paths
        model: Trained occupancy network
        output_dir: Output directory for meshes
        resolution: Marching cubes resolution
        threshold: Occupancy threshold
        device: PyTorch device

    Returns:
        List of output mesh file paths
    """
    try:
        from .generation import VoxelToMeshGenerator
    except ImportError:
        from generation import VoxelToMeshGenerator

    os.makedirs(output_dir, exist_ok=True)
    generator = VoxelToMeshGenerator(model, device)

    output_files = []

    for i, voxel_file in enumerate(voxel_files):
        print(f"Processing {i+1}/{len(voxel_files)}: {voxel_file}")

        # Load voxel data
        voxels = load_voxel_data(voxel_file)

        # Generate mesh
        mesh = generator.generate_mesh(voxels, resolution=resolution, threshold=threshold)

        # Save mesh
        base_name = os.path.splitext(os.path.basename(voxel_file))[0]
        output_file = os.path.join(output_dir, f"{base_name}_mesh.ply")
        save_mesh(mesh, output_file)
        output_files.append(output_file)

    return output_files


def load_model_checkpoint(model, checkpoint_path: str, device=None) -> Tuple[any, dict]:
    """Load model from checkpoint.

    Args:
        model: Model instance
        checkpoint_path: Path to checkpoint file
        device: PyTorch device

    Returns:
        Tuple of (loaded_model, checkpoint_info)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    checkpoint_info = {
        'epoch': checkpoint.get('epoch', 0),
        'train_loss': checkpoint.get('train_loss', None),
        'val_loss': checkpoint.get('val_loss', None)
    }

    return model, checkpoint_info


def save_model_checkpoint(model, optimizer, epoch: int, train_loss: float,
                         val_loss: Optional[float], filepath: str) -> None:
    """Save model checkpoint.

    Args:
        model: Model instance
        optimizer: Optimizer instance
        epoch: Current epoch
        train_loss: Training loss
        val_loss: Validation loss
        filepath: Output file path
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }, filepath)

    print(f"Checkpoint saved to {filepath}")
