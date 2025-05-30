"""
Mesh generation utilities for voxel-conditioned occupancy networks.
Extracted from the original Occupancy Networks codebase.

This module contains mesh extraction and generation utilities that preserve
the exact mesh generation procedures from the original implementation.
"""

import torch
import numpy as np
from typing import Tuple, Optional, Union
import trimesh
from skimage import measure


def make_3d_grid(bb_min: Tuple[float, float, float], 
                bb_max: Tuple[float, float, float],
                shape: Tuple[int, int, int]) -> torch.Tensor:
    """Makes a 3D grid.
    
    Exact replication of original function from im2mesh/utils.py

    Args:
        bb_min: bounding box minimum
        bb_max: bounding box maximum  
        shape: output shape
    """
    size = shape[0] * shape[1] * shape[2]

    pxs = torch.linspace(bb_min[0], bb_max[0], shape[0])
    pys = torch.linspace(bb_min[1], bb_max[1], shape[1])
    pzs = torch.linspace(bb_min[2], bb_max[2], shape[2])

    pxs = pxs.view(-1, 1, 1).expand(*shape).contiguous().view(size)
    pys = pys.view(1, -1, 1).expand(*shape).contiguous().view(size)
    pzs = pzs.view(1, 1, -1).expand(*shape).contiguous().view(size)
    p = torch.stack([pxs, pys, pzs], dim=1)

    return p


def extract_mesh_marching_cubes(model, voxel_input: torch.Tensor,
                               resolution: int = 64, threshold: float = 0.5,
                               bb_min: Tuple[float, float, float] = (-0.6, -0.6, -0.6),
                               bb_max: Tuple[float, float, float] = (0.6, 0.6, 0.6),
                               device=None, batch_size: int = 100000) -> trimesh.Trimesh:
    """Extract mesh using marching cubes from voxel-conditioned occupancy network.
    
    Args:
        model: Trained occupancy network
        voxel_input: Input voxel grid tensor
        resolution: Grid resolution for marching cubes
        threshold: Occupancy threshold for surface extraction
        bb_min: Minimum bounding box coordinates
        bb_max: Maximum bounding box coordinates
        device: PyTorch device
        batch_size: Batch size for inference
        
    Returns:
        Extracted mesh as trimesh object
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    model = model.to(device)
    
    # Ensure voxel input has batch dimension
    if voxel_input.dim() == 3:
        voxel_input = voxel_input.unsqueeze(0)
    voxel_input = voxel_input.to(device)
    
    # Create 3D grid
    shape = (resolution, resolution, resolution)
    p = make_3d_grid(bb_min, bb_max, shape).to(device)
    
    # Encode voxel input
    with torch.no_grad():
        c = model.encode_inputs(voxel_input)
        
        # Predict occupancy for all points
        occupancy_probs = []
        
        for i in range(0, p.shape[0], batch_size):
            p_batch = p[i:i+batch_size].unsqueeze(0)  # Add batch dimension
            
            # Expand conditioning to match batch size
            c_batch = c.expand(p_batch.shape[0], -1)
            
            # Get occupancy predictions
            z = model.get_z_from_prior((p_batch.shape[0],), sample=False)
            p_r = model.decode(p_batch, z, c_batch)
            occupancy_probs.append(p_r.probs.squeeze(0))
        
        occupancy_probs = torch.cat(occupancy_probs, dim=0)
    
    # Reshape to 3D grid
    occupancy_grid = occupancy_probs.view(*shape).cpu().numpy()
    
    # Extract mesh using marching cubes
    try:
        vertices, faces, normals, _ = measure.marching_cubes(
            occupancy_grid, level=threshold, spacing=(
                (bb_max[0] - bb_min[0]) / (resolution - 1),
                (bb_max[1] - bb_min[1]) / (resolution - 1),
                (bb_max[2] - bb_min[2]) / (resolution - 1)
            )
        )
        
        # Adjust vertices to correct coordinate system
        vertices[:, 0] += bb_min[0]
        vertices[:, 1] += bb_min[1]
        vertices[:, 2] += bb_min[2]
        
        # Create mesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
        
        return mesh
        
    except ValueError as e:
        print(f"Warning: Could not extract mesh with threshold {threshold}: {e}")
        # Return empty mesh
        return trimesh.Trimesh()


class VoxelToMeshGenerator:
    """High-level generator for voxel-to-mesh conversion.
    
    Provides a clean interface for mesh generation from voxel inputs.
    """
    
    def __init__(self, model, device=None):
        """Initialize generator.
        
        Args:
            model: Trained occupancy network
            device: PyTorch device
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def generate_mesh(self, voxel_input: Union[torch.Tensor, np.ndarray],
                     resolution: int = 64, threshold: float = 0.5,
                     bb_min: Tuple[float, float, float] = (-0.6, -0.6, -0.6),
                     bb_max: Tuple[float, float, float] = (0.6, 0.6, 0.6),
                     batch_size: int = 100000) -> trimesh.Trimesh:
        """Generate mesh from voxel input.
        
        Args:
            voxel_input: Input voxel grid
            resolution: Grid resolution for marching cubes
            threshold: Occupancy threshold
            bb_min: Minimum bounding box coordinates
            bb_max: Maximum bounding box coordinates
            batch_size: Batch size for inference
            
        Returns:
            Generated mesh
        """
        # Convert to tensor if needed
        if isinstance(voxel_input, np.ndarray):
            voxel_input = torch.from_numpy(voxel_input).float()
        
        return extract_mesh_marching_cubes(
            self.model, voxel_input, resolution, threshold,
            bb_min, bb_max, self.device, batch_size
        )
    
    def generate_mesh_adaptive_threshold(self, voxel_input: Union[torch.Tensor, np.ndarray],
                                       resolution: int = 64,
                                       thresholds: Tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7),
                                       **kwargs) -> trimesh.Trimesh:
        """Generate mesh with adaptive threshold selection.
        
        Tries multiple thresholds and returns the first successful mesh.
        
        Args:
            voxel_input: Input voxel grid
            resolution: Grid resolution
            thresholds: Thresholds to try
            **kwargs: Additional arguments for generate_mesh
            
        Returns:
            Generated mesh
        """
        for threshold in thresholds:
            mesh = self.generate_mesh(voxel_input, resolution, threshold, **kwargs)
            if len(mesh.vertices) > 0:
                print(f"Successfully generated mesh with threshold {threshold}")
                return mesh
        
        print("Warning: Could not generate mesh with any threshold")
        return trimesh.Trimesh()
    
    def batch_generate(self, voxel_inputs: list, **kwargs) -> list:
        """Generate meshes for a batch of voxel inputs.
        
        Args:
            voxel_inputs: List of voxel inputs
            **kwargs: Arguments for generate_mesh
            
        Returns:
            List of generated meshes
        """
        meshes = []
        for i, voxel_input in enumerate(voxel_inputs):
            print(f"Generating mesh {i+1}/{len(voxel_inputs)}")
            mesh = self.generate_mesh(voxel_input, **kwargs)
            meshes.append(mesh)
        return meshes


def save_mesh(mesh: trimesh.Trimesh, filepath: str) -> None:
    """Save mesh to file.
    
    Args:
        mesh: Trimesh object
        filepath: Output file path
    """
    mesh.export(filepath)
    print(f"Mesh saved to {filepath}")


def load_voxel_data(filepath: str) -> np.ndarray:
    """Load voxel data from file.
    
    Args:
        filepath: Path to voxel file
        
    Returns:
        Voxel data as numpy array
    """
    if filepath.endswith('.npy'):
        return np.load(filepath)
    elif filepath.endswith('.npz'):
        data = np.load(filepath)
        if 'voxels' in data:
            return data['voxels']
        elif 'data' in data:
            return data['data']
        else:
            return list(data.values())[0]
    elif filepath.endswith('.binvox'):
        try:
            import binvox_rw
            with open(filepath, 'rb') as f:
                voxels = binvox_rw.read_as_3d_array(f)
            return voxels.data.astype(np.float32)
        except ImportError:
            raise ImportError("binvox_rw package required for .binvox files")
    else:
        raise ValueError(f"Unsupported file format: {filepath}")


def create_voxel_grid(points: np.ndarray, resolution: int = 32,
                     bb_min: Tuple[float, float, float] = (-0.6, -0.6, -0.6),
                     bb_max: Tuple[float, float, float] = (0.6, 0.6, 0.6)) -> np.ndarray:
    """Create voxel grid from point cloud.
    
    Args:
        points: Point cloud as (N, 3) array
        resolution: Voxel grid resolution
        bb_min: Minimum bounding box coordinates
        bb_max: Maximum bounding box coordinates
        
    Returns:
        Binary voxel grid
    """
    voxel_grid = np.zeros((resolution, resolution, resolution), dtype=np.float32)
    
    # Normalize points to voxel coordinates
    points_norm = (points - np.array(bb_min)) / (np.array(bb_max) - np.array(bb_min))
    points_voxel = (points_norm * (resolution - 1)).astype(int)
    
    # Clip to valid range
    points_voxel = np.clip(points_voxel, 0, resolution - 1)
    
    # Set voxels
    voxel_grid[points_voxel[:, 0], points_voxel[:, 1], points_voxel[:, 2]] = 1.0
    
    return voxel_grid
