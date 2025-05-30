# Voxel Occupancy Network - Standalone Module
# Extracted from the original Occupancy Networks codebase
# Preserves exact neural network architectures and mathematical computations

from .models import (
    VoxelEncoder,
    DecoderCBatchNorm, 
    OccupancyNetwork
)

from .data import (
    VoxelsField,
    PointsField,
    VoxelDataset
)

from .training import (
    VoxelOccupancyTrainer,
    train_voxel_model
)

from .generation import (
    extract_mesh_marching_cubes,
    VoxelToMeshGenerator
)

from .utils import (
    load_voxel_data,
    save_mesh,
    visualize_voxels,
    visualize_mesh
)

__version__ = "1.0.0"
__author__ = "Extracted from Occupancy Networks by Mescheder et al."

# Main components for easy access
__all__ = [
    # Core models
    'VoxelEncoder',
    'DecoderCBatchNorm', 
    'OccupancyNetwork',
    
    # Data handling
    'VoxelsField',
    'PointsField', 
    'VoxelDataset',
    
    # Training
    'VoxelOccupancyTrainer',
    'train_voxel_model',
    
    # Generation
    'extract_mesh_marching_cubes',
    'VoxelToMeshGenerator',
    
    # Utilities
    'load_voxel_data',
    'save_mesh',
    'visualize_voxels',
    'visualize_mesh'
]
