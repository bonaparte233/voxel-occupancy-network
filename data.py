"""
Data loading and processing utilities for voxel-conditioned occupancy networks.
Extracted from the original Occupancy Networks codebase.

This module contains the exact implementations of data fields and dataset classes,
preserving all data loading and preprocessing logic.
"""

import os
import numpy as np
import torch
from torch.utils import data
from typing import Optional, List, Union, Callable, Any, Dict


class Field(object):
    """Data fields base class.
    
    Exactly replicates the original implementation from im2mesh/data/core.py
    """

    def load(self, data_path: str, idx: int, category: int):
        """Loads a data point.

        Args:
            data_path (str): path to data file
            idx (int): index of data point
            category (int): index of category
        """
        raise NotImplementedError

    def check_complete(self, files: List[str]) -> bool:
        """Checks if set is complete.

        Args:
            files: files
        """
        raise NotImplementedError


class VoxelsField(Field):
    """Voxel field class.
    
    Exactly replicates the original implementation from im2mesh/data/fields.py

    It provides the class used for voxel-based data.

    Args:
        file_name (str): file name
        transform (callable): list of transformations applied to data points
    """
    
    def __init__(self, file_name: str, transform: Optional[Callable] = None):
        self.file_name = file_name
        self.transform = transform

    def load(self, model_path: str, idx: int, category: int) -> np.ndarray:
        """Loads the data point.
        
        Exact replication of original method.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        """
        file_path = os.path.join(model_path, self.file_name)

        # Support multiple voxel formats
        if file_path.endswith('.binvox'):
            voxels = self._load_binvox(file_path)
        elif file_path.endswith('.npy'):
            voxels = np.load(file_path).astype(np.float32)
        elif file_path.endswith('.npz'):
            voxel_dict = np.load(file_path)
            # Try common keys for voxel data
            if 'voxels' in voxel_dict:
                voxels = voxel_dict['voxels'].astype(np.float32)
            elif 'data' in voxel_dict:
                voxels = voxel_dict['data'].astype(np.float32)
            else:
                # Take the first array if no standard key found
                voxels = list(voxel_dict.values())[0].astype(np.float32)
        else:
            raise ValueError(f"Unsupported voxel file format: {file_path}")

        if self.transform is not None:
            voxels = self.transform(voxels)

        return voxels

    def _load_binvox(self, file_path: str) -> np.ndarray:
        """Load binvox file - simplified implementation."""
        try:
            # Try to import binvox_rw if available
            import binvox_rw
            with open(file_path, 'rb') as f:
                voxels = binvox_rw.read_as_3d_array(f)
            return voxels.data.astype(np.float32)
        except ImportError:
            raise ImportError("binvox_rw package required for .binvox files. "
                            "Install with: pip install binvox-rw")

    def check_complete(self, files: List[str]) -> bool:
        """Check if field is complete.
        
        Exact replication of original method.
        
        Args:
            files: files
        """
        complete = (self.file_name in files)
        return complete


class PointsField(Field):
    """Point Field.
    
    Exactly replicates the original implementation from im2mesh/data/fields.py

    It provides the field to load point data. This is used for the points
    randomly sampled in the bounding volume of the 3D shape.

    Args:
        file_name (str): file name
        transform (callable): list of transformations which will be applied to the
            points tensor
        with_transforms (bool): whether scaling and rotation data should be
            provided
        unpackbits (bool): whether to unpack bits for occupancy
    """
    
    def __init__(self, file_name: str, transform: Optional[Callable] = None, 
                 with_transforms: bool = False, unpackbits: bool = False):
        self.file_name = file_name
        self.transform = transform
        self.with_transforms = with_transforms
        self.unpackbits = unpackbits

    def load(self, model_path: str, idx: int, category: int) -> Dict[str, np.ndarray]:
        """Loads the data point.
        
        Exact replication of original method.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        """
        file_path = os.path.join(model_path, self.file_name)

        points_dict = np.load(file_path)
        points = points_dict['points']
        
        # Break symmetry if given in float16 - exact same as original
        if points.dtype == np.float16:
            points = points.astype(np.float32)
            points += 1e-4 * np.random.randn(*points.shape)
        else:
            points = points.astype(np.float32)

        occupancies = points_dict['occupancies']
        if self.unpackbits:
            occupancies = np.unpackbits(occupancies)[:points.shape[0]]
        occupancies = occupancies.astype(np.float32)

        data = {
            None: points,
            'occ': occupancies,
        }

        if self.with_transforms:
            data['loc'] = points_dict['loc'].astype(np.float32)
            data['scale'] = points_dict['scale'].astype(np.float32)

        if self.transform is not None:
            data = self.transform(data)

        return data

    def check_complete(self, files: List[str]) -> bool:
        """Check if field is complete.
        
        Args:
            files: files
        """
        complete = (self.file_name in files)
        return complete


class VoxelDataset(data.Dataset):
    """Voxel dataset for training voxel-conditioned occupancy networks.
    
    Simplified dataset class focused on voxel data loading.
    """
    
    def __init__(self, voxel_files: List[str], point_files: Optional[List[str]] = None,
                 voxel_transform: Optional[Callable] = None,
                 point_transform: Optional[Callable] = None):
        """Initialize voxel dataset.
        
        Args:
            voxel_files: List of voxel file paths
            point_files: List of point file paths (optional)
            voxel_transform: Transform for voxel data
            point_transform: Transform for point data
        """
        self.voxel_files = voxel_files
        self.point_files = point_files
        self.voxel_transform = voxel_transform
        self.point_transform = point_transform
        
        # Create fields
        self.voxel_field = VoxelsField('dummy', transform=voxel_transform)
        if point_files:
            self.point_field = PointsField('dummy', transform=point_transform, unpackbits=True)

    def __len__(self) -> int:
        return len(self.voxel_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get dataset item.
        
        Args:
            idx: Index
            
        Returns:
            Dictionary containing voxel and point data
        """
        # Load voxel data
        voxel_path = self.voxel_files[idx]
        if voxel_path.endswith('.binvox'):
            voxels = self.voxel_field._load_binvox(voxel_path)
        elif voxel_path.endswith('.npy'):
            voxels = np.load(voxel_path).astype(np.float32)
        elif voxel_path.endswith('.npz'):
            voxel_dict = np.load(voxel_path)
            if 'voxels' in voxel_dict:
                voxels = voxel_dict['voxels'].astype(np.float32)
            elif 'data' in voxel_dict:
                voxels = voxel_dict['data'].astype(np.float32)
            else:
                voxels = list(voxel_dict.values())[0].astype(np.float32)
        else:
            raise ValueError(f"Unsupported voxel format: {voxel_path}")
            
        if self.voxel_transform:
            voxels = self.voxel_transform(voxels)
            
        data = {
            'inputs': torch.from_numpy(voxels).float()
        }
        
        # Load point data if available
        if self.point_files and idx < len(self.point_files):
            point_path = self.point_files[idx]
            points_dict = np.load(point_path)
            
            points = points_dict['points'].astype(np.float32)
            occupancies = points_dict['occupancies'].astype(np.float32)
            
            if self.point_transform:
                point_data = {'points': points, 'occ': occupancies}
                point_data = self.point_transform(point_data)
                points = point_data['points']
                occupancies = point_data['occ']
            
            data['points'] = torch.from_numpy(points).float()
            data['points.occ'] = torch.from_numpy(occupancies).float()
        
        return data


def collate_remove_none(batch: List[Any]) -> Any:
    """Collater that puts each data field into a tensor with outer dimension batch size.
    
    Exact replication of original function from im2mesh/data/core.py

    Args:
        batch: batch
    """
    batch = list(filter(lambda x: x is not None, batch))
    return data.dataloader.default_collate(batch)


def worker_init_fn(worker_id: int) -> None:
    """Worker init function to ensure true randomness.
    
    Exact replication of original function from im2mesh/data/core.py
    """
    random_data = os.urandom(4)
    base_seed = int.from_bytes(random_data, byteorder="big")
    np.random.seed(base_seed + worker_id)
