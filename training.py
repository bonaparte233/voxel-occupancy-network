"""
Training utilities for voxel-conditioned occupancy networks.
Extracted from the original Occupancy Networks codebase.

This module contains training loops and trainer classes that preserve
the exact training procedures from the original implementation.
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import distributions as dist
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, Any
import numpy as np
from tqdm import tqdm
import os


class VoxelOccupancyTrainer:
    """Trainer for voxel-conditioned occupancy networks.
    
    Replicates the training logic from the original im2mesh/onet/training.py
    """
    
    def __init__(self, model, optimizer, device=None, threshold: float = 0.5,
                 eval_sample: bool = False):
        """Initialize trainer.
        
        Args:
            model: OccupancyNetwork model
            optimizer: PyTorch optimizer
            device: PyTorch device
            threshold: Threshold for evaluation
            eval_sample: Whether to sample during evaluation
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device or torch.device('cpu')
        self.threshold = threshold
        self.eval_sample = eval_sample

    def train_step(self, data: Dict[str, torch.Tensor]) -> float:
        """Performs a training step.
        
        Exact replication of original method from im2mesh/onet/training.py

        Args:
            data (dict): data dictionary
        """
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.compute_loss(data)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def eval_step(self, data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Performs an evaluation step.
        
        Exact replication of original method from im2mesh/onet/training.py

        Args:
            data (dict): data dictionary
        """
        self.model.eval()
        device = self.device
        threshold = self.threshold
        eval_dict = {}

        # Get data - exact same as original
        points = data.get('points').to(device)
        occ = data.get('points.occ').to(device)
        inputs = data.get('inputs', torch.empty(points.size(0), 0)).to(device)

        kwargs = {}

        with torch.no_grad():
            # Compute ELBO - exact same as original
            elbo, rec_error, kl = self.model.compute_elbo(
                points, occ, inputs, **kwargs)
            
            eval_dict['loss'] = -elbo.mean().item()
            eval_dict['rec_error'] = rec_error.mean().item()
            eval_dict['kl'] = kl.mean().item()

            # Compute IoU if points_iou available
            if 'points_iou' in data:
                points_iou = data.get('points_iou').to(device)
                occ_iou = data.get('points_iou.occ').to(device)
                
                p_out = self.model(points_iou, inputs, sample=self.eval_sample, **kwargs)
                occ_iou_np = (occ_iou >= 0.5).cpu().numpy()
                occ_iou_hat_np = (p_out.probs >= threshold).cpu().numpy()
                
                iou = compute_iou(occ_iou_np, occ_iou_hat_np).mean()
                eval_dict['iou'] = iou

        return eval_dict

    def compute_loss(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Computes the loss.
        
        Exact replication of original method from im2mesh/onet/training.py

        Args:
            data (dict): data dictionary
        """
        device = self.device
        p = data.get('points').to(device)
        occ = data.get('points.occ').to(device)
        inputs = data.get('inputs', torch.empty(p.size(0), 0)).to(device)

        kwargs = {}

        # Encode inputs - exact same as original
        c = self.model.encode_inputs(inputs)
        q_z = self.model.infer_z(p, occ, c, **kwargs)
        z = q_z.rsample()

        # KL-divergence - exact same as original
        kl = dist.kl_divergence(q_z, self.model.p0_z).sum(dim=-1)
        loss = kl.mean()

        # General points - exact same as original
        logits = self.model.decode(p, z, c, **kwargs).logits
        loss_i = F.binary_cross_entropy_with_logits(
            logits, occ, reduction='none')
        loss = loss + loss_i.sum(-1).mean()

        return loss

    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Performs an evaluation.
        
        Exact replication of original method from im2mesh/training.py
        
        Args:
            val_loader (dataloader): pytorch dataloader
        """
        eval_list = {}

        for data in tqdm(val_loader, desc='Evaluating'):
            eval_step_dict = self.eval_step(data)

            for k, v in eval_step_dict.items():
                if k not in eval_list:
                    eval_list[k] = []
                eval_list[k].append(v)

        eval_dict = {k: np.mean(v) for k, v in eval_list.items()}
        return eval_dict


def compute_iou(occ1: np.ndarray, occ2: np.ndarray) -> np.ndarray:
    """Computes the Intersection over Union (IoU) value for two sets of
    occupancy values.
    
    Exact replication of original function.

    Args:
        occ1 (numpy array): first set of occupancy values
        occ2 (numpy array): second set of occupancy values
    """
    occ1 = np.asarray(occ1)
    occ2 = np.asarray(occ2)

    # Put all data in second dimension
    # Also works for 1-dimensional data
    if occ1.ndim >= 2:
        occ1 = occ1.reshape(occ1.shape[0], -1)
    if occ2.ndim >= 2:
        occ2 = occ2.reshape(occ2.shape[0], -1)

    # Convert to boolean values
    occ1 = (occ1 >= 0.5)
    occ2 = (occ2 >= 0.5)

    # Compute IOU
    area_union = (occ1 | occ2).astype(np.float32).sum(axis=-1)
    area_intersect = (occ1 & occ2).astype(np.float32).sum(axis=-1)

    iou = (area_intersect / area_union)

    return iou


def train_voxel_model(model, train_loader: DataLoader, val_loader: DataLoader,
                     n_epochs: int = 100, lr: float = 1e-4, device=None,
                     save_checkpoints: bool = True, checkpoint_dir: str = 'checkpoints',
                     eval_every: int = 10, print_every: int = 1) -> Tuple[list, list]:
    """Train a voxel-conditioned occupancy network.
    
    Args:
        model: OccupancyNetwork model
        train_loader: Training data loader
        val_loader: Validation data loader
        n_epochs: Number of training epochs
        lr: Learning rate
        device: PyTorch device
        save_checkpoints: Whether to save checkpoints
        checkpoint_dir: Directory to save checkpoints
        eval_every: Evaluate every N epochs
        print_every: Print progress every N epochs
        
    Returns:
        Tuple of (train_losses, val_losses)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    trainer = VoxelOccupancyTrainer(model, optimizer, device)
    
    if save_checkpoints and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        epoch_train_losses = []
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epochs}'):
            loss = trainer.train_step(batch)
            epoch_train_losses.append(loss)
        
        avg_train_loss = np.mean(epoch_train_losses)
        train_losses.append(avg_train_loss)
        
        # Evaluation
        if (epoch + 1) % eval_every == 0:
            eval_dict = trainer.evaluate(val_loader)
            val_loss = eval_dict['loss']
            val_losses.append(val_loss)
            
            # Save best model
            if save_checkpoints and val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': val_loss,
                }, checkpoint_path)
        
        # Print progress
        if (epoch + 1) % print_every == 0:
            print(f'Epoch {epoch+1}/{n_epochs}:')
            print(f'  Train Loss: {avg_train_loss:.6f}')
            if val_losses:
                print(f'  Val Loss: {val_losses[-1]:.6f}')
                if 'iou' in eval_dict:
                    print(f'  Val IoU: {eval_dict["iou"]:.4f}')
    
    return train_losses, val_losses
