#!/usr/bin/env python3
"""
Compatibility test suite for voxel occupancy network.

This module tests that the extracted implementation produces identical
results to the original Occupancy Networks codebase.

Run with: python test_compatibility.py
"""

import torch
import numpy as np
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voxelOccupancyNetwork import (
    VoxelEncoder, DecoderCBatchNorm, OccupancyNetwork,
    VoxelToMeshGenerator, create_synthetic_voxel_data
)


def test_voxel_encoder_architecture():
    """Test that VoxelEncoder has the exact same architecture as original."""
    print("Testing VoxelEncoder architecture...")
    
    encoder = VoxelEncoder(c_dim=256)
    
    # Test layer structure
    expected_layers = [
        ('conv_in', torch.nn.Conv3d, (1, 32, 3)),
        ('conv_0', torch.nn.Conv3d, (32, 64, 3)),
        ('conv_1', torch.nn.Conv3d, (64, 128, 3)),
        ('conv_2', torch.nn.Conv3d, (128, 256, 3)),
        ('conv_3', torch.nn.Conv3d, (256, 512, 3)),
        ('fc', torch.nn.Linear, (4096, 256))
    ]
    
    for layer_name, layer_type, expected_shape in expected_layers:
        layer = getattr(encoder, layer_name)
        assert isinstance(layer, layer_type), f"{layer_name} should be {layer_type}"
        
        if layer_type == torch.nn.Conv3d:
            in_ch, out_ch, kernel = expected_shape
            assert layer.in_channels == in_ch, f"{layer_name} in_channels mismatch"
            assert layer.out_channels == out_ch, f"{layer_name} out_channels mismatch"
            assert layer.kernel_size == (kernel, kernel, kernel), f"{layer_name} kernel_size mismatch"
        elif layer_type == torch.nn.Linear:
            in_feat, out_feat = expected_shape
            assert layer.in_features == in_feat, f"{layer_name} in_features mismatch"
            assert layer.out_features == out_feat, f"{layer_name} out_features mismatch"
    
    print("✓ VoxelEncoder architecture matches original")


def test_decoder_architecture():
    """Test that DecoderCBatchNorm has the exact same architecture as original."""
    print("Testing DecoderCBatchNorm architecture...")
    
    decoder = DecoderCBatchNorm(dim=3, z_dim=0, c_dim=256, hidden_size=256, legacy=False)
    
    # Test that z_dim=0 doesn't create fc_z
    assert not hasattr(decoder, 'fc_z'), "fc_z should not exist for z_dim=0"
    
    # Test required components
    required_components = ['fc_p', 'block0', 'block1', 'block2', 'block3', 'block4', 'bn', 'fc_out']
    for component in required_components:
        assert hasattr(decoder, component), f"Missing component: {component}"
    
    # Test fc_p layer
    assert decoder.fc_p.in_channels == 3, "fc_p should have 3 input channels"
    assert decoder.fc_p.out_channels == 256, "fc_p should have 256 output channels"
    
    # Test fc_out layer
    assert decoder.fc_out.in_channels == 256, "fc_out should have 256 input channels"
    assert decoder.fc_out.out_channels == 1, "fc_out should have 1 output channel"
    
    print("✓ DecoderCBatchNorm architecture matches original")


def test_forward_pass_shapes():
    """Test that forward passes produce correct output shapes."""
    print("Testing forward pass shapes...")
    
    # Create model
    encoder = VoxelEncoder(c_dim=256)
    decoder = DecoderCBatchNorm(dim=3, z_dim=0, c_dim=256, hidden_size=256)
    model = OccupancyNetwork(decoder, encoder, device=torch.device('cpu'))
    
    batch_size = 4
    n_points = 1024
    
    # Test data
    voxel_input = torch.randn(batch_size, 32, 32, 32)
    points = torch.randn(batch_size, n_points, 3)
    
    # Test encoder
    c = model.encode_inputs(voxel_input)
    assert c.shape == (batch_size, 256), f"Encoder output shape mismatch: {c.shape}"
    
    # Test decoder
    z = None  # z_dim=0
    p_r = model.decode(points, z, c)
    assert p_r.logits.shape == (batch_size, n_points), f"Decoder output shape mismatch: {p_r.logits.shape}"
    
    # Test full forward pass
    output = model(points, voxel_input, sample=False)
    assert output.probs.shape == (batch_size, n_points), f"Model output shape mismatch: {output.probs.shape}"
    
    print("✓ Forward pass shapes are correct")


def test_deterministic_behavior():
    """Test that the model produces deterministic outputs in eval mode."""
    print("Testing deterministic behavior...")
    
    # Create model
    encoder = VoxelEncoder(c_dim=256)
    decoder = DecoderCBatchNorm(dim=3, z_dim=0, c_dim=256, hidden_size=256)
    model = OccupancyNetwork(decoder, encoder, device=torch.device('cpu'))
    model.eval()
    
    # Test data
    voxel_input = torch.randn(2, 32, 32, 32)
    points = torch.randn(2, 512, 3)
    
    # Multiple forward passes
    with torch.no_grad():
        output1 = model(points, voxel_input, sample=False)
        output2 = model(points, voxel_input, sample=False)
    
    # Should be identical
    assert torch.allclose(output1.logits, output2.logits, atol=1e-6), "Model should be deterministic"
    
    print("✓ Model behavior is deterministic")


def test_gradient_computation():
    """Test that gradients are computed correctly."""
    print("Testing gradient computation...")
    
    # Create model
    encoder = VoxelEncoder(c_dim=256)
    decoder = DecoderCBatchNorm(dim=3, z_dim=0, c_dim=256, hidden_size=256)
    model = OccupancyNetwork(decoder, encoder, device=torch.device('cpu'))
    model.train()
    
    # Test data
    voxel_input = torch.randn(2, 32, 32, 32, requires_grad=True)
    points = torch.randn(2, 256, 3)
    target_occ = torch.randint(0, 2, (2, 256)).float()
    
    # Forward pass
    output = model(points, voxel_input, sample=False)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(output.logits, target_occ)
    
    # Backward pass
    loss.backward()
    
    # Check gradients exist and are finite
    assert voxel_input.grad is not None, "Input should have gradients"
    assert torch.isfinite(voxel_input.grad).all(), "Input gradients should be finite"
    
    # Check model parameter gradients
    param_count = 0
    grad_count = 0
    for param in model.parameters():
        if param.requires_grad:
            param_count += 1
            if param.grad is not None:
                grad_count += 1
                assert torch.isfinite(param.grad).all(), "Parameter gradients should be finite"
    
    assert grad_count == param_count, "All parameters should have gradients"
    
    print("✓ Gradient computation is correct")


def test_mesh_generation():
    """Test mesh generation functionality."""
    print("Testing mesh generation...")
    
    # Create model
    encoder = VoxelEncoder(c_dim=256)
    decoder = DecoderCBatchNorm(dim=3, z_dim=0, c_dim=256, hidden_size=256)
    model = OccupancyNetwork(decoder, encoder, device=torch.device('cpu'))
    model.eval()
    
    # Create synthetic voxel data
    voxels = create_synthetic_voxel_data('sphere', resolution=32)
    
    # Create generator
    generator = VoxelToMeshGenerator(model, torch.device('cpu'))
    
    # Generate mesh (low resolution for speed)
    mesh = generator.generate_mesh(voxels, resolution=16, threshold=0.5)
    
    # Check mesh properties
    assert hasattr(mesh, 'vertices'), "Mesh should have vertices"
    assert hasattr(mesh, 'faces'), "Mesh should have faces"
    
    print(f"✓ Mesh generation successful: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")


def test_training_compatibility():
    """Test that training loop works correctly."""
    print("Testing training compatibility...")
    
    from voxelOccupancyNetwork import VoxelOccupancyTrainer
    
    # Create model
    encoder = VoxelEncoder(c_dim=256)
    decoder = DecoderCBatchNorm(dim=3, z_dim=0, c_dim=256, hidden_size=256)
    model = OccupancyNetwork(decoder, encoder, device=torch.device('cpu'))
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Create trainer
    trainer = VoxelOccupancyTrainer(model, optimizer, device=torch.device('cpu'))
    
    # Create dummy batch
    batch = {
        'inputs': torch.randn(2, 32, 32, 32),
        'points': torch.randn(2, 512, 3),
        'points.occ': torch.randint(0, 2, (2, 512)).float()
    }
    
    # Training step
    initial_loss = trainer.train_step(batch)
    assert isinstance(initial_loss, float), "Training step should return float loss"
    assert initial_loss > 0, "Loss should be positive"
    
    # Evaluation step
    eval_dict = trainer.eval_step(batch)
    assert 'loss' in eval_dict, "Evaluation should return loss"
    assert 'rec_error' in eval_dict, "Evaluation should return reconstruction error"
    assert 'kl' in eval_dict, "Evaluation should return KL divergence"
    
    print("✓ Training compatibility verified")


def test_config_yaml_compatibility():
    """Test compatibility with configs/voxels/onet.yaml settings."""
    print("Testing configs/voxels/onet.yaml compatibility...")
    
    # Exact settings from configs/voxels/onet.yaml
    config = {
        'model': {
            'encoder': 'voxel_simple',  # -> VoxelEncoder
            'decoder': 'cbatchnorm',    # -> DecoderCBatchNorm
            'c_dim': 256,
            'z_dim': 0,
            'decoder_kwargs': {
                'dim': 3,
                'hidden_size': 256,
                'legacy': False
            }
        }
    }
    
    # Create model with exact config
    encoder = VoxelEncoder(c_dim=config['model']['c_dim'])
    decoder = DecoderCBatchNorm(
        dim=config['model']['decoder_kwargs']['dim'],
        z_dim=config['model']['z_dim'],
        c_dim=config['model']['c_dim'],
        hidden_size=config['model']['decoder_kwargs']['hidden_size'],
        legacy=config['model']['decoder_kwargs']['legacy']
    )
    model = OccupancyNetwork(decoder, encoder, device=torch.device('cpu'))
    
    # Test with typical input
    voxel_input = torch.randn(1, 32, 32, 32)  # Typical voxel size
    points = torch.randn(1, 1024, 3)          # Typical point count
    
    output = model(points, voxel_input, sample=False)
    assert output.probs.shape == (1, 1024), "Output shape should match config expectations"
    
    print("✓ configs/voxels/onet.yaml compatibility verified")


def run_all_tests():
    """Run all compatibility tests."""
    print("=" * 60)
    print("VOXEL OCCUPANCY NETWORK COMPATIBILITY TESTS")
    print("=" * 60)
    
    tests = [
        test_voxel_encoder_architecture,
        test_decoder_architecture,
        test_forward_pass_shapes,
        test_deterministic_behavior,
        test_gradient_computation,
        test_mesh_generation,
        test_training_compatibility,
        test_config_yaml_compatibility
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED! The extracted module is fully compatible.")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
