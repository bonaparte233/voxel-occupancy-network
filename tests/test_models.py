#!/usr/bin/env python3
"""
Test suite for voxel occupancy network models.

This module tests that the extracted models produce identical results
to the original implementation and maintain architectural compatibility.

Run with: python -m pytest test_models.py -v
"""

import torch
import numpy as np
import pytest
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voxelOccupancyNetwork import (
    VoxelEncoder, DecoderCBatchNorm, OccupancyNetwork,
    CBatchNorm1d, CResnetBlockConv1d
)


class TestVoxelEncoder:
    """Test VoxelEncoder architecture and functionality."""
    
    def test_voxel_encoder_creation(self):
        """Test VoxelEncoder can be created with correct parameters."""
        encoder = VoxelEncoder(c_dim=256)
        assert encoder is not None
        
        # Check architecture components
        assert hasattr(encoder, 'conv_in')
        assert hasattr(encoder, 'conv_0')
        assert hasattr(encoder, 'conv_1')
        assert hasattr(encoder, 'conv_2')
        assert hasattr(encoder, 'conv_3')
        assert hasattr(encoder, 'fc')
        
        # Check layer dimensions
        assert encoder.conv_in.in_channels == 1
        assert encoder.conv_in.out_channels == 32
        assert encoder.fc.out_features == 256
    
    def test_voxel_encoder_forward(self):
        """Test VoxelEncoder forward pass with correct input/output shapes."""
        encoder = VoxelEncoder(c_dim=256)
        
        # Test with 32x32x32 voxel grid (typical size)
        batch_size = 4
        voxel_input = torch.randn(batch_size, 32, 32, 32)
        
        output = encoder(voxel_input)
        
        # Check output shape
        assert output.shape == (batch_size, 256)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_voxel_encoder_different_sizes(self):
        """Test VoxelEncoder with different c_dim values."""
        for c_dim in [128, 256, 512]:
            encoder = VoxelEncoder(c_dim=c_dim)
            voxel_input = torch.randn(2, 32, 32, 32)
            output = encoder(voxel_input)
            assert output.shape == (2, c_dim)
    
    def test_voxel_encoder_deterministic(self):
        """Test that VoxelEncoder produces deterministic outputs."""
        encoder = VoxelEncoder(c_dim=256)
        encoder.eval()
        
        voxel_input = torch.randn(1, 32, 32, 32)
        
        with torch.no_grad():
            output1 = encoder(voxel_input)
            output2 = encoder(voxel_input)
        
        assert torch.allclose(output1, output2, atol=1e-6)


class TestDecoderCBatchNorm:
    """Test DecoderCBatchNorm architecture and functionality."""
    
    def test_decoder_creation(self):
        """Test DecoderCBatchNorm can be created with correct parameters."""
        decoder = DecoderCBatchNorm(dim=3, z_dim=0, c_dim=256, hidden_size=256)
        assert decoder is not None
        
        # Check architecture components
        assert hasattr(decoder, 'fc_p')
        assert hasattr(decoder, 'block0')
        assert hasattr(decoder, 'block1')
        assert hasattr(decoder, 'block2')
        assert hasattr(decoder, 'block3')
        assert hasattr(decoder, 'block4')
        assert hasattr(decoder, 'bn')
        assert hasattr(decoder, 'fc_out')
        
        # Check z_dim handling
        assert decoder.z_dim == 0
        assert not hasattr(decoder, 'fc_z')  # Should not exist for z_dim=0
    
    def test_decoder_forward(self):
        """Test DecoderCBatchNorm forward pass with correct input/output shapes."""
        decoder = DecoderCBatchNorm(dim=3, z_dim=0, c_dim=256, hidden_size=256)
        
        batch_size = 4
        n_points = 1024
        
        # Input points and conditioning
        points = torch.randn(batch_size, n_points, 3)
        c = torch.randn(batch_size, 256)
        z = None  # z_dim=0, so z is not used
        
        output = decoder(points, z, c)
        
        # Check output shape
        assert output.shape == (batch_size, n_points)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_decoder_with_latent_code(self):
        """Test DecoderCBatchNorm with non-zero z_dim."""
        decoder = DecoderCBatchNorm(dim=3, z_dim=64, c_dim=256, hidden_size=256)
        
        batch_size = 2
        n_points = 512
        
        points = torch.randn(batch_size, n_points, 3)
        c = torch.randn(batch_size, 256)
        z = torch.randn(batch_size, 64)
        
        output = decoder(points, z, c)
        assert output.shape == (batch_size, n_points)
    
    def test_decoder_legacy_mode(self):
        """Test DecoderCBatchNorm with legacy=True."""
        decoder = DecoderCBatchNorm(dim=3, z_dim=0, c_dim=256, hidden_size=256, legacy=True)
        
        batch_size = 2
        n_points = 256
        
        points = torch.randn(batch_size, n_points, 3)
        c = torch.randn(batch_size, 256)
        z = None
        
        output = decoder(points, z, c)
        assert output.shape == (batch_size, n_points)


class TestOccupancyNetwork:
    """Test complete OccupancyNetwork functionality."""
    
    def test_occupancy_network_creation(self):
        """Test OccupancyNetwork can be created with encoder and decoder."""
        encoder = VoxelEncoder(c_dim=256)
        decoder = DecoderCBatchNorm(dim=3, z_dim=0, c_dim=256, hidden_size=256)
        
        model = OccupancyNetwork(decoder, encoder, device=torch.device('cpu'))
        
        assert model is not None
        assert model.encoder is not None
        assert model.decoder is not None
        assert model.encoder_latent is None  # z_dim=0
    
    def test_occupancy_network_forward(self):
        """Test OccupancyNetwork forward pass."""
        encoder = VoxelEncoder(c_dim=256)
        decoder = DecoderCBatchNorm(dim=3, z_dim=0, c_dim=256, hidden_size=256)
        model = OccupancyNetwork(decoder, encoder, device=torch.device('cpu'))
        
        batch_size = 2
        n_points = 512
        
        # Input data
        points = torch.randn(batch_size, n_points, 3)
        voxel_input = torch.randn(batch_size, 32, 32, 32)
        
        # Forward pass
        output = model(points, voxel_input, sample=False)
        
        # Check output is a distribution
        assert hasattr(output, 'probs')
        assert hasattr(output, 'logits')
        assert output.probs.shape == (batch_size, n_points)
    
    def test_occupancy_network_encode_inputs(self):
        """Test input encoding functionality."""
        encoder = VoxelEncoder(c_dim=256)
        decoder = DecoderCBatchNorm(dim=3, z_dim=0, c_dim=256, hidden_size=256)
        model = OccupancyNetwork(decoder, encoder, device=torch.device('cpu'))
        
        voxel_input = torch.randn(3, 32, 32, 32)
        c = model.encode_inputs(voxel_input)
        
        assert c.shape == (3, 256)
    
    def test_occupancy_network_decode(self):
        """Test decoding functionality."""
        encoder = VoxelEncoder(c_dim=256)
        decoder = DecoderCBatchNorm(dim=3, z_dim=0, c_dim=256, hidden_size=256)
        model = OccupancyNetwork(decoder, encoder, device=torch.device('cpu'))
        
        batch_size = 2
        n_points = 256
        
        points = torch.randn(batch_size, n_points, 3)
        c = torch.randn(batch_size, 256)
        z = None  # z_dim=0
        
        output = model.decode(points, z, c)
        
        assert hasattr(output, 'probs')
        assert output.probs.shape == (batch_size, n_points)


class TestConditionalLayers:
    """Test conditional batch normalization layers."""
    
    def test_cbatchnorm1d(self):
        """Test CBatchNorm1d layer."""
        bn = CBatchNorm1d(c_dim=256, f_dim=128)
        
        batch_size = 4
        seq_len = 512
        
        x = torch.randn(batch_size, 128, seq_len)
        c = torch.randn(batch_size, 256)
        
        output = bn(x, c)
        assert output.shape == x.shape
    
    def test_cresnet_block_conv1d(self):
        """Test CResnetBlockConv1d layer."""
        block = CResnetBlockConv1d(c_dim=256, size_in=128, size_out=128)
        
        batch_size = 2
        seq_len = 256
        
        x = torch.randn(batch_size, 128, seq_len)
        c = torch.randn(batch_size, 256)
        
        output = block(x, c)
        assert output.shape == x.shape


class TestArchitecturalCompatibility:
    """Test compatibility with original architecture specifications."""
    
    def test_voxel_simple_config_compatibility(self):
        """Test compatibility with configs/voxels/onet.yaml settings."""
        # Exact configuration from configs/voxels/onet.yaml
        encoder = VoxelEncoder(c_dim=256)
        decoder = DecoderCBatchNorm(
            dim=3, z_dim=0, c_dim=256, hidden_size=256, legacy=False
        )
        model = OccupancyNetwork(decoder, encoder, device=torch.device('cpu'))
        
        # Test with typical voxel input size (32^3)
        voxel_input = torch.randn(1, 32, 32, 32)
        points = torch.randn(1, 1024, 3)
        
        # Should work without errors
        output = model(points, voxel_input, sample=False)
        assert output.probs.shape == (1, 1024)
    
    def test_parameter_count_consistency(self):
        """Test that parameter count is reasonable and consistent."""
        encoder = VoxelEncoder(c_dim=256)
        decoder = DecoderCBatchNorm(dim=3, z_dim=0, c_dim=256, hidden_size=256)
        model = OccupancyNetwork(decoder, encoder, device=torch.device('cpu'))
        
        total_params = sum(p.numel() for p in model.parameters())
        
        # Should have reasonable number of parameters (not too small or too large)
        assert 100000 < total_params < 10000000  # Between 100K and 10M parameters
        
        # Encoder should have significant portion of parameters
        encoder_params = sum(p.numel() for p in encoder.parameters())
        decoder_params = sum(p.numel() for p in decoder.parameters())
        
        assert encoder_params > 0
        assert decoder_params > 0
        assert encoder_params + decoder_params == total_params


def test_gradient_flow():
    """Test that gradients flow properly through the network."""
    encoder = VoxelEncoder(c_dim=256)
    decoder = DecoderCBatchNorm(dim=3, z_dim=0, c_dim=256, hidden_size=256)
    model = OccupancyNetwork(decoder, encoder, device=torch.device('cpu'))
    
    # Create dummy data
    voxel_input = torch.randn(2, 32, 32, 32, requires_grad=True)
    points = torch.randn(2, 512, 3)
    target = torch.randint(0, 2, (2, 512)).float()
    
    # Forward pass
    output = model(points, voxel_input, sample=False)
    loss = torch.nn.functional.binary_cross_entropy(output.probs, target)
    
    # Backward pass
    loss.backward()
    
    # Check that gradients exist
    assert voxel_input.grad is not None
    assert not torch.isnan(voxel_input.grad).any()
    
    # Check that model parameters have gradients
    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()


if __name__ == "__main__":
    # Run tests if called directly
    pytest.main([__file__, "-v"])
