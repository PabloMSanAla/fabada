"""
Integration tests for FABADA examples - Tests the complete workflow from example files.
"""

import os
import sys
import subprocess
import numpy as np
import pandas as pd
import pytest
import tempfile
import shutil

# Add the parent directory to the path so we can import fabada
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fabada import fabada, PSNR

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class TestFabadaIntegration:
    """Integration tests that run complete FABADA workflows similar to examples."""

    @pytest.fixture
    def examples_dir(self):
        """Get the examples directory path."""
        return os.path.join(os.path.dirname(__file__), '..', 'examples')

    def test_spectrum_example_workflow(self, examples_dir):
        """Test the complete spectrum example workflow."""
        # Load data like in fabada_demo_spectrum.py
        arp256_path = os.path.join(examples_dir, 'arp256.csv')
        
        if not os.path.exists(arp256_path):
            pytest.skip(f"arp256.csv not found at {arp256_path}")
        
        # Import spectrum
        y = np.array(pd.read_csv(arp256_path).flux)[100:1530]
        y = (y / y.max()) * 255  # Normalize to 255
        
        # Add random Gaussian noise (same as example)
        np.random.seed(12431)
        sig = 10  # Standard deviation of noise
        noise = np.random.normal(0, sig, y.shape)
        z = y + noise
        variance = sig ** 2
        
        # Apply FABADA for recovery
        y_recover = fabada(z, variance)
        
        # Validate results
        assert y_recover is not None
        assert y_recover.shape == y.shape
        
        # Check that denoising improves quality
        psnr_noisy = PSNR(z, y, L=255)
        psnr_recovered = PSNR(y_recover, y, L=255)
        
        assert psnr_recovered > psnr_noisy, f"FABADA should improve PSNR: {psnr_noisy:.2f} -> {psnr_recovered:.2f}"
        
        # Should achieve reasonable denoising (based on paper results)
        improvement = psnr_recovered - psnr_noisy
        assert improvement > 2.0, f"FABADA improvement should be significant: {improvement:.2f} dB"

    @pytest.mark.skipif(not HAS_CV2, reason="OpenCV not available")
    def test_image_example_workflow(self, examples_dir):
        """Test the complete image example workflow."""
        # Load data like in fabada_demo_image.py
        bubble_path = os.path.join(examples_dir, 'bubble.png')
        
        if not os.path.exists(bubble_path):
            pytest.skip(f"bubble.png not found at {bubble_path}")
        
        # Import image
        y = cv2.imread(bubble_path, 0)
        
        # Add random Gaussian noise (same as example)
        np.random.seed(12431)
        sig = 15  # Standard deviation of noise
        noise = np.random.normal(0, sig, y.shape)
        z = y + noise
        variance = sig ** 2
        
        # Apply FABADA for recovery
        y_recover = fabada(z, variance)
        
        # Validate results
        assert y_recover is not None
        assert y_recover.shape == y.shape
        
        # Check that denoising improves quality
        psnr_noisy = PSNR(z, y, L=255)
        psnr_recovered = PSNR(y_recover, y, L=255)
        
        assert psnr_recovered > psnr_noisy, f"FABADA should improve PSNR: {psnr_noisy:.2f} -> {psnr_recovered:.2f}"
        
        # Should achieve reasonable denoising for images
        improvement = psnr_recovered - psnr_noisy
        assert improvement > 1.0, f"FABADA improvement should be significant: {improvement:.2f} dB"

    def test_fabada_cmd_script_exists(self, examples_dir):
        """Test that the command line script exists and is readable."""
        cmd_script_path = os.path.join(examples_dir, 'fabadaCMD.py')
        
        assert os.path.exists(cmd_script_path), "fabadaCMD.py should exist in examples"
        assert os.path.isfile(cmd_script_path), "fabadaCMD.py should be a file"
        
        # Check if it's executable (can be read)
        with open(cmd_script_path, 'r') as f:
            content = f.read()
            assert 'fabada' in content.lower(), "Script should contain fabada functionality"
            assert 'argparse' in content, "Script should use argparse for CLI"

    def test_multiple_noise_levels_consistency(self, examples_dir):
        """Test FABADA performance across multiple noise levels for consistency."""
        # Load spectrum data
        arp256_path = os.path.join(examples_dir, 'arp256.csv')
        
        if not os.path.exists(arp256_path):
            pytest.skip(f"arp256.csv not found")
        
        y = np.array(pd.read_csv(arp256_path).flux)[100:1530]
        y = (y / y.max()) * 255
        
        noise_levels = [5, 10, 15, 20, 25]
        improvements = []
        
        for sig in noise_levels:
            np.random.seed(12431)  # Consistent seed
            noise = np.random.normal(0, sig, y.shape)
            z = y + noise
            variance = sig ** 2
            
            y_recover = fabada(z, variance)
            
            psnr_noisy = PSNR(z, y, L=255)
            psnr_recovered = PSNR(y_recover, y, L=255)
            improvement = psnr_recovered - psnr_noisy
            
            improvements.append(improvement)
            
            # Each level should show improvement
            assert improvement > 0, f"No improvement at noise level {sig}"
        
        # Generally, higher noise should show more improvement (with some tolerance)
        # This tests the algorithm's adaptive behavior
        assert improvements[-1] > improvements[0], "Higher noise should generally show more improvement"

    def test_fabada_convergence_stability(self, examples_dir):
        """Test that FABADA converges stably with different max_iter values."""
        arp256_path = os.path.join(examples_dir, 'arp256.csv')
        
        if not os.path.exists(arp256_path):
            pytest.skip(f"arp256.csv not found")
        
        y = np.array(pd.read_csv(arp256_path).flux)[100:1530]
        y = (y / y.max()) * 255
        
        # Add noise
        np.random.seed(12431)
        sig = 10
        noise = np.random.normal(0, sig, y.shape)
        z = y + noise
        variance = sig ** 2
        
        # Test with different iteration limits
        iter_values = [100, 500, 1000, 2000]
        results = []
        
        for max_iter in iter_values:
            y_recover = fabada(z.copy(), variance, max_iter=max_iter)
            psnr = PSNR(y_recover, y, L=255)
            results.append(psnr)
        
        # Results should be stable (not drastically different)
        # and generally improve or stay stable with more iterations
        for i in range(1, len(results)):
            # Each result should be reasonable
            assert results[i] > 0, f"Invalid PSNR at max_iter={iter_values[i]}"
            
            # Results shouldn't vary too wildly (algorithm should be stable)
            variation = abs(results[i] - results[i-1])
            assert variation < 5.0, f"Too much variation between iterations: {variation:.2f} dB"

    def test_data_types_compatibility(self, examples_dir):
        """Test FABADA works with different data types and shapes."""
        arp256_path = os.path.join(examples_dir, 'arp256.csv')
        
        if not os.path.exists(arp256_path):
            pytest.skip(f"arp256.csv not found")
        
        y = np.array(pd.read_csv(arp256_path).flux)[100:1530]
        y = (y / y.max()) * 255
        
        # Add noise
        np.random.seed(12431)
        sig = 10
        noise = np.random.normal(0, sig, y.shape)
        z = y + noise
        variance = sig ** 2
        
        # Test with different data types
        data_types = [np.float32, np.float64]
        
        for dtype in data_types:
            z_typed = z.astype(dtype)
            y_typed = y.astype(dtype)
            
            y_recover = fabada(z_typed, variance)
            
            assert y_recover is not None
            assert y_recover.shape == y.shape
            
            # Should still improve quality
            psnr_noisy = PSNR(z_typed, y_typed, L=255)
            psnr_recovered = PSNR(y_recover, y_typed, L=255)
            assert psnr_recovered > psnr_noisy

    def test_variance_specifications(self, examples_dir):
        """Test different ways of specifying variance."""
        arp256_path = os.path.join(examples_dir, 'arp256.csv')
        
        if not os.path.exists(arp256_path):
            pytest.skip(f"arp256.csv not found")
        
        y = np.array(pd.read_csv(arp256_path).flux)[100:300]  # Smaller for faster testing
        y = (y / y.max()) * 255
        
        # Add noise
        np.random.seed(12431)
        sig = 10
        noise = np.random.normal(0, sig, y.shape)
        z = y + noise
        
        # Test different variance specifications
        variance_specs = [
            sig ** 2,  # Float
            [sig ** 2] * len(y),  # List
            np.full(y.shape, sig ** 2),  # Array
        ]
        
        for variance in variance_specs:
            y_recover = fabada(z.copy(), variance)
            
            assert y_recover is not None
            assert y_recover.shape == y.shape
            
            # Should improve quality
            psnr_noisy = PSNR(z, y, L=255)
            psnr_recovered = PSNR(y_recover, y, L=255)
            assert psnr_recovered > psnr_noisy