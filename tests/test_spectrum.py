"""
Tests for FABADA spectrum denoising using the arp256.csv example data.
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

# Add the parent directory to the path so we can import fabada
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fabada import fabada, PSNR


class TestSpectrumDenoising:
    """Test class for spectrum denoising using FABADA."""

    @pytest.fixture
    def spectrum_data(self):
        """Load and prepare the arp256.csv spectrum data."""
        # Get the path to the examples directory
        examples_path = os.path.join(os.path.dirname(__file__), '..', 'examples')
        arp256_path = os.path.join(examples_path, 'arp256.csv')
        
        # Load the data
        y = np.array(pd.read_csv(arp256_path).flux)[100:1530]
        y = (y / y.max()) * 255  # Normalize to 255
        return y

    @pytest.fixture
    def noisy_spectrum_data(self, spectrum_data):
        """Create noisy version of spectrum data with different noise levels."""
        np.random.seed(12431)  # Fixed seed for reproducible tests
        
        noise_levels = [5, 10, 15, 20]
        noisy_data = {}
        
        for sig in noise_levels:
            noise = np.random.normal(0, sig, spectrum_data.shape)
            z = spectrum_data + noise
            variance = sig ** 2
            noisy_data[sig] = {'data': z, 'variance': variance, 'original': spectrum_data}
            
        return noisy_data

    def test_fabada_spectrum_basic(self, spectrum_data):
        """Test basic FABADA functionality on spectrum data."""
        # Add noise
        np.random.seed(12431)
        sig = 10
        noise = np.random.normal(0, sig, spectrum_data.shape)
        noisy_data = spectrum_data + noise
        variance = sig ** 2
        
        # Apply FABADA
        recovered = fabada(noisy_data, variance)
        
        # Basic checks
        assert recovered is not None
        assert recovered.shape == spectrum_data.shape
        assert isinstance(recovered, np.ndarray)
        assert not np.any(np.isnan(recovered))
        assert not np.any(np.isinf(recovered))

    def test_fabada_improves_psnr(self, noisy_spectrum_data):
        """Test that FABADA improves PSNR for different noise levels."""
        for noise_level, data_dict in noisy_spectrum_data.items():
            noisy_data = data_dict['data']
            variance = data_dict['variance']
            original = data_dict['original']
            
            # Calculate original PSNR
            psnr_noisy = PSNR(noisy_data, original, L=255)
            
            # Apply FABADA
            recovered = fabada(noisy_data, variance)
            
            # Calculate recovered PSNR
            psnr_recovered = PSNR(recovered, original, L=255)
            
            # FABADA should improve PSNR
            assert psnr_recovered > psnr_noisy, f"FABADA did not improve PSNR for noise level {noise_level}. Original: {psnr_noisy:.2f}, Recovered: {psnr_recovered:.2f}"

    def test_fabada_reduces_mse(self, noisy_spectrum_data):
        """Test that FABADA reduces Mean Squared Error."""
        for noise_level, data_dict in noisy_spectrum_data.items():
            noisy_data = data_dict['data']
            variance = data_dict['variance']
            original = data_dict['original']
            
            # Calculate original MSE
            mse_noisy = np.mean((noisy_data - original) ** 2)
            
            # Apply FABADA
            recovered = fabada(noisy_data, variance)
            
            # Calculate recovered MSE
            mse_recovered = np.mean((recovered - original) ** 2)
            
            # FABADA should reduce MSE
            assert mse_recovered < mse_noisy, f"FABADA did not reduce MSE for noise level {noise_level}. Original: {mse_noisy:.2f}, Recovered: {mse_recovered:.2f}"

    def test_fabada_preserves_signal_range(self, noisy_spectrum_data):
        """Test that FABADA preserves reasonable signal range."""
        for noise_level, data_dict in noisy_spectrum_data.items():
            noisy_data = data_dict['data']
            variance = data_dict['variance']
            original = data_dict['original']
            
            # Apply FABADA
            recovered = fabada(noisy_data, variance)
            
            # Check that recovered signal is within reasonable bounds
            # Should be close to the original signal range
            original_min, original_max = original.min(), original.max()
            recovered_min, recovered_max = recovered.min(), recovered.max()
            
            # Allow some tolerance but not too much deviation
            tolerance = 0.2 * (original_max - original_min)
            
            assert recovered_min >= original_min - tolerance, f"Recovered signal minimum too low for noise level {noise_level}"
            assert recovered_max <= original_max + tolerance, f"Recovered signal maximum too high for noise level {noise_level}"

    def test_fabada_with_different_parameters(self, spectrum_data):
        """Test FABADA with different parameter settings."""
        # Add noise
        np.random.seed(12431)
        sig = 10
        noise = np.random.normal(0, sig, spectrum_data.shape)
        noisy_data = spectrum_data + noise
        variance = sig ** 2
        
        # Test with different max_iter values
        for max_iter in [100, 500, 1000]:
            recovered = fabada(noisy_data, variance, max_iter=max_iter)
            assert recovered is not None
            assert recovered.shape == spectrum_data.shape
            
        # Test with verbose mode
        recovered_verbose = fabada(noisy_data, variance, verbose=True)
        assert recovered_verbose is not None
        assert recovered_verbose.shape == spectrum_data.shape

    def test_fabada_with_uniform_variance(self, spectrum_data):
        """Test FABADA with uniform variance (float) instead of array."""
        # Add noise
        np.random.seed(12431)
        sig = 10
        noise = np.random.normal(0, sig, spectrum_data.shape)
        noisy_data = spectrum_data + noise
        variance = sig ** 2  # Single float value
        
        # Apply FABADA
        recovered = fabada(noisy_data, variance)
        
        # Basic checks
        assert recovered is not None
        assert recovered.shape == spectrum_data.shape
        
        # Should improve PSNR
        psnr_noisy = PSNR(noisy_data, spectrum_data, L=255)
        psnr_recovered = PSNR(recovered, spectrum_data, L=255)
        assert psnr_recovered > psnr_noisy

    def test_fabada_reproducibility(self, spectrum_data):
        """Test that FABADA produces reproducible results."""
        # Add noise
        np.random.seed(12431)
        sig = 10
        noise = np.random.normal(0, sig, spectrum_data.shape)
        noisy_data = spectrum_data + noise
        variance = sig ** 2
        
        # Run FABADA twice with same parameters
        recovered1 = fabada(noisy_data.copy(), variance)
        recovered2 = fabada(noisy_data.copy(), variance)
        
        # Results should be identical (or very close due to floating point precision)
        np.testing.assert_array_almost_equal(recovered1, recovered2, decimal=10)

    @pytest.mark.parametrize("noise_level", [5, 10, 15, 20, 25])
    def test_fabada_performance_vs_noise(self, spectrum_data, noise_level):
        """Parametric test to check FABADA performance across different noise levels."""
        # Add noise
        np.random.seed(12431)
        noise = np.random.normal(0, noise_level, spectrum_data.shape)
        noisy_data = spectrum_data + noise
        variance = noise_level ** 2
        
        # Apply FABADA
        recovered = fabada(noisy_data, variance)
        
        # Calculate improvements
        psnr_noisy = PSNR(noisy_data, spectrum_data, L=255)
        psnr_recovered = PSNR(recovered, spectrum_data, L=255)
        improvement = psnr_recovered - psnr_noisy
        
        # FABADA should always improve PSNR, with larger improvements for higher noise
        assert improvement > 0, f"FABADA did not improve PSNR for noise level {noise_level}"
        
        # For higher noise levels, we should see more improvement (within reason)
        if noise_level >= 15:
            assert improvement > 1.0, f"FABADA improvement too small for high noise level {noise_level}: {improvement:.2f} dB"