"""
Tests for FABADA image denoising using the bubble.png example data.
"""

import os
import sys
import numpy as np
import pytest

# Add the parent directory to the path so we can import fabada
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fabada import fabada, PSNR

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


class TestImageDenoising:
    """Test class for image denoising using FABADA."""

    @pytest.fixture
    def image_data(self):
        """Load and prepare the bubble.png image data."""
        if not HAS_CV2:
            pytest.skip("OpenCV (cv2) not available")
            
        # Get the path to the examples directory
        examples_path = os.path.join(os.path.dirname(__file__), '..', 'examples')
        bubble_path = os.path.join(examples_path, 'bubble.png')
        
        if not os.path.exists(bubble_path):
            pytest.skip(f"bubble.png not found at {bubble_path}")
        
        # Load the image
        y = cv2.imread(bubble_path, 0)  # Load as grayscale
        return y

    @pytest.fixture
    def noisy_image_data(self, image_data):
        """Create noisy version of image data with different noise levels."""
        np.random.seed(12431)  # Fixed seed for reproducible tests
        
        noise_levels = [5, 10, 15, 20]
        noisy_data = {}
        
        for sig in noise_levels:
            noise = np.random.normal(0, sig, image_data.shape)
            z = image_data + noise
            variance = sig ** 2
            noisy_data[sig] = {'data': z, 'variance': variance, 'original': image_data}
            
        return noisy_data

    @pytest.mark.skipif(not HAS_CV2, reason="OpenCV not available")
    def test_fabada_image_basic(self, image_data):
        """Test basic FABADA functionality on image data."""
        # Add noise
        np.random.seed(12431)
        sig = 15
        noise = np.random.normal(0, sig, image_data.shape)
        noisy_data = image_data + noise
        variance = sig ** 2
        
        # Apply FABADA
        recovered = fabada(noisy_data, variance)
        
        # Basic checks
        assert recovered is not None
        assert recovered.shape == image_data.shape
        assert isinstance(recovered, np.ndarray)
        assert not np.any(np.isnan(recovered))
        assert not np.any(np.isinf(recovered))

    @pytest.mark.skipif(not HAS_CV2, reason="OpenCV not available")
    def test_fabada_improves_psnr_image(self, noisy_image_data):
        """Test that FABADA improves PSNR for different noise levels on images."""
        for noise_level, data_dict in noisy_image_data.items():
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

    @pytest.mark.skipif(not HAS_CV2, reason="OpenCV not available")
    def test_fabada_reduces_mse_image(self, noisy_image_data):
        """Test that FABADA reduces Mean Squared Error for images."""
        for noise_level, data_dict in noisy_image_data.items():
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

    @pytest.mark.skipif(not (HAS_CV2 and HAS_SKIMAGE), reason="OpenCV or scikit-image not available")
    def test_fabada_improves_ssim(self, noisy_image_data):
        """Test that FABADA improves SSIM (Structural Similarity Index) for images."""
        for noise_level, data_dict in noisy_image_data.items():
            noisy_data = data_dict['data']
            variance = data_dict['variance']
            original = data_dict['original']
            
            # Calculate original SSIM
            ssim_noisy = ssim(noisy_data, original, data_range=255)
            
            # Apply FABADA
            recovered = fabada(noisy_data, variance)
            
            # Calculate recovered SSIM
            ssim_recovered = ssim(recovered, original, data_range=255)
            
            # FABADA should improve SSIM
            assert ssim_recovered > ssim_noisy, f"FABADA did not improve SSIM for noise level {noise_level}. Original: {ssim_noisy:.3f}, Recovered: {ssim_recovered:.3f}"

    @pytest.mark.skipif(not HAS_CV2, reason="OpenCV not available")
    def test_fabada_preserves_image_range(self, noisy_image_data):
        """Test that FABADA preserves reasonable image pixel range."""
        for noise_level, data_dict in noisy_image_data.items():
            noisy_data = data_dict['data']
            variance = data_dict['variance']
            original = data_dict['original']
            
            # Apply FABADA
            recovered = fabada(noisy_data, variance)
            
            # Check that recovered image is within reasonable bounds
            # Images should be in [0, 255] range approximately
            assert np.min(recovered) >= -50, f"Recovered image has pixels too low for noise level {noise_level}"
            assert np.max(recovered) <= 305, f"Recovered image has pixels too high for noise level {noise_level}"

    @pytest.mark.skipif(not HAS_CV2, reason="OpenCV not available")
    def test_fabada_2d_consistency(self, image_data):
        """Test that FABADA works consistently on 2D image data."""
        # Add noise
        np.random.seed(12431)
        sig = 15
        noise = np.random.normal(0, sig, image_data.shape)
        noisy_data = image_data + noise
        variance = sig ** 2
        
        # Apply FABADA
        recovered = fabada(noisy_data, variance)
        
        # Test that the algorithm handles 2D data properly
        assert len(recovered.shape) == 2, "Recovered data should be 2D"
        assert recovered.shape == image_data.shape, "Shape should be preserved"
        
        # Test improvement
        psnr_noisy = PSNR(noisy_data, image_data, L=255)
        psnr_recovered = PSNR(recovered, image_data, L=255)
        assert psnr_recovered > psnr_noisy, "FABADA should improve image quality"

    @pytest.mark.skipif(not HAS_CV2, reason="OpenCV not available")
    def test_fabada_image_with_different_variances(self, image_data):
        """Test FABADA with different variance specifications."""
        # Add noise
        np.random.seed(12431)
        sig = 10
        noise = np.random.normal(0, sig, image_data.shape)
        noisy_data = image_data + noise
        
        # Test with uniform variance (float)
        variance_uniform = sig ** 2
        recovered_uniform = fabada(noisy_data, variance_uniform)
        
        # Test with array variance (same value everywhere)
        variance_array = np.full(image_data.shape, sig ** 2)
        recovered_array = fabada(noisy_data, variance_array)
        
        # Both should work and give similar results
        assert recovered_uniform is not None
        assert recovered_array is not None
        
        # Results should be very similar
        np.testing.assert_array_almost_equal(recovered_uniform, recovered_array, decimal=5)

    @pytest.mark.skipif(not HAS_CV2, reason="OpenCV not available")
    @pytest.mark.parametrize("noise_level", [5, 10, 15, 20])
    def test_fabada_image_performance_scaling(self, image_data, noise_level):
        """Parametric test for FABADA performance on images with different noise levels."""
        # Add noise
        np.random.seed(12431)
        noise = np.random.normal(0, noise_level, image_data.shape)
        noisy_data = image_data + noise
        variance = noise_level ** 2
        
        # Apply FABADA
        recovered = fabada(noisy_data, variance)
        
        # Calculate improvements
        psnr_noisy = PSNR(noisy_data, image_data, L=255)
        psnr_recovered = PSNR(recovered, image_data, L=255)
        improvement = psnr_recovered - psnr_noisy
        
        # FABADA should always improve PSNR
        assert improvement > 0, f"FABADA did not improve PSNR for noise level {noise_level}"
        
        # For higher noise levels, expect reasonable improvement
        if noise_level >= 15:
            assert improvement > 0.5, f"FABADA improvement too small for noise level {noise_level}: {improvement:.2f} dB"

    @pytest.mark.skipif(not HAS_CV2, reason="OpenCV not available")  
    def test_fabada_handles_edge_cases(self, image_data):
        """Test FABADA behavior with edge cases."""
        # Test with very small image
        small_image = image_data[:10, :10]
        np.random.seed(12431)
        noise = np.random.normal(0, 10, small_image.shape)
        noisy_small = small_image + noise
        
        recovered_small = fabada(noisy_small, 100)  # variance = 10^2
        assert recovered_small.shape == small_image.shape
        
        # Test with very low noise
        low_noise = np.random.normal(0, 1, image_data.shape)
        barely_noisy = image_data + low_noise
        
        recovered_low_noise = fabada(barely_noisy, 1)  # variance = 1^2
        assert recovered_low_noise is not None
        assert recovered_low_noise.shape == image_data.shape