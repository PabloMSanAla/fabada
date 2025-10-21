"""
Basic smoke test to verify FABADA installation and imports work correctly.
"""

import os
import sys
import numpy as np

# Add the parent directory to the path so we can import fabada
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_fabada_import():
    """Test that FABADA can be imported successfully."""
    try:
        from fabada import fabada, PSNR
        assert fabada is not None
        assert PSNR is not None
    except ImportError as e:
        assert False, f"Failed to import FABADA: {e}"


def test_fabada_basic_functionality():
    """Test basic FABADA functionality with synthetic data."""
    from fabada import fabada, PSNR
    
    # Create simple synthetic signal
    x = np.linspace(0, 4*np.pi, 100)
    y_true = np.sin(x) + 0.5 * np.sin(3*x)
    y_true = (y_true - y_true.min()) / (y_true.max() - y_true.min()) * 255  # Normalize to [0, 255]
    
    # Add noise
    np.random.seed(42)
    noise_std = 10
    noise = np.random.normal(0, noise_std, y_true.shape)
    y_noisy = y_true + noise
    
    # Apply FABADA
    y_recovered = fabada(y_noisy, noise_std**2)
    
    # Basic validation
    assert y_recovered is not None, "FABADA returned None"
    assert y_recovered.shape == y_true.shape, "Shape mismatch"
    assert not np.any(np.isnan(y_recovered)), "Result contains NaN"
    assert not np.any(np.isinf(y_recovered)), "Result contains infinite values"
    
    # Quality improvement check
    psnr_noisy = PSNR(y_noisy, y_true, L=255)
    psnr_recovered = PSNR(y_recovered, y_true, L=255)
    
    assert psnr_recovered > psnr_noisy, f"FABADA should improve PSNR: {psnr_noisy:.2f} -> {psnr_recovered:.2f}"


def test_examples_directory_exists():
    """Test that examples directory and files exist."""
    examples_dir = os.path.join(os.path.dirname(__file__), '..', 'examples')
    
    assert os.path.exists(examples_dir), "Examples directory does not exist"
    assert os.path.isdir(examples_dir), "Examples path is not a directory"
    
    # Check for key example files
    expected_files = [
        'fabada_demo_spectrum.py',
        'fabada_demo_image.py', 
        'fabadaCMD.py',
        'arp256.csv'
    ]
    
    for filename in expected_files:
        filepath = os.path.join(examples_dir, filename)
        assert os.path.exists(filepath), f"Expected example file not found: {filename}"


if __name__ == '__main__':
    # Run tests directly if script is executed
    print("Running basic FABADA smoke tests...")
    
    try:
        test_fabada_import()
        print("✓ Import test passed")
        
        test_fabada_basic_functionality() 
        print("✓ Basic functionality test passed")
        
        test_examples_directory_exists()
        print("✓ Examples directory test passed")
        
        print("\n✓ All smoke tests passed! FABADA is working correctly.")
        
    except Exception as e:
        print(f"\n✗ Smoke test failed: {e}")
        sys.exit(1)