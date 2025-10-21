# FABADA Tests

This directory contains comprehensive tests for the FABADA algorithm using the provided examples.

## Test Structure

- `test_spectrum.py`: Tests FABADA on 1D spectrum data using `arp256.csv`
- `test_image.py`: Tests FABADA on 2D image data using `bubble.png` 
- `test_integration.py`: Integration tests that run complete workflows from examples

## Running Tests

### Prerequisites

Install test dependencies:
```bash
pip install -e .[test]
```

Or install dependencies manually:
```bash
pip install pytest pandas opencv-python scikit-image matplotlib
```

### Run All Tests

```bash
# From the project root
pytest tests/

# With verbose output
pytest -v tests/

# Run specific test file
pytest tests/test_spectrum.py
pytest tests/test_image.py
pytest tests/test_integration.py
```

### Run Tests with Coverage

```bash
pip install pytest-cov
pytest --cov=fabada tests/
```

## Test Categories

### Spectrum Tests (`test_spectrum.py`)
- Basic FABADA functionality on 1D data
- PSNR improvement validation
- MSE reduction validation
- Parameter variation tests
- Different noise level tests
- Reproducibility tests

### Image Tests (`test_image.py`)
- Basic FABADA functionality on 2D data  
- PSNR and SSIM improvement validation
- 2D data consistency tests
- Different variance specification tests
- Edge case handling

### Integration Tests (`test_integration.py`)
- Complete example workflows
- Multi-noise level consistency
- Convergence stability tests
- Data type compatibility
- Command-line script validation

## Test Features

- **Automatic skipping**: Tests skip gracefully if optional dependencies (OpenCV, scikit-image) are not available
- **Reproducible**: Fixed random seeds for consistent test results
- **Parametric**: Tests run across multiple noise levels and parameters
- **Comprehensive**: Tests cover basic functionality, edge cases, and performance validation

## Expected Results

All tests validate that FABADA:
1. **Improves signal quality**: PSNR increases after denoising
2. **Reduces noise**: MSE decreases compared to noisy input
3. **Preserves structure**: SSIM improves for image data
4. **Handles various inputs**: Works with different data types and variance specifications
5. **Converges stably**: Results are consistent across parameter variations

## Troubleshooting

If tests fail:

1. **Import errors**: Install missing dependencies with `pip install -e .[test]`
2. **File not found**: Ensure example files (`arp256.csv`, `bubble.png`) exist in `examples/`
3. **Performance issues**: Some tests may be slow on large datasets; consider using smaller subsets for debugging
4. **Numerical precision**: Small floating-point differences are normal; tests use appropriate tolerances