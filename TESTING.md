# Testing FABADA

This document describes how to run tests for the FABADA algorithm.

## Quick Test

Run the smoke test to verify basic functionality:

```bash
python3 tests/test_smoke.py
```

## Full Test Suite

### 1. Install Test Dependencies

```bash
# Install minimal test dependencies
pip3 install pytest pandas

# Install all optional dependencies for full testing
pip3 install pytest pandas opencv-python scikit-image matplotlib
```

### 2. Run Tests

```bash
# Run all tests (may take several minutes)
/Users/pablom.sanchezalarcon/Library/Python/3.9/bin/pytest tests/ -v

# Run specific test categories
/Users/pablom.sanchezalarcon/Library/Python/3.9/bin/pytest tests/test_spectrum.py -v
/Users/pablom.sanchezalarcon/Library/Python/3.9/bin/pytest tests/test_integration.py -v

# Run smoke tests only
/Users/pablom.sanchezalarcon/Library/Python/3.9/bin/pytest tests/test_smoke.py -v
```

### 3. Using the Test Runner

```bash
python3 run_tests.py
```

## Test Coverage

The test suite includes:

1. **Smoke Tests** (`test_smoke.py`): Basic functionality verification
2. **Spectrum Tests** (`test_spectrum.py`): 1D signal denoising tests using `arp256.csv`
3. **Image Tests** (`test_image.py`): 2D image denoising tests using `bubble.png` (requires OpenCV)
4. **Integration Tests** (`test_integration.py`): Complete workflow tests

### Key Test Features

- ✅ **PSNR Improvement**: Validates that FABADA improves Peak Signal-to-Noise Ratio
- ✅ **MSE Reduction**: Confirms Mean Squared Error decreases after denoising
- ✅ **Multiple Noise Levels**: Tests performance across different noise intensities
- ✅ **Reproducibility**: Ensures consistent results with fixed random seeds
- ✅ **Edge Cases**: Handles various data types and parameter combinations
- ✅ **Example Workflows**: Replicates the example scripts' functionality

### Expected Results

All tests validate that FABADA:
- Improves signal quality (higher PSNR)
- Reduces noise (lower MSE)  
- Works with both 1D and 2D data
- Handles different variance specifications
- Produces stable, reproducible results

## Dependencies for Testing

### Required
- `numpy` (≥1.22.4)
- `scipy` (≥1.0.0)
- `pandas` (for reading example data)
- `pytest` (≥6.0)

### Optional (for full test coverage)
- `opencv-python` (for image tests)
- `scikit-image` (for SSIM metrics)
- `matplotlib` (for visualization tests)

Tests will automatically skip when optional dependencies are missing.

## Troubleshooting

### Common Issues

1. **Import Errors**: Install missing packages with `pip3 install <package>`
2. **File Not Found**: Ensure example files exist in `examples/` directory
3. **Slow Tests**: Some tests may take 30+ seconds for convergence
4. **Path Issues**: Use the full pytest path if `pytest` command not found

### Performance Notes

- Spectrum tests: ~30-60 seconds each (FABADA convergence time)
- Image tests: ~60-120 seconds each (larger 2D data)  
- Integration tests: ~10-30 seconds each (smaller datasets)

Total test runtime: ~5-15 minutes depending on system and optional dependencies.