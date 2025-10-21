# GitHub Actions CI/CD for FABADA

This repository uses GitHub Actions for Continuous Integration and Continuous Deployment to automatically test the FABADA algorithm.

## Workflows

### 🚀 Quick Tests (`quick-test.yml`)
- **Trigger**: Every push and pull request to `master`/`main`
- **Duration**: ~3-5 minutes
- **Purpose**: Fast validation of core functionality
- **Tests**:
  - Smoke tests (import verification, basic functionality)
  - Core spectrum denoising tests
  - Integration workflow test

### 🧪 Comprehensive Tests (`test.yml`)
- **Trigger**: Push to `master`/`main`/`develop`, pull requests
- **Duration**: ~15-25 minutes
- **Purpose**: Full test matrix across multiple environments
- **Matrix**:
  - OS: Ubuntu, Windows, macOS
  - Python: 3.8, 3.9, 3.10, 3.11, 3.12
- **Tests**:
  - All test categories (smoke, spectrum, image, integration)
  - Coverage reporting with Codecov
  - Performance benchmarks
  - Example script validation
  - Minimal dependency testing

### 🏷️ Release Tests (`release.yml`)
- **Trigger**: Tag pushes (`v*`) and releases
- **Purpose**: Comprehensive validation before release
- **Tests**:
  - Full test suite across all platforms
  - Package installation validation
  - Real-world usage scenarios

## Test Categories

### Core Tests (Always Run)
```bash
# Smoke tests - basic functionality
python tests/test_smoke.py

# Spectrum tests - 1D denoising validation
pytest tests/test_spectrum.py

# Integration tests - workflow validation  
pytest tests/test_integration.py
```

### Optional Tests (Skip if dependencies missing)
```bash
# Image tests - 2D denoising (requires OpenCV)
pytest tests/test_image.py

# Example execution tests (requires matplotlib)
python examples/fabada_demo_spectrum.py
```

## Badges

Add these badges to your README to show test status:

```markdown
[![Tests](https://github.com/PabloMSanAla/fabada/actions/workflows/test.yml/badge.svg)](https://github.com/PabloMSanAla/fabada/actions/workflows/test.yml)
[![Quick Tests](https://github.com/PabloMSanAla/fabada/actions/workflows/quick-test.yml/badge.svg)](https://github.com/PabloMSanAla/fabada/actions/workflows/quick-test.yml)
```

## Local Testing vs CI

### Local Development
```bash
# Quick validation during development
python tests/test_smoke.py

# Run specific tests
pytest tests/test_spectrum.py -v
```

### GitHub Actions Features

1. **Matrix Testing**: Tests across multiple Python versions and OS
2. **Dependency Caching**: Faster builds with pip cache
3. **Graceful Degradation**: Optional dependencies don't fail the build
4. **Performance Monitoring**: Execution time and improvement tracking
5. **Coverage Reporting**: Code coverage analysis with Codecov

## Troubleshooting CI

### Common Issues

1. **Test Timeouts**: Some FABADA convergence tests may take 60+ seconds
2. **Optional Dependencies**: Image tests skip when OpenCV unavailable
3. **Platform Differences**: Minor numerical differences across OS/Python versions
4. **Memory Usage**: Large test matrices may hit memory limits

### Solutions

- Tests use `continue-on-error: true` for optional components
- Timeouts set to reasonable limits (120s for spectrum tests)
- Platform-specific dependency installation
- Numerical tolerances account for floating-point precision

## Status Monitoring

- ✅ **Green**: All tests passing - safe to merge/release
- ⚠️ **Yellow**: Some optional tests failed - core functionality OK
- ❌ **Red**: Core tests failing - requires investigation

The CI ensures FABADA maintains its core promise: **consistently improving signal quality through Bayesian noise reduction** across all supported environments.