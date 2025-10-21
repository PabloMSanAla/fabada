#!/usr/bin/env python3
"""
Simple test runner for FABADA tests.
Run this from the project root directory.
"""

import sys
import os
import subprocess

def main():
    """Run FABADA tests with proper setup."""
    # Change to project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
    
    print("FABADA Test Runner")
    print("=" * 50)
    print(f"Project root: {project_root}")
    
    # Check if pytest is available
    try:
        import pytest
        print("✓ pytest is available")
    except ImportError:
        print("✗ pytest not found. Install with: pip install pytest")
        sys.exit(1)
    
    # Check for required example files
    examples_dir = os.path.join(project_root, 'examples')
    arp256_file = os.path.join(examples_dir, 'arp256.csv')
    bubble_file = os.path.join(examples_dir, 'bubble.png')
    
    if os.path.exists(arp256_file):
        print("✓ arp256.csv found")
    else:
        print("✗ arp256.csv not found in examples/")
    
    if os.path.exists(bubble_file):
        print("✓ bubble.png found")
    else:
        print("✗ bubble.png not found in examples/")
    
    # Check optional dependencies
    optional_deps = {
        'pandas': 'pandas',
        'cv2': 'opencv-python', 
        'skimage': 'scikit-image',
        'matplotlib': 'matplotlib'
    }
    
    for module, package in optional_deps.items():
        try:
            __import__(module)
            print(f"✓ {package} available")
        except ImportError:
            print(f"⚠ {package} not available - some tests will be skipped")
    
    print("\nRunning tests...")
    print("-" * 50)
    
    # Run pytest with verbose output
    cmd = [sys.executable, '-m', 'pytest', 'tests/', '-v']
    
    # Add any command line arguments passed to this script
    if len(sys.argv) > 1:
        cmd.extend(sys.argv[1:])
    
    result = subprocess.run(cmd)
    
    print("-" * 50)
    if result.returncode == 0:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed.")
    
    return result.returncode

if __name__ == '__main__':
    sys.exit(main())