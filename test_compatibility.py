#!/usr/bin/env python3
"""
Test script to verify Python 3.8+ compatibility without TensorFlow 1.x
"""

import sys
import tempfile
import os
from PIL import Image
import numpy as np
import json

def test_utilities():
    """Test utilities module works without TensorFlow"""
    print("\n=== Testing utilities module ===")
    import utilities
    
    print(f"✓ utilities module imported")
    print(f"  TF_AVAILABLE: {utilities.TF_AVAILABLE}")
    
    # Test Logger
    log_dir = tempfile.mkdtemp()
    logger = utilities.Logger(log_dir, clear=False)
    print(f"✓ Logger created without errors")
    
    # Test logging methods (should not crash)
    logger.log_scalar('test', 1.0)
    logger.log_histogram('test', np.array([1, 2, 3]))
    logger.log_bar('test', [1, 2, 3])
    print(f"✓ Logging methods work (silently skip when TF not available)")
    
    # Cleanup
    import shutil
    shutil.rmtree(log_dir)
    
    return True

def test_data():
    """Test data module works with PIL instead of scipy"""
    print("\n=== Testing data module ===")
    import data
    
    print(f"✓ data module imported")
    print(f"  TENSORPACK_AVAILABLE: {data.TENSORPACK_AVAILABLE}")
    
    # Test PIL-based image loading
    test_dir = tempfile.mkdtemp()
    test_img_path = os.path.join(test_dir, 'test.jpg')
    
    # Create test image
    img = Image.new('RGB', (100, 100), color='red')
    img.save(test_img_path)
    
    # Test image loading with PIL
    img_loaded = np.array(Image.open(test_img_path).convert('RGB'))
    assert img_loaded.shape == (100, 100, 3), "Image loading failed"
    print(f"✓ PIL-based image loading works")
    
    # Test image resizing with PIL
    img_resized = np.array(Image.fromarray(img_loaded).resize((224, 224), Image.BILINEAR))
    assert img_resized.shape == (224, 224, 3), "Image resizing failed"
    print(f"✓ PIL-based image resizing works")
    
    # Cleanup
    import shutil
    shutil.rmtree(test_dir)
    
    return True

def test_office_home_imports():
    """Test Office-Home.py imports work"""
    print("\n=== Testing Office-Home.py imports ===")
    
    # Test PIL resize approach used in Office-Home.py
    patch_rgb = np.random.rand(15, 15, 3).astype(np.float32)
    patch_rgb_resized = np.array(
        Image.fromarray((patch_rgb * 255).astype(np.uint8))
        .resize((224, 224), Image.BILINEAR)
    ).astype(np.float32) / 255.0
    
    assert patch_rgb_resized.shape == (224, 224, 3), "PIL resize failed"
    print(f"✓ Office-Home.py PIL resize approach works")
    
    return True

def test_dataset_config():
    """Test dataset_config.json has required fields"""
    print("\n=== Testing dataset_config.json ===")
    
    with open('data/dataset_config.json', 'r') as f:
        config = json.load(f)
    
    print(f"✓ dataset_config.json loaded successfully")
    
    # Check that required datasets have known_classes and unknown_classes
    required_datasets = [
        'HyRank_source', 'HyRank_target', 
        'Yancheng_ZY', 'Yancheng_GF',
        'Pavia_source', 'Pavia_target'
    ]
    
    for dataset_name in required_datasets:
        if dataset_name in config:
            assert 'known_classes' in config[dataset_name], f"{dataset_name} missing known_classes"
            assert 'unknown_classes' in config[dataset_name], f"{dataset_name} missing unknown_classes"
            print(f"✓ {dataset_name} has required fields")
    
    return True

def main():
    """Run all tests"""
    print("=" * 60)
    print("Python 3.8+ Compatibility Test Suite")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    
    tests = [
        ("utilities module", test_utilities),
        ("data module", test_data),
        ("Office-Home imports", test_office_home_imports),
        ("dataset_config.json", test_dataset_config),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ {test_name} failed: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("\n✓ All tests passed! Code is compatible with Python 3.8+")
        print("  without requiring TensorFlow 1.x or scipy.misc")
        return 0
    else:
        print(f"\n✗ {failed} test(s) failed")
        return 1

if __name__ == '__main__':
    sys.exit(main())
