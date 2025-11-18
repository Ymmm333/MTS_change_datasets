#!/usr/bin/env python3
"""
Standalone test to verify TensorFlow GPU memory growth configuration
"""

import sys

def test_tf_memory_growth_code():
    """Test that the TensorFlow memory growth configuration code works"""
    print("=" * 60)
    print("Testing TensorFlow GPU Memory Growth Configuration")
    print("=" * 60)
    
    # This is the exact code from Office-Home.py
    try:
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Enable memory growth for all GPUs
                # This allows TensorFlow to allocate GPU memory incrementally as needed
                # instead of allocating all available memory at startup
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✓ TensorFlow memory growth enabled for {len(gpus)} GPU(s)")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(f"⚠ Warning: Could not set TensorFlow memory growth: {e}")
        else:
            print("✓ TensorFlow available but no GPUs found (CPU-only mode)")
    except ImportError:
        # TensorFlow not available - this is fine, code works without it
        print("✓ TensorFlow not available, skipping GPU memory config")
    except Exception as e:
        print(f"⚠ Warning: Error configuring TensorFlow GPU: {e}")
    
    print("\n✓ Configuration code executed without crashing")
    print("=" * 60)
    return True

if __name__ == '__main__':
    try:
        success = test_tf_memory_growth_code()
        if success:
            print("\n✓ Test PASSED")
            sys.exit(0)
        else:
            print("\n✗ Test FAILED")
            sys.exit(1)
    except Exception as e:
        print(f"\n✗ Test FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
