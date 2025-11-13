# Python 3.8+ Compatibility Guide

## Overview

This repository has been updated to be compatible with Python 3.8+ by making TensorFlow 1.x optional and replacing deprecated scipy functions with PIL/Pillow.

## Changes Made

### 1. Optional TensorFlow/TensorLayer Dependencies

**File: `utilities.py`**

- TensorFlow and TensorLayer imports are now wrapped in try-except blocks
- Added `TF_AVAILABLE` flag to check if TensorFlow is installed
- Created dummy `tl.files` class for basic file operations when TensorLayer is unavailable
- Updated `Logger` class to gracefully handle missing TensorFlow:
  - TensorBoard logging is disabled with warning messages
  - All logging methods (`log_scalar`, `log_histogram`, `log_bar`) check the flag before attempting to use TensorFlow
  - Training continues normally without TensorBoard functionality

### 2. Replace scipy.misc with PIL/Pillow

**File: `data.py`**

- Replaced `from scipy.misc import imread, imresize` with `from PIL import Image`
- Updated `BaseImageDataset._get_one_data()` method:
  - `imread(data, mode='RGB')` → `np.array(Image.open(data).convert('RGB'))`
  - `imresize(im, (size, size))` → `np.array(Image.fromarray(im).resize((size, size), Image.BILINEAR))`
- Made tensorpack/tensorlayer imports optional with fallback parent class

**File: `Office-Home.py`**

- Replaced `from scipy.misc import imresize` with `from PIL import Image`
- Updated `HyperspectralDataLoader.transform()` method:
  - `imresize(patch_rgb, (224, 224))` → `np.array(Image.fromarray((patch_rgb * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)).astype(np.float32) / 255.0`
- Fixed dataset config path to `data/dataset_config.json`

### 3. Dataset Configuration Updates

**File: `data/dataset_config.json`**

Added `known_classes` and `unknown_classes` fields to all datasets:
- HyRank_source
- HyRank_target
- Yancheng_ZY
- Yancheng_GF
- Pavia_source
- Pavia_target

## Dependencies

### Required
- Python 3.8+
- numpy
- torch
- torchvision
- Pillow (PIL)
- scikit-learn
- scipy (for sio.loadmat only, not for image functions)

### Optional (for enhanced features)
- TensorFlow 1.x or 2.x (for TensorBoard logging)
- TensorLayer (for TensorBoard logging)
- tensorpack (for advanced data loading)

## Installation

```bash
# Install required dependencies
pip install numpy torch torchvision pillow scikit-learn scipy

# Optional: Install TensorFlow for TensorBoard logging
# pip install tensorflow tensorlayer tensorpack
```

## Running the Code

The code now works without TensorFlow installed:

```bash
python Office-Home.py PaviaU_7gt PaviaC_OS 0 experiment_name 1.0 1.0
```

**Expected behavior without TensorFlow:**
- Warnings will be printed about TensorFlow not being available
- TensorBoard logging will be disabled
- Training will proceed normally
- All core functionality remains intact

**With TensorFlow installed:**
- All features work as before
- TensorBoard logging is enabled
- No warnings about missing dependencies

## Testing

Run the compatibility test suite:

```bash
python test_compatibility.py
```

This verifies:
- utilities module works without TensorFlow
- Logger class handles missing TensorFlow gracefully
- PIL-based image loading and resizing work correctly
- Office-Home.py imports work
- dataset_config.json has required fields

## Backward Compatibility

The changes maintain full backward compatibility:
- If TensorFlow is installed, it will be used automatically
- If TensorFlow is not installed, the code continues to work without it
- All existing functionality is preserved

## Migration Notes

If you were previously using TensorFlow 1.x:

1. **Option 1: Continue using TensorFlow** (with TensorBoard logging)
   ```bash
   pip install tensorflow==1.15.0 tensorlayer==1.11.0 tensorpack
   ```

2. **Option 2: Run without TensorFlow** (no TensorBoard logging)
   - Simply run the code as-is
   - Warnings will indicate TensorBoard is disabled
   - All training functionality works normally

## Known Limitations

When running without TensorFlow:
- TensorBoard logging is disabled (no visualization during training)
- Must use alternative methods for monitoring training progress (e.g., print statements)

## Support

For issues related to Python 3.8+ compatibility, please check:
1. Ensure all required dependencies are installed
2. Check Python version: `python --version` (should be 3.8+)
3. Review warning messages for missing optional dependencies
