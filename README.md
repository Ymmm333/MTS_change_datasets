# Mind the Gap: Open Set Domain Adaptation via Mutual-to-Separate Framework

Code release for Mind the Gap: Open Set Domain Adaptation via Mutual-to-Separate Framework [TCSVT 2023]

## Dataset
### OFFICE-Home

## Requirements 

- python 3.8+
- PyTorch 1.1.0+
- torchvision 0.3.0+
- scipy
- numpy
- PIL/Pillow
- scikit-learn

### Optional (for TensorBoard logging):
- TensorFlow 1.9.0+ (or 2.x)
- Tensorlayer 1.11+
- Tensorpack

**Note**: TensorFlow is disabled by default to avoid GPU memory conflicts. To enable TensorBoard logging:
```bash
USE_TENSORFLOW=1 python Office-Home.py ...
```

## GPU Version

- 1080ti

## Training

- Download datasets
- Train: `python Office-Home.py PaviaU_7gt PaviaC_OS 0 experiment_name 1.0 1.0`
- Description: PyTorch Open-set domain adaptation training with ResNet50 (PRE-TRAINED WITH IMAGENET).

### Default behavior (TensorFlow disabled):
```bash
$ python Office-Home.py PaviaU_7gt PaviaC_OS 0 experiment_name 1.0 1.0
✓ TensorFlow disabled (logging disabled). Set USE_TENSORFLOW=1 to enable.
gpu(s) to be used: 0
domain_train PaviaU_7gt
domain_test PaviaC_OS
Logger initialized at experiment_name/step_2 (TensorBoard logging disabled)
# Training proceeds normally without CUDA OOM errors
```

### Optional: Enable TensorFlow for logging:
```bash
$ USE_TENSORFLOW=1 python Office-Home.py PaviaU_7gt PaviaC_OS 0 experiment_name 1.0 1.0
✓ TensorFlow enabled for TensorBoard logging
# TensorBoard logging will work
```


## Reference codes
**https://github.com/thuml/easydl**


## Citation
If you find this paper useful in your research, please consider citing:
```
@ARTICLE{chang_Mind, 
author={Dongliang Chang, Aneeshan Sain, Zhanyu Ma, Yi-Zhe Song, Ruiping Wang, and Jun Guo}, 
journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
title={Mind the Gap: Open Set Domain Adaptation via Mutual-to-Separate Framework}, 
year={2023},
doi={10.1109/TCSVT.2023.3326862}} 
```

## Contact
- changdongliang@bupt.edu.cn
- mazhanyu@bupt.edu.cn
