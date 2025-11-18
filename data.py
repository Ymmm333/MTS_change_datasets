import os
import numpy as np
import time
import random
import numbers
from PIL import Image
from six.moves import cPickle

# Try to import tensorpack and tensorlayer (optional)
try:
    import tensorpack
    import tensorlayer as tl
    TENSORPACK_AVAILABLE = True
except ImportError:
    TENSORPACK_AVAILABLE = False
    print("⚠ TensorPack/TensorLayer not available. Using simple data loading.")
    tensorpack = None
    tl = None

# Import utilities AFTER trying tensorpack
from utilities import *

import warnings
warnings.filterwarnings('ignore', message='.*', category=Warning)

class CustomDataLoader(object):
    def __init__(self, dataset, batch_size, num_threads=8,remainder=None):
        if not TENSORPACK_AVAILABLE:
            raise ImportError("tensorpack is required for CustomDataLoader but is not available")
        
        self.ds0 = dataset
        self.batch_size = batch_size
        self.num_threads = num_threads
        
        if not remainder:
            try:
                is_train = self.ds0.is_train
                remainder = False if is_train else True # if is_train, there is no need to set reminder 
            except Exception as e:
                # self.ds0 maybe doesn't have is_train attribute, then it has no test mode, set remainder = False
                remainder = False
        
        # use_list=False, for each in data point, add a batch dimension (return in numpy array)
        self.ds1 = tensorpack.dataflow.BatchData(self.ds0, self.batch_size,remainder=remainder, use_list=False,) 
        
        # use 1 thread in test to avoid randomness (test should be deterministic)
        self.ds2 = tensorpack.dataflow.PrefetchDataZMQ(self.ds1, nr_proc=self.num_threads if not remainder else 1)
        
        # required by tensorlayer package
        self.ds2.reset_state()
    
    def generator(self):
        return self.ds2.get_data()

# Create base class conditionally
if TENSORPACK_AVAILABLE:
    _BaseDatasetParent = tensorpack.dataflow.RNGDataFlow
else:
    # Simple fallback parent class when tensorpack is not available
    class _BaseDatasetParent:
        pass

class BaseDataset(_BaseDatasetParent):
    def __init__(self, is_train=True, skip_pred=None, transform=None, sample_weight=None):
        self.is_train = is_train
        self.skip_pred = skip_pred or (lambda data, label, is_train : False)
        self.transform = transform or (lambda data, label, is_train : (data, label))
        self.sample_weight = sample_weight or (lambda data, label : 1.0)

        self.datas = []
        self.labels = []

        self._fill_data()

        self._post_init()

    def _fill_data(self):
        raise NotImplementedError("not implemented!")

    def _post_init(self):
        tmp = [[data, label]  for (data, label) in zip(self.datas, self.labels) if not self.skip_pred(data, label, self.is_train) ]
        self.datas = [x[0] for x in tmp]
        self.labels = [x[1] for x in tmp]

        if callable(self.sample_weight):
            self._weight = [self.sample_weight(x, y) for (x, y) in zip(self.datas, self.labels)]
        else:
            self._weight = self.sample_weight
        self._weight = np.asarray(self._weight, dtype=np.float32).reshape(-1)
        assert len(self._weight) == len(self.datas), 'dimension not match!'
        self._weight = self._weight / np.sum(self._weight)

    def size(self):
        return len(self.datas)

    def _get_one_data(self, data, label):
        raise NotImplementedError("not implemented!")

    def get_data(self):
        size = self.size()
        ids = list(range(size))
        for _ in range(size):
            id = np.random.choice(ids, p=self._weight) if self.is_train else _
            data, label = self._get_one_data(self.datas[id], self.labels[id])
            data, label = self.transform(data, label, self.is_train)
            yield np.asarray(data), np.asarray([label]) if isinstance(label, numbers.Number) else label

    
class BaseImageDataset(BaseDataset):
    def __init__(self, imsize=224, is_train=True, skip_pred=None, transform=None, sample_weight=None):
        self.imsize = imsize
        super(BaseImageDataset, self).__init__(is_train, skip_pred, transform, sample_weight=sample_weight)

    def _get_one_data(self, data, label):
        im = np.array(Image.open(data).convert('RGB'))
        if self.imsize:
            im = np.array(Image.fromarray(im).resize((self.imsize, self.imsize), Image.BILINEAR))
        return im, label


def one_hot(n_class, index):
    tmp = np.zeros((n_class,), dtype=np.float32)
    tmp[index] = 1.0
    return tmp

class FileListDataset(BaseImageDataset):
    
    def __init__(self, list_path, path_prefix='', imsize=224, is_train=True, skip_pred=None, transform=None, sample_weight=None):
        self.list_path = list_path
        self.path_prefix = path_prefix

        super(FileListDataset, self).__init__(imsize=imsize, is_train=is_train, skip_pred=skip_pred, transform=transform, sample_weight=sample_weight)

    def _fill_data(self):
        with open(self.list_path, 'r') as f:
            data = [[line.split()[0], line.split()[1]] for line in f.readlines() if line.strip()] # avoid empty lines
            self.datas = [os.path.join(self.path_prefix, x[0]) for x in data]
            try:
                self.labels = [int(x[1]) for x in data]
            except ValueError as e:
                print('invalid label number, maybe there is space in image path?')
                raise e


# 在 data.py 中添加超光谱数据集类
class HyperspectralDataset(BaseImageDataset):
    def __init__(self, config, is_train=True, skip_pred=None, transform=None, sample_weight=None):
        self.config = config
        self.data, self.gt = self.load_data()
        self.patches, self.labels = self.extract_patches()
        super(HyperspectralDataset, self).__init__(imsize=224, is_train=is_train,
                                                   skip_pred=skip_pred, transform=transform,
                                                   sample_weight=sample_weight)

    def load_data(self):
        import scipy.io as sio
        data_path = f"/home/b311/data3/liuhui/yimingwang/CompairExperiment/MTS/{self.config['path']}/{self.config['file_name']}"
        gt_path = f"/home/b311/data3/liuhui/yimingwang/CompairExperiment/MTS/{self.config['path']}/{self.config['gt_file_name']}"

        if self.config['type'] == 'mat':
            data = sio.loadmat(data_path)[self.config['mat_name']]
            gt = sio.loadmat(gt_path)[self.config['gt_mat_name']]
        else:  # npy
            data = np.load(data_path)
            gt = np.load(gt_path)

        if self.config.get('norm', False):
            data = (data - np.min(data)) / (np.max(data) - np.min(data))

        return data, gt

    def extract_patches(self, patch_size=15):
        H, W, C = self.data.shape
        half = patch_size // 2
        patches = []
        labels = []

        for i in range(half, H - half):
            for j in range(half, W - half):
                if self.gt[i, j] != 0:
                    patch = self.data[i - half:i + half + 1, j - half:j + half + 1, :]
                    patches.append(patch)
                    labels.append(self.gt[i, j])

        return patches, labels

    def _fill_data(self):
        self.datas = self.patches
        self.labels = self.labels

    def _get_one_data(self, data, label):
        # 这里直接返回patch和标签，transform会处理后续的PCA和resize
        return data, label
