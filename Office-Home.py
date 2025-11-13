# from data import *
# from utilities import *
# from networks import *
# import numpy as np
# import json
# import scipy.io as sio
# from sklearn.decomposition import PCA
# import os
# import tempfile

from data import *
from utilities import *
from networks import *
import numpy as np
import json
import scipy.io as sio
from sklearn.decomposition import PCA
import os
import tempfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from PIL import Image

def skip(data, label, is_train):
    return False


batch_size = 16

import sys

domain_train = sys.argv[1]
domain_test = sys.argv[2]
setGPU(sys.argv[3])
store_name = sys.argv[4]
loss_MSE_value = float(sys.argv[5])
three_domain_loss_value = float(sys.argv[6])

# 加载数据集配置
with open('dataset_config.json', 'r') as f:
    dataset_config = json.load(f)

log = Logger(store_name + '/step_2', clear=True)
print('domain_train', domain_train)
print('domain_test', domain_test)


class HyperspectralDataLoader:
    """超光谱数据加载器"""

    def __init__(self, config, is_train=True, batch_size=16, patch_size=15):
        self.config = config
        self.is_train = is_train
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.data, self.gt = self.load_data()
        self.patches, self.labels, self.positions = self.extract_patches()
        self.current_index = 0
        self.indices = np.arange(len(self.patches))
        if is_train:
            np.random.shuffle(self.indices)

    def load_data(self):
        """加载超光谱数据"""
        data_path = f"/home/username/data/{self.config['path']}/{self.config['file_name']}"
        gt_path = f"/home/username/data/{self.config['path']}/{self.config['gt_file_name']}"

        print(f"Loading data from: {data_path}")
        print(f"Loading GT from: {gt_path}")

        if self.config['type'] == 'mat':
            data = sio.loadmat(data_path)[self.config['mat_name']]
            gt = sio.loadmat(gt_path)[self.config['gt_mat_name']]
        else:  # npy
            data = np.load(data_path)
            gt = np.load(gt_path)

        # 确保数据是float32类型
        data = data.astype(np.float32)

        # 归一化
        if self.config.get('norm', False) or True:  # 默认都进行归一化
            data_min = np.min(data)
            data_max = np.max(data)
            if data_max > data_min:  # 避免除零
                data = (data - data_min) / (data_max - data_min)

        print(f"Data shape: {data.shape}, GT shape: {gt.shape}")
        return data, gt

    def extract_patches(self):
        """从超光谱图像中提取图像块"""
        H, W, C = self.data.shape
        half = self.patch_size // 2
        patches = []
        labels = []
        positions = []

        # 统计每个类别的样本数
        class_counts = {}

        for i in range(half, H - half):
            for j in range(half, W - half):
                label_val = int(self.gt[i, j])
                if label_val != 0:  # 只提取有标签的区域
                    patch = self.data[i - half:i + half + 1, j - half:j + half + 1, :]

                    # 检查patch是否包含有效数据
                    if np.any(np.isnan(patch)) or np.any(np.isinf(patch)):
                        continue

                    patches.append(patch)
                    labels.append(label_val)
                    positions.append((i, j))

                    # 统计类别
                    if label_val not in class_counts:
                        class_counts[label_val] = 0
                    class_counts[label_val] += 1

        print(f"Extracted {len(patches)} patches")
        print("Class distribution:", class_counts)

        return patches, labels, positions

    def hyperspectral_to_rgb(self, patch):
        """将超光谱数据转换为RGB图像"""
        H, W, C = patch.shape

        # 重塑为2D数组进行PCA
        patch_2d = patch.reshape(-1, C)

        # 处理NaN和Inf值
        patch_2d = np.nan_to_num(patch_2d)

        # 使用PCA降维到3个主成分
        try:
            pca = PCA(n_components=3)
            patch_rgb = pca.fit_transform(patch_2d)
        except:
            # 如果PCA失败，使用前三个波段
            patch_rgb = patch_2d[:, :3]

        # 重塑回图像形状
        patch_rgb = patch_rgb.reshape(H, W, 3)

        # 归一化到0-1范围
        patch_min = np.min(patch_rgb)
        patch_max = np.max(patch_rgb)
        if patch_max > patch_min:
            patch_rgb = (patch_rgb - patch_min) / (patch_max - patch_min)

        return patch_rgb

    def transform(self, patch, label, is_train):
        """数据转换"""
        # 转换为RGB
        patch_rgb = self.hyperspectral_to_rgb(patch)

        # 调整大小到224x224
        patch_rgb = np.array(Image.fromarray((patch_rgb * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)).astype(np.float32) / 255.0

        # 数据增强：随机水平翻转
        if is_train and np.random.random() > 0.5:
            patch_rgb = np.fliplr(patch_rgb)

        # 转换为CHW格式
        patch_rgb = np.transpose(patch_rgb, [2, 0, 1])
        patch_rgb = patch_rgb.astype(np.float32)

        return patch_rgb, label

    def get_batch(self):
        """获取一个batch的数据"""
        batch_data = []
        batch_labels = []

        for _ in range(self.batch_size):
            if self.current_index >= len(self.indices):
                if self.is_train:
                    np.random.shuffle(self.indices)
                    self.current_index = 0
                else:
                    # 测试模式下，循环使用数据
                    self.current_index = 0

            idx = self.indices[self.current_index]
            patch = self.patches[idx]
            label = self.labels[idx]

            # 转换数据
            patch_rgb, label_onehot = self.transform(patch, label, self.is_train)

            batch_data.append(patch_rgb)
            batch_labels.append(label_onehot)
            self.current_index += 1

        return np.array(batch_data), np.array(batch_labels)

    def generator(self):
        """数据生成器"""
        while True:
            batch_data, batch_labels = self.get_batch()
            yield batch_data, batch_labels

            # 如果是测试模式且已经遍历完所有数据，则退出
            if not self.is_train and self.current_index >= len(self.indices):
                break


# 获取数据集配置
source_config = dataset_config[domain_train]
target_config = dataset_config[domain_test]

print(f"Source config: {source_config}")
print(f"Target config: {target_config}")

# 创建数据加载器
source_train_loader = HyperspectralDataLoader(source_config, is_train=True, batch_size=batch_size)
target_train_loader = HyperspectralDataLoader(target_config, is_train=True, batch_size=batch_size)
target_test_loader = HyperspectralDataLoader(target_config, is_train=False, batch_size=batch_size)


# 定义标签处理函数
def process_source_label(label, config):
    """处理源域标签"""
    known_classes = config.get('known_classes', list(range(1, config['num_classes'] + 1)))
    if label in known_classes:
        label_index = known_classes.index(label)
        return one_hot(26, label_index)
    else:
        # 源域不应该有未知类别
        return one_hot(26, 25)  # 标记为未知


def process_target_label(label, config, for_training=True):
    """处理目标域标签"""
    known_classes = config.get('known_classes', list(range(1, min(25, config['num_classes']))))
    unknown_classes = config.get('unknown_classes', [])

    if for_training:
        # 训练时：已知类别 + 1个未知类别
        if label in known_classes:
            label_index = known_classes.index(label)
            return one_hot(26, label_index)
        elif label in unknown_classes:
            return one_hot(26, 25)  # 未知类别标记为第25类
        else:
            return one_hot(26, 25)  # 其他情况标记为未知
    else:
        # 测试时：所有类别（已知+未知）
        if label in known_classes:
            label_index = known_classes.index(label)
            return one_hot(65, label_index)
        elif label in unknown_classes:
            # 未知类别映射到25-64
            unk_index = unknown_classes.index(label) + 25
            return one_hot(65, unk_index)
        else:
            return one_hot(65, 64)  # 其他情况标记为最后一个类别


# 包装数据加载器以处理标签
class LabelWrapper:
    def __init__(self, loader, config, is_source=True, for_training=True):
        self.loader = loader
        self.config = config
        self.is_source = is_source
        self.for_training = for_training

    def generator(self):
        for batch_data, batch_labels in self.loader.generator():
            processed_labels = []
            for label in batch_labels:
                if self.is_source:
                    processed_label = process_source_label(label, self.config)
                else:
                    processed_label = process_target_label(label, self.config, self.for_training)
                processed_labels.append(processed_label)

            yield batch_data, np.array(processed_labels)


# 创建最终的数据加载器
source_train = LabelWrapper(source_train_loader, source_config, is_source=True, for_training=True)
target_train = LabelWrapper(target_train_loader, target_config, is_source=False, for_training=True)
target_test = LabelWrapper(target_test_loader, target_config, is_source=False, for_training=False)

# 网络定义
discriminator_p = Discriminator(n=25).cuda()  # 10 binary classifier
discriminator = LargeAdversarialNetwork(256).cuda()

# 直接使用torchvision的预训练权重，不指定model_path
feature_extractor_fix = ResNetFc(model_name='resnet50')
feature_extractor_nofix = ResNetFc(model_name='resnet50')

cls_upper = CLS(feature_extractor_fix.output_num(), 26, bottle_neck_dim=256)
cls_down = CLS(feature_extractor_nofix.output_num(), 26, bottle_neck_dim=256)

net_upper = nn.Sequential(feature_extractor_fix, cls_upper).cuda()
net_down = nn.Sequential(feature_extractor_nofix, cls_down).cuda()

three_domain_discriminator = Discriminator2(n=3).cuda()

# 优化器定义
scheduler = lambda step, initial_lr: inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=10000)

optimizer_discriminator = OptimWithSheduler(
    optim.SGD(discriminator.parameters(), lr=5e-4, weight_decay=5e-4, momentum=0.9, nesterov=True),
    scheduler)

optimizer_feature_extractor_fix = OptimWithSheduler(
    optim.SGD(feature_extractor_fix.parameters(), lr=5e-5, weight_decay=5e-4, momentum=0.9, nesterov=True),
    scheduler)

optimizer_feature_extractor_nofix = OptimWithSheduler(
    optim.SGD(feature_extractor_nofix.parameters(), lr=5e-5, weight_decay=5e-4, momentum=0.9, nesterov=True),
    scheduler)

optimizer_cls_upper = OptimWithSheduler(
    optim.SGD(cls_upper.parameters(), lr=5e-4, weight_decay=5e-4, momentum=0.9, nesterov=True),
    scheduler)

optimizer_cls_down = OptimWithSheduler(
    optim.SGD(cls_down.parameters(), lr=5e-4, weight_decay=5e-4, momentum=0.9, nesterov=True),
    scheduler)

optimizer_discriminator_p = OptimWithSheduler(
    optim.SGD(discriminator_p.parameters(), lr=1e-3, weight_decay=5e-4, momentum=0.9, nesterov=True),
    scheduler)

optimizer_three_domain_discriminator = OptimWithSheduler(
    optim.SGD(three_domain_discriminator.parameters(), lr=5e-4, weight_decay=5e-4, momentum=0.9, nesterov=True),
    scheduler)

KL = nn.KLDivLoss()
MSE = nn.MSELoss()

# 训练循环
k = 0
while k < 6000:
    for (i, ((im_source, label_source), (im_target, label_target))) in enumerate(
            zip(source_train.generator(), target_train.generator())):

        im_source = Variable(torch.from_numpy(im_source)).cuda()
        label_source = Variable(torch.from_numpy(label_source)).cuda()
        im_target = Variable(torch.from_numpy(im_target)).cuda()

        fs1, feature_source, __, predict_prob_source_upper = net_upper.forward(im_source)
        ft1, feature_target, __, predict_prob_target = net_upper.forward(im_target)

        p0_upper = discriminator_p.forward(fs1)
        p1_upper = discriminator_p.forward(ft1)

        p2_upper = torch.sum(p1_upper, dim=-1).detach()
        p3_upper = torch.sum(p0_upper, dim=-1).detach()

        d1_upper = BCELossForMultiClassification(label_source[:, 0:25], p0_upper)
        ce_upper = CrossEntropyLoss(label_source, predict_prob_source_upper)

        fs1, feature_source, __, predict_prob_source_down = net_down.forward(im_source)
        ft1, feature_target, __, predict_prob_target = net_down.forward(im_target)

        domain_prob_discriminator_1_source = discriminator.forward(feature_source)
        domain_prob_discriminator_1_target = discriminator.forward(feature_target)

        p0_down = discriminator_p.forward(fs1)
        p1_down = discriminator_p.forward(ft1)

        r_unk = torch.sort(p2_upper, dim=0)[1][:2]  # 001
        feature_otherep = torch.index_select(ft1, 0, r_unk.view(2))
        feature_unk, feature_target_unkonwn, _, predict_prob_otherep = cls_down.forward(feature_otherep)

        r = torch.sort(p2_upper, dim=0)[1][-2:]  # 010
        feature_otherep = torch.index_select(ft1, 0, r.view(2))
        _, feature_target_konwn, _, _ = cls_down.forward(feature_otherep)

        r = torch.sort(p3_upper, dim=0)[1][-2:]  # 100
        feature_otherep = torch.index_select(fs1, 0, r.view(2))
        _, feature_source_konwn, _, _ = cls_down.forward(feature_otherep)

        feature_target_unkonwn_labels = Variable(torch.from_numpy(
            np.concatenate((np.zeros((2, 2)), np.ones((2, 1))), axis=-1).astype('float32'))).cuda()  # 001
        feature_target_konwn_labels = Variable(torch.from_numpy(
            np.concatenate((np.ones((2, 2)), np.zeros((2, 1))), axis=-1).astype('float32'))).cuda()  # 110
        feature_source_unkonwn_labels = Variable(torch.from_numpy(
            np.concatenate((np.ones((2, 2)), np.zeros((2, 1))), axis=-1).astype('float32'))).cuda()  # 110

        three_domain_discriminator_feature = torch.cat(
            [feature_target_unkonwn, feature_target_konwn, feature_source_konwn], 0)
        three_domain_discriminator_labels = torch.cat(
            [feature_target_unkonwn_labels, feature_target_konwn_labels, feature_source_unkonwn_labels], 0)

        p0 = three_domain_discriminator.forward(three_domain_discriminator_feature)
        three_domain_loss = BCELossForMultiClassification(three_domain_discriminator_labels, p0)

        ce_ep = CrossEntropyLoss(Variable(
            torch.from_numpy(np.concatenate((np.zeros((2, 25)), np.ones((2, 1))), axis=-1).astype('float32'))).cuda(), \
                                 predict_prob_otherep)

        ce_down = CrossEntropyLoss(label_source, predict_prob_source_down)

        loss_MSE_upper = MSE(p0_upper, p0_down.detach())
        loss_MSE_upper += MSE(p1_upper, p1_down.detach())

        entropy = EntropyLoss(predict_prob_target, instance_level_weight=p2_upper.contiguous())
        adv_loss = BCELossForMultiClassification(label=torch.ones_like(domain_prob_discriminator_1_source), \
                                                 predict_prob=domain_prob_discriminator_1_source)
        adv_loss += BCELossForMultiClassification(label=torch.ones_like(domain_prob_discriminator_1_target), \
                                                  predict_prob=1 - domain_prob_discriminator_1_target,
                                                  instance_level_weight=p2_upper.contiguous())

        with OptimizerManager([optimizer_cls_upper, optimizer_discriminator_p, optimizer_feature_extractor_fix]):
            loss = loss_MSE_value * loss_MSE_upper + d1_upper + ce_upper
            loss.backward()

        # ---------------------------------------------------------------------------------------------------------------------------
        fs1, feature_source, __, predict_prob_source_upper = net_upper.forward(im_source)
        ft1, feature_target, __, predict_prob_target = net_upper.forward(im_target)

        p0_upper = discriminator_p.forward(fs1)
        p1_upper = discriminator_p.forward(ft1)

        p2_upper = torch.sum(p1_upper, dim=-1).detach()
        p3_upper = torch.sum(p0_upper, dim=-1).detach()

        d1_upper = BCELossForMultiClassification(label_source[:, 0:25], p0_upper)
        ce_upper = CrossEntropyLoss(label_source, predict_prob_source_upper)

        fs1, feature_source, __, predict_prob_source_down = net_down.forward(im_source)
        ft1, feature_target, __, predict_prob_target = net_down.forward(im_target)

        domain_prob_discriminator_1_source = discriminator.forward(feature_source)
        domain_prob_discriminator_1_target = discriminator.forward(feature_target)

        p0_down = discriminator_p.forward(fs1)
        p1_down = discriminator_p.forward(ft1)

        r_unk = torch.sort(p2_upper, dim=0)[1][:2]  # 001
        feature_otherep = torch.index_select(ft1, 0, r_unk.view(2))
        feature_unk, feature_target_unkonwn, _, predict_prob_otherep = cls_down.forward(feature_otherep)

        r = torch.sort(p2_upper, dim=0)[1][-2:]  # 010
        feature_otherep = torch.index_select(ft1, 0, r.view(2))
        _, feature_target_konwn, _, _ = cls_down.forward(feature_otherep)

        r = torch.sort(p3_upper, dim=0)[1][-2:]  # 100
        feature_otherep = torch.index_select(fs1, 0, r.view(2))
        _, feature_source_konwn, _, _ = cls_down.forward(feature_otherep)

        feature_target_unkonwn_labels = Variable(torch.from_numpy(
            np.concatenate((np.zeros((2, 2)), np.ones((2, 1))), axis=-1).astype('float32'))).cuda()  # 001
        feature_target_konwn_labels = Variable(torch.from_numpy(
            np.concatenate((np.ones((2, 2)), np.zeros((2, 1))), axis=-1).astype('float32'))).cuda()  # 110
        feature_source_unkonwn_labels = Variable(torch.from_numpy(
            np.concatenate((np.ones((2, 2)), np.zeros((2, 1))), axis=-1).astype('float32'))).cuda()  # 110

        three_domain_discriminator_feature = torch.cat(
            [feature_target_unkonwn, feature_target_konwn, feature_source_konwn], 0)
        three_domain_discriminator_labels = torch.cat(
            [feature_target_unkonwn_labels, feature_target_konwn_labels, feature_source_unkonwn_labels], 0)

        p0 = three_domain_discriminator.forward(three_domain_discriminator_feature)
        three_domain_loss = BCELossForMultiClassification(three_domain_discriminator_labels, p0)

        ce_ep = CrossEntropyLoss(Variable(
            torch.from_numpy(np.concatenate((np.zeros((2, 25)), np.ones((2, 1))), axis=-1).astype('float32'))).cuda(), \
                                 predict_prob_otherep)

        ce_down = CrossEntropyLoss(label_source, predict_prob_source_down)

        loss_MSE_down = MSE(p0_down, p0_upper.detach())
        loss_MSE_down += MSE(p1_down, p1_upper.detach())

        entropy = EntropyLoss(predict_prob_target, instance_level_weight=p2_upper.contiguous())
        adv_loss = BCELossForMultiClassification(label=torch.ones_like(domain_prob_discriminator_1_source), \
                                                 predict_prob=domain_prob_discriminator_1_source)
        adv_loss += BCELossForMultiClassification(label=torch.ones_like(domain_prob_discriminator_1_target), \
                                                  predict_prob=1 - domain_prob_discriminator_1_target,
                                                  instance_level_weight=p2_upper.contiguous())

        with OptimizerManager([optimizer_cls_down, optimizer_feature_extractor_nofix, optimizer_discriminator,
                               optimizer_three_domain_discriminator]):
            loss = loss_MSE_value * loss_MSE_down + ce_down + 0.3 * adv_loss + 0.1 * entropy + three_domain_loss_value * three_domain_loss
            loss.backward()

        k += 1
        log.step += 1

        if log.step % 10 == 1:
            counter_upper = AccuracyCounter()
            counter_upper.addOntBatch(variable_to_numpy(predict_prob_source_upper), variable_to_numpy(label_source))
            acc_train_upper = Variable(
                torch.from_numpy(np.asarray([counter_upper.reportAccuracy()], dtype=np.float32))).cuda()

            counter_down = AccuracyCounter()
            counter_down.addOntBatch(variable_to_numpy(predict_prob_source_down), variable_to_numpy(label_source))
            acc_train_down = Variable(
                torch.from_numpy(np.asarray([counter_down.reportAccuracy()], dtype=np.float32))).cuda()
            track_scalars(log, ['ce_down', 'acc_train_down', 'acc_train_upper', 'adv_loss', 'entropy', "d1_upper",
                                "loss_MSE_upper", "loss_MSE_down", "three_domain_loss", "ce_ep"], globals())

        if log.step % 100 == 0:
            clear_output()
    print(k)

# 第二阶段训练
k = 0
while k < 3000:
    for (i, ((im_source, label_source), (im_target, label_target))) in enumerate(
            zip(source_train.generator(), target_train.generator())):

        im_source = Variable(torch.from_numpy(im_source)).cuda()
        label_source = Variable(torch.from_numpy(label_source)).cuda()
        im_target = Variable(torch.from_numpy(im_target)).cuda()

        fs1, feature_source, __, predict_prob_source_upper = net_upper.forward(im_source)
        ft1, feature_target, __, predict_prob_target = net_upper.forward(im_target)

        p0_upper = discriminator_p.forward(fs1)
        p1_upper = discriminator_p.forward(ft1)

        p2_upper = torch.sum(p1_upper, dim=-1).detach()
        p3_upper = torch.sum(p0_upper, dim=-1).detach()

        d1_upper = BCELossForMultiClassification(label_source[:, 0:25], p0_upper)
        ce_upper = CrossEntropyLoss(label_source, predict_prob_source_upper)

        fs1, feature_source, __, predict_prob_source_down = net_down.forward(im_source)
        ft1, feature_target, __, predict_prob_target = net_down.forward(im_target)

        domain_prob_discriminator_1_source = discriminator.forward(feature_source)
        domain_prob_discriminator_1_target = discriminator.forward(feature_target)

        p0_down = discriminator_p.forward(fs1)
        p1_down = discriminator_p.forward(ft1)

        r_unk = torch.sort(p2_upper, dim=0)[1][:2]  # 001
        feature_otherep = torch.index_select(ft1, 0, r_unk.view(2))
        feature_unk, feature_target_unkonwn, _, predict_prob_otherep = cls_down.forward(feature_otherep)

        r = torch.sort(p2_upper, dim=0)[1][-2:]  # 010
        feature_otherep = torch.index_select(ft1, 0, r.view(2))
        _, feature_target_konwn, _, _ = cls_down.forward(feature_otherep)

        r = torch.sort(p3_upper, dim=0)[1][-2:]  # 100
        feature_otherep = torch.index_select(fs1, 0, r.view(2))
        _, feature_source_konwn, _, _ = cls_down.forward(feature_otherep)

        feature_target_unkonwn_labels = Variable(torch.from_numpy(
            np.concatenate((np.zeros((2, 2)), np.ones((2, 1))), axis=-1).astype('float32'))).cuda()  # 001
        feature_target_konwn_labels = Variable(torch.from_numpy(
            np.concatenate((np.ones((2, 2)), np.zeros((2, 1))), axis=-1).astype('float32'))).cuda()  # 110
        feature_source_unkonwn_labels = Variable(torch.from_numpy(
            np.concatenate((np.ones((2, 2)), np.zeros((2, 1))), axis=-1).astype('float32'))).cuda()  # 110

        three_domain_discriminator_feature = torch.cat(
            [feature_target_unkonwn, feature_target_konwn, feature_source_konwn], 0)
        three_domain_discriminator_labels = torch.cat(
            [feature_target_unkonwn_labels, feature_target_konwn_labels, feature_source_unkonwn_labels], 0)

        p0 = three_domain_discriminator.forward(three_domain_discriminator_feature)
        three_domain_loss = BCELossForMultiClassification(three_domain_discriminator_labels, p0)

        ce_ep = CrossEntropyLoss(Variable(
            torch.from_numpy(np.concatenate((np.zeros((2, 25)), np.ones((2, 1))), axis=-1).astype('float32'))).cuda(), \
                                 predict_prob_otherep)

        ce_down = CrossEntropyLoss(label_source, predict_prob_source_down)

        loss_MSE_upper = MSE(p0_upper, p0_down.detach())
        loss_MSE_upper += MSE(p1_upper, p1_down.detach())

        entropy = EntropyLoss(predict_prob_target, instance_level_weight=p2_upper.contiguous())
        adv_loss = BCELossForMultiClassification(label=torch.ones_like(domain_prob_discriminator_1_source), \
                                                 predict_prob=domain_prob_discriminator_1_source)
        adv_loss += BCELossForMultiClassification(label=torch.ones_like(domain_prob_discriminator_1_target), \
                                                  predict_prob=1 - domain_prob_discriminator_1_target,
                                                  instance_level_weight=p2_upper.contiguous())

        with OptimizerManager([optimizer_cls_upper, optimizer_discriminator_p, optimizer_feature_extractor_fix]):
            loss = loss_MSE_value * loss_MSE_upper + d1_upper + ce_upper
            loss.backward()

        # ---------------------------------------------------------------------------------------------------------------------------
        fs1, feature_source, __, predict_prob_source_upper = net_upper.forward(im_source)
        ft1, feature_target, __, predict_prob_target = net_upper.forward(im_target)

        p0_upper = discriminator_p.forward(fs1)
        p1_upper = discriminator_p.forward(ft1)

        p2_upper = torch.sum(p1_upper, dim=-1).detach()
        p3_upper = torch.sum(p0_upper, dim=-1).detach()

        d1_upper = BCELossForMultiClassification(label_source[:, 0:25], p0_upper)
        ce_upper = CrossEntropyLoss(label_source, predict_prob_source_upper)

        fs1, feature_source, __, predict_prob_source_down = net_down.forward(im_source)
        ft1, feature_target, __, predict_prob_target = net_down.forward(im_target)

        domain_prob_discriminator_1_source = discriminator.forward(feature_source)
        domain_prob_discriminator_1_target = discriminator.forward(feature_target)

        p0_down = discriminator_p.forward(fs1)
        p1_down = discriminator_p.forward(ft1)

        r_unk = torch.sort(p2_upper, dim=0)[1][:2]  # 001
        feature_otherep = torch.index_select(ft1, 0, r_unk.view(2))
        feature_unk, feature_target_unkonwn, _, predict_prob_otherep = cls_down.forward(feature_otherep)

        r = torch.sort(p2_upper, dim=0)[1][-2:]  # 010
        feature_otherep = torch.index_select(ft1, 0, r.view(2))
        _, feature_target_konwn, _, _ = cls_down.forward(feature_otherep)

        r = torch.sort(p3_upper, dim=0)[1][-2:]  # 100
        feature_otherep = torch.index_select(fs1, 0, r.view(2))
        _, feature_source_konwn, _, _ = cls_down.forward(feature_otherep)

        feature_target_unkonwn_labels = Variable(torch.from_numpy(
            np.concatenate((np.zeros((2, 2)), np.ones((2, 1))), axis=-1).astype('float32'))).cuda()  # 001
        feature_target_konwn_labels = Variable(torch.from_numpy(
            np.concatenate((np.ones((2, 2)), np.zeros((2, 1))), axis=-1).astype('float32'))).cuda()  # 110
        feature_source_unkonwn_labels = Variable(torch.from_numpy(
            np.concatenate((np.ones((2, 2)), np.zeros((2, 1))), axis=-1).astype('float32'))).cuda()  # 110

        three_domain_discriminator_feature = torch.cat(
            [feature_target_unkonwn, feature_target_konwn, feature_source_konwn], 0)
        three_domain_discriminator_labels = torch.cat(
            [feature_target_unkonwn_labels, feature_target_konwn_labels, feature_source_unkonwn_labels], 0)

        p0 = three_domain_discriminator.forward(three_domain_discriminator_feature)
        three_domain_loss = BCELossForMultiClassification(three_domain_discriminator_labels, p0)

        ce_ep = CrossEntropyLoss(Variable(
            torch.from_numpy(np.concatenate((np.zeros((2, 25)), np.ones((2, 1))), axis=-1).astype('float32'))).cuda(), \
                                 predict_prob_otherep)

        ce_down = CrossEntropyLoss(label_source, predict_prob_source_down)

        loss_MSE_down = MSE(p0_down, p0_upper.detach())
        loss_MSE_down += MSE(p1_down, p1_upper.detach())

        entropy = EntropyLoss(predict_prob_target, instance_level_weight=p2_upper.contiguous())
        adv_loss = BCELossForMultiClassification(label=torch.ones_like(domain_prob_discriminator_1_source), \
                                                 predict_prob=domain_prob_discriminator_1_source)
        adv_loss += BCELossForMultiClassification(label=torch.ones_like(domain_prob_discriminator_1_target), \
                                                  predict_prob=1 - domain_prob_discriminator_1_target,
                                                  instance_level_weight=p2_upper.contiguous())

        with OptimizerManager([optimizer_cls_down, optimizer_feature_extractor_nofix, optimizer_discriminator,
                               optimizer_three_domain_discriminator]):
            loss = loss_MSE_value * loss_MSE_down + ce_down + 0.3 * adv_loss + 0.1 * entropy + three_domain_loss_value * three_domain_loss + 0.3 * ce_ep
            loss.backward()

        k += 1
        log.step += 1

        if log.step % 10 == 1:
            counter_upper = AccuracyCounter()
            counter_upper.addOntBatch(variable_to_numpy(predict_prob_source_upper), variable_to_numpy(label_source))
            acc_train_upper = Variable(
                torch.from_numpy(np.asarray([counter_upper.reportAccuracy()], dtype=np.float32))).cuda()

            counter_down = AccuracyCounter()
            counter_down.addOntBatch(variable_to_numpy(predict_prob_source_down), variable_to_numpy(label_source))
            acc_train_down = Variable(
                torch.from_numpy(np.asarray([counter_down.reportAccuracy()], dtype=np.float32))).cuda()
            track_scalars(log, ['ce_down', 'acc_train_down', 'acc_train_upper', 'adv_loss', 'entropy', "d1_upper",
                                "loss_MSE_upper", "loss_MSE_down", "three_domain_loss", "ce_ep"], globals())

        if log.step % 100 == 0:
            clear_output()
    print(k)

    # 测试
    with TrainingModeManager([feature_extractor_fix, feature_extractor_nofix, cls_down, cls_upper], train=False) \
            as mgr, Accumulator(['predict_prob', 'predict_index', 'label']) as accumulator:
        for (i, (im, label)) in enumerate(target_test.generator()):
            im = Variable(torch.from_numpy(im), volatile=True).cuda()
            label = Variable(torch.from_numpy(label), volatile=True).cuda()

            ft1, feature_target, __, predict_prob = net_down.forward(im)

            p1_upper = discriminator_p.forward(ft1)
            p2_upper = torch.sum(p1_upper, dim=-1).detach()

            predict_prob, label = [variable_to_numpy(x) for x in (predict_prob, label)]
            label = np.argmax(label, axis=-1).reshape(-1, 1)
            predict_index = np.argmax(predict_prob, axis=-1).reshape(-1, 1)
            accumulator.updateData(globals())

    for x in accumulator.keys():
        globals()[x] = accumulator[x]

    y_true = label.flatten()
    y_pred = predict_index.flatten()

    # 获取目标域的已知和未知类别
    known_classes = target_config.get('known_classes', list(range(1, min(25, target_config['num_classes']))))
    unknown_classes = target_config.get('unknown_classes', [])

    # 创建标签映射
    true_labels = known_classes + unknown_classes
    pred_labels = list(range(len(known_classes))) + [25]  # 未知类别在预测中被映射到第25类

    m = extended_confusion_matrix(y_true, y_pred, true_labels=true_labels, pred_labels=pred_labels)

    cm = m
    cm = cm.astype(np.float) / np.sum(cm, axis=1, keepdims=True)

    # 计算已知类别的准确率
    known_acc = sum([cm[i][i] for i in range(len(known_classes))]) / len(known_classes)

    # 计算未知类别的检测率（被正确识别为未知的比例）
    if len(unknown_classes) > 0:
        unk_detection_rate = sum([cm[len(known_classes) + i][25] for i in range(len(unknown_classes))]) / len(
            unknown_classes)
    else:
        unk_detection_rate = 0.0

    # 计算整体准确率
    overall_acc = (known_acc * len(known_classes) + unk_detection_rate * len(unknown_classes)) / (
                len(known_classes) + len(unknown_classes))

    print("Overall Accuracy:", overall_acc)
    print("Known Classes Accuracy:", known_acc)
    print("Unknown Detection Rate:", unk_detection_rate)