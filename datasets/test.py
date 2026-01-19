import sys
import os
import logging
from easydict import EasyDict
import torch
from torch.utils.data import DataLoader
import numpy as np
from functools import partial
import os.path as osp
from collections.abc import Mapping, Sequence
from mmcv.utils import Registry
from torch.utils.data import Dataset
import copy
from abc import ABCMeta, abstractmethod
from collections import defaultdict
import mmcv
from ipdb import set_trace as st
import matplotlib.pyplot as plt
# 将父目录添加到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入相应模块和函数
from pipeline_three import *
from build_gray_PRF import build_dataloader

def save_tensor_images(tensor, save_path='/data2/XCLIP/XCLIP/save_img/test', prefix=""):
    # 检查保存路径是否存在，不存在则创建
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 如果张量的维度大于3，则递归处理
    if tensor.dim() > 3:
        for idx in range(tensor.size(0)):
            new_prefix = f"{prefix}_{idx}"
            save_tensor_images(tensor[idx], save_path, new_prefix)
    else:
        # 处理最后三维为 (3, 224, 224) 的情况
        img_np = tensor.permute(1, 2, 0).cpu().numpy()  # 转换为 (224, 224, 3)

        # 创建图片的保存路径和文件名
        img_filename = f"{prefix}.png"
        img_filepath = os.path.join(save_path, img_filename)
        
        # 使用 matplotlib 保存图片
        plt.imshow(img_np)
        plt.axis('off')  # 隐藏坐标轴
        plt.savefig(img_filepath, bbox_inches='tight', pad_inches=0)
        plt.close()
    
    if prefix == "":
        print(f"所有图片已保存到 {save_path} 目录下")


# 创建一个简单的logger
logger = logging.getLogger('test_logger')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# 模拟一个简单的config对象
config_dict = {
    'DATA': {
        'Data_path': '/data2/NTU_data/resize_mb_not_frame_difference',
        'TRAIN_FILE': '/data2/XCLIP/XCLIP/txt_file/NTU120_XSet_train.txt',
        'VAL_FILE': '/data2/XCLIP/XCLIP/txt_file/NTU120_XSet_val.txt',
        'NUM_FRAMES': 8,
        'NUM_CLASSES': 120,
        'LABEL_LIST': '/data2/XCLIP/XCLIP/ntu120_label_name.txt',
        'BENCHMARK': 'NTU120-XSet'
    },
    'MODEL': {
        'ARCH': 'ViT-B/16'
    },                                                              
    'TRAIN': {
        'BATCH_SIZE': 2,
        'ACCUMULATION_STEPS': 2,
        'NUM_WORKERS': 4
    },
    'TEST': {
        'BATCH_SIZE': 16,
        'ONLY_TEST': False,
        'NUM_CROP': 1,  # 添加默认的NUM_CROP属性
        'NUM_CLIP': 1   # 添加默认的NUM_CLIP属性
    },
    'AUG': {
        'COLOR_JITTER': 0.4,
        'GRAY_SCALE': 0.2
    },
    'DISTRIBUTED': False
}
config = EasyDict(config_dict)

# 调用build_dataloader函数
train_data, val_data, train_loader, val_loader = build_dataloader(logger, config)

# 打印train_loader的结构，取第一个元素
for batch in train_loader:
    print(batch)
    st()
    image = torch.stack([image_1, image_2, image_3], dim=2)
    # save_tensor_images(batch['imgs'])
   
    # break

'''
ipdb> batch['imgs'].shape
torch.Size([2, 8, 3, 224, 224])   BATCH_SIZE,NUM_FRAMES, c h w

'''