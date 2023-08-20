import torch.utils.data as data
import torch
import PIL.Image as Image
import os
import random
from math import ceil
from torchvision.transforms import transforms
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import DataLoader
import cv2
import math
from glob import glob
from tqdm import tqdm,trange
import random
Image.MAX_IMAGE_PIXELS = None
import torch.nn as nn
import matplotlib.pyplot as plt
from skimage import io,transform
import imageio
import pandas as pd
from torchvision.transforms import Compose
import platform
sysstr = platform.system()
import numpy as np
from copy import deepcopy
import numpy as np
import cv2
import batchgenerators.augmentations.crop_and_pad_augmentations as crop
from batchgenerators.augmentations.utils import pad_nd_image
from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.utilities.data_splitting import get_split_deterministic
import numpy as np
from time import time
import os
from collections import OrderedDict
from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.augmentations.utils import pad_nd_image
from batchgenerators.transforms.spatial_transforms import SpatialTransform_2, MirrorTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.abstract_transforms import Compose
from data_augmentation import get_moreDA_augmentation
from nnunet.training.data_augmentation import data_augmentation_moreDA
from random import sample


class MLCDataLoader2D(DataLoader):
    def __init__(self, root, batch_size, img_size, num_threads_in_multithreaded, seed_for_shuffle=1234, return_incomplete=False,
                 shuffle=True):
        super().__init__(root, num_threads_in_multithreaded, seed_for_shuffle, return_incomplete, shuffle,
                         True)
        self.root = root
        self.batch_size = batch_size
        self.img_size = img_size
        self.label_dict = ['M0E0S0C1', 'M0E1S0C0', 'M1E1S1C0','M0E0S1C0', 'M1E1S1C1',
                            'M1E0S1C1', 'M1E0S0C1', 'M1E1S0C0', 'M0E1S1C1', 'M0E1S0C1', 'M0E1S1C0', 'M0E0S1C1', 'M1E1S0C1']
        self.file_lists = []
        for file_dir in np.array([x.path for x in os.scandir(self.root)]):
            if file_dir.split('/')[-1] in self.label_dict:
                files = np.array([x.path.decode('utf-8') for x in os.scandir(file_dir)])
                self.file_lists = np.append(self.file_lists, files)

    def load_patient(self, patient):
        seg = np.zeros((1, *self.img_size), dtype=np.float32)
        img = cv2.imread(patient)
        img = cv2.resize(img, self.img_size)
        img = np.transpose(img, (2, 0, 1))
        img = img / 255.0
        img = img.astype(np.dtype(np.float32))

        return img, seg

    def generate_train_batch(self):

        patients_for_batch = sample(list(self.file_lists), self.batch_size)
        data = np.zeros((self.batch_size, 3, *self.img_size), dtype=np.float32)
        seg = np.zeros((self.batch_size, 1, *self.img_size), dtype=np.float32)
        label = np.zeros((self.batch_size, 4), dtype=np.float32)
        patients_name= []

        for i in range(len(patients_for_batch)):
            img_data, seg_data = self.load_patient(patients_for_batch[i])
            data[i] = img_data
            seg[i] = seg_data
            label_patient = patients_for_batch[i].split('/')[-2]
            if label_patient != 'Normal Glomeruli':
                label[i] = [int(label_patient[1]), int(label_patient[3]), int(label_patient[5]), int(label_patient[7])]
            else:
                label[i] = [0, 0, 0, 0]
            patients_name.append(patients_for_batch[i])
        return {'data': data, 'seg': seg, 'label': label, 'names':patients_name}

aug_params = {'selected_data_channels': None, 'selected_seg_channels': [0], 'do_elastic': False,
              'elastic_deform_alpha': (0.0, 200.0), 'elastic_deform_sigma': (9.0, 13.0), 'p_eldef': 0.2,
              'do_scaling': True,
              'scale_range': (0.7, 1.4), 'independent_scale_factor_for_each_axis': False,
              'p_independent_scale_per_axis': 1,
              'p_scale': 0.2, 'do_rotation': True, 'rotation_x': (-3.141592653589793, 3.141592653589793),
              'rotation_y': (-0.0, 0.0), 'rotation_z': (-0.0, 0.0), 'rotation_p_per_axis': 0.1, 'p_rot': 0.2,
              'random_crop': False,
              'random_crop_dist_to_border': None, 'do_gamma': True, 'gamma_retain_stats': True,
              'gamma_range': (0.7, 1.5),
              'p_gamma': 0.3, 'do_mirror': True, 'mirror_axes': (0, 1), 'dummy_2D': False,
              'mask_was_used_for_normalization': OrderedDict([(0, False), (1, False), (2, False)]),
              'border_mode_data': 'constant', 'all_segmentation_labels': None,
              'move_last_seg_chanel_to_data': False,
              'cascade_do_cascade_augmentations': False, 'cascade_random_binary_transform_p': 0.4,
              'cascade_random_binary_transform_p_per_label': 1, 'cascade_random_binary_transform_size': (1, 8),
              'cascade_remove_conn_comp_p': 0.2, 'cascade_remove_conn_comp_max_size_percent_threshold': 0.15,
              'cascade_remove_conn_comp_fill_with_other_class_p': 0.0, 'do_additive_brightness': False,
              'additive_brightness_p_per_sample': 0.15, 'additive_brightness_p_per_channel': 0.5,
              'additive_brightness_mu': 0.0,
              'additive_brightness_sigma': 0.1, 'num_threads': 12, 'num_cached_per_thread': 2,
              'patch_size_for_spatialtransform': np.array([512, 512])}

import platform
class MLCDataset(data.Dataset):
    def __init__(self, root, img_size, num_class=5):
        self.root = root
        self.label_dict = ['M0E0S0C1', 'M0E1S0C0', 'M1E0S1C0', 'M1E0S0C0', 'M1E1S1C0', 'M0E0S1C0', 'M1E1S1C1',
                           'M1E0S1C1', 'M1E0S0C1', 'M1E1S0C0', 'M0E1S1C1', 'M0E1S0C1', 'M0E1S1C0', 'M0E0S1C1',
                           'M1E1S0C1']
        self.file_lists = []
        for file_dir in np.array([x.path for x in os.scandir(self.root)]):
            file_dir = file_dir.replace('\\', '/')
            if file_dir.split('/')[-1] in self.label_dict:
                # files = np.array([x.path.decode('utf-8') for x in os.scandir(file_dir)])
                files = np.array([x.path for x in os.scandir(file_dir)])
                self.file_lists = np.append(self.file_lists, files)

        self.file_lists.sort()
        self.image_files = self.file_lists
        self.num_img = len(self.file_lists)
        self.img_size = img_size
        self.num_class = num_class

    def __getitem__(self, index):

        img = cv2.imread(self.image_files[index])
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img / 255.0

        img = np.transpose(img, (2, 0, 1))
        img = img.astype(np.dtype(np.float32))

        self.image_files[index] = self.image_files[index].replace('\\', '/')
        label_patient = self.image_files[index].split('/')[-2]
        gt = [int(label_patient[1]), int(label_patient[3]), int(label_patient[5]), int(label_patient[7])]
        return torch.from_numpy(np.array(img)).float(), torch.from_numpy(np.array(gt)).long()

    def __len__(self):
        return len(self.image_files)

if __name__ == "__main__":


    EPOCH = 600
    img_size = 512
    num_class = 2
    LR = 1 * 1e-4
    train_BATCH_SIZE = 8
    test_BATCH_SIZE = 1
    dataset = "_PLAG"
    basic_model = "densenet"
    basic_model = basic_model + '_' + dataset
    device_ids = [1]
    num_batches_per_epoch = 600
    sysstr = platform.system()

    root_path_train = './data_cls/train/PLAG'
    root_path_test = './data_cls/test/PLAG'