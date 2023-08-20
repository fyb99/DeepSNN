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
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2
import numpy as np
from time import time
import os
from collections import OrderedDict
from batchgenerators.dataloading.data_loader import DataLoader
from random import sample
from PIL import Image
from skimage import io

Image.MAX_IMAGE_PIXELS = None

def train_transform(degree=180):

    return transforms.Compose([
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=degree),
        transforms.ColorJitter(),
    ])


def glo_crop(image_path, mask_path, savedir):

    if not os.path.exists(savedir):
        os.mkdir(savedir)
    labelmap = {'Normal Glomeruli':3,  'Sclerotic Glomeruli':4, 'Incomplete Glomeruli':5, 'Other Glomeruli':6,
                'M0E0S0C1':12, 'M0E1S0C0':13, 'M1E0S1C0':14, 'M1E0S0C0':15,  'M1E1S1C0':16,'M0E0S1C0':17, 'M1E1S1C1':18,
                'M1E0S1C1':19, 'M1E0S0C1':20, 'M1E1S0C0':21, 'M0E1S1C1':22, 'M0E1S0C1':23, 'M0E1S1C0':24, 'M0E0S1C1':25, 'M1E1S0C1':26}

    image = io.imread(image_path)
    for label in labelmap:
        if not os.path.exists(savedir +'/' + label):
            os.mkdir(savedir +'/'+ label)
        count = 0
        mask = io.imread(mask_path, 0)
        # if mask.shape != image.shape:
        #     mask = cv2.resize(mask, (image.shape[1], image.shape[0]), cv2.INTER_NEAREST)
        img = np.uint8(np.where(mask==labelmap[label], 255, 0))
        ret, thresh = cv2.threshold(img.copy(), 127, 255, cv2.THRESH_BINARY)
        if np.mean(thresh) != 0:
            retval, labels, stats, centroids = cv2.connectedComponentsWithStats(np.uint8(thresh))
            for c in range(1, retval):
                if stats[c, 4] > 300:
                    io.imsave(savedir + '/' + label + '/' + image_path.split('/')[-1] + '_' + str(count) + '.png',
                              image[stats[c, 1]:stats[c, 1] + stats[c, 3], stats[c, 0]:stats[c, 0] + stats[c, 2]])
                    count += 1
            # contours, hier = cv2.findContours(np.uint8(thresh), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
            # for c in contours:
            #     x, y, w, h = cv2.boundingRect(c)
            #     io.imsave(savedir + '/' + label + '/' + image_path.split('/')[-1] + '_' + str(count) + '.png', image[y:y+h, x:x+w, :])
            #     count = count + 1


class ClsDataLoader2D(DataLoader):
    def __init__(self, root, batch_size, img_size, num_class, num_threads_in_multithreaded, seed_for_shuffle=1234, return_incomplete=False,
                 shuffle=True):
        super().__init__(root, num_threads_in_multithreaded, seed_for_shuffle, return_incomplete, shuffle,
                         True)
        self.root = root
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_class = num_class
        self.label_dict = {'Normal Glomeruli': 0, 'Sclerotic Glomeruli': 1, 'Incomplete Glomeruli': 2,
                               'Other Glomeruli': 3,
                               'M0E0S0C1': 4, 'M0E1S0C0': 4, 'M1E0S1C0': 4, 'M1E0S0C0': 4, 'M1E1S1C0': 4, 'M0E0S1C0': 4,
                               'M1E1S1C1': 4,
                               'M1E0S1C1': 4, 'M1E0S0C1': 4, 'M1E1S0C0': 4, 'M0E1S1C1': 4, 'M0E1S0C1': 4, 'M0E1S1C0': 4,
                               'M0E0S1C1': 4, 'M1E1S0C1': 4}
        self.file_lists = []
        for file_dir in np.array([x.path for x in os.scandir(self.root)]):
            files = np.array([x.path.decode('utf-8') for x in os.scandir(file_dir)])
            self.file_lists = np.append(self.file_lists, files)


    def load_patient(self, patient):
        img = cv2.imread(patient)
        seg = np.zeros((1, *self.img_size), dtype=np.float32)
        img = cv2.resize(img, self.img_size) / 255.0
        img = np.transpose(img, (2, 0, 1))

        return img, seg

    def generate_train_batch(self):

        patients_for_batch = sample(list(self.file_lists), self.batch_size)
        data = np.zeros((self.batch_size, 3, *self.img_size), dtype=np.float32)
        seg = np.zeros((self.batch_size, 1, *self.img_size), dtype=np.float32)
        label = np.zeros((self.batch_size,), dtype=np.float32)
        patients_name= []

        for i in range(len(patients_for_batch)):
            img_data, seg_data = self.load_patient(patients_for_batch[i])
            data[i] = img_data
            seg[i] = seg_data
            label[i] = self.label_dict[patients_for_batch[i].split('/')[-2]]
            patients_name.append(patients_for_batch[i])
        return {'data': data, 'seg': seg, 'label':label, 'names':patients_name}


aug_params = {'selected_data_channels': None, 'selected_seg_channels': [0], 'do_elastic': False,
              'elastic_deform_alpha': (0.0, 200.0), 'elastic_deform_sigma': (9.0, 13.0), 'p_eldef': 0.2,
              'do_scaling': True,
              'scale_range': (0.7, 1.4), 'independent_scale_factor_for_each_axis': False,
              'p_independent_scale_per_axis': 1,
              'p_scale': 0.2, 'do_rotation': True, 'rotation_x': (-3.141592653589793, 3.141592653589793),
              'rotation_y': (-0.0, 0.0), 'rotation_z': (-0.0, 0.0), 'rotation_p_per_axis': 0.15, 'p_rot': 0.2,
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


class ClsDataset(data.Dataset):
    def __init__(self, root, img_size, num_class=5):
        self.root = root
        self.file_lists = []
        for file_dir in np.array([x.path for x in os.scandir(self.root)]):
            # files = np.array([x.path.decode('utf-8') for x in os.scandir(file_dir)])
            files = np.array([x.path for x in os.scandir(file_dir)])
            self.file_lists = np.append(self.file_lists, files)
        self.file_lists.sort()
        self.image_files = self.file_lists
        self.num_img = len(self.file_lists)
        self.img_size = img_size
        self.num_class = num_class
        self.label_dict = {'Normal Glomeruli': 0, 'Sclerotic Glomeruli': 1, 'Incomplete Glomeruli': 2, 'Other Glomeruli': 3,
                           'M0E0S0C1': 4, 'M0E1S0C0': 4, 'M1E0S1C0': 4, 'M1E0S0C0': 4, 'M1E1S1C0': 4, 'M0E0S1C0': 4,
                           'M1E1S1C1': 4,
                           'M1E0S1C1': 4, 'M1E0S0C1': 4, 'M1E1S0C0': 4, 'M0E1S1C1': 4, 'M0E1S0C1': 4, 'M0E1S1C0': 4,
                           'M0E0S1C1': 4, 'M1E1S0C1': 4}

    def __getitem__(self, index):

        img = cv2.imread(self.image_files[index])
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = np.transpose(img, (2, 0, 1))
        img = img / 255.0
        img = img.astype(np.dtype(np.float32))
        self.image_files[index] = self.image_files[index].replace('\\', '/')
        gt = self.label_dict[self.image_files[index].split('/')[-2]]
        # gt = self.label_dict[self.image_files[index].split('\\')[-2]]

        return torch.from_numpy(np.array(img)).float(), torch.from_numpy(np.array(gt)).long()

    def __len__(self):
        return len(self.image_files)

if __name__ == "__main__":

    file_test = np.array([x.path for x in os.scandir('./data/img')])
    for idx in trange(len(file_test)):
        file = './data/img/' + file_test[idx]
        mask_file = file.replace('img', 'mask')
        glo_crop(file, mask_file, './data_cls/PLAG/%s/'%file_test[idx].split('_')[0])