import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from batchgenerators.utilities.file_and_folder_operations import *
import pandas as pd
from tqdm import trange, tqdm
from skimage import io
from PIL import Image
import numpy as np
Image.MAX_IMAGE_PIXELS = None
from skimage import io

target_rank = {'Background': 0, '_background_': 1, 'Normal Glomeruli': 2,
                    'Sclerotic Glomeruli': 3, 'Incomplete Glomeruli': 4, 'Other Glomeruli': 5, 'MESC': 6,
                    'Atrophic Tubuli/Interstitial Fibrosis': 7, 'Interstitial Inflammation': 8,
                    'Medium-Small Sized Artery': 9,
                    'Arteriole': 10, 'Vein': 11, 'Medulla': 12}

label_color = [[0, 0, 0], [255, 255, 255], [0, 0, 255], [0, 128, 0], [255, 0, 0], [127, 0, 255], [128, 0, 0],
                   [255, 255, 0],
                   [255, 0, 127], [94, 38, 18], [255, 127, 0], [255, 0, 255], [0, 127, 255], [127, 255, 0],
                   [127, 0, 127], [0, 127, 127], [127, 127, 0], [80, 80, 160], [80, 160, 160], [160, 160, 160],
                    [160, 80, 80], [160, 160, 80], [80, 80, 240], [80, 240, 240], [240, 80, 80], [240, 240, 80], [240, 240, 160]]


def vis_binary_map(mask, save_path):
    mask_vis = np.zeros((mask.shape[0], mask.shape[1], 3))
    for ch in range(27):
        mask_vis[mask == ch] = label_color[ch]
    io.imsave(save_path, np.uint8(mask_vis))

def mask_convert(mask):
    current_rank = { 'Background':0, '_background_':1, 'Medulla':2,  'Normal Glomeruli':3,
                    'Sclerotic Glomeruli':4, 'Incomplete Glomeruli':5, 'Other Glomeruli':6,
                    'Atrophic Tubuli/Interstitial Fibrosis':7, 'Interstitial Inflammation':8, 'Medium-Small Sized Artery':9,
                    'Arteriole':10, 'Vein':11, 'M0E0S0C1':12, 'M0E1S0C0':13, 'M1E0S1C0':14, 'M1E0S0C0':15,
                    'M1E1S1C0':16, 'M0E0S1C0':17, 'M1E1S1C1':18,
                    'M1E0S1C1':19, 'M1E0S0C1':20, 'M1E1S0C0':21, 'M0E1S1C1':22, 'M0E1S0C1':23, 'M0E1S1C0':24,
                    'M0E0S1C1':25, 'M1E1S0C1':26}
    ori_rank_nnunet = {'Background': 0, '_background_': 1, 'Glomeruli': 2,
         'Atrophic Tubuli_Interstitial Fibrosis': 3, 'Interstitial Inflammation': 4, 'Medium-Small Sized Artery': 5,
         'Arteriole': 6,
         'Vein': 7, 'Medulla': 8}
    mask[mask==2] = 100
    for i in range(3, 7):
        mask[mask == i] = 2
    for i in range(7, 12):
        mask[mask == i] -= 4
    for i in range(12, 27):
        mask[mask == i] = 2
    mask[mask==100] = 8
    return mask

def vis_mesc_map(mask, save_path):
    mask_vis = np.zeros((mask.shape[0], mask.shape[1], 3))
    for ch in range(13):
        mask_vis[mask == ch] = label_color[ch]
    io.imsave(save_path, np.uint8(mask_vis))

if __name__ == "__main__":

    vis_binary_map('./nnunet_output/gt/000.png', 'gt.png')



