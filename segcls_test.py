import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from batchgenerators.utilities.file_and_folder_operations import *
import nibabel as nib
import os
import numpy as np
import pandas as pd
from tqdm import tqdm,trange
import cv2
from PIL import Image
import numpy as np
Image.MAX_IMAGE_PIXELS = None
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
from skimage import io
import warnings
from cls_net import *
import cv2
warnings.filterwarnings(action='ignore')
from PIL import Image
from skimage import io
import numpy as np
import os
Image.MAX_IMAGE_PIXELS = None

def mask_convert(mask_ori, mode, test_center = 'PLAG'):
    rank_nnunet = {'Background': 0, '_background_': 1, 'Glomeruli': 2,
         'Atrophic Tubuli_Interstitial Fibrosis': 3, 'Interstitial Inflammation': 4, 'Medium-Small Sized Artery': 5,
         'Arteriole': 6,
         'Vein': 7, 'Medulla': 8}
    target_rank = {'Background': 0, '_background_': 1, 'Normal Glomeruli': 2,
                    'Sclerotic Glomeruli': 3, 'Incomplete Glomeruli': 4, 'Other Glomeruli': 5, 'MESC': 6,
                    'Atrophic Tubuli/Interstitial Fibrosis': 7, 'Interstitial Inflammation': 8,
                    'Medium-Small Sized Artery': 9,
                    'Arteriole': 10, 'Vein': 11, 'Medulla': 12}
    all_rank = {'Background':0, '_background_':1, 'Medulla':2,  'Normal Glomeruli':3,
                    'Sclerotic Glomeruli':4, 'Incomplete Glomeruli':5, 'Other Glomeruli':6,
                    'Atrophic Tubuli/Interstitial Fibrosis':7, 'Interstitial Inflammation':8, 'Medium-Small Sized Artery':9,
                    'Arteriole':10, 'Vein':11, 'M0E0S0C1':12, 'M0E1S0C0':13, 'M1E0S1C0':14, 'M1E0S0C0':15,
                    'M1E1S1C0':16, 'M0E0S1C0':17, 'M1E1S1C1':18,
                    'M1E0S1C1':19, 'M1E0S0C1':20, 'M1E1S0C0':21, 'M0E1S1C1':22, 'M0E1S0C1':23, 'M0E1S1C0':24,
                    'M0E0S1C1':25, 'M1E1S0C1':26}
    mask = mask_ori.copy()
    if mode == 'nnunet':
        for i in range(3, 9):
            mask[mask_ori == i] = mask[mask_ori == i] + 4
        return mask
    else:
        mask[mask_ori==2] = 12
        for i in range(3, 7):
            mask[mask_ori == i] -= 1
        for i in range(12, 27):
            mask[mask_ori == i] = 6
        mask[mask_ori==12] = 2
        if test_center != 'PLAG':
            mask[mask_ori == 0] = 1
            mask[mask_ori == 1] = 0
        return mask

def cm_convert(mask_ori):
    target_rank = {'Background': 0, '_background_': 1, 'Normal Glomeruli': 2,
                    'Sclerotic Glomeruli': 3, 'Incomplete Glomeruli': 4, 'Other Glomeruli': 5, 'MESC': 6,
                    'Atrophic Tubuli/Interstitial Fibrosis': 7, 'Interstitial Inflammation': 8,
                    'Medium-Small Sized Artery': 9,
                    'Arteriole': 10, 'Vein': 11, 'Medulla': 12}

    mask = mask_ori.copy()
    mask[mask_ori==12] = 2
    for i in range(2, 7):
        mask[mask_ori == i] = 3
    for i in range(7, 12):
        mask[mask_ori == i] = i-3
    return mask


def glo_mask_generate(mask, img_whole, threshold, glo_index, device_ids, center):

    glo_mask = np.where(mask==glo_index, 255, 0)
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(np.uint8(glo_mask))
    net = clsNet(classes_num=5, basic_model="resnet_cbam", pretrain=False).cuda(device_ids[0])
    net.load_state_dict(torch.load('cls.pth', map_location='cpu'))
    net.eval()
    for c in range(1, retval):
        if stats[c, 4] > threshold:
            img = img_whole[stats[c, 1]:stats[c, 1] + stats[c, 3], stats[c, 0]:stats[c, 0] + stats[c, 2]]
            img = cv2.resize(img, (512, 512))
            img = img / 255.0
            img = np.transpose(img, (2, 0, 1))
            img = torch.from_numpy(np.array(img.astype(np.dtype(np.float32)))).unsqueeze(0).cuda(device_ids[0])
            outputs = net(img)
            _, predicted = torch.max(outputs.data, 1)
            predicted = int(predicted.cpu().numpy()[0])
            mask[labels == c] = predicted + 2
        else:
            mask[labels == c] = 1

    return mask


def rotate(mask):
    dst1 = cv2.rotate(mask, 0)
    v_flip = cv2.flip(dst1, 1)
    return v_flip

from utils import label_color
def vis_binary_map(mask, save_path):
    mask_vis = np.zeros((mask.shape[0], mask.shape[1], 3))
    for ch in range(13):
        mask_vis[mask == ch] = label_color[ch]
    io.imsave(save_path, np.uint8(mask_vis))

if __name__ == "__main__":

    test_Center = 'PLAG'
    root_gt = './data_test/gt/'
    root_pred = './data_test/nnunet_output/'
    img_path = './data_test/img/'
    seg_class = list({'Background': 0, '_background_': 1, 'Normal Glomeruli': 2,
                    'Sclerotic Glomeruli': 3, 'Incomplete Glomeruli': 4, 'Other Glomeruli': 5, 'MESC': 6,
                    'Atrophic Tubuli/Interstitial Fibrosis': 7, 'Interstitial Inflammation': 8,
                    'Medium-Small Sized Artery': 9,
                    'Arteriole': 10, 'Vein': 11, 'Medulla': 12}.keys())

    test_files = [file for file in os.listdir(root_pred)]
    result_dice = pd.DataFrame(columns=seg_class, index=test_files)
    device_ids = [1]
    for i in trange(len(result_dice)):
        seg_name = result_dice.index[i]
        seg_gt = io.imread(root_gt + seg_name, 0)
        seg_pred = io.imread(root_pred + seg_name, 0)
        WSI = io.imread(img_path + seg_name)
        WSI = cv2.cvtColor(WSI, cv2.COLOR_RGB2BGR)

        if seg_pred.shape != (WSI.shape[0], WSI.shape[1]):
            WSI = cv2.resize(WSI, (seg_pred.shape[1], seg_pred.shape[0]))
        if seg_pred.shape != seg_gt.shape:
            seg_gt = cv2.resize(seg_gt, (seg_pred.shape[1], seg_pred.shape[0]), cv2.INTER_NEAREST)

        seg_gt_cls = mask_convert(seg_gt, mode='gt', test_center=seg_name.split('_')[0])
        seg_pred_cls = mask_convert(seg_pred, mode='nnunet', test_center=seg_name.split('_')[0])
        WSI = io.imread(img_path + seg_name)
        seg_glo_cls = glo_mask_generate(seg_pred_cls, WSI, 3000, 2, device_ids, center=seg_name.split('_')[0])
        io.imsave('./data_test/segcls/' + seg_name, np.uint8(seg_glo_cls))
