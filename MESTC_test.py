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
from cls_net import *
import numpy as np
Image.MAX_IMAGE_PIXELS = None
from skimage import io
import warnings
import cv2
warnings.filterwarnings(action='ignore')

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
        if test_center == 'CJF':
            mask[mask_ori == 0] = 1
            mask[mask_ori == 1] = 0
        return mask

def glo_predict(img, device_ids):

    threshold = [0.5, 0.5, 0.5, 0.5]
    mesc_pred = [0, 0, 0, 0]
    net = clsNet(classes_num=4, basic_model="densenet", pretrain=True).cuda(device_ids[0])
    net.load_state_dict(torch.load('MESC.pth', map_location='cpu'))
    net.eval()
    img = cv2.resize(img, (512, 512))
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(np.array(img.astype(np.dtype(np.float32)))).unsqueeze(0).cuda(device_ids[0])
    outputs = net(img)
    for idx in range(4):
        cls_out_sigmoid = F.sigmoid(outputs[0][idx])
        cls_out_sigmoid_per = torch.where(cls_out_sigmoid > threshold[idx], 1, 0)
        mesc_pred[idx] = cls_out_sigmoid_per.item()
    return mesc_pred

def rotate(mask):
    dst1 = cv2.rotate(mask, 0)
    v_flip = cv2.flip(dst1, 1)
    return v_flip

from utils import label_color
def vis_binary_map(mask, save_path):
    mask_vis = np.zeros((mask.shape[0], mask.shape[1], 3))
    for ch in range(13):
        mask_vis[mask == ch] = label_color[ch]
    cv2.imwrite(save_path, mask_vis)

if __name__ == "__main__":

    import pandas as pd
    root_pred = './data_test/segcls/'
    img_path = './data_test/img/'
    save_path = './data_test/'
    seg_class = ['M', 'E', 'S', 'C', 'IG', 'All', 'T',]
    rank= {'Background': 0, '_background_': 1, 'Normal Glomeruli': 2,
            'Sclerotic Glomeruli': 3, 'Incomplete Glomeruli': 4, 'Other Glomeruli': 5, 'MESC': 6,
            'Atrophic Tubuli/Interstitial Fibrosis': 7, 'Interstitial Inflammation': 8,
            'Medium-Small Sized Artery': 9,
            'Arteriole': 10, 'Vein': 11, 'Medulla': 12}
    test_files = [file for file in os.listdir(root_pred)]
    result_mestc = pd.DataFrame(columns=seg_class, index=test_files)
    device_ids = [0]
    for i in trange(len(result_mestc)):
        MESC_number = [0, 0, 0, 0]
        seg_name = result_mestc.index[i]
        seg_pred = io.imread(root_pred + seg_name, 0)
        seg_pred_post = io.imread(root_pred + seg_name, 0)
        WSI = io.imread(img_path + seg_name)
        WSI = cv2.cvtColor(WSI, cv2.COLOR_RGB2BGR)
        # if seg_pred.shape != (WSI.shape[0], WSI.shape[1]):
        #     WSI = cv2.resize(WSI, (seg_pred.shape[1], seg_pred.shape[0]))
        mesc_mask = np.where(seg_pred == 6, 255, 0)
        normal_mask = np.where(seg_pred == 2, 255, 0)
        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(np.uint8(mesc_mask))
        for c in range(1, retval):
            if stats[c, 4] > 300:
                img = WSI[stats[c, 1]:stats[c, 1] + stats[c, 3], stats[c, 0]:stats[c, 0] + stats[c, 2]]
                mesc_pred = glo_predict(img, device_ids)
                if np.sum(np.array([mesc_pred])) == 0:
                    seg_pred[labels == c] = 2
                else:
                    MESC_number[0] += mesc_pred[0]
                    MESC_number[1] += mesc_pred[1]
                    MESC_number[2] += mesc_pred[2]
                    MESC_number[3] += mesc_pred[3]
            else:
                seg_pred_post[labels == c] = 1
        glo_all, IG_all = 0, 0
        for glo_idx in range(2, 7):
            glo_mask_idx = np.where(seg_pred==glo_idx, 255, 0)
            ret, thresh = cv2.threshold(np.uint8(glo_mask_idx), 127, 255, cv2.THRESH_BINARY)
            if np.mean(thresh) != 0:
                retval, labels, stats, centroids = cv2.connectedComponentsWithStats(np.uint8(thresh))
                for c in range(1, retval):
                    if stats[c, 4] > 300:
                        # contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
                        glo_all = glo_all + 1
                    else:
                        seg_pred_post[labels == c] = 1
        glo_mask_idx = np.where(seg_pred == 4, 255, 0)
        ret, thresh = cv2.threshold(np.uint8(glo_mask_idx), 127, 255, cv2.THRESH_BINARY)
        if np.mean(thresh) != 0:
            retval, labels, stats, centroids = cv2.connectedComponentsWithStats(np.uint8(thresh))
            for c in range(1, retval):
                if stats[c, 4] > 300:
                    # contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
                    IG_all = IG_all + 1
                else:
                    seg_pred_post[labels == c] = 1

        result_mestc.iloc[i, 0] = MESC_number[0]
        result_mestc.iloc[i, 1] = MESC_number[1]
        result_mestc.iloc[i, 2] = MESC_number[2]
        result_mestc.iloc[i, 3] = MESC_number[3]
        result_mestc.iloc[i, 4] = IG_all
        result_mestc.iloc[i, 5] = glo_all

        medulla_mask = np.where(seg_pred == 12, 1, 0)
        Bg_mask = np.where(seg_pred == 0, 1, 0)
        fore = seg_pred.shape[0] * seg_pred.shape[1] - np.sum(medulla_mask) - np.sum(Bg_mask)
        ATIF = np.where(seg_pred == 7, 1, 0)
        ratio = np.sum(ATIF) / fore
        result_mestc.iloc[i, 6] = ratio
        result_mestc.to_excel(save_path + 'MESC_score.xlsx')

