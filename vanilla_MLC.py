import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import platform
from time import *
import pandas as pd
from tqdm import tqdm, trange
import basic_net as basic_net
import torch.cuda
from torchvision.transforms import transforms
from tensorboardX import SummaryWriter
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
from MLC_dataset import *
from cls_net import *
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import argparse
import warnings
from vit_pytorch import ViT
warnings.filterwarnings(action='ignore')
""" classification"""


def maybe_to_torch(d):
    if isinstance(d, list):
        d = [maybe_to_torch(i) if not isinstance(i, torch.Tensor) else i for i in d]
    elif not isinstance(d, torch.Tensor):
        d = torch.from_numpy(d).float()
    return d

def metricCLS(y, y_pred):

    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    acc = accuracy_score(y, y_pred)
    kappa = cohen_kappa_score(y, y_pred)

    return acc, precision, recall, f1, kappa

EPOCH = 600
img_size = 512
num_class = 4
LR = 1 * 1e-4
train_BATCH_SIZE = 8
test_BATCH_SIZE = 1
dataset = "PLAG"
basic_model = "densenet"
basic_model = basic_model + '_' + dataset
device_ids = [6]
num_batches_per_epoch = 400
sysstr = platform.system()

root_path_train= './data_cls/train/PLAG'
root_path_test= './data_cls/test/PLAG'

def main( ):

    best_pre = 0.0
    if not os.path.exists("./saved_models/" + basic_model):
        if not os.path.exists("./saved_models"):
            os.mkdir("./saved_models")
        os.mkdir("./saved_models/" + basic_model)

    if not os.path.exists("./test_metrics/" + basic_model):
        if not os.path.exists("./test_metrics"):
            os.mkdir("./test_metrics")
        os.mkdir("./test_metrics/" + basic_model)
    txtfile = "./test_metrics/" + basic_model + "/results.txt"
    print("loading dataset-------------------------------------")
    dataloader = MLCDataLoader2D(root_path_train, train_BATCH_SIZE, (img_size, img_size), 4)
    tr_gen, _ = data_augmentation_moreDA.get_moreDA_augmentation(dataloader, dataloader, (img_size, img_size), aug_params)
    dr_dataset_train = MLCDataset(root=root_path_train, img_size=img_size, num_class=num_class)
    dr_dataset_test = MLCDataset(root=root_path_test, img_size=img_size, num_class=num_class)
    loader_test = torch.utils.data.DataLoader(dr_dataset_test, batch_size=test_BATCH_SIZE, num_workers=4, shuffle=False)
    net = clsNet(classes_num=num_class, basic_model=basic_model.split('_')[0], pretrain=True).cuda(device_ids[0])
    count=0
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    writer = SummaryWriter(log_dir='runs/vanilla_cls/' + basic_model)

    with open(txtfile, "a+") as f:
        for epoch in range(EPOCH):
            if epoch % 100 == 0:
                new_lr = LR / 2
                for para_group in optimizer.param_groups:
                    para_group['lr'] = new_lr
                print("Learning weight decays to %f"%(new_lr))
            writer.add_scalar('scalar/lr', optimizer.param_groups[0]['lr'] , epoch)
            _ = tr_gen.next()
            with trange(num_batches_per_epoch) as train_bar:
                for _ in train_bar:
                    count += 1
                    data_dict = next(tr_gen)
                    data = data_dict['data']
                    label = data_dict['label']
                    data = maybe_to_torch(data).cuda(device_ids[0])
                    label = maybe_to_torch(label).cuda(device_ids[0])
                    net.train()
                    outputs = net(data)
                    loss = F.binary_cross_entropy_with_logits(outputs, label, weight=torch.from_numpy(np.array([1.5, 1, 3, 1.5])).float().cuda(device_ids[0]))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    writer.add_scalar('scalar/CE_loss', loss.item(), count)
                    train_bar.set_description(
                        desc=basic_model + ' [%d/%d] ce_loss: %.4f ' % (
                            epoch, EPOCH,
                            loss.item(),
                        ))

            if epoch % 4 == 0:
                with torch.no_grad():
                    val_bar = tqdm(loader_test)
                    y1, y2, y3, y4= [], [], [], []
                    pred1, pred2, pred3, pred4 = [], [], [], []
                    for packs in val_bar:
                        input, labels = packs[0].cuda(device_ids[0]), packs[1].cuda(device_ids[0])
                        net.eval()
                        cls_out = net(input)
                        cls_out_sigmoid = F.sigmoid(cls_out)
                        cls_out_sigmoid = torch.where(cls_out_sigmoid> 0.5, 1, 0)
                        y1 = np.append(y1, labels[:, 0].cpu().numpy())
                        y2 = np.append(y2, labels[:, 1].cpu().numpy())
                        y3 = np.append(y3, labels[:, 2].cpu().numpy())
                        y4 = np.append(y4, labels[:, 3].cpu().numpy())
                        pred1 = np.append(pred1, cls_out_sigmoid[:, 0].cpu().numpy())
                        pred2 = np.append(pred2, cls_out_sigmoid[:, 1].cpu().numpy())
                        pred3 = np.append(pred3, cls_out_sigmoid[:, 2].cpu().numpy())
                        pred4 = np.append(pred4, cls_out_sigmoid[:, 3].cpu().numpy())

                    acc1 = metricCLS(y1, pred1)[0]
                    acc2 = metricCLS(y2, pred2)[0]
                    acc3 = metricCLS(y3, pred3)[0]
                    acc4 = metricCLS(y4, pred4)[0]
                    pre1 = metricCLS(y1, pred1)[4]
                    pre2 = metricCLS(y2, pred2)[4]
                    pre3 = metricCLS(y3, pred3)[4]
                    pre4 = metricCLS(y4, pred4)[4]
                    writer.add_scalar('scalar/acc', (acc1 + acc2 + acc3 + acc4)/4, epoch)
                    writer.add_scalar('scalar/pre', (pre1 + pre2 + pre3 + pre4)/4, epoch)
                    f.write("Test: EPOCH=%03d | OA1=%.3f%%, OA2=%.3f%%, OA3 =%.3f%%, OA4=%.3f%%" % (
                        epoch, acc1, acc2, acc3, acc4))
                    f.write("Test: EPOCH=%03d | Pe1=%.3f%%, Pe2=%.3f%%, Pe3 =%.3f%%, Pe4=%.3f%%" % (
                        epoch, pre1, pre2, pre3, pre4))

                    f.write('\n')
                    f.flush()

                    if (pre1 + pre2 + pre3 + pre4)/4 >= best_pre:
                        best_pre = (pre1 + pre2 + pre3 + pre4)/4
                        print("saving best model.....")
                        save = './saved_models/' + basic_model + '/best_cls.pth'
                        torch.save(net.state_dict(), save)

                    save= "./saved_models/" + basic_model + "/epoch_"+str(epoch)+'_cls.pth'
                    torch.save(net.state_dict(), save)
    f.close()



if __name__ == "__main__":

    init_seed=1217
    np.random.seed(init_seed)
    torch.manual_seed(init_seed)
    torch.cuda.manual_seed_all(init_seed)
    main()