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
from nnunet.training.data_augmentation import data_augmentation_moreDA
import torch.cuda
from cls_net import clsNet
from cls_dataset import ClsDataset
from torchvision.transforms import transforms
from tensorboardX import SummaryWriter
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
from cls_dataset import *
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import argparse
import warnings
warnings.filterwarnings(action='ignore')
""" classification"""

def maybe_to_torch(d):
    if isinstance(d, list):
        d = [maybe_to_torch(i) if not isinstance(i, torch.Tensor) else i for i in d]
    elif not isinstance(d, torch.Tensor):
        d = torch.from_numpy(d).float()
    return d

def metricCLS(y, y_pred):

    precision = precision_score(y, y_pred, average='macro')
    recall = recall_score(y, y_pred, average="macro")
    f1 = f1_score(y, y_pred, average="macro")
    acc = accuracy_score(y, y_pred)
    kappa = cohen_kappa_score(y, y_pred)

    return acc, precision, recall, f1, kappa


EPOCH = 600
img_size = 512
num_class = 5
LR = 1 * 1e-4
train_BATCH_SIZE = 16
test_BATCH_SIZE = 1
dataset = ""
basic_model = "rescbam"
basic_model = basic_model + '_' + dataset
device_ids = [0]
num_batches_per_epoch = 300
sysstr = platform.system()

root_path_train= './data_cls/train/PLAG'
root_path_test= './data_cls/test/PLAG'


def main( ):

    best_OA = 0.0
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

    dataloader = ClsDataLoader2D(root_path_train, train_BATCH_SIZE, (img_size, img_size), num_class, 4)
    tr_gen, _ = data_augmentation_moreDA.get_moreDA_augmentation(dataloader, dataloader, (img_size, img_size), aug_params)
    dr_dataset_test = ClsDataset(root=root_path_test, img_size=img_size, num_class=num_class)
    loader_test = torch.utils.data.DataLoader(dr_dataset_test, batch_size=test_BATCH_SIZE, num_workers=4, shuffle=False)
    net = clsNet(classes_num=num_class, basic_model='resnet_cbam', pretrain=False).cuda(device_ids[0])
    count=0
    CE_criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1.6, 1, 1, 1.8 ,1.3])).float().cuda(device_ids[0])).cuda(device_ids[0])
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    writer = SummaryWriter(log_dir='runs/vanilla_cls/' + basic_model)

    with open(txtfile, "a+") as f:
        for epoch in range(1,EPOCH+1):
            if epoch % 100 == 0:
                new_lr = LR / 2
                for para_group in optimizer.param_groups:
                    para_group['lr'] = new_lr
                print("Learning weight decays to %f"%(new_lr))
            writer.add_scalar('scalar/lr', optimizer.param_groups[0]['lr'] , epoch)
            _ = tr_gen.next()
            with trange(num_batches_per_epoch) as train_bar:
                for _ in train_bar:
                    data_dict = next(tr_gen)
                    data = data_dict['data']
                    label = data_dict['label']
                    data = maybe_to_torch(data).cuda(device_ids[0])
                    label = maybe_to_torch(label).long().cuda(device_ids[0])
                    net.train()
                    optimizer.zero_grad()
                    outputs = net(data)
                    loss = CE_criterion(outputs, label)
                    _, predicted = torch.max(outputs.data, 1)
                    loss.backward()
                    optimizer.step()
                    total = label.size(0)
                    correct = predicted.eq(label.data).cpu().sum()

                    writer.add_scalar('scalar/loss',loss.item(), count)
                    writer.add_scalar('scalar/CE_loss', CE_criterion(outputs, label).item(), count)
                    writer.add_scalar('scalar/Acc_batchwise', 100.*correct/total, count)
                    train_bar.set_description(
                        desc=basic_model + ' [%d/%d] ce_loss: %.4f  | batch_acc: %.4f' % (
                            epoch, EPOCH,
                            CE_criterion(outputs, label).item(),
                            100.*correct/total,
                        ))

            if epoch % 4 == 0:
                with torch.no_grad():
                    val_bar = tqdm(loader_test)
                    y = []
                    pred = []
                    for packs in val_bar:
                        input, labels = packs[0].cuda(device_ids[0]), packs[1].cuda(device_ids[0])
                        net.eval()
                        cls_out = net(input)
                        _, predicted = torch.max(cls_out.data, 1)
                        y = np.append(y, labels.cpu().numpy())
                        pred = np.append(pred, predicted.cpu().numpy())
                    acc, precision, recall, f1, kappa = metricCLS(y, pred)

                    print("Test: %s EPOCH=%03d | OA=%.3f%%, Precision=%.3f%%, Recall =%.3f%%, F1=%.3f%%, kappa=%.3f%%" % (
                    basic_model, epoch, acc, precision, recall, f1, kappa))
                    writer.add_scalar('scalar/acc', acc, epoch)
                    writer.add_scalar('scalar/f1', f1, epoch)
                    f.write("Test: EPOCH=%03d | OA=%.3f%%, Precision=%.3f%%, Recall =%.3f%%, F1=%.3f%%, kappa=%.3f%%" % (
                        epoch , acc, precision, recall, f1, kappa))

                    f.write('\n')
                    f.flush()

                    if acc > best_OA:
                        best_OA = acc
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