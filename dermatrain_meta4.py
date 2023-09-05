import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tqdm.auto import tqdm
import torchvision.transforms as transforms
import datetime
import os
from multiprocessing import Process, freeze_support
import sys
import numpy as np
import torchvision.models as models
import time
import copy
import matplotlib.pyplot as plt
# from dataloader import DTXDataset, transform, DTXNorm, ImbalancedDatasetSampler
from dataloader_4 import DTX_clinic_Dataset, DTXDataset
from torch.utils.data import Dataset, DataLoader, random_split
#import torchmetrics
import copy
from torch import topk
from tqdm.auto import tqdm
import argparse
from model3 import *
from train4 import train_model
import datetime
from efficientnet_pytorch import EfficientNet


### without META DATA
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
os.chdir(current_directory)
parser = argparse.ArgumentParser()
parser.add_argument('--image_resize',default=640, type=int, metavar = 'N')
parser.add_argument('--max_len',default=6, type=int, metavar = 'N')
parser.add_argument('--num_classes',default=9, type=int, metavar = 'N')
parser.add_argument('--crop_size',default=480, type=int, metavar = 'N')
parser.add_argument('--train_data_path',default='../../clinic_data/train1618', type=str, metavar = 'N')
parser.add_argument('--valid_data_path',default='../../clinic_data/clinic_data2022', type=str, metavar = 'N')
parser.add_argument('--model_path',default='./models/', type=str, metavar = 'N')
parser.add_argument('--train_csv',default='unique1618.csv', type=str, metavar = 'N')
parser.add_argument('--valid_csv',default='unique20.csv', type=str, metavar = 'N')
parser.add_argument('--epoch',default=50, type=int, metavar = 'N')
parser.add_argument('--gpu_id', default=0)
args = parser.parse_args()


if torch.cuda.is_available():
    device = f"cuda:{args.gpu_id}"
else:
    device = 'cpu'

train_data = DTX_clinic_Dataset(args.train_csv,args.train_data_path,max_len = args.max_len)
valid_data = DTXDataset(args.valid_csv,args.valid_data_path)
# import pdb;pdb.set_trace()
num_classes = args.num_classes
# meta_size = train_data[0][2].size(dim=1)
# import pdb;pdb.set_trace()
image_datasets = {'train':train_data, 'val':valid_data}

dataloaders = { 'train' : DataLoader(image_datasets["train"], batch_size=6,
                                     num_workers=2, shuffle=True),
                'val' : DataLoader(image_datasets['val'], batch_size=12,
                                   #sampler=ImbalancedDatasetSampler(image_datasets['val']),
                                    num_workers=2,  shuffle=True) }

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

###########################################################################################

# resnet50_pretrained = models.resnet50(pretrained=True)   
# efficientnet = EfficientNet.from_pretrained("efficientnet-b4", advprop=True)
efficientnet = EfficientNet.from_pretrained("efficientnet-b3", advprop=True)
model = DTmodel1(cnn_model = efficientnet, img_size = args.image_resize, n_labels = num_classes)

model.to(device)

print("num_classses: ", num_classes)
criterion = FocalLoss(gamma=0)
# criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(resnet18_pretrained.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=0.0005)
cs_scheduler =optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
best_acc = 0



if __name__ == '__main__':
    epochs = args.epoch
    path = '../models/'
    save_model_name = f"efficient_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ptpath = os.path.join(path, save_model_name)
    model_ft = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, cs_scheduler,args.train_csv, args.valid_csv, 
                           args.train_data_path, args.valid_data_path,device = device, num_epochs=epochs, model_path = args.model_path)
    torch.save(model_ft.state_dict(), ptpath)

