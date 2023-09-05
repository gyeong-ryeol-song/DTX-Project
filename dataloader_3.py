#from cvxpy import deep_flatten
from ast import AsyncFunctionDef
import torch
import pandas as pd
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, models, transforms
import numpy as np
import os
import imageio
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
import torchvision.transforms.functional
import random
from PIL import Image
import copy
from typing import Callable

import pandas as pd
import torch
import torch.utils.data
import torchvision

def transform(args, mode):
    data_transform = {
                'train': transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize((args.image_resize,args.image_resize)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomCrop((args.crop_size, args.crop_size)),
                                        transforms.RandomRotation(90, expand=False),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                        ])
                ,
                'val': transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize((args.image_resize,args.image_resize)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                        ])
            }
    return data_transform[mode]

class TorchvisionNormalize():
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        imgarr = np.asarray(img)
        proc_img = np.empty_like(imgarr, np.float32)

        proc_img[..., 0] = (imgarr[..., 0] / 255. - self.mean[0]) / self.std[0]
        proc_img[..., 1] = (imgarr[..., 1] / 255. - self.mean[1]) / self.std[1]
        proc_img[..., 2] = (imgarr[..., 2] / 255. - self.mean[2]) / self.std[2]

        return proc_img
    
def TorchNorm(img):
    
    mean=(0.485, 0.456, 0.406) 
    std=(0.229, 0.224, 0.225)
    imgarr = np.asarray(img)
    proc_img = np.empty_like(imgarr, np.float32)

    proc_img[..., 0] = (imgarr[..., 0] / 255. - mean[0]) / std[0]
    proc_img[..., 1] = (imgarr[..., 1] / 255. - mean[1]) / std[1]
    proc_img[..., 2] = (imgarr[..., 2] / 255. - mean[2]) / std[2]

    return proc_img

class DTXDataset(Dataset):
    def __init__(self,csvpath,file_dir,to_torch=True,resize_long=(224,224)):
        self.to_torch = to_torch
        self.resize_long = resize_long
        
        df = pd.read_csv(csvpath,encoding="cp949")
        
        img_name_list = np.array(df["image_info"])
        
        df_y = torch.tensor(np.array(df["Label"]))
        y = F.one_hot(df_y.to(torch.int64), num_classes=df["Label"].nunique())
        self.img_name_list = img_name_list
        self.dir = file_dir
        
        self.y = y
        
    def __len__(self):
        return len(self.img_name_list)
    
    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        name_str = os.path.join(self.dir, name)
        img = np.asarray(imageio.imread(name_str))      
        label = self.y[idx]
        if self.to_torch:
            img = HWC_to_CHW(img)
        labels = copy.deepcopy(label)
        return {'img': torch.from_numpy(img.copy()), 'label': labels}

# class DTXMulti_Label_Dataset(Dataset):
#     def __init__(self,csvpath,file_dir,to_torch=True,resize_long=(224,224)):
#         self.to_torch = to_torch
#         self.resize_long = resize_long
#         df = pd.read_csv(csvpath,encoding="cp949")
#         img_name_list = np.array(df["image_info"])
#         df_y = df.iloc[:,2:]
#         self.img_name_list = img_name_list
#         self.dir = file_dir
#         self.y = df_y
        
        
#     def __len__(self):
#         return len(self.img_name_list)
    
    
#     def __getitem__(self, idx):
#         name = self.img_name_list[idx]
#         name_str = os.path.join(self.dir, name)
#         img = np.asarray(imageio.imread(name_str))
#         label = torch.tensor(self.y.iloc[idx][:])
#         if self.to_torch:
#             img = HWC_to_CHW(img)
#         labels = copy.deepcopy(label.float())
        
#         return {'img': torch.from_numpy(img.copy()), 'label': labels}
    
"""   
class DTX_dermameta_Dataset(Dataset):
    def __init__(self,csvpath,file_dir,to_torch=True,norm = TorchvisionNormalize()):
        self.df = pd.read_csv(csvpath,encoding="cp949")
        self.img_name_list = self.df["image_info"]
        self.site_list = self.df["site"]
        self.sex_list = self.df["sex"]
        self.age_list = self.df["age_"]
        self.id_list = self.df["con"].unique()
        self.norm = norm
        self.dir = file_dir
        self.to_torch = to_torch
        #self.transform = transform[self.mode]
        #assert len(self.id_list)==929

        # df_y = torch.tensor(np.array(self.df["Label"]))
        # y = F.one_hot(df_y.to(torch.int64), num_classes=self.df["Label"].nunique())
        # site = F.one_hot(df_y.to(torch.int64), num_classes=self.df["site"].nunique())    ##TODO: 희윤오면 바꿔주기
        # self.y=y

        
    def __len__(self):
        # return len(self.img_name_list)
        return len(self.id_list)
    
    def __getitem__(self, idx):
        # if self.first:
        #     Async
        # 인덱스로 유니크한 아이디를 불러오면
        # 그 아이디랑 같은 image_info를 갖다가 리스트화해서 하나씩 뽑은 담에 리스트를 리턴해
        # 
        # id_list_uniq = self.id_list.unique()
        id_to_read = self.id_list[idx]
        img_paths_to_read = list(self.df[self.df["con"]==id_to_read]["image_info"])
        meta = self.df[self.df['con'] == id_to_read].iloc[0:1, 16:] # 60개
        meta = np.array(meta)
        meta = torch.from_numpy(meta)
        meta = meta.to(torch.float32)
        label = self.df[self.df['con'] == id_to_read]['Label']
        
        
        img_list = []
        for img_path in img_paths_to_read:
            name_str = os.path.join(self.dir, img_path)
            img = Image.open(name_str)
            img = img.resize((448, 448),Image.BICUBIC)
            img = np.asarray(img)
            img = self.norm(img)
            img = random_lr_flip(img)
            img = center_crop(img, 336)
            if self.to_torch:
                img = HWC_to_CHW(img)
            

            img_list.append(img)
            
        img_list = np.array(img_list)
        
        # name = self.img_name_list[idx]
        # name_str = os.path.join(self.dir, name)
        # img = np.asarray(imageio.imread(name_str))
        # site = self.site[idx] 
        # label = self.y[idx]
        
        label = torch.tensor(np.array(label))
        label = F.one_hot(label.to(torch.int64), num_classes=self.df["Label"].nunique())
        labels = copy.deepcopy(label)
        labels = labels[0]
        
    
        img = torch.from_numpy(img_list.copy())
        
        return img, meta, labels
"""    

class DTX_dermameta_Dataset(Dataset):
    def __init__(self,csvpath,file_dir,to_torch=True,norm = TorchvisionNormalize()):
        self.df = pd.read_csv(csvpath,encoding="cp949")
        self.img_name_list = self.df["image_info"]
        self.site_list = self.df["site"]
        self.sex_list = self.df["sex"]
        self.age_list = self.df["age_"]
        self.id_list = self.df["con"].unique()
        self.norm = norm
        self.dir = file_dir
        self.to_torch = to_torch
        #self.transform = transform[self.mode]
        #assert len(self.id_list)==929

        # df_y = torch.tensor(np.array(self.df["Label"]))
        # y = F.one_hot(df_y.to(torch.int64), num_classes=self.df["Label"].nunique())
        # site = F.one_hot(df_y.to(torch.int64), num_classes=self.df["site"].nunique())    ##TODO: 희윤오면 바꿔주기
        # self.y=y

        
    def __len__(self):
        # return len(self.img_name_list)
        return len(self.id_list)
    
    def __getitem__(self, idx):
       
        # 인덱스로 유니크한 아이디를 불러오면
        # 그 아이디랑 같은 image_info를 갖다가 리스트화해서 하나씩 뽑은 담에 리스트를 리턴해
        # 
        # id_list_uniq = self.id_list.unique()
        id_to_read = self.id_list[idx]
        img_paths_to_read = list(self.df[self.df["con"]==id_to_read]["image_info"])
        id_length = len(img_paths_to_read)

        meta = self.df[self.df['con'] == id_to_read].iloc[0:1, 16:] # 60개
        meta = np.array(meta)
        meta = torch.from_numpy(meta)
        meta = meta.to(torch.float32)
        label = self.df[self.df['con'] == id_to_read]['Label']
        
        
        """
        img_list = []
        for img_path in img_paths_to_read:
            name_str = os.path.join(self.dir, img_path)
            img = Image.open(name_str)
            img = img.resize((448, 448),Image.BICUBIC)
            img = np.asarray(img)
            img = self.norm(img)
            img = random_lr_flip(img)
            img = center_crop(img, 336)
            if self.to_torch:
                img = HWC_to_CHW(img)
            

            img_list.append(img)
            
        img_list = np.array(img_list)
        """
        # name = self.img_name_list[idx]
        # name_str = os.path.join(self.dir, name)
        # img = np.asarray(imageio.imread(name_str))
        # site = self.site[idx] 
        # label = self.y[idx]
        
        label = torch.tensor(np.array(label))
        label = F.one_hot(label.to(torch.int64), num_classes=self.df["Label"].nunique())
        labels = copy.deepcopy(label)
        labels = labels[0]
        
        
        return id_to_read, id_length, meta, labels
    
class DTX_clinic_Dataset(Dataset):
    def __init__(self,csvpath,file_dir,max_len,to_torch=True):
        self.df = pd.read_csv(csvpath,encoding="cp949")
        self.img_name_list = self.df["image_info"]
        self.id_list = self.df["unique_idx"].unique()
        self.dir = file_dir
        self.to_torch = to_torch
        self.max_len = max_len
        self.over = False
    def __len__(self):
        # return len(self.img_name_list)
        return len(self.id_list)
    
    def __getitem__(self, idx):
       
        # 인덱스로 유니크한 아이디를 불러오면
        # 그 아이디랑 같은 image_info를 갖다가 리스트화해서 하나씩 뽑은 담에 리스트를 리턴해
        # 
        # id_list_uniq = self.id_list.unique()
        over = False
        id_to_read = self.id_list[idx]
        img_paths_to_read = list(self.df[self.df["unique_idx"]==id_to_read]["image_info"])
        id_length = len(img_paths_to_read)
        if id_length > self.max_len:
            id_length = self.max_len
            over = True
        label = self.df[self.df['unique_idx'] == id_to_read]['Label']
        
        label = torch.tensor(np.array(label))
        label = F.one_hot(label.to(torch.int64), num_classes=self.df["Label"].nunique())
        labels = copy.deepcopy(label)
        labels = labels[0]
        return id_to_read, id_length, labels, over
    
class DTX_clinic_Dataset(Dataset):
    def __init__(self,csvpath,file_dir,max_len,to_torch=True):
        self.df = pd.read_csv(csvpath,encoding="cp949")
        self.img_name_list = self.df["image_info"]
        self.id_list = self.df["unique_idx"].unique()
        self.dir = file_dir
        self.to_torch = to_torch
        self.max_len = max_len
        self.over = False
    def __len__(self):
        # return len(self.img_name_list)
        return len(self.id_list)
    
    def __getitem__(self, idx):
       
        # 인덱스로 유니크한 아이디를 불러오면
        # 그 아이디랑 같은 image_info를 갖다가 리스트화해서 하나씩 뽑은 담에 리스트를 리턴해
        # 
        # id_list_uniq = self.id_list.unique()
        over = False
        id_to_read = self.id_list[idx]
        img_paths_to_read = list(self.df[self.df["unique_idx"]==id_to_read]["image_info"])
        id_length = len(img_paths_to_read)
        if id_length > self.max_len:
            id_length = self.max_len
            over = True
        label = self.df[self.df['unique_idx'] == id_to_read]['Label']
        
        label = torch.tensor(np.array(label))
        label = F.one_hot(label.to(torch.int64), num_classes=self.df["Label"].nunique())
        labels = copy.deepcopy(label)
        labels = labels[0]
        return id_to_read, id_length, labels, over
    
class DTXNorm(Dataset):
    def __init__(self,dataset,transforms):
        self.data = dataset
        self.transform = transforms
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = self.transform(self.data[idx]["img"])
        label = self.data[idx]["label"]
        
        return img,label

# def pil_resize(img, size, order):
#     if size[0] == img.shape[0] and size[1] == img.shape[1]:
#         return img

#     if order == 3:
#         resample = Image.BICUBIC
#     elif order == 0:
#         resample = Image.NEAREST

#     return np.asarray(Image.fromarray(img).resize(size[::-1], resample))


def HWC_to_CHW(img):
    return np.transpose(img, (2, 0, 1))






class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices: a list of indices
        num_samples: number of samples to draw
        callback_get_label: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(
        self,
        dataset,
        labels: list = None,
        indices: list = None,
        num_samples: int = None,
        callback_get_label: Callable = None,
        path = None
    ):
        self.csvpath = path
        
        df = pd.read_csv(self.csvpath,encoding="cp949")
        df_y = torch.tensor(np.array(df["Label"]))
        # if indices is not provided, all elements in the dataset will be considered
        
        self.indices = list(range(len(dataset))) if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples
        self.num_samples *= 4
        # distribution of classes in the dataset
        df = pd.DataFrame()
        df["label"] = df_y
        df.index = self.indices
        df = df.sort_index()
        #import pdb;pdb.set_trace()
        label_to_count = df["label"].value_counts()

        weights = 1.0 / label_to_count[df["label"]]

        self.weights = torch.DoubleTensor(weights.to_list())
   
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


def make_weights_for_balanced_classes(dataset, nclasses):
    n_images = len(dataset)
    count_per_class = [0] * nclasses
    for _, image_class in dataset:
        
        count_per_class[image_class.argmax()] += 1
    weight_per_class = [0.] * nclasses
    for i in range(nclasses):
        weight_per_class[i] = float(n_images) / float(count_per_class[i])
    weights = [0] * n_images
    for idx, (image, image_class) in enumerate(dataset):
        weights[idx] = weight_per_class[image_class.argmax()]
    return weights




def pil_resize(img, size, order):
    if size[0] == img.shape[0] and size[1] == img.shape[1]:
        return img

    if order == 3:
        resample = Image.BICUBIC
    elif order == 0:
        resample = Image.NEAREST

    return np.asarray(Image.fromarray(img).resize(size[::-1], resample))

def pil_rescale(img, scale, order):
    height, width = img.shape[:2]
    target_size = (int(np.round(height*scale)), int(np.round(width*scale)))
    return pil_resize(img, target_size, order)

def random_resize_long(img, min_long, max_long):
    target_long = random.randint(min_long, max_long)
    h, w = img.shape[:2]

    if w < h:
        scale = target_long / h
    else:
        scale = target_long / w

    return pil_rescale(img, scale, 3)

def random_scale(img, scale_range, order):

    target_scale = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])

    if isinstance(img, tuple):
        return (pil_rescale(img[0], target_scale, order[0]), pil_rescale(img[1], target_scale, order[1]))
    else:
        return pil_rescale(img[0], target_scale, order)

def random_lr_flip(img):

    if bool(random.getrandbits(1)):
        if isinstance(img, tuple):
            return [np.fliplr(m) for m in img]
        else:
            return np.fliplr(img)
    else:
        return img

def get_random_crop_box(imgsize, cropsize):
    h, w = imgsize

    ch = min(cropsize, h)
    cw = min(cropsize, w)

    w_space = w - cropsize
    h_space = h - cropsize

    if w_space > 0:
        cont_left = 0
        img_left = random.randrange(w_space + 1)
    else:
        cont_left = random.randrange(-w_space + 1)
        img_left = 0

    if h_space > 0:
        cont_top = 0
        img_top = random.randrange(h_space + 1)
    else:
        cont_top = random.randrange(-h_space + 1)
        img_top = 0

    return cont_top, cont_top+ch, cont_left, cont_left+cw, img_top, img_top+ch, img_left, img_left+cw

def random_crop(images, cropsize, default_values):

    if isinstance(images, np.ndarray): images = (images,)
    if isinstance(default_values, int): default_values = (default_values,)

    imgsize = images[0].shape[:2]
    box = get_random_crop_box(imgsize, cropsize)

    new_images = []
    for img, f in zip(images, default_values):

        if len(img.shape) == 3:
            cont = np.ones((cropsize, cropsize, img.shape[2]), img.dtype)*f
        else:
            cont = np.ones((cropsize, cropsize), img.dtype)*f
        cont[box[0]:box[1], box[2]:box[3]] = img[box[4]:box[5], box[6]:box[7]]
        new_images.append(cont)

    if len(new_images) == 1:
        new_images = new_images[0]

    return new_images

def top_left_crop(img, cropsize, default_value):

    h, w = img.shape[:2]

    ch = min(cropsize, h)
    cw = min(cropsize, w)

    if len(img.shape) == 2:
        container = np.ones((cropsize, cropsize), img.dtype)*default_value
    else:
        container = np.ones((cropsize, cropsize, img.shape[2]), img.dtype)*default_value

    container[:ch, :cw] = img[:ch, :cw]

    return container

def center_crop(img, cropsize, default_value=0):

    h, w = img.shape[:2]

    ch = min(cropsize, h)
    cw = min(cropsize, w)

    sh = h - cropsize
    sw = w - cropsize

    if sw > 0:
        cont_left = 0
        img_left = int(round(sw / 2))
    else:
        cont_left = int(round(-sw / 2))
        img_left = 0

    if sh > 0:
        cont_top = 0
        img_top = int(round(sh / 2))
    else:
        cont_top = int(round(-sh / 2))
        img_top = 0

    if len(img.shape) == 2:
        container = np.ones((cropsize, cropsize), img.dtype)*default_value
    else:
        container = np.ones((cropsize, cropsize, img.shape[2]), img.dtype)*default_value

    container[cont_top:cont_top+ch, cont_left:cont_left+cw] = \
        img[img_top:img_top+ch, img_left:img_left+cw]

    return container