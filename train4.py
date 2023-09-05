import torch
import time
from tqdm.auto import tqdm
import copy
import torch.nn as nn
import numpy as np
import pandas as pd
from dataloader_3 import *
from datetime import datetime

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, train_csv, valid_csv, train_path,valid_path,
                device='cpu', num_epochs=25, model_path=None):
    since = time.time() #시작 시간을 기록(총 소요 시간 계산을 위해)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in tqdm(range(num_epochs),desc="Epoch"):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1)) #epoch를 카운트
        print('-' * 10)
        start = time.time()
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:  #train mode와 validation mode 순으로 진행
            train_loss = 0.0
            train_corrects = 0.0
            if phase == 'train':
                
                data_info = pd.read_csv(train_csv, encoding = "cp949")
                # preds_list = []
                model.train()  # Set model to training mode
                print("train")
                for id_to_read, id_length, labels, over in tqdm(dataloaders[phase], desc="Iteration"): #dataloader로부터 dataset과 그에 해당되는 label을 불러
                    
                    img_list = []
                    for index,id in enumerate(id_to_read):
                        img_paths_to_read = list(data_info[data_info["unique_idx"]==id]["image_info"])
                        if over[index]:
                            img_paths_to_read = random.sample(img_paths_to_read, id_length[index])
                        for img_path in img_paths_to_read:
                            name_str = os.path.join(train_path, img_path)
                            img = Image.open(name_str)
                            img = img.resize((224, 224),Image.BICUBIC)
                            img = np.asarray(img)
                            img = TorchNorm(img)
                            img = random_lr_flip(img)
                            img = center_crop(img, 168)
                            img = HWC_to_CHW(img)

                            img_list.append(img)
                    
                    img_list = np.array(img_list)
                    inputs = torch.from_numpy(img_list.copy())    
                    #inputs = torch.squeeze(inputs)
                    inputs = inputs.to(device) #GPU로 입력데이터를 올림
                    #meta = torch.squeeze(meta)
                    id_length = id_length.to(device)
                    # meta = meta.to(device) 
                    labels = labels.to(device) #GPU로 label을 올림
                    # zero the parameter gradients
                    optimizer.zero_grad() #Gradient를 0으로 초기화
                    
                    with torch.set_grad_enabled(phase == 'train'):
                        # outputs = model(inputs, meta)
                        outputs = model(inputs, id_length)
                        _, preds = torch.max(outputs, 1) #마지막 layer에서 가장 값이 큰 1개의 class를 예측 값으로 지정
                     
                        loss = criterion(outputs, torch.max(labels, 1)[1])

                        # backward + optimize only if in training phase
                        loss.backward() #backward
                        optimizer.step()
                        scheduler.step()
                    #     preds_list.append(preds.item())

                    
                    train_loss += loss.item() * inputs.size(0)
                    train_corrects += torch.sum(preds == torch.where(labels.data==1)[1])
                epoch_loss1 = train_loss / dataset_sizes[phase]
                epoch_acc1 = train_corrects.double() / dataset_sizes[phase]
                
                print('Top1 : {} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss1, epoch_acc1))
            
            else:
                model.eval()
                val_loss = 0.0
                val_corrects = 0.0
                preds_list = []
                data_info = pd.read_csv(valid_csv, encoding = "cp949")
                for inputs, labels in tqdm(dataloaders[phase], desc="Iteration"): #dataloader로부터 dataset과 그에 해당되는 label을 불러
                    #inputs = torch.squeeze(inputs)
                    inputs = inputs.to(device) #GPU로 입력데이터를 올림
                    labels = labels.to(device) #GPU로 label을 올림
                    # zero the parameter gradients
                    optimizer.zero_grad() #Gradient를 0으로 초기화
                    
                    outputs = model(inputs, [1 for i in range(inputs.size(0))])
                    # import pdb;pdb.set_trace()
                    _, preds = torch.max(outputs, 1)        
                    loss = criterion(outputs, torch.max(labels, 1)[1]) 
                    val_loss += loss.item() * inputs.size(0)
                    val_corrects += torch.sum(preds == torch.where(labels.data==1)[1])
                    preds_list += preds.tolist()
                    
                preds_list = np.array(preds_list)
                pred_label_list = [np.where(preds_list == i)[0].shape[0] for i in range(9)]

                val_epoch_loss = val_loss / dataset_sizes[phase]
                val_epoch_acc = val_corrects.double() / dataset_sizes[phase]        
                if val_epoch_acc > best_acc:
                    path = os.path.join(model_path,'bestmodel_{}.pt'.format(datetime.today().day))
                    torch.save(model.state_dict(), path)
                    print(f"model saved at {path}")
                    best_acc = val_epoch_acc       
                    best_epoch = epoch
                print('[Valid] Top 1 | Loss: {:.4f} | Acc: {:.4f} | # Labels: {} | Best Acc: {:.4f} at epoch {}'.format(val_epoch_loss, val_epoch_acc, pred_label_list, best_acc, best_epoch))

        end = time.time()
        print('epoch train time : {:.4f} '.format(end-start))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

