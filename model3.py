import torch.nn as nn
import torch

import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        # if isinstance(alpha,(float,int,long)): self.alpha = torch.Tensor([alpha,1-alpha])
        # if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()



class DTmodel(nn.Module):
    def __init__(self, cnn_model,img_size, meta_size=None, n_labels=9):
        super(DTmodel, self).__init__()
        self.img_size = img_size
        self.meta_size = meta_size
        # self.out = cnn_model.fc.out_features # 1000 
        self.meta_out = 20
        
        self.image_encoder = cnn_model
        self.flatten = nn.Flatten()
        self.classification_head = nn.Linear(38400, n_labels)

    def forward(self, image, id_length):
        output = []
        idx = 0
        for i in id_length:
            img_feat = self.image_encoder.extract_features(image[idx:idx + i,: , :, :])
            
            img_feat = self.flatten(img_feat)
            features_mean = img_feat.mean(dim=0)
            x = self.classification_head(features_mean)
            output.append(x)
            idx += i

        output = torch.stack(output, 0)   # [4, 1, 1000]
        # x = torch.concat([features_mean], dim=1)
        
        
        return x

class DTmodel1(nn.Module):
    def __init__(self, cnn_model, img_size, n_labels=9):
        super(DTmodel1, self).__init__()
        self.img_size = img_size
        self.img_output_size = 38400
        
        self.image_encoder = cnn_model
        self.flatten = nn.Flatten()
        self.classification_head = nn.Linear(self.img_output_size, n_labels)

    def forward(self, image, id_length):
        output = []
        pointer = 0
        idx = 0
        img_feat = self.image_encoder.extract_features(image)
        
        for i in id_length:
            hidden = self.flatten(img_feat[pointer:pointer + i,:,:,:])
            hidden = hidden.mean(dim=0)
            x = self.classification_head(hidden)
            output.append(x)
            pointer += i
            idx += 1
        
        output = torch.stack(output, 0)
        return output