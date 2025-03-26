import torch
import torch.nn as nn
import torch.nn.functional
from torch.utils.data import Dataset, DataLoader
import numpy as np

# import os
# import matplotlib.pyplot as plt

# from tqdm import tqdm
# from glob import glob
# from torchsummary import summary


# import argparse
# from torch.utils.tensorboard import SummaryWriter
# from pytorchtools import EarlyStopping
# import random

# # 무작위 샘플 선택 함수 (seed 해제 포함)
# def random_sample(predictions, num_samples=5):
#     random.seed(None)  # 랜덤 시드 해제
#     return random.sample(predictions, min(num_samples, len(predictions)))

# parser = argparse.ArgumentParser(description='')
# parser.add_argument('-lr', '--learningrate', default=0.001 , type=float)
# parser.add_argument('-t', '--threshold', default='3_sm_0.1', type= str )
# parser.add_argument('--smoothing', default=0.1, type=float, help="Label smoothing value")
# parser.add_argument('-a', '--alpha', default=0.5, type=float)
# args = parser.parse_args()


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda:0')

current_path = os.getcwd()
# EarlyStopping 경로 생성 확인 및 수정
early_stopping_path = current_path + '/{}/checkpoint/ResNeXt_best/'.format(str(args.learningrate) + '_' + str(args.threshold))
os.makedirs(early_stopping_path, exist_ok=True)
early_stopping = EarlyStopping(patience=20, verbose=True, path=early_stopping_path)


# 수정중=============================        
image_root = ./img
target_root = ./target
trf_root = ./trf/

class ImageDataset(Dataset):
    def __init__(self, image_root, target_root, trf_root, patch_size=5):

        self.image_root = image_root
        self.target_root = target_root
        self.trf_root = trf_root
        self.patch_size = patch_size

        self.images_list = sorted(glob(os.path.join(image_root, '/*.npy')))
        # self.target_list = sorted(glob(os.path.join(target_root, '/*.npy')))
        # self.trf_list = sorted(glob(os.path.join(trf_root, '/*.npy')))

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):

        image_path = self.image_list[idx]
        image = np.load(image_path)  # shape: (2048, 2048)
        H, W = image.shape

        # 이미지 패치 생성
        image_patches = []
        for i in range(H - self.patch_size + 1):
            for j in range(W - self.patch_size + 1):
                patch = image[i:i+self.patch_size, j:j+self.patch_size]
                patch = patch[np.newaxis, :, :]  # (1, 5, 5)
                image_patches.append(patch)

        return



    cond_patch = self.image_patches[patch_idx] # (1, 5, 5)
    dem_vector = self.loaded_trf[patch_idx] # (81,)

# -----------------------------
# Encoder
# -----------------------------

class PixelDEMEncoder(nn.Module):
    def __init__(self, TRF_size=81):
        super(PixelDEMEncoder, self).__init__()

        self.fc1 = nn.Linear(in_features = TRF_size, out_features = TRF_size//4) # 81>>20
        self.fc2 = nn.Linear(in_features = TRF_size//4, out_features = TRF_size//16) # 20>>5
        self.fc3 = nn.Linear(in_features = TRF_size//16, out_features = 1) # 5>>1
        self.act = nn.SiLU()
    
    def forward(self, x):
        x1 = self.act(self.fc1(x))
        x2 = self.act(self.fc2(x1))
        x3 = self.fc3(x2)

        return x1, x2, x3
    
# -----------------------------
# Condition Encoder (5x5 >> 1x1)
# -----------------------------    

class PixelConditionEncoder(nn.Module):
    def __init__(self):
        super(PixelConditionEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5) # 5x5>>1x1

    def forward(self, condition_img):
        c = self.conv1(condition_img)         # shape: (batch, 1, 1, 1)
        return c.view(c.size(0), -1) 

# -----------------------------
# Decoder
# -----------------------------    

class PixelDEMDecoder(nn.Module):
        def __init__(self, TRF_size=81, latent_dim=1, condition_dim=1):
            super(PixelDEMDecoder, self).__init__()

            self.fc1 = nn.Linear(in_features = condition_dim + latent_dim, out_features = TRF_size//16) # 1+1>>5
            self.fc2 = nn.Linear(in_features = TRF_size//16 + TRF_size//16, out_features = TRF_size//4) # 5+5>>20
            self.fc3 = nn.Linear(in_features = TRF_size//4 + TRF_size//4, out_features = TRF_size) #20+20>>81
            self.act = nn.SiLU()
        
        def forward(self, x1, x2, x3, cond_scalar):
            concat = torch.cat([x3, cond_scalar], dim=1)
            d1 = self.act(self.fc1(concat))
            d2 = self.act(self.fc2(torch.cat([d1, x2], dim=1)))
            d3 = self.fc3(torch.cat([d2, x1], dim=1))

            return d1, d2, d3
        

# temp    
encoder = PixelDEMEncoder()
decoder = PixelDEMDecoder()
cond_encoder = PixelConditionEncoder()

x1, x2, x3 = encoder(x)
cond_scalar = cond_encoder(condition_img) # (batch_size, 1)
d1, d2, d3 = decoder(x1, x2, x3, cond_scalar)


# 예시
train_dataset = ImageDataset(image_root='./img', trf_root='./trf', patch_size=5)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)  # 이미지 2장 단위로 불러옴

# =============================   
