import os

import torch
import torch.nn as nn
import torch.nn.functional
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from glob import glob


# import matplotlib.pyplot as plt


# from torchsummary import summary


# import argparse
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



# 
# # EarlyStopping 경로 생성 확인 및 수정
# early_stopping_path = current_path + '/{}/checkpoint/ResNeXt_best/'.format(str(args.learningrate) + '_' + str(args.threshold))
# os.makedirs(early_stopping_path, exist_ok=True)
# early_stopping = EarlyStopping(patience=20, verbose=True, path=early_stopping_path)


# 수정중=============================      


# TensorBoard 설정
current_path = os.getcwd()
log_dir = current_path + "/{}/logs/".format("#model_name")  # 로그 저장 경로
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir)

# -----------------------------
# CUDA Configuration
# -----------------------------
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda:0')

# -----------------------------
# Dataset
# -----------------------------
class ImageDataset(Dataset):
    def __init__(self, image_root, trf_root, patch_size=5):

        self.image_root = image_root
        self.target_root = target_root
        self.trf_root = trf_root
        self.patch_size = patch_size

        self.images_list = sorted(glob(os.path.join(image_root, '/*.npy')))


    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):

        image_path = self.images_list[idx]
        image = np.load(image_path)  # shape: (2048, 2048)
        H, W = image.shape

        # 이미지 패치 생성
        image_patches = []
        for i in range(H - self.patch_size + 1):
            for j in range(W - self.patch_size + 1):
                patch = image[i:i+self.patch_size, j:j+self.patch_size]
                patch = patch[np.newaxis, :, :]  # (1, 5, 5)
                image_patches.append(patch) 
        
        image_patches = np.stack(image_patches)  # (n, 1, 5, 5)

        #TRF 로드
        tmp_folder, _ = os.path.split(image_root)
        tmp_wvl = tmp_folder.split('/')[-1] # ex) 174

        trf = np.load(os.path.join(self.trf_root, tmp_wvl + '.npy'))  # shape: (81,)
        trf = trf.reshape(-1, 81)

        return image_patches, trf

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
        c = self.conv1(condition_img) # shape: (batch, 1, 1, 1)
        return c.view(c.size(0), -1) # shape: (batch, 1)

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


# =============================   
# Training
# =============================
image_root = ./img
target_root = ./target
trf_root = ./trf

batch_size=2
num_epochs=10

train_dataset = ImageDataset(image_root=image_root, trf_root=trf_root, patch_size=5)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

def Train(num_epochs, dataloader, device, patch_size=5):
    encoder = PixelDEMEncoder().to(device)
    cond_encoder = PixelConditionEncoder().to(device)
    decoder = PixelDEMDecoder().to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(cond_encoder.parameters()) + list(decoder.parameters()), lr=1e-3)


    for epoch in range(num_epochs):
        encoder.train()
        cond_encoder.train()
        decoder.train()

        for img_patches, trf in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            B, N, C, H, W = img_patches.shape
            img_patches = img_patches.view(-1, C, H, W).to(device)      # (B*N, 1, 5, 5)
            trf = trf.view(-1, 81).to(device)                   # (B*N, 81)

            x1, x2, x3 = encoder(trf)
            cond_scalar = cond_encoder(img_patches)
            _, _, DEMs = decoder(x1, x2, x3, cond_scalar)

            # center pixel과 의 차이를 최소화하는 방향으로 학습
            loss = criterion(torch.matmul(DEMs, trf.T), img_patches[(patch_size-1)//2:(patch_size-1)//2])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        average_loss = loss.item() / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
        writer.add_scalar('Loss/train', average_loss, epoch)

        if epoch % 3 == 0:
            try:
                torch.save({
                    'encoder': encoder.state_dict(),
                    'cond_encoder': cond_encoder.state_dict(),
                    'decoder': decoder.state_dict()
                }, current_path + '/{}/checkpoint/DEM4HRI/'.format('#model'))
            except:
                os.makedirs(current_path + '/{}/checkpoint/DEM4HRI/'.format('#model'))
                torch.save({
                    'encoder': encoder.state_dict(),
                    'cond_encoder': cond_encoder.state_dict(),
                    'decoder': decoder.state_dict()
                }, current_path + '/{}/checkpoint/DEM4HRI/'.format('#model'))

        # validation

        encoder.eval()
        cond_encoder.eval()
        decoder.eval()

        with torch.no_grad():
            for img_patches, trf in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                B, N, C, H, W = img_patches.shape
                img_patches = img_patches.view(-1, C, H, W).to(device)      # (B*N, 1, 5, 5)
                trf = trf.view(-1, 81).to(device)                   # (B*N, 81)

                x1, x2, x3 = encoder(trf)
                cond_scalar = cond_encoder(img_patches)
                _, _, DEMs = decoder(x1, x2, x3, cond_scalar)

                # center pixel과 의 차이를 최소화하는 방향으로 학습
                loss = criterion(torch.matmul(DEMs, trf.T), img_patches[(patch_size-1)//2:(patch_size-1)//2])

            average_loss = loss.item() / len(dataloader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
            writer.add_scalar('Loss/val', average_loss, epoch)

        

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = ImageDataset(image_root='./img', trf_root='./trf', patch_size=5)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    Train(num_epochs=num_epochs, dataloader=dataloader, device=device, patch_size=5)
