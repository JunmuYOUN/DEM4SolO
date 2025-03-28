import os
import astropy.io.fits as fits

import torch
import torch.nn as nn
import torch.nn.functional
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from glob import glob

# =============================

# 수정중=============================      
  
# TensorBoard 설정
# current_path = os.getcwd()
# log_dir = current_path + "/{}/logs/".format("#model_name")  # 로그 저장 경로
# os.makedirs(log_dir, exist_ok=True)
# writer = SummaryWriter(log_dir)

# # -----------------------------
# # CUDA Configuration
# # -----------------------------
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# device = torch.device('cuda:0')

# -----------------------------
# Dataset
# -----------------------------
class ImageDataset(Dataset):
    def __init__(self, image_root, trf_root, patch_size=5, patches_per_step=100000):

        self.image_root = image_root
        self.trf_root = trf_root
        self.patch_size = patch_size
        self.patches_per_step = patches_per_step
        self.images_list = sorted(glob(os.path.join(image_root, '*.fits')))

        tmp_wvl = os.path.basename(os.path.normpath(self.image_root))
        self.trf = np.load(os.path.join(self.trf_root, tmp_wvl + '.npy')).astype(np.float64)
        
        
        example_image = fits.open(self.images_list[0])[1].data
        H, W = example_image.shape
        self.H_patches = H - self.patch_size + 1
        self.W_patches = W - self.patch_size + 1
        self.patches_per_image = self.H_patches * self.W_patches
        
        self.steps_per_image = int(np.ceil(self.patches_per_image / self.patches_per_step))
        
    def __len__(self):
        return len(self.images_list) * self.steps_per_image  #이미지당 steps_per_image개

    def __getitem__(self, idx):
        
        image_idx = idx // self.steps_per_image
        step_idx = idx % self.steps_per_image
        
        # 이미지 로드
        image_path = self.images_list[image_idx]
        image = fits.open(image_path)[1].data
        
        
        start_patch = step_idx * self.patches_per_step
        end_patch = min(start_patch + self.patches_per_step, self.patches_per_image)
        current_step_patches = end_patch - start_patch
        
        # 패치 추출
        patches = []
        for patch_idx in range(start_patch, end_patch):
            i = patch_idx // self.W_patches
            j = patch_idx % self.W_patches
            patch = image[i:i+self.patch_size, j:j+self.patch_size]
            patch = patch[np.newaxis, :, :]  # (1, 5, 5)
            patches.append(patch)

        patches = np.stack(patches)  # (current_step_patches, 1, 5, 5)
        
        return patches.astype(np.float64), self.trf.astype(np.float64)


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
        

# =============================   
# Training
# =============================


def Train(num_epochs, train_loader, val_loader, device, patch_size=5, patches_per_step=100000):
    encoder = PixelDEMEncoder().to(device).double() #double
    cond_encoder = PixelConditionEncoder().to(device).double()
    decoder = PixelDEMDecoder().to(device).double()
    
    # DEM calculation init
    
    AIA_mlogt = np.arange(4.0, 8.05, 0.05) # if Tbin == (81,)    
    dlogT = []
    
    for i in range(len(AIA_mlogt)-1):
        delta = 10**(AIA_mlogt[i + 1]) - 10**(AIA_mlogt[i])
        dlogT.append(delta)
    dlogT = torch.tensor(dlogT, dtype=torch.float64, device=device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) +
        list(cond_encoder.parameters()) +
        list(decoder.parameters()), lr=1e-3)

    num_epochs=10
    
    for epoch in range(num_epochs):
        encoder.train()
        cond_encoder.train()
        decoder.train()
        
        
        train_loss = 0.0
        for img_patches, trf in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):

            img_patches = img_patches.squeeze(0).double().to(device)  # (100000, 1, 5, 5)
            trf = trf.double().to(device)                             # (1, 81)
        
            optimizer.zero_grad()
            
            trf_expanded = trf.expand(img_patches.size(0), -1)
            x1, x2, x3 = encoder(trf_expanded)
            cond_scalar = cond_encoder(img_patches)
            _, _, DEMs = decoder(x1, x2, x3, cond_scalar)

            # center pixel과 의 차이를 최소화하는 방향으로 학습
            center_idx = (patch_size**2) // 2
            img_patches_flat = img_patches.view(-1, patch_size**2)
            center_pixel = img_patches_flat[:, center_idx]
            
            predicted_pixel = torch.sum(DEMs * trf_expanded, dim=1)
            loss = criterion(predicted_pixel, center_pixel)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

        average_loss = loss.item() / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {average_loss:.5f}")
        writer.add_scalar('Loss/train', average_loss, epoch)

        if epoch % 3 == 0:
            try:
                torch.save({
                    'encoder': encoder.state_dict(),
                    'cond_encoder': cond_encoder.state_dict(),
                    'decoder': decoder.state_dict()
                }, current_path + '/{}/checkpoint/DEM4HRI/{}.pth'.format('#model', epoch))
            except:
                os.makedirs(current_path + '/{}/checkpoint/DEM4HRI/aaa.pth'.format('#model'))
                torch.save({
                    'encoder': encoder.state_dict(),
                    'cond_encoder': cond_encoder.state_dict(),
                    'decoder': decoder.state_dict()
                }, current_path + '/{}/checkpoint/DEM4HRI/{}'.format('#model', epoch))

                
        # validation
        encoder.eval()
        cond_encoder.eval()
        decoder.eval()

        
        val_loss = 0.0
        with torch.no_grad():
            for img_patches, trf in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                
                img_patches = img_patches.squeeze(0).double().to(device)  # (100000, 1, 5, 5)
                trf = trf.double().to(device)                             # (1, 81)

                optimizer.zero_grad()

                trf_expanded = trf.expand(img_patches.size(0), -1)
                x1, x2, x3 = encoder(trf_expanded)
                cond_scalar = cond_encoder(img_patches)
                _, _, DEMs = decoder(x1, x2, x3, cond_scalar)

                # center pixel과 의 차이를 최소화하는 방향으로 학습
                center_idx = (patch_size**2) // 2
                img_patches_flat = img_patches.view(-1, patch_size**2)
                center_pixel = img_patches_flat[:, center_idx]                
                
                #Temperature bin
                predicted_pixel = torch.sum(DEMs * trf_expanded, dim=1)
                loss = criterion(predicted_pixel, center_pixel)
                val_loss += loss.item()

            Val_avg_loss = loss.item() / len(dataloader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {Val_avg_loss:.5f}")
            writer.add_scalar('Loss/val', Val_avg_loss, epoch)



if __name__ == '__main__':
    
    current_path = os.getcwd()
    log_dir = os.path.join(current_path, "model_name/logs")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    image_root = '/userhome/youn_j/Dataset/HRI/L2_test/174/'
    trf_root = '/userhome/youn_j/Dataset/HRI/TRF/'

    batch_size=1
    patch_size=5
    num_epochs=100

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    device = torch.device('cuda:0')
    
    dataset = ImageDataset(image_root= image_root, 
                           trf_root= trf_root, 
                           patch_size=patch_size, 
                           patches_per_step=200000)
    
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle=True,
                            num_workers=8)
    

 
    Train(num_epochs=num_epochs, 
          train_loader=dataloader, 
          val_loader=dataloader,
          device=device, 
          patch_size=patch_size,
          patches_per_step=200000)

