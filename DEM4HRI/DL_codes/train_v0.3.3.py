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

import warnings
warnings.simplefilter("once")

# =============================

class ImageDataset(Dataset):
    def __init__(self, image_root, trf_root, patch_size=5, batch_size = 1, patches_per_step=1):

        self.image_root = image_root
        self.trf_root = trf_root
        self.patch_size = patch_size
        self.patches_per_step = patches_per_step
        self.images_list = sorted(glob(os.path.join(image_root, '*.npy')))
        _, ext = os.path.splitext(self.images_list[0])
        self.batch_size = batch_size

        tmp_wvl = os.path.basename(os.path.normpath(self.image_root))
        self.trf = np.load(os.path.join(self.trf_root, tmp_wvl + '.npy')).astype(np.float64)
        
        if ext == '.fits':
            example_image = fits.open(self.images_list[0])[1].data
        elif ext == '.npy':
            example_image = np.load(self.images_list[0])
        else:
            raise ValueError(f'Unsupported file extension: {ext}')

        H, W = example_image.shape
        self.H_patches = H - self.patch_size + 1
        self.W_patches = W - self.patch_size + 1
        self.patches_per_image = self.H_patches * self.W_patches
        self.patches_per_step = patches_per_step
        
        if self.patches_per_image >= self.patches_per_step:
            if self.patches_per_image % self.patches_per_step != 0:
                warnings.warn(f"steps (patches) does not divided with potential patches in image: \n {self.patches_per_image}/{self.patches_per_step} \n will be set to the maximum number of patches per steps")
                self.patches_per_step = self.patches_per_image
                self.total_patches = len(self.images_list) * self.patches_per_image // self.patches_per_step
            else:
                self.total_patches = len(self.images_list) * self.patches_per_image // self.patches_per_step
        else:
            warnings.warn(f"steps (patches) does not divided with potential patches in image: \n {self.patches_per_image}/{self.patches_per_step} \n will be set to the maximum number of patches per steps")

            self.total_steps = len(self.images_list) * self.patches_per_image // self.patches_per_step

    def __len__(self):
        return len(self.images_list)
        # return self.total_patches

    def __getitem__(self, idx):
        # 한 스텝에서 몇 개의 패치를 뽑는지
        patches_per_step = self.patches_per_step

        # 한 이미지에서 나올 수 있는 전체 패치 수
        patches_per_image = self.patches_per_image

        # 이 인덱스가 어떤 이미지에 속하는지 계산
        image_idx = idx   #(idx * patches_per_step) // patches_per_image

        # 그 이미지 안에서 시작 패치 번호
        start_patch = (idx * patches_per_step) % patches_per_image
        end_patch = start_patch + patches_per_step

        # 이미지 로드
        image_path = self.images_list[image_idx]
        _, ext = os.path.splitext(image_path)

        if ext == '.fits':
            image = fits.open(image_path)[1].data
        elif ext == '.npy':
            image = np.load(image_path)
        else:
            raise ValueError(f'Unsupported file extension: {ext}')
        
        # 패치 추출
        total_indices = np.arange(patches_per_image)
        selected_indices = np.random.choice(total_indices, size=patches_per_step, replace=False)
        patches = []
        for patch_idx in selected_indices:
            i = patch_idx // self.W_patches
            j = patch_idx % self.W_patches
            patch = image[i:i + self.patch_size, j:j + self.patch_size]
            patch = patch[np.newaxis, :, :]
            patches.append(patch)

        patches = np.stack(patches)  # (patches_per_step, 1, 5, 5)
        
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
    def __init__(self, kernel_size=3, stride=1, padding=0, dilation=1):
        # 이후 모델에 추가할 것 : stride=1, padding=0, dilation=1

        super(PixelConditionEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size) # 5x5>>1x1
        self.pool = nn.AdaptiveAvgPool2d((1, 1)) # 1x1


    def forward(self, condition_img):
        c = self.conv1(condition_img) # shape: (batch, 1, 1, 1)
        c = self.pool(c)  # shape: (batch, 1, 1, 1) 
        return c.view(c.size(0), -1) # shape: (batch, 1)

# -----------------------------
# Decoder
# -----------------------------    

class PixelDEMDecoder(nn.Module):
        def __init__(self, TRF_size=81, latent_dim=1, condition_dim=1):
            super(PixelDEMDecoder, self).__init__()

            self.fc1 = nn.Linear(in_features = condition_dim + latent_dim, out_features = TRF_size//16) # 1+1>>5
            self.fc2 = nn.Linear(in_features = TRF_size//16 + TRF_size//16, out_features = TRF_size//4) # 5+5>>20
            self.fc3 = nn.Linear(in_features = TRF_size//4 + TRF_size//4, out_features = TRF_size) #20+20>>81-1
            self.fc4 = nn.Linear(in_features = TRF_size, out_features = TRF_size-1) # 80
            self.act = nn.SiLU()
        
        def forward(self, x1, x2, x3, cond_scalar):
            concat = torch.cat([x3, cond_scalar], dim=1)
            d1 = self.act(self.fc1(concat))
            d2 = self.act(self.fc2(torch.cat([d1, x2], dim=1)))
            d3 = self.act(self.fc3(torch.cat([d2, x1], dim=1)))
            d4 = self.fc4(d3)
            return d1, d2, d3, d4

# =============================   
# Training
# =============================


def Train(num_epochs, train_loader, val_loader, device, patch_size=5, patches_per_step=0, kernel_size=5):
    encoder = PixelDEMEncoder().to(device).double() #double
    cond_encoder = PixelConditionEncoder(kernel_size=kernel_size).to(device).double()
    decoder = PixelDEMDecoder().to(device).double()
    
    # DEM calculation init
    AIA_mlogt = np.arange(4.0, 8.0 + 1e-5, 0.05) # if Tbin == (80,)    
    dlogT = []
    
    for i in range(len(AIA_mlogt)-1):
        delta = 10**(AIA_mlogt[i + 1]) - 10**(AIA_mlogt[i])
        dlogT.append(delta)
    dlogT = np.array(dlogT)
    # print(dlogT.shape)
    dlogT = torch.tensor(dlogT, dtype=torch.float64, device=device)
    
    criterion = nn.MSELoss()
    encoder_lr = 5e-5
    cond_encoder_lr = 1e-2
    decoder_lr = 5e-2

    optimizer = torch.optim.Adam([
        {'params': encoder.parameters(),       'lr': encoder_lr},
        {'params': cond_encoder.parameters(), 'lr': cond_encoder_lr},
        {'params': decoder.parameters(),      'lr': decoder_lr}])

    # num_epochs=10
    
    for epoch in range(num_epochs):
        encoder.train()
        cond_encoder.train()
        decoder.train()
            
        train_loss = 0.0
        i = 0
        for img_patches, trf in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):

            img_patches = img_patches.squeeze(0).double().to(device)   #
            trf = trf.double().to(device)                          
            trf_expanded = trf.expand(img_patches.size(0), -1)

            # 임시
            denorm_trf = 10**((trf[:,:-1]*7)-30).detach()
            denorm_trf = denorm_trf * dlogT
            optimizer.zero_grad()
            
            # trf_expanded = trf.expand(img_patches.size(0), -1)
            x1, x2, x3 = encoder(trf_expanded)
            cond_scalar = cond_encoder(img_patches)
            _, _, _, DEMs = decoder(x1, x2, x3, cond_scalar)
            DEMs = torch.clamp(DEMs, min=0)

            DEMs = DEMs * 1e18 

            # center pixel과 의 차이를 최소화하는 방향으로 학습
            center_idx = (patch_size**2) // 2
            img_patches_flat = img_patches.view(-1, patch_size**2)
            center_pixel = img_patches_flat[:, center_idx]
            predicted_pixel = torch.sum(DEMs * denorm_trf, dtype=torch.float64, dim=1)
            loss = criterion(torch.log2(predicted_pixel+1)/15, center_pixel)
            
            if i < 1:
                # print(center_pixel)
                # print(np.min(DEMs.cpu().detach().numpy()))
                print(predicted_pixel[10])
                print(denorm_pix[10])
                print("===================================")
                i+=1


            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

        average_loss = train_loss / len(train_loader)
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
            for img_patches, trf in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                
                img_patches = img_patches.squeeze(0).double().to(device)  # (100000, 1, 5, 5) 
                trf = trf.double().to(device)                             # (1, 81)
                trf_expanded = trf.expand(img_patches.size(0), -1)

                # 임시
                denorm_trf = 10**((trf[:, :-1]*7)-30).detach()
                denorm_trf = denorm_trf * dlogT

                x1, x2, x3 = encoder(trf_expanded)
                cond_scalar = cond_encoder(img_patches)
                _, _, _, DEMs = decoder(x1, x2, x3, cond_scalar)
                DEMs = torch.clamp(DEMs, min=0)

                DEMs = DEMs * 1e18

                # center pixel과 의 차이를 최소화하는 방향으로 학습
                center_idx = (patch_size**2) // 2
                img_patches_flat = img_patches.view(-1, patch_size**2)
                center_pixel = img_patches_flat[:, center_idx]                
                predicted_pixel = torch.sum(DEMs * denorm_trf, dtype=torch.float64, dim=1)
                loss = criterion(torch.log2(predicted_pixel+1)/15, center_pixel)

                val_loss += loss.item()

            Val_avg_loss = val_loss / len(val_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {Val_avg_loss:.5f}")
            writer.add_scalar('Loss/val', Val_avg_loss, epoch)

if __name__ == '__main__':
    
    current_path = os.getcwd()
    log_dir = os.path.join(current_path, "model_name/logs")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    image_root = '/userhome/youn_j/Dataset/HRI/174_L2_npy/174/'
    val_root = '/userhome/youn_j/Dataset/HRI/174_L2_val/174/'
    trf_root = '/userhome/youn_j/Dataset/HRI/TRF/'

    # ========================================
    # 모델 동작 테스트용
    image_root = val_root
    # ========================================


    batch_size= 1
    patch_size= 21  #가급적 홀수 (center pixel을 위해)
    kernel_size = 5

    image_size = (2048, 2048)

    def patch_num(image_size, patch_size):
        H_patches = image_size[0] - patch_size + 1
        W_patches = image_size[1] - patch_size + 1
        patches_per_image = H_patches * W_patches
        
        return patches_per_image
    
    # 필요시 쓰기
    # def get_divisors_optimized(n):
    #     divisors = set()
    #     for i in range(1, int(n**0.5) + 1):
    #         if n % i == 0:
    #             divisors.add(i)
    #             divisors.add(n // i)
    #     return sorted(divisors)


    slice_num = (image_size[0] - patch_size + 1)  #adjust something
    patches_per_step = patch_num(image_size=image_size, patch_size=patch_size) // slice_num
    num_epochs= 100

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    device = torch.device('cuda:0')
    
    dataset = ImageDataset(image_root= image_root, 
                           trf_root= trf_root, 
                           patch_size=patch_size,
                           batch_size=1,
                           patches_per_step=patches_per_step)
    
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle=True,
                            drop_last=True,
                            num_workers=4)
    
    val_dataset = ImageDataset(image_root= val_root, 
                           trf_root= trf_root, 
                           patch_size=patch_size,
                           batch_size=1,
                           patches_per_step=patches_per_step)
    
    val_dataloader = DataLoader(val_dataset, 
                            batch_size=batch_size, 
                            shuffle=True,
                            drop_last=False,
                            num_workers=4)

 
    Train(num_epochs=num_epochs, 
          train_loader=dataloader, 
          val_loader=val_dataloader,
          device=device, 
          patch_size=patch_size,
          patches_per_step=patches_per_step,
          kernel_size=kernel_size)