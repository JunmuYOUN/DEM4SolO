print('Prep_HRIL2_v0.2.0_250328.py')
#0.2.1 minor error revised

import os
import numpy as np
import astropy.io.fits as fits
# from astropy.visualization import ImageNormalize, AsinhStretch
from glob import glob
from tqdm import tqdm

hri_list = sorted(glob('/userhome/youn_j/Dataset/HRI/174_L2/*.fits'))
data_num = len(hri_list)

print(data_num)

for i in tqdm(range(data_num)):

    img = fits.open(hri_list[i])
    dat = img[1].data
    hdr = img[1].header
    data_name = hri_list[i].split('/')[-1].split('.')[0]

    if dat.shape[0] == 2048 and dat.shape[1] == 2048:
        
        
        dat[dat<0] = 0
        data = np.nan_to_num(dat)

        dsun = hdr["DSUN_OBS"]  # 태양과의 거리 (단위: m)
        au = 1.496e11  # 1 AU (m)
        
        correction_factor = (dsun / au)**2  # Conrrection factor to 1 AU
        data_corrected = data * correction_factor

        UpLim = 15
        LoLim = 0

        log2_data = np.log2(data_corrected + 1) # log2
        log2_data = (np.clip(log2_data, LoLim, UpLim))/((UpLim-LoLim)) #normalize to 0~1
        log2_data = np.float32(log2_data)   # float32
#         print(np.min(log2_data))
        os.makedirs('/userhome/youn_j/Dataset/HRI/174_L2_npy/174/', exist_ok = True)
        os.makedirs('/userhome/youn_j/Dataset/HRI/174_L2_val/174/', exist_ok = True)
    
        if i % 100 == 0:
            print(data_name, f"{i}/{data_num}")
            np.save(f'/userhome/youn_j/Dataset/HRI/174_L2_val/174/{data_name}.npy', log2_data)
        else:
            np.save(f'/userhome/youn_j/Dataset/HRI/174_L2_npy/174/{data_name}.npy', log2_data)
    else:
        print(data_name, dat.shape)
