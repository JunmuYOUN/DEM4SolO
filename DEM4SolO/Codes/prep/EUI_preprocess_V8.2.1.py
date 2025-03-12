#Update 24 06 10

#V8 Degradation 수정
# Correlation ==> Calibration

#headers
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import astropy.io.fits as fits
#%matplotlib inline

import sunpy.visualization.colormaps as cm

from astropy.time import Time
import warnings
warnings.filterwarnings(action='ignore')

from tqdm import tqdm

import sunpy.map
from sunpy.map import Map
from aiapy.calibrate import register, update_pointing, fix_observer_location
from skimage.transform import resize
import astropy.units as u
import sunpy.visualization.colormaps as cm
from scipy import ndimage
from scipy.spatial.transform import Rotation as R
from sunpy.image.transform import affine_transform

from astropy.coordinates import SkyCoord

import argparse


parser = argparse.ArgumentParser(description='')
parser.add_argument('-ver', '--version', default= '8.1')
args = parser.parse_args()

global VER
VER = args.verison  #8.1 For model #8.3 For DEM

#options
# input_folder = '/userhome/youn_j/Dataset/SO_FSI_2022/304/'
input_folder = '/userhome/youn_j/Dataset/SO_FSI_2022/304_202401/'
# input_folder = '/userhome/youn_j/Code/AIA_set/SO_FSI_2022/304/solo_L1_eui-fsi304-image_20230328T210020197_V01'
output_folder = '/userhome/youn_j/Dataset/EUV/EUI_prepV8_231129/caseV8.3/304/'

file_path = input_folder+'*.fits'
file_name = sorted(glob(file_path)) 
series = len(file_name) 
print(series)

#functions
def Time2JD(_t):
    _temp = str(_t)
    _temp = Time(_temp)
    _temp = float(_temp.jd)
    return _temp


def data_select(data_num):
    temp_file = []
#     temp_max = []

    for i in range(data_num):
        temp_fits = fits.open(file_name[i])
        temp_time = temp_fits[1].header['DATE-OBS']
#         data_max = temp_fits[1].header['datamax']
        temp_time = Time2JD(temp_time)
        temp_time = round(temp_time, 3)
        temp_time = Time(temp_time, format = 'jd')
        temp_time = temp_time.fits
        
        if 'T00' in temp_time:
            temp_file.append(file_name[i])
#             temp_max.append(data_max)
        temp_series = len(temp_file)
        
    return temp_file, temp_series


def disk_median(data, isize, _rad):
    arr = np.arange(isize) - (isize//2 -1)
    xv, yv = np.meshgrid(arr, arr)
    fv = np.sqrt(xv**2 + yv**2)
    temp = fv < 0.995 * _rad
    med = np.float32(np.median(data[temp]))
    
#     data[temp] = 1.
#     plt.imshow(data ,cmap = 'sdoaia211')
#     plt.gca().invert_yaxis()
    
    return med

#V8.1
# def calibration(data,WaveL):
#     if WaveL == '174': 
#         data = (7.87e-08)*(data)**3 + (0.0003)* (data)**2 + (data)
        
#     elif WaveL == '304': 
#         data = (2.03e-06)*(data)**3 + (-0.002)*(data)**2 + (1.1)*(data)
#     return data


def Preprocess(file, ref_median, isize = 1024, Rsize = 400, output_folder = output_folder, order = 3): 
    '''EUI Preprocessing:
    1. Crop 3040 x 3040 (If init size is not 3072 but other value, assume as bad quallity)
    '''    
# load data
    fits_data = fits.open(file)
    SO_header = fits_data[1].header
    image = fits_data[1].data
    
    Imsize = 3072
    crop = 32
    Imsize -= crop
    SO_header['NAXIS1'] = SO_header['NAXIS1'] - crop
    SO_header['NAXIS2'] = SO_header['NAXIS2'] - crop
    image = image[:-crop,:-crop] 
    image[image<0] = 0
    image = np.nan_to_num(image)
    
    PreMap = sunpy.map.Map(np.array(image, dtype = float), SO_header)
    
    if PreMap.meta['NAXIS1'] == Imsize and PreMap.meta['NAXIS2'] == Imsize:
        quality = 0 #EUI라서 임의로..
    else:
        print(PreMap.meta['NAXIS1'],' x ',PreMap.meta['NAXIS2'])
        quality = 1

        
#-----------------------------------------------    
# [1] Time fiting, Path2save 


    r_sun = np.floor((PreMap.meta["RSUN_OBS"]/PreMap.meta["CDELT1"])+0.5) #0.5는 반올림을 위해
    exposure = PreMap.meta["XPOSURE"]
    NAXIS = PreMap.meta['NAXIS1']
    _X = NAXIS/2
    T_OBS = PreMap.meta["DATE-OBS"]
    T_OBS = Time2JD(T_OBS)
    T_OBS = round(T_OBS,3)
    T_OBS = Time(T_OBS, format = 'jd')
    T_OBS = T_OBS.fits
    
    
    WaveL = str(PreMap.meta['WAVELNTH'])
    save = output_folder+T_OBS.replace(":",'')[:17] +'_' +WaveL

    
    if quality == 0: 
        exposure = PreMap.meta["XPOSURE"]
        NAXIS = PreMap.meta['NAXIS1']
        _X = NAXIS/2

#-----------------------------------------------
# [2] Rotate, Resize
        
        # angle
        angle = PreMap.meta['crota']*u.deg
        c = np.cos(np.deg2rad(angle))
        s = np.sin(np.deg2rad(angle))
        rmatrix = np.array([[c, -s],
                            [s, c]])
        inv_rmatrix = np.linalg.inv(rmatrix)

        #copy
        new_data = PreMap.data
        new_header = PreMap.meta

        #rotate
        pixel_center = (PreMap.meta["euxcen"], PreMap.meta["euycen"])

        new_data = affine_transform(new_data, np.asarray(inv_rmatrix), order= 3,
                                    scale=1.0,image_center=pixel_center,recenter=True, 
                                    missing=0) #method= 'scipy', clip = True)
    

        PreMap = sunpy.map.Map(np.array(new_data, dtype = float), PreMap.meta)
# ---------------------------------------------------------           
    #div exposure
        PreMap= PreMap/exposure #[DN/s]
                      
# ---------------------------------------------------------
# [3] Degradation, Correlation, Normalize, Resize
    
        UpLim = 14
        LoLim = 0
    
        disk_med = disk_median(PreMap.data, isize = Imsize, _rad = r_sun)

# =========================================================================================

# Degradation 

# V8.1 Reference median of AIA at 2011.01.01 T00 
        if VER == "8.1":
            if WaveL == '174':
                ref_median = 151.73
            if WaveL == '304':
                ref_median =  24.65

# V8.2 Ref_med at 2021.01.01
        if VER == "8.2":
            if WaveL == '174':
                ref_median = 137.8
            if WaveL == '304':
                ref_median =  66.6
                
# V8.1 Reference median of AIA at 2011.01.01 T00                 
        if VER == "8.3":
            if WaveL == '174':
                ref_median = 151.73
            if WaveL == '304':
                ref_median =  24.65
# =========================================================================================            

        deg_factor = ref_median/disk_med
        PreMap = PreMap * deg_factor
        
        if VER == "8.1":
            cali = calibration(PreMap.data, WaveL) #V8.1
            PreMap = sunpy.map.Map(np.array(cali, dtype = float), PreMap.meta)   
        
        if VER == "8.3":
            PreMap = sunpy.map.Map(np.array(PreMap.data, dtype = float), PreMap.meta)  
 
        
        

    #normalize 
        log2_data = np.log2(PreMap.data + 1) # log2
        log2_data = (np.clip(log2_data, LoLim, UpLim) - (UpLim-LoLim)/2)/((UpLim-LoLim)/2) #normalize
        log2_data = np.float32(log2_data) #bit32

    #resize(downscale and upscale)
#         print(r_sun,'r_sun')
#         if r_sun >= Rsize:  #Rsize == 400

        resize_ratio = Rsize/r_sun
        tmp_size = int(Imsize*resize_ratio)
        tmp_size_half = int(np.floor((tmp_size/2)+0.5))
        log2_data = resize(log2_data, (tmp_size, tmp_size), order = order, mode='constant', preserve_range=True)


        SO_header['cdelt1'] = float(PreMap.meta['cdelt1']/(resize_ratio))
        SO_header['cdelt2'] = float(PreMap.meta['cdelt2']/(resize_ratio))
        SO_header['rsun_obs'] = float(r_sun * (resize_ratio)/SO_header['cdelt1'])

# ===================================================================================================


        if tmp_size < isize:   # isize == 1024
            if (isize - tmp_size) % 2 != 0:
                log2_data = np.pad(log2_data
                                   , (((isize-tmp_size)//2+1,(isize-tmp_size)//2),
                                   ((isize-tmp_size)//2+1,(isize-tmp_size)//2))
                                   , mode='constant'
                                   , constant_values = -1)
            else:
                log2_data = np.pad(log2_data
                                   , (((isize-tmp_size)//2,(isize-tmp_size)//2),
                                   ((isize-tmp_size)//2,(isize-tmp_size)//2))
                                   , mode='constant'
                                   , constant_values = -1)
                
        else:
            log2_data = log2_data[(tmp_size_half)-(isize//2) : (isize//2)+(tmp_size_half),
                              (tmp_size_half)-(isize//2) : (isize//2)+(tmp_size_half)]



# ====================================================================================================


    
#         print('1', AIA_map.meta['CRPIX1'])#
#         print(np.shape(log2_data))#  
#         print(np.min(log2_data))#  
        
#         plt.imshow(log2_data,cmap = 'sdoaia304')#
#         plt.gca().invert_yaxis()#
        
    
    #Mapping
        PreMap = Map(log2_data, PreMap.meta)   
        
        SO_header['lvl_num'] = 2.0     
        SO_header['bitpix'] = int(-32)
        
        SO_header['naxis1'] = 1024
        SO_header['naxis2'] = 1024
        SO_header['crpix1'] = 512
        SO_header['crpix2'] = 512
        SO_header['crota'] = 0

        SO_header['cunit1'] = 'arcsec' 
        SO_header['cunit2'] = 'arcsec'
        SO_header["crval1"] = 0
        SO_header["crval2"] = 0
        SO_header["euxcen"] = 512
        SO_header["euycen"] = 512
        SO_header["pc1_1"] = 1
        SO_header["pc1_2"] = 0
        SO_header["pc2_1"] = 0
        SO_header["pc2_2"] = 1
        
        
        
        
    else:
        PreMap = False
        save = False
        print("Check Data")

    return PreMap, SO_header, save, ref_median

# ========================================================================


# series = 1  #For test
err_li = []

resume = 0
ref_median = 0
#---------------------------------------
#Run Preprocess Code
for n in tqdm(range(series)):
    
    n += resume
    print(n+1)
    try: 
        print(ref_median)
        prep_map, hdr, save_path, ref_median = Preprocess(file_name[n], ref_median)
        
        
        if prep_map == False:
            raise ValueError("Bad Data")
        if True in np.isnan(prep_map.data):
            raise ValueError('nan is in data')

        print('after_shape',np.shape(prep_map.data))
#----------------------------------------    
    #Save
        print(save_path)
        
        save_path = save_path + '.fits'
#         np.save(save_path, prep_map.data)
        OBS = hdr["DATE-OBS"]
        WVL = str(hdr['wavelnth'])
        fits.writeto(save_path, prep_map.data, hdr, overwrite=True)
#         np.save(bad_pix_folder + OBS.replace(":",'')[:17] + '_' +WVL+'.npy', bad_pixel)
        
#         print(ref_median)
#         print(np.median(prep_map.data))
        print('-------------')
        
#---------------------------------------- 
    #Error
    except Exception as e:

        try:
            err_file = str(file_name[n])
            err_li.append(err_file)
            print(err_file)
            print(e)
#             print(traceback.format_exc())

        except IndexError:
            print('No additional data')
            
            break
            
                
  
print('error file:\n',err_li)
print('Finish')
