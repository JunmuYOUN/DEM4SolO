#headers

from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import astropy.io.fits as fits
#%matplotlib inline

import sunpy.visualization.colormaps as cm
import sunpy.map
from sunpy.map import Map

from aiapy.calibrate import register, update_pointing, fix_observer_location

from skimage.transform import resize
from scipy.interpolate import NearestNDInterpolator


import astropy.units as u
from astropy.time import Time
import warnings
import traceback

from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser(description='')
parser.add_argument('-w', '--wave')
args = parser.parse_args()

warnings.filterwarnings(action='ignore')


# folder



# input_folder = '/userhome/youn_j/Code/AIA_set/aia_fits_22_23/221001_4/094/'
# input_folder = '/userhome/youn_j/Dataset/DEM_pair/230410T00/{}/'.format(args.wave)
# output_folder = '/userhome/youn_j/DEM/DEM_testsetV240610/AIA/20240320T0840/'.format(args.wave)

# input_folder = '/userhome/youn_j/Dataset/AIA/240320T0843/'
# output_folder = '/userhome/youn_j/DEM/DEM_testsetV240610/AIA/20240320T0840/'

input_folder = '/userhome/youn_j/2025SW/304/'
output_folder = '/userhome/youn_j/2025SW/prep/'


# bad_pix_folder = '/userhome/youn_j/Code/AIA_set/Prep/AIA_PrepV4_230329/bad_pix/094/'

file_path = input_folder+'/*.fits'
# file_path = input_folder+'*image_lev1.fits'

file_name = sorted(glob(file_path)) 
series = len(file_name) 
print(series)

#Functions

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
        temp_time = temp_fits[1].header['T_OBS']
#         data_max = temp_fits[1].header['datamax']
        temp_time = Time2JD(temp_time)
        temp_time = round(temp_time, 2)
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

def disk_fitting(data, isize, _rad):
    return None

def Preprocess(file, ref_median, isize = 1024, Rsize = 400, output_folder = output_folder, order = 3): 
    
# load data
    fits_data = fits.open(file)
    AIA_header = fits_data[1].header
    AIA_data = fits_data[1].data
    AIA_map = Map(AIA_data, AIA_header)

#-----------------------------------------------    
# [1] Time fiting, Path2save 

    T_OBS = AIA_map.meta["T_OBS"]
    T_OBS = Time2JD(T_OBS)
    T_OBS = round(T_OBS,4)
    T_OBS = Time(T_OBS, format = 'jd')
    T_OBS = T_OBS.fits
    
    WaveL = str(AIA_map.meta['WAVELNTH'])
    if WaveL == '94':
        WaveL = '094'
    save = output_folder+T_OBS.replace(":",'')[:17] +'_' +WaveL
    quality = AIA_map.meta['QUALITY']
    
    if quality == 0: 
        
        _rad = int(AIA_map.meta["R_SUN"])
        exposure = AIA_map.meta["EXPTIME"]
#         distance = AIA_map.meta["DSUN_OBS"]

        AIA_data = np.nan_to_num(AIA_data)
        AIA_data[AIA_data<0] = 0
        AIA_map = Map(AIA_data, AIA_header)

#-----------------------------------------------
# [2] (Align) Rotate, Center
  
        AIA_map = AIA_map.rotate(recenter = True, scale = 1. , order = order, missing = 0) #rotate, header changed
        
    # Align Center (4096*4096)
        xcenter = np.floor(AIA_map.meta['CRPIX1']) 
        ycenter = np.floor(AIA_map.meta['CRPIX2']) 
        xrange = (xcenter + np.array([-1, 1]) * 4096 / 2) * u.pix
        yrange = (ycenter + np.array([-1, 1]) * 4096 / 2) * u.pix
        
        AIA_map = AIA_map.submap(u.Quantity([xrange[0], yrange[0]])
                               , top_right=u.Quantity([xrange[1], yrange[1]]))
        AIA_map = AIA_map.resample([4096,4096]*u.pix)
#         print(np.shape(AIA_map.data))
        
    #bad_pixel save : Remove it from V7
#         bad_pixel = np.where(AIA_map.data <= 0)
#         np.append(bad_pixel,np.where(np.isnan(AIA_map.data)))
#         AIA_map.data[bad_pixel] = 0.1
#         print(np.shape(bad_pixel))
        

# ---------------------------------------------------------
# [3] Exposure, Degradation, Normalize, Resize
    
        
    #div exposure
        AIA_map= AIA_map/exposure #[DN/s]
    
        UpLim = 14
        LoLim = 0
        disk_med = disk_median(AIA_map.data, isize = 4096, _rad = _rad)
        
            #call reference median
        
        if WaveL == '94' or WaveL == '094':
            ref_median = 0.62489337
        elif WaveL ==  '131': 
            ref_median = 3.3768058
        elif WaveL == '171' : 
            ref_median = 151.73026
        elif WaveL == '193' : 
            ref_median = 105.85915
        elif WaveL == '211' : 
            ref_median = 23.901197
        elif WaveL == '304' : 
            ref_median = 24.65
        elif WaveL == '335' : 
            ref_median = 1.286345
        else:
            print('No waveL')
#             raise ValueError('No waveL')
            
        print(ref_median)
        
    #call reference median

        if T_OBS.replace(":",'')[:17] == '2011-01-01T000000':
            ref_median = disk_med
            print(ref_median)
            
        if ref_median == np.nan:
            print('No ref. median')
            raise ValueError('No ref. median')
            
    #degradation        
        deg_factor = ref_median/disk_med
        AIA_map = AIA_map * deg_factor
            
    #normalize 
        log2_data = np.log2(AIA_map.data + 1) # log2
        log2_data = (np.clip(log2_data, LoLim, UpLim) - (UpLim-LoLim)/2)/((UpLim-LoLim)/2) #normalize
        log2_data = np.float32(log2_data) #bit32
        print('after_shape1',np.shape(log2_data))
        

    #resize (also disk size)
        resize_ratio = Rsize/_rad  #0.~~~
        tmp_size = int(np.floor(4096*resize_ratio))
        tmp_size_half = int(np.floor((tmp_size/2)+0.5))
        
        log2_data = resize(log2_data, (tmp_size, tmp_size), order = order, mode='constant', preserve_range=True)
        print('after_shape4',np.shape(log2_data))
        
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

        
        
        
        print('after_shape4',np.shape(log2_data))
#         print('1', AIA_map.meta['CRPIX1'])#
#         print(np.shape(log2_data))#  
#         print(np.max(log2_data))#  
        
#         plt.imshow(log2_data,cmap = 'sdoaia211')#
#         plt.gca().invert_yaxis()#
        
    
    #Mapping
        AIA_map = Map(log2_data, AIA_map.meta)   
        
        AIA_header['lvl_num'] = 2.0     
        AIA_header['bitpix'] = int(-32)
        AIA_header['cdelt1'] = float(AIA_map.meta['cdelt1']/(resize_ratio))
        AIA_header['cdelt2'] = float(AIA_map.meta['cdelt2']/(resize_ratio))
        #V7 changed
#         AIA_header['cdelt1'] = float(AIA_map.meta['cdelt1']/(isize/4096))
#         AIA_header['cdelt2'] = float(AIA_map.meta['cdelt2']/(isize/4096))
        AIA_header['r_sun'] = float(_rad * resize_ratio) 
        AIA_header['crpix1'] = float(isize//2)
        AIA_header['crpix2'] = float(isize//2)
        
        
    else:
        AIA_map = False
        save = False
        print("Check Data")

    return AIA_map, AIA_header, save, ref_median







# ========================================================================
    
# series = 3  #For test
err_li = []

resume = 0
ref_median = np.nan
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
        OBS = hdr['T_OBS']
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
            print(traceback.format_exc())

        except IndexError:
            print('No additional data')
            
            break
            
                
    

print('error file:\n',err_li)
print('Finish')

