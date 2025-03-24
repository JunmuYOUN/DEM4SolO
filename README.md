# DEM4SO

📰 [Can we properly determine differential emission measures from Solar Orbiter/EUI/FSI with deep learning? (A&A)](https://doi.org/10.1051/0004-6361/202452304)

**Abstract:**


In this study, we address the question of whether we can properly determine differential emission measures (DEMs) using Solar Orbiter/Extreme Ultraviolet Imager (EUI)/Full Sun Imager (FSI) and AI-generated extreme UV (EUV) data. The FSI observes only two full-disk EUV channels (174 and 304 Å), which is insufficient for accurately determining DEMs and can lead to significant uncertainties. To solve this problem, we trained and tested deep learning models based on Pix2PixCC using the Solar Dynamics Observatory (SDO)/Atmospheric Imaging Assembly (AIA) dataset. The models successfully generated five-channel (94, 131, 193, 211, and 335 Å) EUV data from 171 and 304 Å EUV observations with high correlation coefficients. Then we applied the trained models to the Solar Orbiter/EUI/FSI dataset and generated the five-channel data that the FSI cannot observe. We used the regularized inversion method to compare the DEMs from the SDO/AIA dataset with those from the Solar Orbiter/EUI/FSI dataset, which includes AI-generated data. We demonstrate that, when SDO and Solar Orbiter are at the inferior conjunction, the main peaks and widths of both DEMs are consistent with each other at the same coronal structures. Our study suggests that deep learning can make it possible to properly determine DEMs using Solar Orbiter/EUI/FSI and AI-generated EUV data.


**CITATION**
```
@article{youn2025can,
  title={Can we properly determine differential emission measures from Solar Orbiter/EUI/FSI with deep learning?},
  author={Youn, Junmu and Lee, Harim and Jeong, Hyun-Jin and Lee, Jin-Yi and Park, Eunsu and Moon, Yong-Jae},
  journal={Astronomy \& Astrophysics},
  volume={695},
  pages={A125},
  year={2025},
  publisher={EDP Sciences}
}
```
