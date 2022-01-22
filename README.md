# KI-EBM
# k-Space and Image Domain Collaborative Energy Based Model for MRI Reconstruction
The Code is created based on the method described in the following paper: k-Space and Image Domain Collaborative Energy Based Model for MRI Reconstruction

Author: Zongjiang Tu, Chen Jiang, Yu Guan, Qiegen Liu, Minghui Zhang, Dong Liang.
Date : January. 21, 2022  
Version : 1.0   
The code and the algorithm are for non-comercial use only.   
Copyright 2022, Department of Electronic Information Engineering, Nanchang University. 

## Overview of parallel KI-EBM for MRI reconstruction.
 <div align="center"><img src="https://github.com/yqx7150/KI-EBM/tree/main/Figures/PKI-EBM.png" width = "815" height = "470"> </div>
## Overview of series KI-EBM for MRI reconstruction.
 <div align="center"><img src="https://github.com/yqx7150/KI-EBM/tree/main/Figures/SKI-EBM.png" width = "815" height = "470"> </div>
## Calibration-free parallel imaging results in T1 GE Brain at acceleration factor R=6 2D random under-sampling mask.
 <div align="center"><img src="https://github.com/yqx7150/KI-EBM/tree/main/Figures/8chR6.png" width = "815" height = "470"> </div>

## Rconstruction results at R=3 1D cartesian sampling percentages in 15 coils parallel imaging. 
 <div align="center"><img src="https://github.com/yqx7150/KI-EBM/tree/main/Figures/DDP_R3.png" width = "815" height = "470"> </div>
## Reconstruction results at R=6 pseudo radial sampling in 12 coils parallel imaging
 <div align="center"><img src="https://github.com/yqx7150/KI-EBM/tree/main/Figures/MODL.png" width = "815" height = "470"> </div>

# Test
If you want to test the code，please
```bash
# 8ch random2D GRBrain R4
CUDA_VISIBLE_DEVICES=0 python3 Test_PKI_8ch_demo.py --swish_act --exp_I=SIAT_I --resume_iter_I=169500 --exp_K=SIAT_K --resume_iter_K=124500 --step_lr_I=300 --step_lr_K=100

# MODL R6
CUDA_VISIBLE_DEVICES=0 python3 PKI_compare_modl.py --swish_act --exp_I=SIAT_I --resume_iter_I=169500 --exp_K=SIAT_K --resume_iter_K=124500 --step_lr_I=10 --step_lr_K=10
CUDA_VISIBLE_DEVICES=0 python3 SKI_compare_modl.py --swish_act --exp_I=SIAT_I --resume_iter_I=169500 --exp_K=SIAT_K --resume_iter_K=124500 --step_lr_I=10 --step_lr_K=10

# We use matlab to calculate PSNR and SSIM.
PSNR_SSIM_8ch_random2D_GEBrain.m
```
