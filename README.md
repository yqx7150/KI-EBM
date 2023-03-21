# KI-EBM
Date : March. 21, 2023  
Version : 2.0   
The code and the algorithm are for non-comercial use only.   
Copyright 2022, Department of Electronic Information Engineering, Nanchang University. 

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
