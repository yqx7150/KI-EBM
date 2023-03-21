# KI-EBM
Paper: K-space and image domain collaborative energy-based model for parallel MRI reconstruction
https://www.sciencedirect.com/science/article/abs/pii/S0730725X23000383

Authors: Zongjiang Tu, Chen Jiang, Yu Guan, Jijun Liu, Qiegen Liu 

Date: March. 21, 2023  
Version: 2.0   
The code and the algorithm are for non-comercial use only.   
Copyright 2022, Department of Electronic Information Engineering, Nanchang University. 

# Checkpoints
We provide pretrained checkpoints. You can download pretrained models from [Google Drive] (https://drive.google.com/file/d/1DmRTPmc_xYaVO3pX1R_CE0ZpiBRFkCwG/view?usp=sharing)

# Test
If you want to test the code，please
```bash
# 8ch random-2D GRBrain R4
CUDA_VISIBLE_DEVICES=0 python3 Test_PKI_8ch_demo.py --swish_act --exp_I=SIAT_I --resume_iter_I=169500 --exp_K=SIAT_K --resume_iter_K=124500 --step_lr_I=300 --step_lr_K=100

# Compare with MoDL random-2D R6
CUDA_VISIBLE_DEVICES=0 python3 PKI_compare_modl.py --swish_act --exp_I=SIAT_I --resume_iter_I=169500 --exp_K=SIAT_K --resume_iter_K=124500 --step_lr_I=10 --step_lr_K=10
CUDA_VISIBLE_DEVICES=0 python3 SKI_compare_modl.py --swish_act --exp_I=SIAT_I --resume_iter_I=169500 --exp_K=SIAT_K --resume_iter_K=124500 --step_lr_I=10 --step_lr_K=10

# We use matlab to calculate PSNR and SSIM.
PSNR_SSIM_8ch_random2D_GEBrain.m
```

# Acknowledgement
The implementation is based on: 
```bash
https://github.com/openai/ebm_code_release
https://github.com/yqx7150/EBMRec
```
