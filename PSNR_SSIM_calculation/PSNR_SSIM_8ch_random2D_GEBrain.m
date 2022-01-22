clear; close all; clc;

load('data1_GE_brain_8channel.mat');
ori = DATA./max(max(max(abs(DATA))));

load('.\result\other_R6\ESPIRiT.mat')
ESPIRiT = resESPIRiT./max(abs(resESPIRiT(:)));

load('.\result\other_R6\SAKE.mat')
SAKE_rec = SAKE_rec./max(abs(SAKE_rec(:)));

load('.\result\other_R6\LINDBREG8h_GE.mat_Random2D_Results.uFct=6.mat')
LIN = paramsa.u0;

load('.\result\other_R6\EBM_rec.mat')
ebm = zeros(256,256,8); 
for i =1:8
   ebm(:,:,i) =  squeeze(im_complex(i,:,:));
end

load('.\result\PKI\R6\KI_EBM_rec.mat')
KI_EBM_BL = zeros(256,256,8); 
for i =1:8
   KI_EBM_BL(:,:,i) =  squeeze(im_complex(i,:,:));
end
% 
load('.\result\SKI\R6\KI_CL_EBM_rec.mat')
KI_EBM_CL = zeros(256,256,8); 
for i =1:8
   KI_EBM_CL(:,:,i) =  squeeze(im_complex(i,:,:));
end

%% SOS
ori_sos = sos(ori);  
ESPIRiT_sos = sos(ESPIRiT); 
SAKE_sos = sos(SAKE_rec); 
LIN_sos = LIN;
ebm_sos = sos(ebm);
KI_EBM_BL_sos =  sos(KI_EBM_BL);
KI_EBM_CL_sos =  sos(KI_EBM_CL);

ori_sos = ori_sos./max(max(ori_sos));
ESPIRiT_sos = ESPIRiT_sos./max(max(ESPIRiT_sos));
SAKE_sos = SAKE_sos./max(max(SAKE_sos));
LIN_sos = LIN_sos./max(max(LIN_sos));
ebm_sos = ebm_sos./max(max(ebm_sos));
KI_EBM_BL_sos = KI_EBM_BL_sos./max(max(KI_EBM_BL_sos));
KI_EBM_CL_sos = KI_EBM_CL_sos./max(max(KI_EBM_CL_sos));

%%
[psnr_ESPIRiT,ssim_ESPIRiT] = MSIQA(abs(ori_sos)*255,abs(ESPIRiT_sos)*255);
[psnr_sake,ssim_sake] = MSIQA(abs(ori_sos)*255,abs(SAKE_sos)*255);
[psnr_lin,ssim_lin] = MSIQA(abs(ori_sos)*255,abs(LIN_sos)*255);
[psnr_ebm,ssim_ebm] = MSIQA(abs(ori_sos)*255,abs(ebm_sos)*255);
[psnr_kiebm_BL,ssim_kiebm_BL] = MSIQA(abs(ori_sos)*255,abs(KI_EBM_BL_sos)*255);
[psnr_kiebm_CL,ssim_kiebm_CL] = MSIQA(abs(ori_sos)*255,abs(KI_EBM_CL_sos)*255);


PSNR = [psnr_ESPIRiT;psnr_lin;psnr_sake;psnr_ebm;psnr_kiebm_BL;psnr_kiebm_CL];
SSIM=[ssim_ESPIRiT;ssim_lin;ssim_sake;ssim_ebm;ssim_kiebm_BL;ssim_kiebm_CL];
RESULT = [PSNR,SSIM];
