from models import ResNet32Large,ResNet128
import numpy as np
import os.path as osp
from tensorflow.python.platform import flags
import tensorflow as tf
import imageio
import scipy.io as io
import cv2
import matplotlib.pyplot as plt
from utils import optimistic_restore
from skimage.measure import compare_psnr,compare_ssim
import glob
from US_pattern import US_pattern
import h5py as h5

flags.DEFINE_string('logdir_I', 'cachedir', 'location where log of experiments will be stored')
flags.DEFINE_string('logdir_K', 'cachedir','location where log of experiments will be stored')
flags.DEFINE_integer('num_steps', 800, 'num of steps for conditional imagenet sampling')
flags.DEFINE_float('step_lr_I', 10., 'step size for Langevin dynamics')
flags.DEFINE_float('step_lr_K', 10., 'step size for Langevin dynamics')
flags.DEFINE_integer('batch_size', 1, 'number of steps to run')
flags.DEFINE_string('exp_I', 'SIAT_I', 'name of experiments')
flags.DEFINE_string('exp_K', 'SIAT_K', 'name of experiments')
flags.DEFINE_integer('resume_iter_I', 169500, 'iteration to resume training from')
flags.DEFINE_integer('resume_iter_K', 124500, 'iteration to resume training from')
flags.DEFINE_bool('spec_norm', True, 'whether to use spectral normalization in weights in a model')
flags.DEFINE_bool('cclass', True, 'conditional models')
flags.DEFINE_bool('use_attention', False, 'using attention')
FLAGS = flags.FLAGS

def show(image):
    plt.imshow(image,cmap='gray')
    plt.xticks([])
    plt.yticks([])
def write_images(x,image_save_path):
    x = np.clip(x * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(image_save_path, x)
def rescale_im(im):
    return np.clip(im * 256, 0, 255)#.astype(np.uint8)
    
def compare_hfen(rec,ori):
    operation = np.array(io.loadmat("./input_data/loglvbo.mat")['h1'],dtype=np.float32)
    ori = cv2.filter2D(ori.astype('float32'), -1, operation,borderType=cv2.BORDER_CONSTANT)
    rec = cv2.filter2D(rec.astype('float32'), -1, operation,borderType=cv2.BORDER_CONSTANT)
    hfen = np.linalg.norm(ori-rec, ord = 'fro')
    return hfen
    

    
    
def write_Data(psnr,ssim,hfen):
    with open(osp.join('./result/compare_modl/PKI/',"psnr_compare_modl.txt"),"w+") as f:
        f.writelines('['+str(round(psnr, 3))+' '+str(round(ssim, 4))+' '+str(round(hfen, 3))+']')
        f.write('\n')

def write_Data2(step,psnr,ssim,hfen):
    with open(osp.join('./result/compare_modl/PKI/',"psnr_T.txt"),"w+") as f:
        f.writelines('step='+str(step)+' ['+str(round(psnr, 3))+' '+str(round(ssim, 4))+' '+str(round(hfen, 3))+']')
        f.write('\n')

def FT(x,csm):
    """ This is a the A operator as defined in the paper"""
    ncoil,nrow,ncol = csm.shape
    ccImg=np.reshape(x,(nrow,ncol) )
    coilImages=np.tile(ccImg,[ncoil,1,1])*csm;
    kspace=np.fft.fft2(coilImages)/np.sqrt(nrow * ncol)

    #io.savemat(osp.join('./result/compare_ddp/'+'modl_csm'),{'csm':csm})    
    #io.savemat(osp.join('./result/compare_ddp/'+'modl_coilImages'),{'coilImages':coilImages})    
    #assert 0
    return kspace

def tFT(kspaceUnder,csm):
    """ This is a the A^T operator as defined in the paper"""
    ncoil,nrow,ncol = csm.shape
    #temp=np.zeros((ncoil,nrow,ncol),dtype=np.complex64)
    img=np.fft.ifft2(kspaceUnder)*np.sqrt(nrow*ncol)
    coilComb=np.sum(img*np.conj(csm),axis=0).astype(np.complex64)
    #coilComb=coilComb.ravel();
    return coilComb

def k2wgt(X,W):
    Y = np.multiply(X,W) 
    return Y

def wgt2k(X,W,DC):
    DC = np.squeeze(DC)
    Y = np.multiply(X,1./W)
    Y[W==0] = DC[W==0]    
    return Y

 
filename='./MoDL_share_data/demoImage.hdf5' #set the correct path here

with h5.File(filename,'r') as f:
    org,csm,mask=f['tstOrg'][:],f['tstCsm'][:],f['tstMask'][:]

#print(org.shape,csm.shape,mask.shape)
#(1, 256, 232) (1, 12, 256, 232) (1, 256, 232)
orim = org[0]
csm = csm[0]
patt = mask[0]

'''
for ii in range(12):
    plt.ion()
    plt.imshow(abs(csm[ii,:,:]),cmap='gray')
    plt.pause(0.3)
'''


if __name__ == "__main__":

    model = ResNet128(num_filters=64)
    X_NOISE = tf.placeholder(shape=(None, 256, 232, 2), dtype=tf.float32)
    LABEL = tf.placeholder(shape=(None, 1), dtype=tf.float32)
    STD = tf.placeholder(shape=(None,1), dtype=tf.float32)
    CLIP_MAX = tf.placeholder(shape=(None, 1), dtype=tf.float32)
    CLIP_MIN = tf.placeholder(shape=(None, 1), dtype=tf.float32)
    LR = tf.placeholder(shape=(None, 1), dtype=tf.float32)
    
    sess_I = tf.InteractiveSession()
    sess_K = tf.InteractiveSession()

    # Langevin dynamics algorithm
    weights = model.construct_weights("context_0")  
    x_mod = X_NOISE
    x_mod1 = x_mod + tf.random_normal(tf.shape(x_mod),mean=0.0,stddev=STD[0,0])
    x_mod2 = x_mod + tf.random_normal(tf.shape(x_mod),mean=0.0,stddev=STD[0,0])
    x_mod3 = x_mod + tf.random_normal(tf.shape(x_mod),mean=0.0,stddev=STD[0,0])
    x_mod4 = x_mod + tf.random_normal(tf.shape(x_mod),mean=0.0,stddev=STD[0,0])
    x_mod5 = x_mod + tf.random_normal(tf.shape(x_mod),mean=0.0,stddev=STD[0,0])
    x_mod6 = x_mod + tf.random_normal(tf.shape(x_mod),mean=0.0,stddev=STD[0,0])
    x_mod7 = x_mod + tf.random_normal(tf.shape(x_mod),mean=0.0,stddev=STD[0,0])
    x_mod8 = x_mod + tf.random_normal(tf.shape(x_mod),mean=0.0,stddev=STD[0,0])  
    
    energy_noise1 = energy_start = model.forward(x_mod1, weights, label=LABEL, reuse=True, stop_at_grad=False, stop_batch=True)
    x_grad1 = tf.gradients(energy_noise1, [x_mod1])[0]
    energy_noise2 = energy_start = model.forward(x_mod2, weights, label=LABEL, reuse=True, stop_at_grad=False, stop_batch=True)
    x_grad2 = tf.gradients(energy_noise2, [x_mod2])[0]
    energy_noise3 = energy_start = model.forward(x_mod3, weights, label=LABEL, reuse=True, stop_at_grad=False, stop_batch=True)
    x_grad3 = tf.gradients(energy_noise3, [x_mod3])[0]
    energy_noise4 = energy_start = model.forward(x_mod4, weights, label=LABEL, reuse=True, stop_at_grad=False, stop_batch=True)
    x_grad4 = tf.gradients(energy_noise4, [x_mod4])[0]
    energy_noise5 = energy_start = model.forward(x_mod5, weights, label=LABEL, reuse=True, stop_at_grad=False, stop_batch=True)
    x_grad5 = tf.gradients(energy_noise5, [x_mod5])[0]
    energy_noise6 = energy_start = model.forward(x_mod6, weights, label=LABEL, reuse=True, stop_at_grad=False, stop_batch=True)
    x_grad6 = tf.gradients(energy_noise6, [x_mod6])[0]
    energy_noise7 = energy_start = model.forward(x_mod7, weights, label=LABEL, reuse=True, stop_at_grad=False, stop_batch=True)
    x_grad7 = tf.gradients(energy_noise7, [x_mod7])[0]
    energy_noise8 = energy_start = model.forward(x_mod8, weights, label=LABEL, reuse=True, stop_at_grad=False, stop_batch=True)
    x_grad8 = tf.gradients(energy_noise8, [x_mod8])[0]
    
    energy_noise_old = energy_noise1
    energy_noise = energy_noise1
    
    lr = LR[0,0]
    x_last = x_mod - (lr) * (x_grad1 + x_grad2 + x_grad3 + x_grad4 + x_grad5 + x_grad6 + x_grad7 + x_grad8)/8
    
    x_mod = x_last
    x_out = tf.clip_by_value(x_mod, CLIP_MIN[0,0], CLIP_MAX[0,0])

    # channel mean
    x_real=x_out[:,:,:,0]
    x_imag=x_out[:,:,:,1]
    x_complex = tf.complex(x_real,x_imag)
    x_output  = x_complex

    sess_I.run(tf.global_variables_initializer())
    sess_K.run(tf.global_variables_initializer())
    
    saver_I = loader_I = tf.train.Saver()
    saver_K = loader_K = tf.train.Saver()
    
    logdir_I = osp.join(FLAGS.logdir_I, FLAGS.exp_I)
    logdir_K = osp.join(FLAGS.logdir_K, FLAGS.exp_K)
    
    model_file_I = osp.join(logdir_I, 'model_{}'.format(FLAGS.resume_iter_I))
    model_file_K = osp.join(logdir_K, 'model_{}'.format(FLAGS.resume_iter_K))
    
    saver_I.restore(sess_I, model_file_I)
    saver_K.restore(sess_K, model_file_K)

#============================================================================================

    write_psnr=0
    write_ssim=0
    write_hfen=9999
    np.random.seed(1)
    lx = np.random.randint(0, 1, (FLAGS.batch_size))

    ims = []
    PSNR=[]
    im_complex=np.zeros((FLAGS.batch_size,256,232),dtype=np.complex128)

    #==========================================================================
    ori_complex = orim
    ori_complex = ori_complex/np.max(np.abs(ori_complex))
    write_images(abs(ori_complex),osp.join('./result/compare_modl/PKI/'+'ori'+'.png'))
    io.savemat(osp.join('./result/compare_modl/PKI/'+'MODL_ori'),{'img':ori_complex})
    
    ww = io.loadmat('./MoDL_share_data/MODL_weight10.mat')['weight']   

    kdata = np.fft.fft2(ori_complex)
    ksample = np.multiply(kdata,mask)
    k_w = k2wgt(np.fft.fftshift(ksample),ww)
    
    mask = patt #0.1666217672413793# R=6
    io.savemat(osp.join('./MoDL_share_data/'+'random_mask_R6'),{'mask':mask})

    print(np.sum(mask)/(256*232))
    ksp = FT(ori_complex,csm)  

    if len(mask.shape)==2:
        mask=np.tile(mask,(csm.shape[0],1,1))  
 
    #get multi coil undersample kspace by mask
    usksp = np.multiply(ksp,mask)
    undersample_kspace = usksp
    
    zero_fiiled = tFT(usksp,csm)
    write_images(abs(zero_fiiled),osp.join('./result/compare_modl/PKI/'+'zero_fiiled'+'.png'))
    io.savemat(osp.join('./result/compare_modl/PKI/'+'zero_fiiled'),{'img':zero_fiiled})

    # use for getting degrade img and psnr,ssim,hfen in iteration
    psnr_zerofill = compare_psnr(255*abs(zero_fiiled),255*abs(ori_complex),data_range=255)
    print('psnr_zerofill = ',psnr_zerofill) #25.95079970708028    
    
    x_mod_I = np.random.uniform(-1, 1, size=(FLAGS.batch_size, 256, 232, 2))
    x_mod_I[:,:,:,0] = np.real(zero_fiiled)
    x_mod_I[:,:,:,1] = np.imag(zero_fiiled)

    x_mod_K = np.random.uniform(-10, 10, size=(FLAGS.batch_size, 256, 232, 2))
    x_mod_K[:,:,:,0] = np.real(k_w)
    x_mod_K[:,:,:,1] = np.imag(k_w)

    labels = np.eye(1)[lx]

    std_I = np.eye(1)[lx]
    std_I[0,0] = 0.005
    clip_MAX_I = np.eye(1)[lx]
    clip_MAX_I[0,0] = 1
    clip_MIN_I = np.eye(1)[lx]
    clip_MIN_I[0,0] = -1
    LR_I = np.eye(1)[lx]
    LR_I[0,0] = FLAGS.step_lr_I

    std_K = np.eye(1)[lx]
    std_K[0,0] = 0.08
    clip_MAX_K = np.eye(1)[lx]
    clip_MAX_K[0,0] = 10
    clip_MIN_K = np.eye(1)[lx]
    clip_MIN_K[0,0] = -10
    LR_K = np.eye(1)[lx]
    LR_K[0,0] = FLAGS.step_lr_K 

    for i in range(FLAGS.num_steps):
        _, K_out= sess_K.run([energy_noise,x_output],{X_NOISE:x_mod_K, LABEL:labels,STD:std_K,CLIP_MAX:clip_MAX_K,CLIP_MIN:clip_MIN_K,LR:LR_K})
        _, I_out= sess_I.run([energy_noise,x_output],{X_NOISE:x_mod_I, LABEL:labels,STD:std_I,CLIP_MAX:clip_MAX_I,CLIP_MIN:clip_MIN_I,LR:LR_I})  
        
        #K
        K_temp = np.squeeze(K_out)
        K_temp2 = wgt2k(K_temp,ww,np.fft.fftshift(ksample))
        
        I_temp = np.fft.ifft2(np.fft.fftshift(K_temp2))
        K_complex = FT(I_temp,csm)
        
        #I
        I_complex = np.squeeze(I_out)
                
        #
        K_mean = (K_complex+FT(I_complex,csm))/2
        #K_mean = FT(I_complex,csm)
        #K_mean = K_complex
        
        # Data Consistance 
        iterkspace = undersample_kspace + K_mean*(1-mask)
        im_complex  = tFT(iterkspace,csm)
        
        # Back
        k_back = np.fft.fftshift(np.fft.fft2(im_complex))
        k_w = k2wgt(k_back,ww)    
        x_mod_K[:,:,:,0],x_mod_K[:,:,:,1]=np.real(k_w),np.imag(k_w)
        
        im_complex = np.expand_dims(im_complex, 0)
        x_mod_I[:,:,:,0],x_mod_I[:,:,:,1]=np.real(im_complex),np.imag(im_complex)        
        
        im_complex = im_complex[0]
        #im_complex=im_complex/np.max(abs(im_complex))
        #print(np.max(abs(im_complex)),np.min(abs(im_complex)))

        ################################################################################# SSIM
   
        ssim=compare_ssim(abs(im_complex),abs(ori_complex),data_range=1)
        
        if write_ssim<ssim:
            write_ssim=ssim
        ################################################################################# HFEN
        
        hfen=compare_hfen(abs(im_complex),abs(ori_complex))
        
        if write_hfen>hfen:
            write_hfen=hfen
        ################################################################################# PSNR
        
        psnr=compare_psnr(255*abs(im_complex),255*abs(ori_complex),data_range=255)
        err = abs(im_complex) -abs(ori_complex)

        if write_psnr<psnr:
            write_psnr=psnr
            write_Data2(i,psnr,ssim,hfen)
            write_images(abs(im_complex),osp.join('./result/compare_modl/PKI/'+'PKI_EBMrec_'+'.png'))
            write_images(abs(err)*5,osp.join('./result/compare_modl/PKI/'+'erro_CompareMmodl'+'.png'))
            io.savemat(osp.join('./result/compare_modl/PKI/'+'PKI_EBM_rec'),{'img':im_complex})
                                 
        print("step:{}".format(i),' PSNR:', psnr,' SSIM:', ssim,' HFEN:', hfen)
        write_Data(write_psnr,write_ssim,write_hfen)
    
        
