from models import ResNet128
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
import time

flags.DEFINE_string('logdir_I', 'cachedir', 'location where log of experiments will be stored')
flags.DEFINE_string('logdir_K', 'cachedir', 'location where log of experiments will be stored')
flags.DEFINE_integer('num_steps', 150, 'num of steps for conditional imagenet sampling')
flags.DEFINE_float('step_lr_I', 300., 'step size for Langevin dynamics')
flags.DEFINE_float('step_lr_K', 300., 'step size for Langevin dynamics')
flags.DEFINE_integer('batch_size', 1, 'number of steps to run')
flags.DEFINE_string('exp_I', 'default', 'name of experiments')
flags.DEFINE_string('exp_K', 'default', 'name of experiments')
flags.DEFINE_integer('resume_iter_I', -1, 'iteration to resume training from')
flags.DEFINE_integer('resume_iter_K', -1, 'iteration to resume training from')
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
    return np.clip(im * 256, 0, 255)
   
def write_Data(i,psnr,ssim):
    filedir="result_psnr.txt"
    with open(osp.join('./result/parallel-8ch/',filedir),"w+") as f:
        f.writelines('step='+str(i)+' '+'['+str(round(psnr, 3))+' '+str(round(ssim, 5))+']')
        f.write('\n')

def write_zero_Data(psnr,ssim):
    with open(osp.join('./result/parallel-8ch/random_R3/',"zero_psnr1.txt"),"w+") as f:
        f.writelines('['+str(round(psnr, 3))+' '+str(round(ssim, 5))+']')
        f.write('\n')
def k2w(X,W):
    Y = np.multiply(X,W) 
    return Y

def w2k(X,W,DC):
    Y = np.multiply(X,1./W)
    Y[W==0] = DC[W==0]   
    return Y

if __name__ == "__main__":
    #========================================================================================

    model = ResNet128(num_filters=64)
    X_NOISE = tf.placeholder(shape=(None, 256, 256, 2), dtype=tf.float32)
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
    x_last = x_mod - (lr) * (x_grad1 + x_grad2 + x_grad3 + x_grad4 + x_grad5 +x_grad6 +x_grad7 +x_grad8)/8

    x_mod = x_last
    x_out = tf.clip_by_value(x_mod, CLIP_MIN[0,0], CLIP_MAX[0,0])

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

    for pp in range(1):
        write_psnr=0
        write_ssim=0
        np.random.seed(1)
        lx = np.random.randint(0, 1, (FLAGS.batch_size))

        coil = 8
        
        x_uncomplex = np.zeros((coil,256,256,2),dtype=np.complex64)
        im_complex = np.zeros((coil,256,256),dtype=np.complex64)
        Kdata = np.zeros((coil,256,256),dtype=np.complex64)
        Ksample = np.zeros((coil,256,256),dtype=np.complex64)
        zeorfilled_data = np.zeros((coil,256,256),dtype=np.complex64)
        Kdata_wused = np.zeros((coil,256,256),dtype=np.complex64)
        Ksample_wused = np.zeros((coil,256,256),dtype=np.complex64)
        k_w = np.zeros((coil,256,256),dtype=np.complex64)
   
        file_path='./inputdata/contract_data_8h/'+'data1_GE_brain'+'.mat'
            
        ori_data = np.zeros([256,256,coil],dtype=np.complex64)
        ori_data = io.loadmat(file_path)['DATA']
        ori_data = ori_data/np.max(np.abs(ori_data))
        ori_data = np.swapaxes(ori_data,0,2)
        ori_data = np.swapaxes(ori_data,1,2)
        ori_data_sos = np.sqrt(np.sum(np.square(np.abs(ori_data)),axis=0))     
                
        mask_item = io.loadmat('./inputdata/contract_mask/random2D/R6.mat')['mask']  # R4 R6

        mask = np.zeros((coil,256,256))
        for i in range(coil):
            mask[i,:,:] = np.fft.fftshift(mask_item)

        mask_wused = np.zeros((coil,256,256))
        for i in range(coil):
            mask_wused[i,:,:] = mask_item
                        
        ww = io.loadmat('./inputdata/weight10_testdata.mat')['weight']
        weight = np.zeros((coil,256,256))       
        for i in range(coil):
            weight[i,:,:] = ww
            
        for i in range(coil):
            # Image domain used
            Kdata[i,:,:] = np.fft.fft2(ori_data[i,:,:])
            Ksample[i,:,:] = np.multiply(mask[i,:,:],Kdata[i,:,:])
            zeorfilled_data[i,:,:] = np.fft.ifft2(Ksample[i,:,:])
            
            # Kspace domain used
            Kdata_wused[i,:,:] = np.fft.fftshift(Kdata[i,:,:])
            Ksample_wused[i,:,:] = np.fft.fftshift(Ksample[i,:,:])
            k_w[i,:,:] = k2w(Ksample_wused[i,:,:],weight[i,:,:])
            
        zeorfilled_data_sos = np.sqrt(np.sum(np.square(np.abs(zeorfilled_data)),axis=0))  
           
        ori_data_sos = ori_data_sos/np.max(np.abs(ori_data_sos))
        zeorfilled_data_sos = zeorfilled_data_sos/np.max(np.abs(zeorfilled_data_sos))
                    
        psnr_zero=compare_psnr(255*abs(zeorfilled_data_sos),255*abs(ori_data_sos),data_range=255)
        ssim_zero=compare_ssim(abs(zeorfilled_data_sos),abs(ori_data_sos),data_range=1)

        print('psnr_zerofilled: ',psnr_zero)

        x_mod_I = np.random.uniform(-1, 1, size=(coil, 256, 256, 2))
        for i in range(coil):
            x_mod_I[i,:,:,0] = np.real(zeorfilled_data[i,:,:])
            x_mod_I[i,:,:,1] = np.imag(zeorfilled_data[i,:,:])
        
        x_mod_K = np.random.uniform(-10, 10, size=(coil, 256, 256, 2))
        for i in range(coil):
            x_mod_K[i,:,:,0] = np.real(k_w[i,:,:])
            x_mod_K[i,:,:,1] = np.imag(k_w[i,:,:])
                      
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
        
        x_complex = np.zeros((coil,256,256),dtype=np.complex64)
        iterkspace = np.zeros((coil,256,256),dtype=np.complex64)
        k_complex = np.zeros((coil,256,256),dtype=np.complex64)
        k_complex2 = np.zeros((coil,256,256),dtype=np.complex64)
        iterkspace = np.zeros((coil,256,256),dtype=np.complex64) 
        iterkspace_shift = np.zeros((coil,256,256),dtype=np.complex64) 
        kspace_mean = np.zeros((coil,256,256),dtype=np.complex64) 
        Kspace_DC = np.zeros((coil,256,256),dtype=np.complex64)
        
        start1 = time.time()
        #=====================================================================================================================================
        for i in range(FLAGS.num_steps):
            start2 = time.time()
            
            _, x_output_K= sess_K.run([energy_noise,x_out],{X_NOISE:x_mod_K, LABEL:labels,STD:std_K,CLIP_MAX:clip_MAX_K,CLIP_MIN:clip_MIN_K,LR:LR_K})
            _, x_output_I= sess_I.run([energy_noise,x_out],{X_NOISE:x_mod_I, LABEL:labels,STD:std_I,CLIP_MAX:clip_MAX_I,CLIP_MIN:clip_MIN_I,LR:LR_I})
            
            k_uncomplex = x_output_K
            x_uncomplex = x_output_I/np.max(np.abs(x_output_I))              
           
            
            for j in range(coil): 
                # K                
                k_complex[j,:,:] = k_uncomplex[j,:,:,0]+1j*k_uncomplex[j,:,:,1]                
                k_complex2[j,:,:] = w2k(k_complex[j,:,:],weight[j,:,:],Ksample_wused[j,:,:])            
    
                # I
                x_complex[j,:,:] = x_uncomplex[j,:,:,0]+1j*x_uncomplex[j,:,:,1]                
                iterkspace[j,:,:] = np.fft.fft2(x_complex[j,:,:])
                iterkspace_shift[j,:,:] = np.fft.fftshift(iterkspace[j,:,:])
                
                # kspace mean
                kspace_mean[j,:,:] = (k_complex2[j,:,:]+iterkspace_shift[j,:,:])/2
            
                # data consistance
                Kspace_DC[j,:,:] = Ksample_wused[j,:,:] + kspace_mean[j,:,:]*(1-mask_wused[j,:,:])
                im_complex[j,:,:] = np.fft.ifft2(np.fft.fftshift(Kspace_DC[j,:,:])) 
                
                
            for j in range(coil):
                # back
                x_mod_I[j,:,:,0] = np.real(im_complex[j,:,:])
                x_mod_I[j,:,:,1] = np.imag(im_complex[j,:,:])                
                
                k_w[j,:,:] = k2w(kspace_mean[j,:,:],weight[j,:,:])
                x_mod_K[j,:,:,0] = np.real(k_w[j,:,:])
                x_mod_K[j,:,:,1] = np.imag(k_w[j,:,:])  
                                             
            im_complex_sos = np.sqrt(np.sum(np.square(np.abs(im_complex)),axis=0))
            im_complex_sos = im_complex_sos/np.max(np.abs(im_complex_sos))
            
            end2 = time.time()

            ################################################################################ PSNR & SSIM
            psnr=compare_psnr(255*abs(im_complex_sos),255*abs(ori_data_sos),data_range=255)
            ssim=compare_ssim(abs(im_complex_sos),abs(ori_data_sos),data_range=1)
            print("step:{}".format(i),' PSNR:', psnr,' SSIM:', ssim)

            if write_psnr<=psnr:
                write_psnr=psnr
                write_ssim=ssim
                #write_images(abs(im_complex_sos),osp.join('./result/parallel-8ch/random_R3/Image/'+'PKI_EBM_rec.png'))
                io.savemat(osp.join('./result/parallel-8ch/'+'PKI_EBM_rec.mat'),{'im_complex':im_complex})
                write_Data(i,write_psnr,write_ssim)                            
            
            #print('Iter time:',end2-start2,' s')   
        end1 = time.time()   
        print('====================================================')
        #print('Total time:',end1-start1,' s')


        
