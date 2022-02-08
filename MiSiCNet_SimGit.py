# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 19:13:50 2021

@author: behnood
"""

#from __future__ import print_function
import matplotlib.pyplot as plt
#%matplotlib inline
# from numpy import linalg as LA
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import numpy as np
#from models import *
#import math
import torch
import torch.optim
import torch.nn as nn

# from skimage.measure import compare_psnr
# from skimage.measure import compare_mse
#from utils.denoising_utils import *

# from skimage._shared import *
# from skimage.util import *
# from skimage.metrics.simple_metrics import _as_floats
# from skimage.metrics.simple_metrics import mean_squared_error

#from UtilityMine import add_noise
# from UtilityMine import find_endmember
# from UtilityMine import add_noise
from UtilityMine import *
# from VCA import *
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

PLOT = True
#%% Load image
import scipy.io
import scipy.linalg
#%%
fname2  = "HS Data/Sim1/Y_clean.mat"
mat2 = scipy.io.loadmat(fname2)
img_np_gt = mat2["Y_clean"]
img_np_gt = img_np_gt.transpose(2,0,1)
[p1, nr1, nc1] = img_np_gt.shape
#%%
# fname3  = "C:/Users/behnood/Desktop/VMCNN/Easy/A_true.mat"
# mat3 = scipy.io.loadmat(fname3)
# A_true_np = mat3["A_true"]
# A_true_np = A_true_np.transpose(2,0,1)
#%%
# fname4  = "C:/Users/behnood/Desktop/VMCNN/Easy/E.mat"
# mat4 = scipy.io.loadmat(fname4)
# E_np = mat4["E"]
rmax=6#E_np.shape[1] 
#%%
npar=np.zeros((1,4))
npar[0,0]=17.5
npar[0,1]=55.5
npar[0,2]=175
npar[0,3]=555
tol1=npar.shape[1]
tol2=1
save_result=False
from tqdm import tqdm

for fi in tqdm(range(tol1)):
    for fj in tqdm(range(tol2)):
            #%%
        #img_noisy_np = get_noisy_image(img_np_gt, 1/10)
        img_noisy_np = add_noise(img_np_gt, 1/npar[0,fi])#11.55 20 dB, 36.7 30 dB, 116.5 40 dB
        #print(compare_snr(img_np_gt, img_noisy_np))
        img_resh=np.reshape(img_noisy_np,(p1,nr1*nc1))
        V, SS, U = scipy.linalg.svd(img_resh, full_matrices=False)
        PC=np.diag(SS)@U
        # img_resh_DN=V[:,:rmax]@PC[:rmax,:]
        img_resh_DN=V[:,:rmax]@V[:,:rmax].transpose(1,0)@img_resh
        img_resh_np_clip=np.clip(img_resh_DN, 0, 1)
        II,III = Endmember_extract(img_resh_np_clip,rmax)
        E_np1=img_resh_np_clip[:,II]
        #%% Set up Simulated 
        INPUT = 'noise' # 'meshgrid'
        pad = 'reflection'
        need_bias=True
        OPT_OVER = 'net' 
        
        # 
        LR1 = 0.001
        show_every = 100
        exp_weight=0.99
        
        num_iter1 = 8000
        input_depth =  img_noisy_np.shape[0]
        class CAE_EndEst(nn.Module):
            def __init__(self):
                super(CAE_EndEst, self).__init__()
                self.conv1 = nn.Sequential(
                    conv(input_depth, 256,3,1,bias=need_bias, pad=pad),
                    nn.BatchNorm2d(256,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.LeakyReLU(0.1, inplace=True),
                )
                self.conv2 = nn.Sequential(
                    conv(256, 256,3,1,bias=need_bias, pad=pad),
                    nn.BatchNorm2d(256,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.LeakyReLU(0.1, inplace=True),
                )
                self.conv3 = nn.Sequential(
                    conv(input_depth, 4, 1,1,bias=need_bias, pad=pad),
                    nn.BatchNorm2d(4,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.LeakyReLU(0.1, inplace=True),
                )
                self.dconv2 = nn.Sequential(
                    nn.Upsample(scale_factor=1),
                    conv(260, 256, 3,1,bias=need_bias, pad=pad),
                    nn.BatchNorm2d(256,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.LeakyReLU(0.1, inplace=True),
                )
        
                self.dconv3 = nn.Sequential(
                    nn.Upsample(scale_factor=1),
                    conv(256, rmax, 3,1,bias=need_bias, pad=pad),
                    nn.BatchNorm2d(rmax,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.Softmax(),
                )
                self.dconv4 = nn.Sequential(
                    nn.Linear(rmax, p1,bias=False),
                )
            def forward(self, x):
                x1 = self.conv3(x)
                x = self.conv1(x)
                x = torch.cat([x,x1], 1)
                x = self.dconv2(x)
                x2 = self.dconv3(x)
                x3 = torch.transpose(x2.view((rmax,nr1*nc1)),0,1)
                x3 = self.dconv4(x3)
                return x2,x3

        net1 = CAE_EndEst()
        net1.cuda()
        
        # Loss
        def my_loss(target, End2, lamb, out_):
            loss1 = 0.5*torch.norm((out_.transpose(1,0).view(1,p1,nr1,nc1) - target), 'fro')**2
            O = torch.mean(target.view(p1,nr1*nc1),1).type(dtype).view(p1,1)
            B = torch.from_numpy(np.identity(rmax)).type(dtype)
            loss2 = torch.norm(torch.mm(End2,B.view((rmax,rmax)))-O, 'fro')**2
            return loss1+lamb*loss2
        img_noisy_torch = torch.from_numpy(img_resh_DN).view(1,p1,nr1,nc1).type(dtype)
        net_input1 = get_noise(input_depth, INPUT,
            (img_noisy_np.shape[1], img_noisy_np.shape[2])).type(dtype).detach()
        E_torch = torch.from_numpy(E_np1).type(dtype)
        #%%
        # net_input_saved = net_input1.detach().clone()
        # noise = net_input1.detach().clone()
        out_avg = True
        
        i = 0
        def closure1():
            
            global i, out_LR_np, out_avg, out_avg_np, Eest
            
            out_LR,out_spec = net1(net_input1)
#            out_HR=torch.mm(E_torch.view(p1,rmax),out_LR.view(rmax,nr1*nc1))
            # Smoothing
            if out_avg is None:
                out_avg = out_LR.detach()
                # out_HR_avg = out_HR.detach()
            else:
                out_avg = out_avg * exp_weight + out_LR.detach() * (1 - exp_weight)
                # out_HR_avg = out_HR_avg * exp_weight + out_HR.detach() * (1 - exp_weight)

        #%%
            total_loss = my_loss(img_noisy_torch, net1.dconv4[0].weight,.1,out_spec)
            total_loss.backward()
         
          

            # print ('Iteration %05d    Loss %f   RMSE_LR: %f   RMSE_LR_avg: %f  SRE: %f SRE_avg: %f' % (i, total_loss.item(), RMSE_LR, RMSE_LR_avg, SRE, SRE_avg), '\r', end='')
            if  PLOT and i % show_every == 0:
                out_LR_np = out_LR.detach().cpu().squeeze().numpy()
                out_avg_np = out_avg.detach().cpu().squeeze().numpy()
                out_LR_np = np.clip(out_LR_np, 0, 1)
                out_avg_np = np.clip(out_avg_np, 0, 1)    
                f, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2, 2, sharey=True, figsize=(10,10))
                ax1.imshow(np.stack((out_LR_np[2,:,:],out_LR_np[1,:,:],out_LR_np[0,:,:]),2))
                ax2.imshow(np.stack((out_LR_np[5,:,:],out_LR_np[4,:,:],out_LR_np[3,:,:]),2))
                ax3.imshow(np.stack((out_avg_np[2,:,:],out_avg_np[1,:,:],out_avg_np[0,:,:]),2))
                ax4.imshow(np.stack((out_avg_np[5,:,:],out_avg_np[4,:,:],out_avg_np[3,:,:]),2))
                plt.show()                   
            i += 1       
            return total_loss
        net1.dconv4[0].weight=torch.nn.Parameter(E_torch.view(p1,rmax))       
        p11 = get_params(OPT_OVER, net1, net_input1)
        optimizer = torch.optim.Adam(p11, lr=LR1, betas=(0.9, 0.999), eps=1e-8,
                  weight_decay= 0, amsgrad=False)
        for j in range(num_iter1):
                optimizer.zero_grad()
                closure1()  
                optimizer.step()
                net1.dconv4[0].weight.data[net1.dconv4[0].weight <= 0] = 0
                net1.dconv4[0].weight.data[net1.dconv4[0].weight >= 1] = 1
                if j>0:
                  Eest=net1.dconv4[0].weight.detach().cpu().squeeze().numpy()
                  if j % show_every== 0: 
                    plt.plot(Eest)
                    plt.show()
                  
        out_avg_np = out_avg.detach().cpu().squeeze().numpy()
       

    #%%
        if  save_result is True:
                  scipy.io.savemat("Result/EestdB%01d%01d.mat" % (fi+2, fj+1),
                                    {'Eest%01d%01d' % (fi+2, fj+1):Eest})
                  scipy.io.savemat("Result/out_avg_npdB%01d%01d.mat" % (fi+2, fj+1),
                                    {'out_avg_np%01d%01d' % (fi+2, fj+1):out_avg_np.transpose(1,2,0)})
        #
