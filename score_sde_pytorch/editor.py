from tokenize import Exponent
from models import utils as mutils
import os
import gc
import io
import numpy as np
import tensorflow as tf
import tensorflow_gan as tfgan
import logging

# Keep the import below for registering all model definitions
import torch
from torch.utils import tensorboard
from torchvision.utils import make_grid, save_image
from utils import save_checkpoint, restore_checkpoint
import datasets
import evaluation
import likelihood
import sde_lib
from models import ddpm,ncsnv2,ncsnpp

def imagenorm(img):
    s=img.shape
    if len(s)==1: #fix for when img is single dimensional (batch_size,) -> (batch_size,1)                                
        img=img[:,None]
    n=torch.linalg.norm(torch.flatten(img, start_dim=1, end_dim=-1),dim=-1) #flattens non-batch dims and calculates norm
    n/=np.sqrt(np.prod(s[1:]))
    return n
def mom2norm(sqsums):
    #sqsums should have shape L,C,H,W or L,A for activations                                                                              
    s=sqsums.shape
    if len(s)==2: #fix for when activations is single dimensional (L,2048) -> (L,1,2048)                                                  
        sqsums=sqsums[:,None]
    return torch.sum(torch.flatten(sqsums, start_dim=1, end_dim=-1),dim=-1)/np.prod(s[1:])


# Directory to save means and norms

Nsamples=1000
M=2
for expdir in ['exp/eval/cifar10mean_DDIM0',
               'exp/eval/cifar10secondmoment_DDIM0',
               'exp/eval/cifar10mean_DDIM1',
               'exp/eval/cifar10secondmoment_DDIM1',
               'exp/eval/cifar10acts_DDIM1',
               'exp/eval/cifar10acts_DDIM0']:

    this_sample_dir = os.path.join(expdir,f"VarMean_M_{M}_Nsamples_{Nsamples}")
    with open(os.path.join(this_sample_dir, "averages.pt"), "rb") as fout:
        avgs=torch.load(fout)
    Lmax=len(avgs)-1
    with open(os.path.join(this_sample_dir, "sqaverages.pt"), "rb") as fout:
        sqavgs=torch.load(fout)
    
    
    means_p=imagenorm(avgs[:,1])
    V_p=mom2norm(sqavgs[:,1])-means_p**2 
    means_dp=imagenorm(avgs[:,0])
    V_dp=mom2norm(sqavgs[:,0])-means_dp**2  
    
    cutoff=np.argmax(V_dp<(np.sqrt(M)-1.)**2*V_p[-1]/(1+M))-1 #index of optimal lmin                                              
    means_p=means_p[cutoff:]
    V_p=V_p[cutoff:]
    means_dp=means_dp[cutoff:]
    V_dp=V_dp[cutoff:]
    
    X=np.ones((Lmax-cutoff,2))
    X[:,0]=np.arange(1.,Lmax-cutoff+1)
    a = np.linalg.lstsq(X,np.log(means_dp[1:]),rcond=None)[0]
    alpha = -a[0]/np.log(M)
    Y0=np.exp(a[1])
    b = np.linalg.lstsq(X,np.log(V_dp[1:]),rcond=None)[0]
    beta = -b[0]/np.log(M)
    
#    with open(os.path.join(this_sample_dir, "info_text.txt"),'a') as f:
 #       f.write(f'Estimated Y0={Y0}. Estimated Lmin={cutoff}.')
    
    with tf.io.gfile.GFile(os.path.join(this_sample_dir, "alphabetagamma.pt"), "wb") as fout:
        io_buffer = io.BytesIO()
        print(alpha,beta,1.,Y0,float(cutoff))
        torch.save(torch.tensor([alpha,beta,1.,Y0,float(cutoff.item())]),io_buffer)
        fout.write(io_buffer.getvalue())
