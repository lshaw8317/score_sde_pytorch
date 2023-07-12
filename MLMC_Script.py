# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 10:05:03 2023

@author: lshaw
"""

from models import utils as mutils
import os
import gc
import io
import numpy as np
import tensorflow as tf
import tensorflow_gan as tfgan
import logging
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.legend import Legend

# Keep the import below for registering all model definitions
from models import ddpm, ncsnv2, ncsnpp
import datasets
import evaluation
import likelihood
import sde_lib
import torch
from torch.utils import tensorboard
from torchvision.utils import make_grid, save_image
from utils import save_checkpoint, restore_checkpoint
from configs.vp import cifar10_ddpmpp_continuous as config

def imagenorm(img):
    s=img.shape
    n=torch.linalg.vector_norm(img,dim=(-3,-2,-1)) #flattens and calculates norm
    n/=np.prod(s[-3:])
    return n

# Create data normalizer and its inverse
denoise=True
alpha_0=0;beta_0=0 #orders of convergence of sde solvers
scaler = datasets.get_data_scaler(config)
inverse_scaler = datasets.get_data_inverse_scaler(config)

# Use inceptionV3 for images with resolution higher than 256.
inceptionv3 = config.data.image_size >= 256
inception_model = evaluation.get_inception_model(inceptionv3=inceptionv3)

ckpt=8
workdir='exp'
checkpoint_dir = os.path.join(workdir, "checkpoints")
ckpt_dir = os.path.join(checkpoint_dir, f'checkpoint_{ckpt}.pth')
eval_dir = os.path.join(workdir, 'eval')
tf.io.gfile.makedirs(eval_dir)

sampling_eps = 1e-3

# Initialize model
model = mutils.create_model(config)
loaded_state = torch.load(ckpt_dir, map_location=config.device)
model.load_state_dict(loaded_state['model'], strict=False)

sampling_shape = (config.eval.batch_size,
                  config.data.num_channels,
                  config.data.image_size, config.data.image_size)

sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
score_fn=mutils.get_score_fn(sde, model)
rsde = sde.reverse(score_fn, probability_flow=False)

def EulerMaruyama(x, t, dt, dW):
    drift, diffusion = rsde.sde(x, t)
    x_mean = x + drift * dt
    x = x_mean + diffusion[:, None, None, None] * dW
    return x, x_mean

def mlmc_sample(bs,l,M,sde=sde,sampling_eps=sampling_eps,sampling_shape=sampling_shape,denoise=False):
    """ 
    The path function for Euler-Maruyama diffusion, which calculates final samples \sim p(x_0).

    Parameters:
        bs(int): batch size to generate number of samples
        l(int) : discretisation level
        M(int) : coarseness factor, number of fine steps = M**l
    Returns:
        Xf,Xc (numpy.array) : final samples for N_loop sample paths (Xc=X0 if l==0)
    """
    xf = sde.prior_sampling((bs,sampling_shape[1:])).to(config.device)
    xc = xf.clone().detach().to(config.device)
    Nf=M**l
    #Nc=M**(l-1) implicitly
    fine_timesteps = torch.linspace(sde.T, sampling_eps,Nf, device=xf.device)
    dWc=torch.zeros_like(xf).to(config.device)
    dtc=0
    tc=sde.T
    for i in range(Nf-1):
      tf = fine_timesteps[i]
      dt=fine_timesteps[i+1]-tf
      dtc+=dt #running sum of coarse timestep
      vec_t = torch.ones(sampling_shape[0], device=tf.device) * tf
      
      dWf = torch.randn_like(xf)*np.sqrt(-dt)
      dWc+=dWf
      xf,xf_mean=EulerMaruyama(xf,vec_t,dt,dWf)
      if i%M==0: #if i is integer multiple of M...
          vec_t = torch.ones(sampling_shape[0], device=tc.device) * tc
          xc,xc_mean=EulerMaruyama(xc,vec_t,dtc,dWc) #...Develop coarse path
          dWc=torch.zeros_like(xc) #...Re-initialise coarse BI to 0
          tc=tf #coarse solution now advanced to current fine time
          dtc=0
    if denoise:
        return inverse_scaler(xf_mean),inverse_scaler(xc_mean)
    else:
        return inverse_scaler(xf),inverse_scaler(xc)

def looper(Nl,l,M):
    """ 
    Interfaces with mlmc function to implement loop over Nl samples and generate payoff sums.
  
    Parameters:
        Nl(int): total number of sample paths to generate
        l(int) : discretisation level
        M(int) : coarseness factor, number of fine steps = M**l
    Returns:
        sums,sqsums (torch.Tensors) = [np.sum(dP_l),np.sum(dP_l**2),sumPf,sumPf2,sumPc,sumPc2,sum(Pf*Pc)]
        3d and 4d Tensors of various payoff sums and payoff-squared sums for Nl samples at level l/l-1
        Returns [sumPf,sumPf2,sumPf,sumPf2,0,0,0] is l=0.
     """
    sqsums=torch.zeros((4,1))
    sums=torch.zeros((3,sampling_shape[-3:]))
    num_sampling_rounds = Nl // config.eval.batch_size + 1
    numrem=Nl % config.eval.batch_size
    for r in range(num_sampling_rounds):
        bs=numrem if r==num_sampling_rounds-1 else config.eval.batch_size

        Xf,Xc=mlmc_sample(bs,l,M) #should automatically use cuda
            
        sumXf=torch.sum(Xf,axis=0) #sum over batch size
        sumXf2=torch.sum(imagenorm(Xf)**2,axis=0)
        if l==0:
            sums[:4,...]+=torch.stack([sumXf,sumXf2,sumXf,sumXf2])
        else:
            dX_l=Xf-Xc #Image difference
            sumdX_l=torch.sum(dX_l,axis=0) #sum over batch size
            sumdX_l2=torch.sum(imagenorm(dX_l)**2,axis=0)
            sumXc=torch.sum(Xc,axis=0)
            sumXc2=torch.sum(imagenorm(Xc)**2,axis=0)
            sumXcXf=torch.sum(Xc*Xf)
            sums+=torch.stack([sumdX_l,sumXf,sumXc])
            sqsums+=torch.stack([sumdX_l2,sumXf2,sumXc2,sumXcXf])
    logging.info("sampling -- ckpt: %d, round: %d" % (ckpt, r))

    # Directory to save samples. Repeatedly overwrites, just to save some example samples for debugging
    this_sample_dir = os.path.join(eval_dir, f"level_{l}")
    if not tf.io.gfile.exists(this_sample_dir):
        tf.io.gfile.makedirs(this_sample_dir)
    samples=np.clip(Xf.permute(0, 2, 3, 1).cpu().numpy() * 255., 0, 255).astype(np.uint8)
    samples = samples.reshape(
      (-1, config.data.image_size, config.data.image_size, config.data.num_channels))
    # Write samples to disk or Google Cloud Storage
    with tf.io.gfile.GFile(os.path.join(this_sample_dir, "samples.npz"), "wb") as fout:
      io_buffer = io.BytesIO()
      np.savez_compressed(io_buffer, samples=samples)
      fout.write(io_buffer.getvalue())
            
    return sums,sqsums 

##MLMC function
def mlmc(acc,M=2,N0=10**2, warm_start=True):
    """
    Runs MLMC algorithm which returns an array of sums at each level.
    ________________
    ________________
    
    Parameters:
        acc(float) : desired accuracy
        M(int) = 2 : coarseness factor
        N0(int) = 10**3 : default number of samples to use when initialising new level
        warm_start(bool) = True: whether to save calculated alpha as alpha_0 for future function calls

    Returns: sums=[np.sum(dP_l),np.sum(dP_l**2),sumPf,sumPf2,sumPc,sumPc2,sum(Pf*Pc)],N
        sums(np.array) : sums of payoff diffs at each level and sum of payoffs at fine level, each column is a level
        N(np.array of ints) : final number of samples at each level
    """
    global alpha_0,beta_0

    #Orders of convergence
    alpha=max(0,alpha_0)
    beta=max(0,beta_0)
    
    L=2

    V=torch.zeros(L+1) #Initialise variance vector of each levels' variance
    N=torch.zeros(L+1) #Initialise num. samples vector of each levels' num. samples
    dN=N0*torch.ones(L+1) #Initialise additional samples for this iteration vector for each level
    sqsums=torch.zeros((L+1,4)) #Initialise sqsums array of normed [dX^2,Xf^2,Xc^2,XcXf], each column is a level
    sqrt_h=torch.sqrt(M**(torch.arange(0,L+1)))
    sums=torch.zeros((L+1,3,sampling_shape[-3:])) #Initialise sums array of unnormed [dX,Xf,Xc], each column is a level

    while (torch.sum(dN)>0): #Loop until no additional samples asked for
        for l in range(L+1):
            num=dN[l]
            if num>0: #If asked for additional samples...
                tempsums,tempsqsums=looper(int(num),l,M) #Call function which gives sums
                sqsums[l,...]+=tempsqsums
                sums[l,...]+=tempsums
                
        N+=dN #Increment samples taken counter for each level
        Yl=imagenorm(sums[:,0]/N)
        V=torch.maximum((sqsums[:,0]/N)-(Yl)**2,0) #Calculate variance based on updated samples
        
        ##Fix to deal with zero variance or mean by linear extrapolation
        # Yl[3:]=np.maximum(Yl[3:],Yl[2:L]*M**(-alpha))
        # V[3:]=np.maximum(V[3:],V[2:L]*M**(-beta))
        
        if alpha_0==0: #Estimate order of weak convergence using LR
            #Yl=(M^alpha-1)khl^alpha=(M^alpha-1)k(TM^-l)^alpha=((M^alpha-1)kT^alpha)M^(-l*alpha)
            #=>log(Yl)=log(k(M^alpha-1)T^alpha)-alpha*l*log(M)
            X=torch.ones((L,2))
            X[:,0]=torch.arange(1,L+1)
            a = torch.linalg.lstsq(X,torch.log(Yl[1:]),rcond=None)[0]
            alpha = max(0.5,-a[0]/torch.log(M))
        if beta_0==0: #Estimate order of variance convergence using LR
            X=torch.ones((L,2))
            X[:,0]=torch.arange(1,L+1)
            b = torch.linalg.lstsq(X,torch.log(V[1:]),rcond=None)[0]
            beta = max(0.5,-b[0]/torch.log(M))

        sqrt_V=torch.sqrt(V)
        Nl_new=torch.ceil((2*acc**-2)*torch.sum(sqrt_V*sqrt_h)*(sqrt_V/sqrt_h)) #Estimate optimal number of samples/level
        dN=torch.maximum(0,Nl_new-N) #Number of additional samples
    
        if sum(dN > 0.01*N) == 0: #Almost converged
            if max(Yl[-2]/(M**alpha),Yl[-1])>(M**alpha-1)*acc*torch.sqrt(0.5):
                L+=1
                #Add extra entries for the new level and estimate sums with N0 samples 
                V=torch.concatenate((V,torch.zeros(1)), axis=0)
                N=torch.concatenate((N,N0*torch.zeros(1)),axis=0)
                dN=torch.concatenate((dN,N0*torch.ones(1)),axis=0)
                sqrt_h=torch.concatenate((sqrt_h,[torch.sqrt(M**L)]),axis=0)
                sums=torch.concatenate((sums,torch.zeros_like(sums[0])),axis=0)
                sqsums=torch.concatenate((sqsums,torch.zeros_like(sqsums[0])),axis=0)
                
    print(f'Estimated alpha = {alpha}')
    print(f'Estimated beta = {beta}')

    if warm_start:
        alpha_0=alpha #update with estimate of option alpha
        beta_0=beta #update with estimate of option beta
        print(f'    Saved estimated alpha_0 = {alpha}')
        print(f'    Saved estimated beta_0 = {beta}')
    return sums,sqsums,N

def Giles_plot(acc,markers,M=2,N0=10**3,Lmax=8,Nsamples=10**5):
    """
    Plots variance/mean and cost/number of levels plots a la Giles 2008.
    
    Parameters:
        acc(list-like) : desired accuracy of MLMC
        label(str) : plot title
        fig : figure to plot onto
        M(int) = 2 : coarseness factor
        N0(int) = 10**3 : min samples per level
        Lmax(int) = 8 : max level to estimate variance/mean of P_l-P_l-1
        Nsamples(int) = 10**5 : number of samples to use to estimate variance/mean
    """
    #Set plotting params
    fig,_=plt.subplots(2,2)
    label='Testing MLMC Diffusion Models'
    markersize=(fig.get_size_inches()[0])
    if len(acc)!=len(markers):
        raise ValueError("Length of markers argument must be same as length of accuracy argument.")
    axis_list=fig.axes
    if len(axis_list)!=4:
        print('Expected 4 subplots in fig, attempting to proceed but may fail.')
    
    #Initialise complexity lists
    cost_mlmc=[]
    cost_mc=[]
    
    #Do the calculations and simulations for num levels and complexity plot
    for i in range(len(acc)):
        e=acc[i]
        sums,sqsums,N=mlmc(e,M,warm_start=False,N0=N0) #sums=[dX,Xf,Xc], sqsums=[||dX||^2,||Xf||^2,||Xc||^2]
        L=len(N)-1
        means_p=imagenorm(sums[1,:]/N) #Norm of mean of fine discretisations
        V_p=(sqsums[1,:]/N)-means_p**2
        
        cost_mlmc+=[torch.sum(N*(M**np.arange(0,L+1)))*e**2]
        cost_mc+=[2*torch.sum(V_p*(M**np.arange(L+1)))]
        
        axis_list[2].semilogy(range(L+1),N,'k-',marker=markers[i],label=f'{e}',markersize=markersize,
                       markerfacecolor="None",markeredgecolor='k', markeredgewidth=1)
        
        # Directory to save means, norms and N
        this_sample_dir = os.path.join(eval_dir, f"M_{M}_accuracy_{e}")
        if not tf.io.gfile.exists(this_sample_dir):
            tf.io.gfile.makedirs(this_sample_dir)        
        # Write samples to disk or Google Cloud Storage
        with tf.io.gfile.GFile(os.path.join(this_sample_dir, "averages.pt"), "wb") as fout:
            io_buffer = io.BytesIO()
            torch.save(sums/N,io_buffer)
            fout.write(io_buffer.getvalue())
        # Write samples to disk or Google Cloud Storage
        with tf.io.gfile.GFile(os.path.join(this_sample_dir, "sqaverages.pt"), "wb") as fout:
            io_buffer = io.BytesIO()
            torch.save(sqsums/N,io_buffer)
            fout.write(io_buffer.getvalue())
        # Write samples to disk or Google Cloud Storage
        with tf.io.gfile.GFile(os.path.join(this_sample_dir, "N.pt"), "wb") as fout:
            io_buffer = io.BytesIO()
            torch.save(N,io_buffer)
            fout.write(io_buffer.getvalue())
        
    #Variance and mean samples
    sums=torch.zeros((Lmax+1,sums.shape[1:]))
    sqsums=torch.zeros((Lmax+1,sqsums.shape[1:]))

    for l in range(Lmax+1):
        sums[l],sqsums[l] = looper(Nsamples,l,M)
    
    means_p=imagenorm(sums[:,1]/Nsamples)
    V_p=(sqsums[:,1]/Nsamples)-means_p**2 
    means_dp=imagenorm(sums[:,0]/Nsamples)
    V_dp=(sqsums[:,0]/Nsamples)-means_dp**2  
    
    #Plot variances
    axis_list[0].plot(range(Lmax+1),np.log(V_p)/np.log(M),'k:',label='$P_{l}$',
                      marker=(8,2,0),markersize=markersize,markerfacecolor="None",markeredgecolor='k', markeredgewidth=1)
    axis_list[0].plot(range(1,Lmax+1),np.log(V_dp[1:])/np.log(M),'k-',label='$P_{l}-P_{l-1}$',
                      marker=(8,2,0), markersize=markersize, markerfacecolor="None", markeredgecolor='k',
                      markeredgewidth=1)
    #Plot means
    axis_list[1].plot(range(Lmax+1),np.log(means_p)/np.log(M),'k:',label='$P_{l}$',
                      marker=(8,2,0), markersize=markersize, markerfacecolor="None",markeredgecolor='k',
                      markeredgewidth=1)
    axis_list[1].plot(range(1,Lmax+1),np.log(means_dp[1:])/np.log(M),'k-',label='$P_{l}-P_{l-1}$',
                      marker=(8,2,0),markersize=markersize,markerfacecolor="None",markeredgecolor='k', markeredgewidth=1)
    
    # Directory to save means and norms
    this_sample_dir = os.path.join(eval_dir, f"VarMean_M_{M}_Nsamples_{Nsamples}")
    if not tf.io.gfile.exists(this_sample_dir):
        tf.io.gfile.makedirs(this_sample_dir)        
    
    # Write samples to disk or Google Cloud Storage
    with tf.io.gfile.GFile(os.path.join(this_sample_dir, "averages.pt"), "wb") as fout:
        io_buffer = io.BytesIO()
        torch.save(sums/Nsamples,io_buffer)
        fout.write(io_buffer.getvalue())
    with tf.io.gfile.GFile(os.path.join(this_sample_dir, "sqaverages.pt"), "wb") as fout:
        io_buffer = io.BytesIO()
        torch.save(sqsums/Nsamples,io_buffer)
        fout.write(io_buffer.getvalue())
        
    #Estimate orders of weak (alpha from means) and strong (beta from variance) convergence using LR
    X=np.ones((Lmax,2))
    X[:,0]=np.arange(1,Lmax+1)
    a = np.linalg.lstsq(X,np.log(means_dp[1:]),rcond=None)[0]
    alpha = -a[0]/np.log(M)
    b = np.linalg.lstsq(X,np.log(V_dp[1:]),rcond=None)[0]
    beta = -b[0]/np.log(M) 
    
    with open(os.path.join(this_sample_dir, "info_text.txt"),'w') as f:
        f.write(f'Estimated alpha={alpha}\n Estimated beta={beta}')
                
    #Label variance plot
    axis_list[0].set_xlabel('$l$')
    axis_list[0].set_ylabel(f'log$_{M}$(var)')
    axis_list[0].legend(framealpha=0.6, frameon=True)
    axis_list[0].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    #Add estimated beta
    # s='$\\beta$ = {}'.format(round(beta,2))
    # t = axis_list[0].annotate(s, (Lmax/2, np.log(V_dp[2])/np.log(M)),fontsize=markersize,
    #         size=2*markersize, bbox=dict(ec='None',facecolor='None',lw=2))
    
    #Label means plot
    axis_list[1].set_xlabel('$l$')
    axis_list[1].set_ylabel(f'log$_{M}$(mean)')
    axis_list[1].legend(framealpha=0.6, frameon=True)
    axis_list[1].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    #Add estimated alpha
    # s='$\\alpha$ = {}'.format(round(alpha,2))
    # t = axis_list[1].annotate(s, (Lmax/2, np.log(means_dp[1])/np.log(M)), fontsize=markersize,
    #         size=2*markersize, bbox=dict(ec='None',facecolor='None',lw=2))
    
    #Label number of levels plot
    axis_list[2].set_xlabel('$l$')
    axis_list[2].set_ylabel('$N_l$')
    xa=axis_list[2].xaxis
    xa.set_major_locator(ticker.MaxNLocator(integer=True))
    (lines,labels)=axis_list[2].get_legend_handles_labels()
    ncol=1
    leg = Legend(axis_list[2], lines, labels, ncol=ncol, title='Accuracy',
                 frameon=True, framealpha=0.6)
    leg._legend_box.align = "right"
    axis_list[2].add_artist(leg)
        
    #Label and plot complexity plot
    axis_list[3].loglog(acc,cost_mc,'k:',marker=(8,2,0),markersize=markersize,
                 markerfacecolor="None",markeredgecolor='k', markeredgewidth=1,label='Std. MC')
    axis_list[3].loglog(acc,cost_mlmc,'k-',marker=(8,2,0),markersize=markersize,
                 markerfacecolor="None",markeredgecolor='k', markeredgewidth=1,label='Std. MLMC')
    axis_list[3].set_xlabel('Acc. $a$')
    axis_list[3].set_ylabel('$a^{2}$cost')
    axis_list[3].legend(frameon=True,framealpha=0.6)
    
    #Add title and space out subplots
    fig.suptitle(label+f'\n$M={M}$')
    fig.tight_layout(rect=[0, 0.03, 1, 0.94],h_pad=1,w_pad=1,pad=1)
    
    
    with open(os.path.join(eval_dir,'GilesPlot.pdf'),'w') as f:
        plt.savefig(f, format='pdf', bbox_inches='tight')
    return None

# eval_dir = os.path.join(workdir, 'eval')
# tf.io.gfile.makedirs(eval_dir)
# for r in range(num_sampling_rounds):
#     logging.info("sampling -- ckpt: %d, round: %d" % (ckpt, r))

#     # Directory to save samples. Different for each host to avoid writing conflicts
#     this_sample_dir = os.path.join(
#       eval_dir, f"ckpt_{ckpt}")
#     tf.io.gfile.makedirs(this_sample_dir)
#     samples = sample()
#     samples = np.clip(samples.permute(0, 2, 3, 1).cpu().numpy() * 255., 0, 255).astype(np.uint8)
#     samples = samples.reshape(
#       (-1, config.data.image_size, config.data.image_size, config.data.num_channels))
#     # Write samples to disk or Google Cloud Storage
#     with tf.io.gfile.GFile(
#         os.path.join(this_sample_dir, f"samples_{r}.npz"), "wb") as fout:
#       io_buffer = io.BytesIO()
#       np.savez_compressed(io_buffer, samples=samples)
#       fout.write(io_buffer.getvalue())
      
    # # Force garbage collection before calling TensorFlow code for Inception network
    # gc.collect()
    # latents = evaluation.run_inception_distributed(samples, inception_model,
    #                                                inceptionv3=inceptionv3)
    # # Force garbage collection again before returning to JAX code
    # gc.collect()
    # # Save latent represents of the Inception network to disk or Google Cloud Storage
    # with tf.io.gfile.GFile(
    #     os.path.join(this_sample_dir, f"statistics_{r}.npz"), "wb") as fout:
    #   io_buffer = io.BytesIO()
    #   np.savez_compressed(
    #     io_buffer, pool_3=latents["pool_3"], logits=latents["logits"])
    #   fout.write(io_buffer.getvalue())

# Compute inception scores, FIDs and KIDs.
# Load all statistics that have been previously computed and saved for each host
# all_logits = []
# all_pools = []
# this_sample_dir = os.path.join(eval_dir, f"ckpt_{ckpt}")
# stats = tf.io.gfile.glob(os.path.join(this_sample_dir, "statistics_*.npz"))
# for stat_file in stats:
#   with tf.io.gfile.GFile(stat_file, "rb") as fin:
#     stat = np.load(fin)
#     if not inceptionv3:
#       all_logits.append(stat["logits"])
#     all_pools.append(stat["pool_3"])

# if not inceptionv3:
#   all_logits = np.concatenate(all_logits, axis=0)[:config.eval.num_samples]
# all_pools = np.concatenate(all_pools, axis=0)[:config.eval.num_samples]

# # Load pre-computed dataset statistics.
# data_stats = evaluation.load_dataset_stats(config)
# data_pools = data_stats["pool_3"]

# # Compute FID/KID/IS on all samples together.
# if not inceptionv3:
#   inception_score = tfgan.eval.classifier_score_from_logits(all_logits)
# else:
#   inception_score = -1

# fid = tfgan.eval.frechet_classifier_distance_from_activations(
#   data_pools, all_pools)
# # Hack to get tfgan KID work for eager execution.
# tf_data_pools = tf.convert_to_tensor(data_pools)
# tf_all_pools = tf.convert_to_tensor(all_pools)
# kid = tfgan.eval.kernel_classifier_distance_from_activations(
#   tf_data_pools, tf_all_pools).numpy()
# del tf_data_pools, tf_all_pools

# logging.info(
#   "ckpt-%d --- inception_score: %.6e, FID: %.6e, KID: %.6e" % (
#     ckpt, inception_score, fid, kid))

# with tf.io.gfile.GFile(os.path.join(eval_dir, f"report_{ckpt}.npz"),
#                        "wb") as f:
#   io_buffer = io.BytesIO()
#   np.savez_compressed(io_buffer, IS=inception_score, fid=fid, kid=kid)
#   f.write(io_buffer.getvalue())

   

