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
plt.rc('text', usetex=False)
plt.rc('font',**{'serif':['cm']})
plt.style.use('seaborn-paper')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

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
alpha_0=0;beta_0=0

def imagenorm(img):
    s=img.shape
    n=torch.linalg.norm(torch.flatten(img, start_dim=-3, end_dim=-1),dim=-1) #flattens non-batch dims and calculates norm
    n/=np.prod(s[-3:])
    return n

def mlmc_test(config,eval_dir,checkpoint_dir):
    acc=config.mlmc.acc
    #config.device=torch.device("cuda:1")
    
    torch.cuda.empty_cache()
    # Create data normalizer and its inverse
    denoise=True
    alpha_0=0;beta_0=0 #orders of convergence of sde solvers
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)
    
    # Use inceptionV3 for images with resolution higher than 256.
    inceptionv3 = config.data.image_size >= 256
    inception_model = evaluation.get_inception_model(inceptionv3=inceptionv3)

    dirs=os.listdir(checkpoint_dir)
    ckpt = np.min(np.array([int(d.split('_')[-1][:-4]) for d in dirs]))
    ckpt_dir = os.path.join(checkpoint_dir, f'checkpoint_{ckpt}.pth')
    if not tf.io.gfile.exists(eval_dir):
        tf.io.gfile.makedirs(eval_dir)
    
    # Initialize model
    model = mutils.get_model(config.model.name)(config)
    loaded_state = torch.load(ckpt_dir, map_location='cpu')
    model.load_state_dict(loaded_state['model'], strict=False)
    model.to(config.device)
    #model = torch.nn.DataParallel(model,device_ids=[1,2,3])
    model=torch.nn.DataParallel(model)

    sampling_shape = (config.eval.batch_size,
                      config.data.num_channels,
                      config.data.image_size, config.data.image_size)
    if config.training.sde.lower() == 'vpsde':
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        def getbetas(x, t, dt):
            timestep = (t * (sde.N - 1) / sde.T).long()
            timestepm1 = ((t+dt) * (sde.N - 1) / sde.T).long()

            beta = sde.discrete_betas.to(x.device)[timestep]
            stdt =sde.sqrt_1m_alphas_cumprod.to(x.device)[timestep]
            stdtm1 =sde.sqrt_1m_alphas_cumprod.to(x.device)[timestepm1]
            
            return beta, stdt,stdtm1
        eta=config.mlmc.DDIM_eta
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'subvpsde':
        sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'vesde':
        sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
        sampling_eps = 1e-5
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    score_fn=mutils.get_score_fn(sde, model,continuous=config.training.continuous)
    rsde = sde.reverse(score_fn, probability_flow=False)
    
    def EulerMaruyama(x, t, dt, dW):
        drift, diffusion = rsde.sde(x, t)
        x_mean = x + drift * dt
        x = x_mean + diffusion[:, None, None, None] * dW
        return x, x_mean
    
    def DDIMSampler(x, t, dt, dW):
        beta, stdt, stdtm1 = getbetas(x,t[0],dt) #t should be vector of copies of times so just get first element
        stheta=score_fn(x,t)
        x_mean = (x + stheta)/torch.sqrt(1.-beta)-torch.sqrt(stdt**2-eta**2*beta)*stdtm1*stheta
        x = x_mean + eta * stdtm1*torch.sqrt(beta)/stdt*dW/torch.sqrt(-dt)
        return x, x_mean
    if config.mlmc.sampler.lower()=='ddim':
        samplerfun=DDIMSampler
    else:
        samplerfun=EulerMaruyama
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

        with torch.no_grad():
            xf = sde.prior_sampling((bs,*sampling_shape[-3:])).to(config.device)
            xc = xf.clone().detach().to(config.device)
            Nf=M**l
            #Nc=M**(l-1) implicitly
            fine_times = torch.linspace(sde.T, sampling_eps,Nf+1, device=xf.device,dtype=torch.float32)
            dWc=torch.zeros_like(xf).to(xc.device)
            dtc=0.
            tc=torch.tensor([sde.T],dtype=torch.float32).to(xc.device)
            for i in range(Nf):
                tf = fine_times[i]
                dt=fine_times[i+1]-tf
                dtc+=dt #running sum of coarse timestep
                vec_t = torch.ones(bs, device=tf.device, dtype=torch.float32) * tf
                dWf = torch.randn_like(xf)*torch.sqrt(-dt)
                dWc+=dWf
                xf,xf_mean=samplerfun(xf,vec_t,dt,dWf)
                if i%M==0: #if i is integer multiple of M...
                    vec_t = torch.ones(bs, device=tc.device,dtype=torch.float32) * tc
                    xc,xc_mean=samplerfun(xc,vec_t,dtc,dWc) #...Develop coarse path
                    dWc=torch.zeros_like(xc) #...Re-initialise coarse BI to 0
                    tc=tf.clone().detach() #coarse solution now advanced to current fine time
                    dtc=0.
            if denoise:
                return inverse_scaler(xf_mean),inverse_scaler(xc_mean)
            else:
                return inverse_scaler(xf),inverse_scaler(xc)
        
    def looper(Nl,l,M,min_l=0):
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
        sums=torch.zeros((3,*sampling_shape[-3:]))
        num_sampling_rounds = Nl // config.eval.batch_size + 1
        numrem=Nl % config.eval.batch_size
        for r in range(num_sampling_rounds):
            bs=numrem if r==num_sampling_rounds-1 else config.eval.batch_size
    
            Xf,Xc=mlmc_sample(bs,l,M) #should automatically use cuda
            sumXf=torch.sum(Xf,axis=0).to('cpu') #sum over batch size
            sumXf2=torch.sum(imagenorm(Xf)**2,axis=0).to('cpu').item()
            if l==min_l:
                sqsums+=torch.tensor([sumXf2,sumXf2,0,0]).reshape(sqsums.shape)
                sums+=torch.stack([sumXf,sumXf,torch.zeros_like(sumXf)])
            elif l<min_l:
                raise ValueError("l must be at least min_l")
            else:
                dX_l=Xf-Xc #Image difference
                sumdX_l=torch.sum(dX_l,axis=0).to('cpu') #sum over batch size
                sumdX_l2=torch.sum(imagenorm(dX_l)**2,axis=0).to('cpu').item()
                sumXc=torch.sum(Xc,axis=0).to('cpu')
                sumXc2=torch.sum(imagenorm(Xc)**2,axis=0).to('cpu').item()
                sumXcXf=torch.sum(Xc*Xf).to('cpu')
                sums+=torch.stack([sumdX_l,sumXf,sumXc])
                sqsums+=torch.tensor([sumdX_l2,sumXf2,sumXc2,sumXcXf]).reshape(sqsums.shape)
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
    def mlmc(acc,M=2,N0=10**2, warm_start=True,min_l=0):
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
        
        L=min_l+2

        mylen=L+1-min_l
        V=torch.zeros(mylen) #Initialise variance vector of each levels' variance
        N=torch.zeros(mylen) #Initialise num. samples vector of each levels' num. samples
        dN=N0*torch.ones(mylen) #Initialise additional samples for this iteration vector for each level
        sqsums=torch.zeros((mylen,4,1)) #Initialise sqsums array of normed [dX^2,Xf^2,Xc^2,XcXf], each column is a level
        sqrt_h=torch.sqrt(M**(torch.arange(min_l,L+1,dtype=torch.float32)))
        sums=torch.zeros((mylen,3,*sampling_shape[-3:])) #Initialise sums array of unnormed [dX,Xf,Xc], each column is a level
    
        while (torch.sum(dN)>0): #Loop until no additional samples asked for
            mylen=L+1-min_l
            for i,l in enumerate(torch.arange(min_l,L+1)):
                num=dN[i]
                if num>0: #If asked for additional samples...
                    tempsums,tempsqsums=looper(int(num),l,M,min_l=min_l) #Call function which gives sums
                    sqsums[i,...]+=tempsqsums
                    sums[i,...]+=tempsums
                    
            N+=dN #Increment samples taken counter for each level
            Yl=imagenorm(sums[:,0])/N
            V=torch.clip((sqsums[:,0].squeeze())/N-(Yl)**2,min=0) #Calculate variance based on updated samples
            
            ##Fix to deal with zero variance or mean by linear extrapolation
            # Yl[3:]=np.maximum(Yl[3:],Yl[2:L]*M**(-alpha))
            # V[3:]=np.maximum(V[3:],V[2:L]*M**(-beta))
            
            if alpha_0==0: #Estimate order of weak convergence using LR
                #Yl=(M^alpha-1)khl^alpha=(M^alpha-1)k(TM^-l)^alpha=((M^alpha-1)kT^alpha)M^(-l*alpha)
                #=>log(Yl)=log(k(M^alpha-1)T^alpha)-alpha*l*log(M)
                X=torch.ones((mylen-1,2))
                X[:,0]=torch.arange(min_l+1,L+1)
                a = torch.lstsq(torch.log(Yl[1:]),X)[0]
                alpha = max(-a[0]/np.log(M),.5)
            if beta_0==0: #Estimate order of variance convergence using LR
                X=torch.ones((mylen-1,2))
                X[:,0]=torch.arange(min_l+1,L+1)
                b = torch.lstsq(torch.log(V[1:]),X)[0]
                beta= max(-b[0]/np.log(M),.5)
    
            sqrt_V=torch.sqrt(V)
            Nl_new=torch.ceil((2*acc**-2)*torch.sum(sqrt_V*sqrt_h)*(sqrt_V/sqrt_h)) #Estimate optimal number of samples/level
            dN=torch.clip(Nl_new-N,min=0) #Number of additional samples
        
            if torch.sum(dN > 0.01*N).item() == 0: #Almost converged
                if max(Yl[-2]/(M**alpha),Yl[-1])>(M**alpha-1)*acc*np.sqrt(0.5):
                    L+=1
                    #Add extra entries for the new level and estimate sums with N0 samples 
                    V=torch.cat((V,torch.zeros(1)), dim=0)
                    N=torch.cat((N,N0*torch.zeros(1)),dim=0)
                    dN=torch.cat((dN,N0*torch.ones(1)),dim=0)
                    sqrt_h=torch.cat((sqrt_h,torch.tensor([M**(L/2)])),dim=0)
                    sums=torch.cat((sums,torch.zeros((1,*sums[0].shape))),dim=0)
                    sqsums=torch.cat((sqsums,torch.zeros((1,*sqsums[0].shape))),dim=0)
                    
        print(f'Estimated alpha = {alpha}')
        print(f'Estimated beta = {beta}')
    
        if warm_start:
            alpha_0=alpha #update with estimate of option alpha
            beta_0=beta #update with estimate of option beta
            print(f'    Saved estimated alpha_0 = {alpha}')
            print(f'    Saved estimated beta_0 = {beta}')
        return sums,sqsums,N
    
    def Giles_plot(acc):
        #Set plotting params
        M=config.mlmc.M
        N0=config.mlmc.N0
        Lmax=config.mlmc.Lmax
        Nsamples=config.mlmc.Nsamples
        print('Successfully called GilesPlot')
        
        #Initialise complexity lists
        cost_mlmc=[]
        cost_mc=[]
        min_l=config.mlmc.min_l
        '''
        #Do the calculations and simulations for num levels and complexity plot
        for i in range(len(acc)):
            e=acc[i]
            sums,sqsums,N=mlmc(e,M,warm_start=False,N0=N0,min_l=min_l) #sums=[dX,Xf,Xc], sqsums=[||dX||^2,||Xf||^2,||Xc||^2]
            L=len(N)-1+min_l
            means_p=imagenorm(sums[:,1])/N #Norm of mean of fine discretisations
            V_p=(sqsums[:,1]/N)-means_p**2
            
            #e^2*cost
            cost_mlmc+=[torch.sum(N*(M**np.arange(min_l,L+1)+np.hstack((0,M**np.arange(min_l,L)))))*e**2] #cost is number of NFE
            cost_mc+=[2*torch.sum(V_p*(M**np.arange(min_l,L+1)))]
            
            # Directory to save means, norms and N
            this_sample_dir = os.path.join(eval_dir, f"M_{M}_accuracy_{e}")
            if not tf.io.gfile.exists(this_sample_dir):
                tf.io.gfile.makedirs(this_sample_dir)        
            # Write samples to disk or Google Cloud Storage
            with tf.io.gfile.GFile(os.path.join(this_sample_dir, "averages.pt"), "wb") as fout:
                io_buffer = io.BytesIO()
                torch.save(sums/N[...,None,None,None,None],io_buffer)
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
                
            meanimg=torch.sum(sums[:,0]/N[...,None,None,None],axis=0)
            meanimg=np.clip(meanimg.permute(1, 2, 0).cpu().numpy() * 255., 0, 255).astype(np.uint8)
            # Write samples to disk or Google Cloud Storage
            with tf.io.gfile.GFile(os.path.join(this_sample_dir, "meanimg.npz"), "wb") as fout:
              io_buffer = io.BytesIO()
              np.savez_compressed(io_buffer, meanimg=meanimg)
              fout.write(io_buffer.getvalue())
        '''
        #Variance and mean samples
        sums=torch.zeros((1,3,*sampling_shape[1:]))
        sqsums=torch.zeros((1,4,1))
        sums=torch.zeros((Lmax+1-min_l,*sums.shape[1:]))
        sqsums=torch.zeros((Lmax+1-min_l,*sqsums.shape[1:]))
    
        for i,l in enumerate(torch.arange(min_l,Lmax+1)):
            sums[i],sqsums[i] = looper(Nsamples,l,M,min_l=min_l)
        
        means_p=imagenorm(sums[:,1]/Nsamples)
        V_p=(sqsums[:,1].squeeze()/Nsamples)-means_p**2 
        means_dp=imagenorm(sums[:,0]/Nsamples)
        V_dp=(sqsums[:,0].squeeze()/Nsamples)-means_dp**2  
        
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
        X=np.ones((Lmax-min_l,2))
        X[:,0]=np.arange(min_l+1,Lmax+1)
        a = np.linalg.lstsq(X,np.log(means_dp[1:]),rcond=None)[0]
        alpha = -a[0]/np.log(M)
        b = np.linalg.lstsq(X,np.log(V_dp[1:]),rcond=None)[0]
        beta = -b[0]/np.log(M) 
        
        with open(os.path.join(this_sample_dir, "info_text.txt"),'a') as f:
            f.write(f'Estimated alpha={alpha}\n Estimated beta={beta}\n')
        
        return None
    
    Giles_plot(acc)
    
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

   

