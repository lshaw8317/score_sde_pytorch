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
    #sqsums should have shape L,C,H,W
    s=sqsums.shape
    if len(s)!=4:
        raise Exception('shape of sqsums likely not LHCW')
    return torch.sum(torch.flatten(sqsums, start_dim=1, end_dim=-1),dim=-1)/np.prod(s[1:])

def activations_payoff(samples,inception_model,inceptionv3,config):
    samples=tf.convert_to_tensor(np.clip(samples.permute(0, 2, 3, 1).cpu().numpy() * 255., 0, 255).astype(np.uint8))
    # Force garbage collection before calling TensorFlow code for Inception network
    gc.collect()
    latents = evaluation.run_inception_distributed(samples, inception_model, inceptionv3=inceptionv3)
    # Force garbage collection again before returning to JAX code
    gc.collect()
    
    # Compute inception scores, FIDs and KIDs.
    # Load all statistics that have been previously computed and saved for each host
    # all_logits = latents["logits"]
    all_pools = latents["pool_3"]

    # if not inceptionv3:
    #   all_logits = np.concatenate(all_logits, axis=0)
    # all_pools = np.concatenate(all_pools, axis=0)
    
    # Load pre-computed dataset statistics.
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
    #   " --- inception_score: %.6e, FID: %.6e, KID: %.6e" % (inception_score, fid, kid))
    all_pools=tf.convert_to_tensor(all_pools).numpy()
    return torch.tensor(all_pools) #should have (batch_size, 2048)

def mlmc_test(config,eval_dir,checkpoint_dir,payoff_arg,acc=[],sampler='EM',adaptive=False, DDIMeta=0.,MLMC_=True):
    torch.cuda.empty_cache()
    tf.keras.backend.clear_session()
    
    # Create data normalizer and its inverse
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)
    
    # Use inceptionV3 for images with resolution higher than 256.
    inceptionv3 = config.data.image_size >= 256
    inception_model = evaluation.get_inception_model(inceptionv3=inceptionv3)
    accsplit=np.sqrt(.5) #default even bias-variance split
    config.model.num_scales=(config.mlmc.M)**(config.mlmc.Lmax)
    if payoff_arg=='activations':
        print('activations payoff selected for MLMC. Altering config file defaults correspondingly.')
        payoff = lambda samples: activations_payoff(samples, inception_model=inception_model, 
                                          inceptionv3=inceptionv3, config=config)
        config.mlmc.min_l=5
        config.eval.batch_size=128
        config.mlmc.N0=100
        accsplit=np.sqrt(0.01) #since beta<gamma, let error in bias be large and force error onto variance 
    elif payoff_arg=='variance':
        print('Pixel-wise variance payoff selected for MLMC. Altering config file defaults correspondingly.')
        config.mlmc.N0=1000
        config.mlmc.min_l=7
        config.eval.batch_size=1800
        payoff = lambda samples: torch.clip(samples,0.,1.)**2
        alpha_0=.8
        beta_0=1.5
    elif payoff_arg=='images':
        config.mlmc.N0=1000
        config.mlmc.min_l=7
        config.eval.batch_size=1800
        alpha_0=.8
        beta_0=1.5
        print('Setting payoff function to images for MLMC.')
        payoff = lambda samples: torch.clip(samples,0.,1.) #default to calculating mean image
    else:
        raise ValueError('payoff_arg not recognised. Should be one of variance, activations, images.')
    
    dirs=os.listdir(checkpoint_dir)
    ckpt = np.min(np.array([int(d.split('_')[-1][:-4]) for d in dirs]))
    ckpt_dir = os.path.join(checkpoint_dir, f'checkpoint_{ckpt}.pth')
    if not tf.io.gfile.exists(eval_dir):
        tf.io.gfile.makedirs(eval_dir)
    
    # Initialize model
    model = mutils.create_model(config)
    loaded_state = torch.load(ckpt_dir, map_location=config.device)
    model.load_state_dict(loaded_state['model'], strict=False)

    sampling_shape = (config.eval.batch_size,
                      config.data.num_channels,
                      config.data.image_size, config.data.image_size)
    if config.training.sde.lower() == 'vpsde':
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        def getbetas(x, t, dt):
            timestep = (t * (sde.N - 1) / sde.T).long()
            timestepm1 = ((t+dt) * (sde.N - 1) / sde.T).long()
            sqrtalphat=sde.sqrt_alphas_cumprod[timestep].to(x.device)
            sqrtalphatm1=sde.sqrt_alphas_cumprod[timestepm1].to(x.device)
            
            return sqrtalphat, sqrtalphatm1
        sampling_eps = 0
        def EIfactor(dt, t):
            #dt<0
            beta_t = sde.beta_0 + (t+.5*dt) * (sde.beta_1 - sde.beta_0)
            return torch.exp(.5*(-dt)*beta_t)
        
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
        #dt is negative
        drift, diffusion = rsde.sde(x, t)
        x_mean = x + drift * dt
        x = x_mean + diffusion[:, None, None, None] * dW
        return x, x_mean
    
    def TamedEulerMaruyama(x, t, dt, dW):
        drift, diffusion = rsde.sde(x, t)
        norm_diff=imagenorm(drift)[:,None,None,None]
        x_mean = x + drift * dt/(1-dt*norm_diff) #dt is negative so -dt=abs(dt)
        x = x_mean + diffusion[:, None, None, None] * dW
        return x, x_mean
    
    def SKROCK(x, t, dt, dW):
        return 
    
    def ExponentialIntegrator(x, t, dt, dW):
        #should only work for vpsde
        factor=EIfactor(dt,t)[:, None, None, None]
        stheta=score_fn(x,t)
        x_mean=factor*x+2*(factor-1.)*stheta
        x=x_mean+torch.sqrt(factor**2-1.)*dW/torch.sqrt(-dt)
        return x, x_mean

    def DDIMSampler(x, t, dt, dW,eta=DDIMeta):
        sat,satm1 = getbetas(x,t[0],dt) #t should be vector of copies of times so just get first element
        stheta=score_fn(x,t)
        stdt=torch.sqrt(1.-sat**2)
        stdtm1=torch.sqrt(1.-satm1**2)
        b=(sat/satm1)
        x_mean = (1./b)*(x + stdt**2*stheta)-torch.sqrt(stdt**2-eta**2*(1.-b**2))*stdtm1*stheta
        x = x_mean + eta * (stdtm1/stdt)*torch.sqrt(1.-b**2)*(dW/torch.sqrt(-dt))
        return x, x_mean
    
    if sampler.lower()=='skrock':
        samplerfun=SKROCK
    elif sampler.lower()=='expint':
        samplerfun=ExponentialIntegrator
    elif sampler.lower()=='tem':
        samplerfun=TamedEulerMaruyama
    elif sampler.lower()=='ddim':
        samplerfun=DDIMSampler
    else:
        print('Setting sampler for MLMC to Euler-Maruyama.')
        samplerfun=EulerMaruyama
    
    def nonadaptivemlmc_sample(bs,l,M,sde=sde,sampling_eps=sampling_eps,sampling_shape=sampling_shape,denoise=False,saver=True):
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
            xc = xf.clone()
            Nf=M**l
            #Nc=M**(l-1) implicitly
            fine_times = torch.linspace(sde.T, sampling_eps,Nf+1, device=xf.device,dtype=torch.float32)
            dWc=torch.zeros_like(xf).to(xc.device)
            dtc=fine_times[0]*0.
            tc=torch.tensor([sde.T],dtype=torch.float32).to(xc.device)
            if saver:
                coarselist=inverse_scaler(xc)[0][None,...].cpu()
                finelist=inverse_scaler(xf)[0][None,...].cpu()
                coarsetimes=torch.tensor([sde.T])[None,...].cpu()
                finetimes=torch.tensor([sde.T])[None,...].cpu()
            for i in range(Nf):
                tf_ = fine_times[i]
                dt=fine_times[i+1]-tf_
                dtc+=dt #running sum of coarse timestep
                vec_t = torch.ones(bs, device=tf_.device, dtype=torch.float32) * tf_
                dWf = torch.randn_like(xf)*torch.sqrt(-dt)
                dWc+=dWf
                xf,xf_mean=samplerfun(xf,vec_t,dt,dWf)
                if saver:
                    finelist=torch.cat((finelist,inverse_scaler(xf)[0][None,...].cpu()),dim=0)
                    finetimes=torch.cat((finetimes,tf_[None,...].cpu()),dim=0)
            
                if i%M==(M-1): #if i is integer multiple of M...
                    vec_t = torch.ones(bs, device=tc.device,dtype=torch.float32) * tc
                    xc,xc_mean=samplerfun(xc,vec_t,dtc,dWc) #...Develop coarse path
                    dWc=torch.zeros_like(xc) #...Re-initialise coarse BI to 0
                    tc=tf_.clone() #coarse solution now advanced to current fine time
                    dtc=0.
                    if saver:
                        coarselist=torch.cat((coarselist,inverse_scaler(xc)[0][None,...]),dim=0)
                        coarsetimes=torch.cat((coarsetimes,tc[None,...].cpu()),dim=0)
            #if denoise:
            # return inverse_scaler(xf_mean),inverse_scaler(xc_mean)
            #else:
            if saver:
                this_sample_dir = os.path.join(eval_dir, f"level_{l}")
                if not tf.io.gfile.exists(this_sample_dir):
                    tf.io.gfile.makedirs(this_sample_dir)
                with tf.io.gfile.GFile(os.path.join(this_sample_dir, "sample_progression.npz"), "wb") as fout:
                    io_buffer = io.BytesIO()
                    np.savez_compressed(io_buffer, coarsesamples=coarselist.numpy(),finesamples=finelist.numpy(),coarsetimes=coarsetimes.numpy(),finetimes=finetimes.numpy())
                    fout.write(io_buffer.getvalue())
                    
            return inverse_scaler(xf),inverse_scaler(xc)

    def adaptivemlmc_sample(bs,l,M,sde=sde,sampling_eps=sampling_eps,sampling_shape=sampling_shape,denoise=False,saver=True):
        """ 
        The path function for Euler-Maruyama diffusion, which calculates final samples \sim p(x_0).
    
        Parameters:
            bs(int): batch size to generate number of samples
            l(int) : discretisation level
            M(int) : coarseness factor, number of fine steps = M**l
        Returns:
            Xf,Xc (numpy.array) : final samples for N_loop sample paths (Xc=X0 if l==0)
        """
        def hfunc(x,t):
            _,std=sde.marginal_prob(x,t)
            _,diffusion=sde.sde(x,t)
            h=(2/diffusion**2)/(1.+2./(std*torch.max(imagenorm(x))))
            return h
        
        with torch.no_grad():
            xf = sde.prior_sampling((bs,*sampling_shape[-3:])).to(config.device)
            xc = xf.clone().detach().to(config.device)
            dWc=torch.zeros_like(xf).to(xc.device)
            dWf=torch.zeros_like(xf).to(xc.device)
            dtc=torch.zeros(1).to(xc.device)
            dtf=torch.zeros(1).to(xc.device)
            t=torch.tensor([sde.T],dtype=torch.float32).to(xc.device)
            tc=torch.tensor([sde.T],dtype=torch.float32).to(xc.device)
            tf_=torch.tensor([sde.T],dtype=torch.float32).to(xc.device)
            if saver:
                coarselist=inverse_scaler(xc)[0][None,...].cpu()
                finelist=inverse_scaler(xf)[0][None,...].cpu()
                coarsetimes=torch.tensor([sde.T])[None,...].cpu()
                finetimes=torch.tensor([sde.T])[None,...].cpu()
            
            while t>sampling_eps:
                told=t
                t=torch.max(tc,tf_)                
                dW = torch.randn_like(xf)*torch.sqrt(told-t)
                dWf +=dW
                dWc +=dW
                if t==tc:#...Develop coarse path
                    vec_t = torch.ones(bs, device=xc.device,dtype=torch.float32) * (tc-dtc)
                    xc,xc_mean=samplerfun(xc,vec_t,dtc,dWc) 
                    dtc=-hfunc(xc,tc)/(M**(l-1))
                    dtc=torch.max(dtc,sampling_eps-t) #dtc negative
                    tc+=dtc #tc should decrease
                    dWc*=0.
                    if saver:
                        coarselist=torch.cat((coarselist,inverse_scaler(xc)[0][None,...].cpu()),dim=0)
                        coarsetimes=torch.cat((coarsetimes,t[None,...].cpu()),dim=0)
                if t==tf_:
                    vec_t = torch.ones(bs, device=xf.device,dtype=torch.float32) * (tf_-dtf)
                    xf,xf_mean=samplerfun(xf,vec_t,dtf,dWf)
                    dtf=-hfunc(xf,tf_)/(M**l)
                    dtf=torch.max(dtf,sampling_eps-t) #dtf negative
                    tf_+=dtf #tf_ should decrease
                    dWf*=0.
                    if saver:
                        finelist=torch.cat((finelist,inverse_scaler(xf)[0][None,...].cpu()),dim=0)
                        finetimes=torch.cat((finetimes,t[None,...].cpu()),dim=0)
                
            if saver:
                this_sample_dir = os.path.join(eval_dir, f"level_{l}")
                if not tf.io.gfile.exists(this_sample_dir):
                    tf.io.gfile.makedirs(this_sample_dir)
                with tf.io.gfile.GFile(os.path.join(this_sample_dir, "sample_progression.npz"), "wb") as fout:
                    io_buffer = io.BytesIO()
                    np.savez_compressed(io_buffer, coarsesamples=coarselist.numpy(),finesamples=finelist.numpy(),coarsetimes=coarsetimes.numpy(),finetimes=finetimes.numpy())
                    fout.write(io_buffer.getvalue())
            
            if denoise:
                return inverse_scaler(xf_mean),inverse_scaler(xc_mean)
            else: 
                return inverse_scaler(xf),inverse_scaler(xc)
        
    mlmc_sample = adaptivemlmc_sample if adaptive else nonadaptivemlmc_sample

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
        num_sampling_rounds = Nl // config.eval.batch_size + 1
        numrem=Nl % config.eval.batch_size
        for r in range(num_sampling_rounds):
            bs=numrem if r==num_sampling_rounds-1 else config.eval.batch_size
    
            Xf,Xc=mlmc_sample(bs,l,M) #should automatically use cuda
            fine_payoff=payoff(Xf)
            coarse_payoff=payoff(Xc)
            if r==0:
                sums=torch.zeros((3,*fine_payoff.shape[1:])) #skip batch_size
                sqsums=torch.zeros((4,*fine_payoff.shape[1:]))
            sumXf=torch.sum(fine_payoff,axis=0).to('cpu') #sum over batch size
            sumXf2=torch.sum(fine_payoff**2,axis=0).to('cpu')
            if l==min_l:
                sqsums+=torch.stack([sumXf2,sumXf2,torch.zeros_like(sumXf2),torch.zeros_like(sumXf2)])
                sums+=torch.stack([sumXf,sumXf,torch.zeros_like(sumXf)])
            elif l<min_l:
                raise ValueError("l must be at least min_l")
            else:
                dX_l=fine_payoff-coarse_payoff #Image difference
                sumdX_l=torch.sum(dX_l,axis=0).to('cpu') #sum over batch size
                sumdX_l2=torch.sum(dX_l**2,axis=0).to('cpu')
                sumXc=torch.sum(coarse_payoff,axis=0).to('cpu')
                sumXc2=torch.sum(coarse_payoff**2,axis=0).to('cpu')
                sumXcXf=torch.sum(coarse_payoff*fine_payoff,axis=0).to('cpu')
                sums+=torch.stack([sumdX_l,sumXf,sumXc])
                sqsums+=torch.stack([sumdX_l2,sumXf2,sumXc2,sumXcXf])
    
        # Directory to save samples. Repeatedly overwrites, just to save some example samples for debugging
        if l>min_l:
            this_sample_dir = os.path.join(eval_dir, f"level_{l}")
            if not tf.io.gfile.exists(this_sample_dir):
                tf.io.gfile.makedirs(this_sample_dir)
            samples_f=np.clip(Xf.permute(0, 2, 3, 1).cpu().numpy() * 255., 0, 255).astype(np.uint8)
            samples_f = samples_f.reshape(
                (-1, config.data.image_size, config.data.image_size, config.data.num_channels))
            samples_c=np.clip(Xc.permute(0, 2, 3, 1).cpu().numpy() * 255., 0, 255).astype(np.uint8)
            samples_c = samples_c.reshape(
                (-1, config.data.image_size, config.data.image_size, config.data.num_channels))
            with tf.io.gfile.GFile(os.path.join(this_sample_dir, "samples_f.npz"), "wb") as fout:
                io_buffer = io.BytesIO()
                np.savez_compressed(io_buffer, samplesf=samples_f)
                fout.write(io_buffer.getvalue())
            with tf.io.gfile.GFile(os.path.join(this_sample_dir, "samples_c.npz"), "wb") as fout:
                io_buffer = io.BytesIO()
                np.savez_compressed(io_buffer, samplesc=samples_c)
                fout.write(io_buffer.getvalue())
                
        return sums,sqsums 
    
    ##MLMC function
    def mlmc(accuracy,M=2,N0=10**2,alpha_0=-1,beta_0=-1,min_l=0,Lmax=11,accsplit=accsplit):
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
        #Orders of convergence
        alpha=max(0,alpha_0)
        beta=max(0,beta_0)
        
        L=min_l+1

        mylen=L+1-min_l
        V=torch.zeros(mylen) #Initialise variance vector of each levels' variance
        N=torch.zeros(mylen) #Initialise num. samples vector of each levels' num. samples
        dN=N0*torch.ones(mylen) #Initialise additional samples for this iteration vector for each level
        sqrt_cost=torch.sqrt(M**torch.arange(min_l,L+1.)+torch.hstack((torch.tensor([0.]),M**torch.arange(min_l,1.*L))))
        it0_ind=False
        while (torch.sum(dN)>0): #Loop until no additional samples asked for
            mylen=L+1-min_l
            for i,l in enumerate(torch.arange(min_l,L+1)):
                num=dN[i]
                if num>0: #If asked for additional samples...
                    tempsums,tempsqsums=looper(int(num),l,M,min_l=min_l) #Call function which gives sums
                    if not it0_ind:
                        sums=torch.zeros((mylen,*tempsums.shape)) #Initialise sums array of unnormed [dX,Xf,Xc], each column is a level
                        sqsums=torch.zeros((mylen,*tempsqsums.shape)) #Initialise sqsums array of normed [dX^2,Xf^2,Xc^2,XcXf], each column is a level
                        it0_ind=True
                    sqsums[i]+=tempsqsums
                    sums[i]+=tempsums
                    
            N+=dN #Increment samples taken counter for each level
            Yl=imagenorm(sums[:,0])/N
            V=torch.clip(mom2norm(sqsums[:,0])/N-(Yl)**2,min=0) #Calculate variance based on updated samples
        
            ##Fix to deal with zero variance or mean by linear extrapolation
            Yl[2:]=torch.maximum(Yl[2:],.5*Yl[1:-1]*M**(-alpha))
            V[2:]=torch.maximum(V[2:],.5*V[1:-1]*M**(-beta))

            #Estimate order of weak convergence using LR
            #Yl=(M^alpha-1)khl^alpha=(M^alpha-1)k(TM^-l)^alpha=((M^alpha-1)kT^alpha)M^(-l*alpha)
            #=>log(Yl)=log(k(M^alpha-1)T^alpha)-alpha*l*log(M)
            X=torch.ones((mylen-1,2))
            X[:,0]=torch.arange(1,mylen)
            a = torch.lstsq(torch.log(Yl[1:]),X)[0]
            alpha_ = max(-a[0]/np.log(M),0.)
            b = torch.lstsq(torch.log(V[1:]),X)[0]
            beta_= -b[0]/np.log(M)
            if alpha_0==-1:
                alpha=alpha_
            if beta_0==-1:
                beta=beta_
                
            sqrt_V=torch.sqrt(V)
            Nl_new=torch.ceil(((accsplit*accuracy)**-2)*torch.sum(sqrt_V*sqrt_cost)*(sqrt_V/sqrt_cost)) #Estimate optimal number of samples/level
            dN=torch.clip(Nl_new-N,min=0) #Number of additional samples
            print(f'Estimated std = {torch.sqrt(torch.sum(V/N))}. Estimated bias={Yl[-1]/(M**alpha-1)}')
            print(f'Asking for {dN} new samples for l={min_l,L}')
            if torch.sum(dN > 0.01*N).item() == 0: #Almost converged
                test=max(Yl[-2]/(M**alpha),Yl[-1]) if len(N)>2 else Yl[-1]
                if test>(M**alpha-1)*accuracy*np.sqrt(1-accsplit**2):
                    L+=1
                    print(f'Increased L to {L}.')
                    if (L>Lmax):
                        print('Asked for an L greater than maximum allowed Lmax. Ending MLMC algorithm.')
                        break
                    #Add extra entries for the new level and estimate sums with N0 samples 
                    V=torch.cat((V,V[-1]*M**(-beta)*torch.ones(1)), dim=0)
                    sqrt_V=torch.sqrt(V)
                    newcost=torch.sqrt(torch.tensor([M**L+M**(L-1.)]))
                    sqrt_cost=torch.cat((sqrt_cost,newcost),dim=0)
                    Nl_new=torch.ceil(((accsplit*accuracy)**-2)*torch.sum(sqrt_V*sqrt_cost)*(sqrt_V/sqrt_cost)) #Estimate optimal number of sample
                    N=torch.cat((N,torch.tensor([0])),dim=0)
                    dN=torch.clip(Nl_new-N,min=0) #Number of additional samples
                    print(f'With new L, estimate of {dN} new samples for l={min_l,L}')
                    sums=torch.cat((sums,torch.zeros((1,*sums[0].shape))),dim=0)
                    sqsums=torch.cat((sqsums,torch.zeros((1,*sqsums[0].shape))),dim=0)
                    
        print(f'Estimated alpha = {alpha_}')
        print(f'Estimated beta = {beta_}')
        return sums,sqsums,N
    
    def Giles_plot(acc):
        #Set mlmc params
        M=config.mlmc.M
        N0=config.mlmc.N0
        Lmax=config.mlmc.Lmax
        Nsamples=100#config.mlmc.Nsamples
        min_l=config.mlmc.min_l

        #Variance and mean samples
        tpayoff=payoff(torch.randn(*sampling_shape[1:]))
        sums=torch.zeros((1,3,*tpayoff.shape))
        sqsums=torch.zeros((1,4,*tpayoff.shape))

        # Directory to save means and norms                          
        this_sample_dir = os.path.join(eval_dir, f"VarMean_M_{M}_Nsamples_{Nsamples}")
        if not tf.io.gfile.exists(this_sample_dir):
            tf.io.gfile.makedirs(this_sample_dir)
            print(f'Proceeding to calculate variance and means with {Nsamples} estimator samples')
            sums=torch.zeros((Lmax+1,*sums.shape[1:]))
            sqsums=torch.zeros((Lmax+1,*sqsums.shape[1:]))
            for i,l in enumerate(range(0,Lmax+1)):
                print(f'l={l}')
                sums[i],sqsums[i] = looper(Nsamples,l,M,min_l=0)

            means_p=imagenorm(sums[:,1])/Nsamples
            V_p=mom2norm(sqsums[:,1])/Nsamples-means_p**2
            means_dp=imagenorm(sums[:,0])/Nsamples
            V_dp=mom2norm(sqsums[:,0])/Nsamples-means_dp**2  
        
            # Write samples to disk or Google Cloud Storage
            with tf.io.gfile.GFile(os.path.join(this_sample_dir, "averages.pt"), "wb") as fout:
                io_buffer = io.BytesIO()
                torch.save(sums/Nsamples,io_buffer)
                fout.write(io_buffer.getvalue())
            with tf.io.gfile.GFile(os.path.join(this_sample_dir, "sqaverages.pt"), "wb") as fout:
                io_buffer = io.BytesIO()
                torch.save(sqsums/Nsamples,io_buffer)
                fout.write(io_buffer.getvalue())
            with tf.io.gfile.GFile(os.path.join(this_sample_dir, "Ls.pt"), "wb") as fout:
                io_buffer = io.BytesIO()
                torch.save(torch.arange(0,Lmax+1,dtype=torch.int32),io_buffer)
                fout.write(io_buffer.getvalue())
            
            #Estimate orders of weak (alpha from means) and strong (beta from variance) convergence using LR
            X=np.ones((Lmax,2))
            X[:,0]=np.arange(1,Lmax+1)
            a = np.linalg.lstsq(X,np.log(means_dp[1:]),rcond=None)[0]
            alpha = -a[0]/np.log(M)
            b = np.linalg.lstsq(X,np.log(V_dp[1:]),rcond=None)[0]
            beta = -b[0]/np.log(M) 

            print(f'Estimated alpha={alpha}\n Estimated beta={beta}\n')
            with open(os.path.join(this_sample_dir, "info_text.txt"),'w') as f:
                extrastr="continuous" if config.training.continuous else ''
                f.write(f'Dataset:{config.data.dataset}. Model: {config.model.name}, {extrastr}, {config.training.sde}.\n')
                f.write(f'Payoff:{payoff_arg}\n')
                f.write(f'Sampler:{sampler}. DDIM_eta={DDIMeta}. Sampling eps={sampling_eps}.\n')
                f.write(f'MLMC params: N0={N0}, Lmax={Lmax}, Lmin={min_l}, Nsamples={Nsamples}, M={M}, accsplit={accsplit}.\n')
                f.write(f'Estimated alpha={alpha}\n Estimated beta={beta}. Plotting Lmin=1.')
            with tf.io.gfile.GFile(os.path.join(this_sample_dir, "alphabeta.pt"), "wb") as fout:
                io_buffer = io.BytesIO()
                torch.save(torch.tensor([alpha,beta]),io_buffer)
                fout.write(io_buffer.getvalue())
                
        with open(os.path.join(this_sample_dir, "alphabeta.pt"),'rb') as f:
            temp=torch.load(f)
            alpha=temp[0].item()
            beta=temp[1].item()
        
        #Do the calculations and simulations for num levels and complexity plot
        sums=torch.zeros((Lmax+1-min_l,*sums.shape[1:]))
        sqsums=torch.zeros((Lmax+1-min_l,*sqsums.shape[1:]))
        for i in range(len(acc)):
            e=acc[i]
            print(f'Performing mlmc for accuracy={e}')
            sums,sqsums,N=mlmc(e,M,alpha_0=alpha_0,beta_0=beta_0,N0=N0,min_l=min_l,Lmax=Lmax) #sums=[dX,Xf,Xc], sqsums=[||dX||^2,||Xf||^2,||Xc||^2]
            L=len(N)-1+min_l
            means_p=imagenorm(sums[:,1])/N #Norm of mean of fine discretisations
            V_p=mom2norm(sqsums[:,1])/N-means_p**2

            #e^2*cost
            cost_mlmc=torch.sum(N*(M**np.arange(min_l,L+1)+np.hstack((0,M**np.arange(min_l,L)))))*e**2 #cost is number of NFE
            cost_mc=V_p[-1]*(M**L)/accsplit**2
            
            # Directory to save means, norms and N
            dividerN=N.clone() #add axes to N to broadcast correctly on division
            for i in range(len(sums.shape[1:])):
                dividerN.unsqueeze_(-1)
            this_sample_dir = os.path.join(eval_dir, f"M_{M}_accuracy_{e}")
            
            if not tf.io.gfile.exists(this_sample_dir):
                tf.io.gfile.makedirs(this_sample_dir)        
            # Write samples to disk or Google Cloud Storage
            with tf.io.gfile.GFile(os.path.join(this_sample_dir, "averages.pt"), "wb") as fout:
                io_buffer = io.BytesIO()
                torch.save(sums/dividerN,io_buffer)
                fout.write(io_buffer.getvalue())
            # Write samples to disk or Google Cloud Storage
            with tf.io.gfile.GFile(os.path.join(this_sample_dir, "sqaverages.pt"), "wb") as fout:
                io_buffer = io.BytesIO()
                torch.save(sqsums/dividerN,io_buffer) #sums has shape (L,4,C,H,W) if img (L,4,2048) if activations
                fout.write(io_buffer.getvalue())
            # Write samples to disk or Google Cloud Storage
            with tf.io.gfile.GFile(os.path.join(this_sample_dir, "N.pt"), "wb") as fout:
                io_buffer = io.BytesIO()
                torch.save(N,io_buffer)
                fout.write(io_buffer.getvalue())
            # Write samples to disk or Google Cloud Storage        
            with open(os.path.join(this_sample_dir, "costs.npz"), "wb") as fout:
                np.savez_compressed(fout,costmlmc=np.array(cost_mlmc),costmc=np.array(cost_mc))
            
            with open(os.path.join(this_sample_dir, "info_text.txt"),'w') as f:
                f.write(f'MLMC params:epsilon={e}, alpha={alpha_0}, beta={beta_0}, N0={N0}, Lmax={Lmax}, Lmin={min_l}, M={M}, accsplit={accsplit}.\n')
            
            meanimg=torch.sum(sums[:,0]/dividerN[:,0,...],axis=0)#cut off one dummy axis
            if payoff_arg=='images' or payoff_arg=='variance':
                meanimg=np.clip(meanimg.permute(1, 2, 0).cpu().numpy() * 255., 0, 255).astype(np.uint8)
            # Write samples to disk or Google Cloud Storage
            with tf.io.gfile.GFile(os.path.join(this_sample_dir, "meanpayoff.npz"), "wb") as fout:
                io_buffer = io.BytesIO()
                np.savez_compressed(io_buffer, meanpayoff=meanimg)
                fout.write(io_buffer.getvalue())
       
        return None

    def MC_sample(bs,l,M,sde=sde,sampling_eps=sampling_eps,sampling_shape=sampling_shape,denoise=False):
        with torch.no_grad():
            xf = sde.prior_sampling((bs,*sampling_shape[-3:])).to(config.device)
            Nf=M**l
            fine_times = torch.linspace(sde.T, sampling_eps,Nf+1, device=xf.device,dtype=torch.float32)
            for i in range(Nf):
                tf_ = fine_times[i]
                dt=fine_times[i+1]-tf_
                vec_t = torch.ones(bs, device=tf_.device, dtype=torch.float32) * tf_
                dWf = torch.randn_like(xf)*torch.sqrt(-dt)
                xf,xf_mean=samplerfun(xf,vec_t,dt,dWf)
            return inverse_scaler(xf)
        
    def MC(Nl):
        this_sample_dir = os.path.join(eval_dir,'MCsamples')
        if not tf.io.gfile.exists(this_sample_dir):
            tf.io.gfile.makedirs(this_sample_dir)
        l=8
        M=config.mlmc.M
        with open(os.path.join(this_sample_dir, "info_text.txt"),'w') as f:
            extrastr="continuous" if config.training.continuous else ''
            f.write(f'Dataset:{config.data.dataset}. Model: {config.model.name}, {extrastr}, {config.training.sde}. \n')
            f.write(f'Sampler:{sampler}. DDIM_eta={DDIMeta}. Sampling eps={sampling_eps}.\n')
            f.write(f'MC params:L={l}, Nsamples={Nl}, M={M}.')
        num_sampling_rounds = Nl // config.eval.batch_size + 1
        numrem=Nl % config.eval.batch_size
        for r in range(num_sampling_rounds):
            bs=numrem if r==num_sampling_rounds-1 else config.eval.batch_size
            Xf=MC_sample(bs,l,M) #should automatically use cuda
            #acts=actspayoff(Xf)
            # Directory to save samples.
            with tf.io.gfile.GFile(os.path.join(this_sample_dir, f"samples_{r}.npz"), "wb") as fout:
                io_buffer = io.BytesIO()
                np.savez_compressed(io_buffer, samples=Xf.cpu().numpy())
                fout.write(io_buffer.getvalue())
        
        return None
    
    if MLMC_:
        Giles_plot(acc)
    else: #MC
        print('Doing MC estimates.')
        MC(int(1e6))
