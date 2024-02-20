from tokenize import Exponent
from models import utils as mutils
import os
import gc
import io
import numpy as np
import tensorflow as tf
import tensorflow_gan as tfgan
import logging
from easydict import EasyDict as edict

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


def mlmc_test(config,eval_dir,checkpoint_dir,payoff_arg,acc=[],M=2,Lmax=11,
              sampler='EM',probflow=False,abL=None,
              accsplit=np.sqrt(.5),conditional=None):
    torch.cuda.empty_cache()
    tf.keras.backend.clear_session()
    
    # Create data normalizer and its inverse
    inverse_scaler = datasets.get_data_inverse_scaler(config)
    
    #MLMC algo params
    mlmcopts=edict()
    if abL is not None:
        mlmcopts.alpha_0=abL[0]
        mlmcopts.beta_0=abL[1]
        mlmcopts.Lmin=abL[2]
    else:
        mlmcopts.alpha_0=-1
        mlmcopts.beta_0=-1
        mlmcopts.Lmin=3
    mlmcopts.M=M
    mlmcopts.Lmax=Lmax
    mlmcopts.N0=50
    mlmcopts.batch_size=1600
    mlmcopts.accsplit=accsplit
    config.model.num_scales=(M)**(Lmax)
    
    if payoff_arg=='secondmoment':
        print('Pixel-wise second moment payoff selected for MLMC.')
        payoff = lambda samples: torch.clip(samples,-1.,1.)**2
    elif payoff_arg=='mean':
        print('Setting payoff function to mean image for MLMC.')
        payoff = lambda samples: torch.clip(samples,-1.,1.) #default to calculating mean image
    else:
        raise ValueError('payoff_arg not recognised. Should be one of variance, activations, images.')
    
    dirs=os.listdir(checkpoint_dir)
    ckpt = np.min(np.array([int(d.split('_')[-1][:-4]) for d in dirs]))
    ckpt_dir = os.path.join(checkpoint_dir, f'checkpoint_{ckpt}.pth')
    os.makedirs(eval_dir,exist_ok=True)
    
    # Initialize model
    model = mutils.create_model(config)
    loaded_state = torch.load(ckpt_dir, map_location=config.device)
    model.load_state_dict(loaded_state['model'], strict=False)

    sampling_shape = (config.eval.batch_size,
                      config.data.num_channels,
                      config.data.image_size, config.data.image_size)    
    if config.training.sde.lower() == 'vpsde':
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 0.
        def diffusion(t):
            beta_t=sde.beta_0 + t * (sde.beta_1 - sde.beta_0)
            return torch.sqrt(beta_t)
        def EIfactor(dt, t):
            #dt<0
            beta_t = sde.beta_0 + (t+.5*dt) * (sde.beta_1 - sde.beta_0)
            return torch.exp(.5*(-dt)*beta_t)
        def std(t):
            log_mean_coeff = -0.25 * t ** 2 * (sde.beta_1 - sde.beta_0) - 0.5 * t * sde.beta_0
            std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
            return std
        MASK=torch.ones(*sampling_shape[1:])
        if conditional is not None:
            def log_mean_coeff(t,s):
                #int_t^s A
                #-(1/2)int_t^s beta_0 + (beta_1-beta_0)*t
                beta_t = sde.beta_0 + .5*(t+s) * (sde.beta_1 - sde.beta_0)
                return -.5*(beta_t)*(s-t)
            MASK=conditional.MASK
            obsT=(-sde.beta_0+torch.sqrt(sde.beta_0**2 + 2*(sde.beta_1 - sde.beta_0)*torch.log1p(torch.tensor([conditional.noise])**2)))/(sde.beta_1 - sde.beta_0)
            sde.T=obsT.item()
            obsV=(conditional.obsV*torch.exp(log_mean_coeff(0,obsT))).to(torch.float32) #convert to v_
            obsT=obsT.to(config.device)
            sde.prior_sampling = lambda shape: obsV.repeat((shape[0],*tuple([1 for i in range(len(obsV.shape))])))
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown. Only VPSDE allowed.")

    score_fn=mutils.get_score_fn(sde, model,continuous=config.training.continuous)
    #score_fn=lambda x,t:std(t[0])*torch.sin(t)[...,None,None,None]*torch.ones_like(x)
    rsde = sde.reverse(score_fn, probability_flow=probflow)

    def imagenorm(img):
        s=MASK.sum()
        if len(img.shape)==1: #fix for when img is single dimensional (batch_size,) -> (batch_size,1)
            img=img[:,None]
        n=torch.linalg.norm(torch.flatten(img*MASK[None,...], start_dim=1, end_dim=-1),dim=-1) #flattens non-batch dims and calculates norm
        n/=np.sqrt(s)
        return n
    
    def mom2norm(sqsums):
        #sqsums should have shape L,C,H,W
        s=MASK.sum()
        if len(sqsums.shape)!=4:
            raise Exception('shape of sqsums likely not LHCW')
        return torch.sum(torch.flatten(sqsums*MASK[None,...], start_dim=1, end_dim=-1),dim=-1)/s

    
    @torch.no_grad()
    def EulerMaruyama(x, t, dt, dW):
        #dt is negative
        d=diffusion(t)
        stheta=-score_fn(x,t*torch.ones(x.shape[0],device=x.device,dtype=torch.float32))/std(t)
        x_mean = x - d**2*(stheta+x/2) * dt
        x = x_mean + d*dW
        return x, x_mean

    def EIBrownianIncrement(t,dt,xi):
        factor=EIfactor(dt,t)
        stdt=std(t)
        stdtm1=std(t+dt)
        scaler=(stdtm1/stdt)**2/factor
        dBnew=stdtm1*torch.sqrt(1.-(stdtm1/(stdt*factor))**2)*xi
        return dBnew,scaler
        
    @torch.no_grad()
    def ExponentialIntegrator(x, t, dt, dB):
        #should only work for vpsde
        factor=EIfactor(dt,t)
        stdt=std(t)
        stdtm1=std(t+dt)
        stheta=score_fn(x,t*torch.ones(x.shape[0],device=x.device,dtype=torch.float32)) #=-grad_logP*stdt
        drift=stdtm1-stdt*factor
        noise=torch.zeros_like(dB)
        if not probflow:
            drift=(stdtm1**2/(factor*stdt)-stdt*factor)
            noise+=dB
        x_mean=factor*x+drift*stheta
        x=x_mean+noise
        return x, x_mean

    if sampler.lower()=='em':
        samplerfun=EulerMaruyama
        BIfun=lambda t,dt,xi: torch.sqrt(-dt)*xi,1.
    else:
        print('Setting sampler for MLMC to ExpInt.')
        samplerfun=ExponentialIntegrator
        BIfun=EIBrownianIncrement
    if conditional is not None:
        if sampler.lower()=='em':
            print('Setting sampler for MLMC to ConditionalEM.')
            samplerfun=EulerMaruyama
            BIfun=lambda t,dt,xi: torch.sqrt(-dt)*xi,1.
        else:
            print('Setting sampler for MLMC to ConditionalExpInt.')
            samplerfun=ExponentialIntegrator
            BIfun=EIBrownianIncrement
        
    def mlmc_sampler(bs,l,M,sde=sde,sampling_eps=sampling_eps,sampling_shape=sampling_shape,saver=False):
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
            fine_times = torch.linspace(sde.T, sampling_eps,Nf+1,device=xf.device,dtype=torch.float32)
            dBc=torch.zeros_like(xf).to(xc.device)
            dtc=fine_times[0]*0.
            tc=torch.tensor([sde.T],dtype=torch.float32).to(xc.device)
            if saver:
                coarselist=inverse_scaler(xc)[0][None,...].cpu()
                finelist=inverse_scaler(xf)[0][None,...].cpu()
                coarsetimes=torch.tensor([sde.T])[None,...].cpu()
                finetimes=torch.tensor([sde.T])[None,...].cpu()
            tf_ = fine_times[0]
            for i in range(Nf):
                dt=fine_times[i+1]-tf_
                dtc+=dt #running sum of coarse timestep
                xi = torch.randn_like(xf)
                dBf,scaler=BIfun(tf_,dt,xi)
                xf,xf_mean=samplerfun(xf,tf_,dt,dBf)
                tf_ = fine_times[i+1] #fine solution now advanced to this time
                dBc=dBf+scaler*dBc
                if saver:
                    finelist=torch.cat((finelist,inverse_scaler(xf)[0][None,...].cpu()),dim=0)
                    finetimes=torch.cat((finetimes,tf_[None,None,...].cpu()),dim=0)
            
                if i%M==(M-1): #if i is integer multiple of M...
                    xc,xc_mean=samplerfun(xc,tc,dtc,dBc) #...Develop coarse path
                    dBc=torch.zeros_like(xc) #...Re-initialise coarse BI to 0
                    tc=tf_  #coarse solution now advanced to current fine time
                    dtc=0.
                    if saver:
                        coarselist=torch.cat((coarselist,inverse_scaler(xc)[0][None,...].cpu()),dim=0)
                        coarsetimes=torch.cat((coarsetimes,tc[None,None,...].cpu()),dim=0)

            if saver:
                this_sample_dir = os.path.join(eval_dir, f"level_{l}")
                if not tf.io.gfile.exists(this_sample_dir):
                    tf.io.gfile.makedirs(this_sample_dir)
                with open(os.path.join(this_sample_dir, "sample_progression.npz"), "wb") as fout:
                    np.savez_compressed(fout, coarsesamples=coarselist.numpy(),finesamples=finelist.numpy(),coarsetimes=coarsetimes.numpy(),finetimes=finetimes.numpy())
                    
            return xf,xc,1.*M**l,1.*M**(l-1) #finecost and coarsecost

    mlmc_sample = mlmc_sampler

    def looper(Nl,l,M=mlmcopts.M,Lmin=mlmcopts.Lmin):
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
        num_sampling_rounds = Nl // mlmcopts.batch_size + 1
        numrem=Nl % mlmcopts.batch_size
        finecost=0.
        coarsecost=0.
        for r in range(num_sampling_rounds):
            bs=numrem if r==num_sampling_rounds-1 else mlmcopts.batch_size
            if bs==0:
                break
            Xf,Xc,fc,cc=mlmc_sample(bs,l,M) #should automatically use cuda
            finecost+=fc*bs
            coarsecost+=cc*bs
            fine_payoff=payoff(Xf)
            coarse_payoff=payoff(Xc)
            if r==0:
                sums=torch.zeros((3,*fine_payoff.shape[1:])) #skip batch_size
                sqsums=torch.zeros((4,*fine_payoff.shape[1:]))
            sumXf=torch.sum(fine_payoff,axis=0).to('cpu') #sum over batch size
            sumXf2=torch.sum(fine_payoff**2,axis=0).to('cpu')
            if l==Lmin:
                coarsecost=0.
                sqsums+=torch.stack([sumXf2,sumXf2,torch.zeros_like(sumXf2),torch.zeros_like(sumXf2)])
                sums+=torch.stack([sumXf,sumXf,torch.zeros_like(sumXf)])
            elif l<Lmin:
                raise ValueError("l must be at least Lmin")
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
        if l>Lmin:
            this_sample_dir = os.path.join(eval_dir, f"level_{l}")
            if not tf.io.gfile.exists(this_sample_dir):
                tf.io.gfile.makedirs(this_sample_dir)
            ns=int(.1*Xf.shape[0])+1
            samples_f=np.clip(inverse_scaler(Xf[:ns]).permute(0, 2, 3, 1).cpu().numpy(), 0., 1.)
            samples_c=np.clip(inverse_scaler(Xc[:ns]).permute(0, 2, 3, 1).cpu().numpy(), 0., 1.)
            with open(os.path.join(this_sample_dir, "samples_f.npz"), "wb") as fout:
                np.savez_compressed(fout, samplesf=samples_f)
            with open(os.path.join(this_sample_dir, "samples_c.npz"), "wb") as fout:
                np.savez_compressed(fout, samplesc=samples_c)
                
        return sums,sqsums, finecost+coarsecost #total cost
    
    ##MLMC function
    def mlmc(accuracy,M=mlmcopts.M,N0=mlmcopts.N0,alpha_0=mlmcopts.alpha_0,beta_0=mlmcopts.beta_0,Lmin=mlmcopts.Lmin,
             Lmax=mlmcopts.Lmax,accsplit=mlmcopts.accsplit):
        """
        Runs MLMC algorithm which returns an array of sums at each level.
        ________________
        ________________
        Returns: sums=[np.sum(dP_l),np.sum(dP_l**2),sumPf,sumPf2,sumPc,sumPc2,sum(Pf*Pc)],N
            sums(np.array) : sums of payoff diffs at each level and sum of payoffs at fine level, each column is a level
            N(np.array of ints) : final number of samples at each level
        """
        #Orders of convergence
        alpha=max(0.,alpha_0)
        beta=max(0.,beta_0)
        L=Lmin+2


        mylen=L+1-Lmin
        V=torch.zeros(mylen) #Initialise variance vector of each levels' variance
        N=torch.zeros(mylen) #Initialise num. samples vector of each levels' num. samples
        dN=N0*torch.ones(mylen) #Initialise additional samples for this iteration vector for each level
        cost=torch.zeros(mylen)
        it0_ind=False
        while (torch.sum(dN)>0): #Loop until no additional samples asked for
            mylen=L+1-Lmin
            for i,l in enumerate(torch.arange(Lmin,L+1)):
                num=dN[i]
                if num>0: #If asked for additional samples...
                    tempsums,tempsqsums,c=looper(int(num),l,M,Lmin=Lmin) #Call function which gives sums
                    if not it0_ind:
                        sums=torch.zeros((mylen,*tempsums.shape)) #Initialise sums array of unnormed [dX,Xf,Xc], each column is a level
                        sqsums=torch.zeros((mylen,*tempsqsums.shape)) #Initialise sqsums array of normed [dX^2,Xf^2,Xc^2,XcXf], each column is a level
                        it0_ind=True
                    sqsums[i]+=tempsqsums
                    sums[i]+=tempsums
                    #c is total cost
                    cost[i]=(cost[i]*N[i]+c)/(num+N[i])
            
            N+=dN #Increment samples taken counter for each level
            Yl=imagenorm(sums[:,0])/N
            V=torch.clip(mom2norm(sqsums[:,0])/N-(Yl)**2,min=0) #Calculate variance based on updated samples
        
            ##Fix to deal with zero variance or mean by linear extrapolation
            Yl[2:]=torch.maximum(Yl[2:],.5*Yl[1:-1]*M**(-alpha))
            V[2:]=torch.maximum(V[2:],.5*V[1:-1]*M**(-beta))

            #Estimate order of weak convergence using LR
            #Yl=(M^alpha-1)khl^alpha=(M^alpha-1)k(TM^-l)^alpha=((M^alpha-1)kT^alpha)M^(-l*alpha)
            #=>log(Yl)=log(k(M^alpha-1)T^alpha)-alpha*l*log(M)
            X=torch.ones((mylen-1,2),dtype=torch.float64)
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
            sqrt_cost=torch.sqrt(cost) #cost per sample on average at each level
            Nl_new=torch.ceil(((accsplit*accuracy)**-2)*torch.sum(sqrt_V*sqrt_cost)*(sqrt_V/sqrt_cost)) #Estimate optimal number of samples/level
            dN=torch.clip(Nl_new-N,min=0) #Number of additional samples
            print(f'Std={torch.sqrt(torch.sum(2*V/N))}. Bias={np.sqrt(2)*Yl[-1]/(M**alpha-1)}. Error={torch.sqrt((Yl[-1]/(M**alpha-1))**2+torch.sum(V/N))}.')
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
                    newcost=torch.tensor([cost[-1]*M])
                    cost=torch.cat((cost,newcost),dim=0)
                    sqrt_cost=torch.sqrt(cost) #cost per sample on average at each level
                    Nl_new=torch.ceil(((accsplit*accuracy)**-2)*torch.sum(sqrt_V*sqrt_cost)*(sqrt_V/sqrt_cost)) #Estimate optimal number of sample
                    N=torch.cat((N,torch.tensor([0])),dim=0)
                    dN=torch.clip(Nl_new-N,min=0) #Number of additional samples
                    print(f'With new L, estimate of {dN} new samples for l={Lmin,L}')
                    sums=torch.cat((sums,torch.zeros((1,*sums[0].shape))),dim=0)
                    sqsums=torch.cat((sqsums,torch.zeros((1,*sqsums[0].shape))),dim=0)
                    
        return sums,sqsums,N,cost #average cost
    
    def Giles_plot(acc):
        #Set mlmc params
        M=mlmcopts.M
        N0=mlmcopts.N0
        Lmax = mlmcopts.Lmax
        Nsamples=1000
        Lmin=mlmcopts.Lmin

        #Variance and mean samples
        tpayoffshape=payoff(torch.randn(*sampling_shape)).shape[1:]
        sums=torch.zeros((1,3,*tpayoffshape))
        sqsums=torch.zeros((1,4,*tpayoffshape))
        cost=torch.zeros((1,))

        # Directory to save means and norms                          
        this_sample_dir = os.path.join(eval_dir, f"VarMean_M_{M}_Nsamples_{Nsamples}")
        if not tf.io.gfile.exists(this_sample_dir):
            tf.io.gfile.makedirs(this_sample_dir)
            print(f'Proceeding to calculate variance and means with {Nsamples} estimator samples')
            sums=torch.zeros((Lmax+1,*sums.shape[1:]))
            sqsums=torch.zeros((Lmax+1,*sqsums.shape[1:]))
            cost=torch.zeros((Lmax+1,))
            for i,l in enumerate(range(0,Lmax+1)):
                print(f'l={l}') #total cost
                sums[i],sqsums[i],cost[i] = looper(Nsamples,l,M,Lmin=0)
        
                # Write samples to disk or Google Cloud Storage
                with open(os.path.join(this_sample_dir, "averages.pt"), "wb") as fout:
                    torch.save(sums/Nsamples,fout)
                with open(os.path.join(this_sample_dir, "sqaverages.pt"), "wb") as fout:
                    torch.save(sqsums/Nsamples,fout)
                with open(os.path.join(this_sample_dir, "Ls.pt"), "wb") as fout:
                    torch.save(torch.arange(0,Lmax+1,dtype=torch.int32),fout)
                with open(os.path.join(this_sample_dir, "avgcost.pt"), "wb") as fout:
                    torch.save(cost/Nsamples,fout)
            
            means_p=imagenorm(sums[:,1])/Nsamples
            V_p=mom2norm(sqsums[:,1])/Nsamples-means_p**2
            means_dp=imagenorm(sums[:,0])/Nsamples
            V_dp=mom2norm(sqsums[:,0])/Nsamples-means_dp**2  
            
            #Estimate orders of weak (alpha from means) and strong (beta from variance) convergence using LR
            Lmincond=V_dp[1:]<(np.sqrt(M)-1.)**2*V_p[-1]/(1+M) #index of optimal lmin
            cutoff=np.argmax(Lmincond[1:]*Lmincond[:-1])   
            means_p=means_p[cutoff:]
            V_p=V_p[cutoff:]
            means_dp=means_dp[cutoff:]
            V_dp=V_dp[cutoff:]
            
            X=np.ones((Lmax-cutoff,2))
            X[:,0]=np.arange(1.,Lmax-cutoff+1)
            a = np.linalg.lstsq(X,np.log(means_dp[1:]),rcond=None)[0]
            alpha = -a[0]/np.log(M)
            b = np.linalg.lstsq(X,np.log(V_dp[1:]),rcond=None)[0]
            beta = -b[0]/np.log(M)

            print(f'Estimated alpha={alpha}\n Estimated beta={beta}\n Estimate Lmin={cutoff}.')
            with open(os.path.join(this_sample_dir, "info_text.txt"),'w') as f:
                extrastr="continuous" if config.training.continuous else ''
                pflow="True" if probflow else 'False'
                f.write(f'Dataset:{config.data.dataset}. Model: {config.model.name}, {extrastr}, {config.training.sde}.\n')
                f.write(f'Payoff:{payoff_arg}\n')
                f.write(f'Sampler:{sampler}. Sampling eps={sampling_eps}. Probflow={pflow}\n')
                f.write(f'MLMC params: Nsamples={Nsamples}, M={M}, accsplit={accsplit}.\n')
                f.write(f'Estimated alpha={alpha}. Estimated beta={beta}. Estimated Lmin={cutoff}.')
            with open(os.path.join(this_sample_dir, "alphabetaLmin.pt"), "wb") as fout:
                torch.save(torch.tensor([alpha,beta,cutoff]),fout)
        if abL is None:
            try:
                with open(os.path.join(this_sample_dir, "alphabetaLmin.pt"),'rb') as f:
                    temp=torch.load(f)
                    alpha=temp[0].item()
                    beta=temp[1].item()
                    Lmin=int(temp[-1])
                
            except:
                print('Will estimate alpha,beta during algorithm. Setting Lmin=3')
                alpha=-1
                beta=-1
                Lmin=3
            
        #Do the calculations and simulations for num levels and complexity plot
        sums=torch.zeros((Lmax+1-Lmin,*sums.shape[1:]))
        sqsums=torch.zeros((Lmax+1-Lmin,*sqsums.shape[1:]))
        accsplitdir = os.path.join(eval_dir, f"Experiment")
        if not tf.io.gfile.exists(accsplitdir):
            tf.io.gfile.makedirs(accsplitdir)
        for i in range(len(acc)):
            e=acc[i]
            print(f'Performing mlmc for accuracy={e}')
            print(f'abLmin={alpha,beta,Lmin}')
            sums,sqsums,N,cost=mlmc(e,M,alpha_0=alpha,beta_0=beta,N0=N0,Lmin=Lmin,Lmax=Lmax) #sums=[dX,Xf,Xc], sqsums=[||dX||^2,||Xf||^2,||Xc||^2]
            means_p=imagenorm(sums[:,1])/N #Norm of mean of fine discretisations
            V_p=mom2norm(sqsums[:,1])/N-means_p**2
            means_dp=imagenorm(sums[:,1])/N

            #cost
            cost_mlmc=torch.sum(N*cost) #cost is number of NFE
            cost_mc=(1/(e*accsplit))**2*V_p[-1]*cost[-1]/(1+1./M)

            # Directory to save means, norms and N
            dividerN=N.clone() #add axes to N to broadcast correctly on division
            for i in range(len(sums.shape[1:])):
                dividerN.unsqueeze_(-1)
            this_sample_dir = os.path.join(accsplitdir, f"M_{M}_accuracy_{e}")
            
            if not tf.io.gfile.exists(this_sample_dir):
                tf.io.gfile.makedirs(this_sample_dir)        
            # Write samples to disk or Google Cloud Storage
            with open(os.path.join(this_sample_dir, "averages.pt"), "wb") as fout:
                torch.save(sums/dividerN,fout)
            with open(os.path.join(this_sample_dir, "sqaverages.pt"), "wb") as fout:
                torch.save(sqsums/dividerN,fout) #sums has shape (L,4,C,H,W)
            with open(os.path.join(this_sample_dir, "N.pt"), "wb") as fout:
                torch.save(N,fout)        
            with open(os.path.join(this_sample_dir, "costs.npz"), "wb") as fout:
                np.savez_compressed(fout,costmlmc=np.array(cost_mlmc),costmc=np.array(cost_mc))
            
            with open(os.path.join(this_sample_dir, "info_text.txt"),'w') as f:
                f.write(f'MLMC params:epsilon={e}, alpha={alpha}, beta={beta}, N0={mlmcopts.N0}, Lmax={mlmcopts.Lmax}, Lmin={mlmcopts.Lmin}, M={mlmcopts.M}, accsplit={mlmcopts.accsplit}.\n')
            
            meanimg=torch.sum(sums[:,0]/dividerN[:,0,...],axis=0)#cut off one dummy axis
            meanimg=np.clip(inverse_scaler(meanimg).permute(1, 2, 0).cpu().numpy() * 255., 0, 255).astype(np.uint8)
            with open(os.path.join(this_sample_dir, "meanpayoff.npz"), "wb") as fout:
                np.savez_compressed(fout, meanpayoff=meanimg)
       
        return None

    Giles_plot(acc)
