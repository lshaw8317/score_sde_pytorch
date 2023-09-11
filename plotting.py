# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 08:51:39 2023

@author: lshaw
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib.ticker as ticker
from matplotlib.legend import Legend
plt.rc('text', usetex=True)
plt.rc('font',**{'serif':['cm']})
plt.style.use('seaborn-paper')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
M=2
Nsamples=10**3

def imagenorm(img):
    s=img.shape
    if len(s)==1: #fix for when img is single dimensional (batch_size,) -> (batch_size,1)                                
        img=img[:,None]
    n=torch.linalg.norm(torch.flatten(img, start_dim=1, end_dim=-1),dim=-1) #flattens non-batch dims and calculates norm
    n/=np.sqrt(np.prod(s[1:]))
    return n

#Set plotting params
fig,_=plt.subplots(2,2)
markersize=(fig.get_size_inches()[0])
axis_list=fig.axes
expdir='cifar10_ddpmpp_continuous_means'
switcher= 'means'#expdir.split('_')[-1]
label='Testing MLMC Diffusion Models '+ switcher


#Do the calculations and simulations for num levels and complexity plot
files = [os.path.join(expdir,f) for f in os.listdir(expdir) if f.startswith('M_2')]
acc=np.zeros(len(files))
realvar=np.zeros(len(files))
realbias=np.zeros(len(files))

cost_mlmc=np.zeros(len(files))
cost_mc=np.zeros(len(files))
plt.rc('axes', titlesize=4)     # fontsize of the axes title
larr=np.array([int(f.split('_')[-1]) for f in os.listdir(expdir) if f.startswith('level')])
Lmax = 11

if switcher=='means':
    with np.load('CIFAR10_MCmean.npz') as data:
        CIFAR10mean=np.clip(data['mean'],0.,1.)
    fig2,ax=plt.subplots(1,len(files)+1)
    plt.figure(fig2)
    ax[-1].imshow(CIFAR10mean)
    ax[-1].set_title('CIFAR10 MC Mean')
    ax[-1].set_axis_off()
    Lmin=3
    
elif switcher=='secondmoments':
    with np.load('CIFAR10_MCmoment2.npz') as data:
        CIFAR10mean=np.clip(data['mean'],0.,1.)
    fig2,ax=plt.subplots(1,len(files)+1)
    plt.figure(fig2)
    ax[-1].imshow(CIFAR10mean)
    ax[-1].set_title('CIFAR10 MC pw second moment')
    ax[-1].set_axis_off()
    Lmin=4
    
elif switcher=='acts':
    actserrs=np.zeros(len(files))
    with np.load('CIFAR10_MCactsmean.npz') as data:
        CIFAR10mean=data['mean']
    fig2=plt.figure()
    plt.title('CIFAR10 MC activations error')
    Lmin=5


# Directory to save means and norms
this_sample_dir = os.path.join(expdir,f"VarMean_M_{M}_Nsamples_{Nsamples}")
with open(os.path.join(this_sample_dir, "averages.pt"), "rb") as fout:
    avgs=torch.load(fout)
with open(os.path.join(this_sample_dir, "sqaverages.pt"), "rb") as fout:
    sqavgs=torch.load(fout)

sumdims=tuple(range(1,len(sqavgs[:,0].shape))) #sqsums is output of payoff element-wise squared, so reduce                        
s=sqavgs[:,0].shape
cutoff=0
means_p=imagenorm(avgs[cutoff:,1])
V_p=(torch.sum(sqavgs[cutoff:,1],axis=sumdims)/np.prod(s[1:]))-means_p**2 
means_dp=imagenorm(avgs[cutoff:,0])
V_dp=(torch.sum(sqavgs[cutoff:,0],axis=sumdims)/np.prod(s[1:]))-means_dp**2  
plottingLmin=Lmax-len(V_p)+1

#Plot variances
axis_list[0].semilogy(range(plottingLmin,Lmax+1),V_p,'k:',label='$F_{l}$',
                  marker=(8,2,0),markersize=markersize,markerfacecolor="None",markeredgecolor='k', markeredgewidth=1,base=2)
axis_list[0].semilogy(range(plottingLmin+1,Lmax+1),V_dp[1:],'k-',label='$F_{l}-F_{l-1}$',
                  marker=(8,2,0), markersize=markersize, markerfacecolor="None", markeredgecolor='k',base=2,
                  markeredgewidth=1)
#Plot means
axis_list[1].semilogy(range(plottingLmin,Lmax+1),means_p,'k:',label='$F_{l}$',
                  marker=(8,2,0), markersize=markersize, markerfacecolor="None",markeredgecolor='k',base=2,
                  markeredgewidth=1)
axis_list[1].semilogy(range(plottingLmin+1,Lmax+1),means_dp[1:],'k-',label='$F_{l}-F_{l-1}$',
                  marker=(8,2,0),markersize=markersize,markerfacecolor="None",markeredgecolor='k', markeredgewidth=1,base=2)

    
#Estimate orders of weak (alpha from means) and strong (beta from variance) convergence using LR
X=np.ones((Lmax-plottingLmin,2))
X[:,0]=np.arange(plottingLmin+1,Lmax+1)
a = np.linalg.lstsq(X,np.log(means_dp[1:]),rcond=None)[0]
alpha = -a[0]/np.log(M)
b = np.linalg.lstsq(X,np.log(V_dp[1:]),rcond=None)[0]
beta = -b[0]/np.log(M)

#Label variance plot
axis_list[0].set_xlabel('$l$')
axis_list[0].set_ylabel(f'var')
axis_list[0].legend(framealpha=0.6, frameon=True)
axis_list[0].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
#Add estimated beta
s='$\\beta$ = {}'.format(round(beta,2))
t = axis_list[0].annotate(s, (Lmax/2+2, V_dp[4]),
                          fontsize=markersize,bbox=dict(ec='None',facecolor='None',lw=2))

#Label means plot
axis_list[1].set_xlabel('$l$')
axis_list[1].set_ylabel(f'mean')
axis_list[1].legend(framealpha=0.6, frameon=True)
axis_list[1].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
#Add estimated alpha
s='$\\alpha$ = {}'.format(round(alpha,2))
t = axis_list[1].annotate(s, (Lmax/2, means_dp[2]), 
                          fontsize=markersize,bbox=dict(ec='None',facecolor='None',lw=2))


bias=(means_dp[-4]/(M**alpha-1.))**2
sampling_error=1e-6*V_p[-4]
Mcerror=np.sqrt(sampling_error+bias).item()

for i,f in enumerate(reversed(files)):
    e=float(f.split('_')[-1])
    acc[i]=e
    # Load saved data
    with open(os.path.join(f, "averages.pt"), "rb") as fout:
        avgs=torch.load(fout)
    with open(os.path.join(f, "sqaverages.pt"), "rb") as fout:
        sqavgs=torch.load(fout)
    with open(os.path.join(f, "N.pt"), "rb") as fout:
        N=torch.load(fout)
    with np.load(os.path.join(f,'meanpayoff.npz')) as data:
        meanimg=data['meanpayoff']
    if switcher=='secondmoments' or switcher=='means':
        plt.figure(fig2)
        ax[i].imshow(meanimg/255.)
        reala=imagenorm(torch.tensor(meanimg/255.-CIFAR10mean)[None,...])
        ax[i].set_title(f'${e},{round(reala.item(),4)}$')
        ax[i].set_axis_off()
    elif switcher=='acts':
        reala=imagenorm(torch.tensor(meanimg-CIFAR10mean.astype(float))[None,...])
        actserrs[i]=reala
    else:
        raise Exception('switcher not recognised')
    L=3+len(N)-1 
    if e==.001:
        Lmin=5
        L=5+len(N)-1
    
    sumdims=tuple(range(1,len(sqavgs[:,0].shape))) #sqsums is output of payoff element-wise squared, so reduce                        
    s=sqavgs[:,0].shape
    cutoff=0
    means_p=imagenorm(avgs[cutoff:,1])
    V_p=(torch.sum(sqavgs[cutoff:,1],axis=sumdims)/np.prod(s[1:]))-means_p**2 
    means_dp=imagenorm(avgs[cutoff:,0])
    V_dp=(torch.sum(sqavgs[cutoff:,0],axis=sumdims)/np.prod(s[1:]))-means_dp**2  
    
    cost_mlmc[i]=torch.sum(N*(M**np.arange(Lmin,L+1)+np.hstack((0,M**np.arange(Lmin,L)))))*e**2 #cost is number of NFE
    accsplit=(1/.01) if switcher=='acts' else 2
    cost_mc[i]=accsplit*V_p[-1]*(M**L)

    realvar[i]=torch.sum(V_dp/N)
    realbias[i]=(max(means_dp[-2]/M**(alpha),means_dp[-1])/(M**alpha-1))**2
    
    axis_list[2].semilogy(range(Lmin,L+1),N,'k-',marker=i,label=f'{e}',markersize=markersize,
                   markerfacecolor="None",markeredgecolor='k', markeredgewidth=1)


plt.figure(fig2)

if switcher=='means' or switcher=='secondmoments':
    ax[-1].set_title(ax[-1].get_title()+f'\n $MSE=\pm {round(Mcerror,4)}$')
else: #activations
    indices=np.argsort(acc)
    sortacc=acc[indices]
    sorterrs=actserrs[indices]
    plt.plot(sortacc,sorterrs)
    plt.title(plt.gca().get_title()+f'\n $MSE=\pm {round(Mcerror,4)}$')
fig2.tight_layout(pad=.1)
plt.savefig(os.path.join(expdir,'MeanImages.pdf'),bbox_inches='tight',format='pdf')

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
indices=np.argsort(acc)
sortcost_mc=cost_mc[indices]
sortcost_mlmc=cost_mlmc[indices]
sortacc=acc[indices]

axis_list[3].loglog(sortacc,sortcost_mc,'k:',marker=(8,2,0),markersize=markersize,
             markerfacecolor="None",markeredgecolor='k', markeredgewidth=1,label='Std. MC',base=2)
axis_list[3].loglog(sortacc,sortcost_mlmc,'k-',marker=(8,2,0),markersize=markersize,
             markerfacecolor="None",markeredgecolor='k', markeredgewidth=1,label='Std. MLMC',base=2)
axis_list[3].set_xlabel('Acc. $\\varepsilon$')
axis_list[3].set_ylabel('$\\varepsilon^{2}$cost')
axis_list[3].legend(frameon=True,framealpha=0.6)

#Add title and space out subplots
fig.suptitle(label+f'\n$M={M}$')
fig.tight_layout(rect=[0, 0.03, 1, 0.94],h_pad=1,w_pad=1,pad=1)

fig.savefig(os.path.join(expdir,'GilesPlot.pdf'), format='pdf', bbox_inches='tight')

files = np.array([os.path.join(expdir,f) for f in os.listdir(expdir) if f.startswith('level') and not f.endswith('pdf')])
orderer=np.array([int(f.split('_')[-1]) for f in files])
indices=np.argsort(orderer)
files=files[indices]
cutoff=0
fig,ax=plt.subplots(nrows=2,ncols=len(files)-cutoff,sharey=True,sharex=True,tight_layout=True)
# plt.rc('axes', titlesize=4)     # fontsize of the axes title

for i,f in enumerate(files):
    l=int(f.split('_')[-1])
    with np.load(os.path.join(f,'samples_f.npz')) as data:
        num=data['samplesf'].shape[0]
        r=np.random.randint(low=0,high=num)
        ax[0,i].imshow(data['samplesf'][r])
        ax[0,i].set_title(f'{M**l} steps')
        ax[0,i].tick_params(
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False, 
            left=False,# ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off
    with np.load(os.path.join(f,'samples_c.npz')) as data:
        ax[1,i].set_title(f'{M**(l-1)} steps')
        ax[1,i].imshow(data['samplesc'][r])
        ax[1,i].tick_params(
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,
            left=False,# ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off
    
ax[1,0].set_ylabel('coarse')
ax[0,0].set_yticklabels(())
ax[0,0].set_ylabel('fine')
ax[1,0].set_yticklabels(())

fig.tight_layout(w_pad=.1,h_pad=.1)
plt.savefig(os.path.join(expdir,f'SampleLevels.pdf'),format='pdf',bbox_inches='tight')
        