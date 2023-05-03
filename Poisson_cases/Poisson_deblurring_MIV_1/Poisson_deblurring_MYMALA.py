"""
IMAGE DEBLURRING EXPERIMENT UNDER POISSON NOISE

We implement the Reflected MY-MALA algorithm described in: "Efficient Bayesian Computation 
for low-photon imaging problems", Savvas Melidonis, Paul Dobson, Yoann Altmann, 
Marcelo Pereyra, Konstantinos C. Zygalakis, arXiv https://arxiv.org/abs/2206.05350 , June 2022

@author: Savvas Melidonis

"""

#%% Setting the necessary packages.

import numpy as np
import torch
import math
from methods.RMYMALA import MYMALA_refl
from methods.RMYULA import MYULA_refl
from scipy.io import savemat

from functions.norml12 import tv
from functions.Grad_Image import Grad_Image
from functions.chambolle_prox_TV import chambolle_prox_TV
from functions.plots import plots
from functions.ac_and_variance_plots import ac_var_plots
from functions.welford import welford
from functions.blur_operators import blur_operators

from skimage.metrics import structural_similarity as ssim

import matplotlib.pyplot as plt

from tqdm import tqdm
import time

import os

SEED = 666
np.random.seed(SEED)
torch.manual_seed(SEED)

# Cuda 
device = torch.device('cuda:0')
print('Using device:', device)

torch.set_default_tensor_type(torch.DoubleTensor)

path = '../Poisson_deblurring_MIV_1_RMYMALA'
if not os.path.exists(path):
    os.makedirs(path)
    print("The new directory is created!")

#%% Load the true image

# Load image
im_np = np.loadtxt('images/cameraman.txt', dtype=float)

# Define dimensions of image
nx = im_np.shape[0]
ny = im_np.shape[1]
dim=nx*ny

# Scale the image to have the desired Mean Intensity Value (M.I.V) 
lambda_mean = 1
im_np = (im_np/np.mean(im_np)*lambda_mean)
im = torch.Tensor(im_np.astype(np.float64)).cuda(device)

# Show the true image
fig, ax = plt.subplots()
ax.set_title("Ground truth")
plt.imshow(im.cpu().numpy(), cmap="gray")

#%% Create synthetic data

kernel_len = [5,5]
size = [im.shape[0],im.shape[1]]
type_blur = "uniform"
K, KT, K_norm = blur_operators(kernel_len, size, type_blur, device)

y = torch.poisson(K(im))

fig, ax = plt.subplots()
ax.set_title("Noisy image")
plt.imshow(y.cpu().numpy(),cmap="gray")

#%% Set up parameters and gradients

# Algorithm parameters
beta = lambda_mean*0.01                            # parameter for the well-defined poisson likelihood
lambda_prox = 1/((torch.max(y)/beta**2)*(K_norm))  # MY enveloppe parameter 
theta = 5.65                                       # hyperparameter of the prior

# Lipschitz Constants

L_F = (torch.max(y)/beta**2)*(K_norm) # Lipshcitz constant
L_G = 1/lambda_prox

L_Phi = L_F+L_G

F = lambda z: torch.sum(K(z)-torch.log(K(z))*y)    # Negative log-likelihood
G = lambda z: theta*tv(Grad_Image(z,device))       # Regularization term

logPi = lambda z:  (- F(z) - G(z))               # Log of posterior distribution

def gradF(z):
    
    Kz= K(z)
    inv_Kz= 1/(Kz+ beta)
    return  1 - KT(y*inv_Kz) 

varagrin = {'lambda' : theta*lambda_prox, 'MaxIter' : 25}
proxG = lambda x: chambolle_prox_TV(x,varagrin,device)

gradG = lambda x: (x -proxG(x))/lambda_prox 			 # gradient of the prior
    
grad_Phi = lambda x: gradF(x) + gradG(x) 				 # gradient of the model

#%% Setting the experiment

nSamples_sr = int(1e6)                       # number of samples to produce in the sampling stage (short run)
nSamples_burnin = int(nSamples_sr*0.05)      # number of samples to produce in the burnin stage

# save the trace of the chain
store_samples= int(5000)    # number of samples to be saved
RMYMALA_trace = torch.zeros([store_samples,nx,ny])
k=0

# save the log-pi trace
logPiTrace_sr = np.zeros(0)         

# to save the NRMSE,PSNR, SSIM of the posterior mean in the sampling stage
NRMSE_trace_sr = np.zeros(0)                              
PSNR_trace_sr = np.zeros(0)                              
SSIM_trace_sr = np.zeros(0)                              

# Store the minimum value of each sample to check the range of negative values bfore reflection.
min_values_trace = np.zeros(0)

# Store the acceptance probabilities
alpha_trace = np.zeros(0)

# Change verbose_lr to 1 if you wish a long-run MALA for bias check
verbose_lr = 1
if verbose_lr == 1:
    nSamples_lr = int(1e6)  # additional number of samples to produce in the sampling stage (long run)
    logPiTrace_lr = np.zeros(0)         
    NRMSE_trace_lr = np.zeros(0)                             

#%% Metrics
MSE = lambda z: (1/dim)*np.linalg.norm((z-im_np).ravel(),2)**2
NRMSE = lambda z: np.linalg.norm((z-im_np).ravel(),2)/np.linalg.norm(im_np.ravel(),2)  # NRMSE (Normalized RMSE)
RSNR = lambda z: 10*np.log10(np.sum(im_np.ravel()**2)/np.sum((z-im_np).ravel()**2))    # Reconstructed signal-Noise ratio
PSNR = lambda z: 20*math.log10(np.max(im_np))-10*math.log10(MSE(z))                    # Reconstructed signal-Noise ratio
SSIM = lambda z: ssim(im_np, z, data_range=float(np.max(im_np)-np.min(im_np)))

#%% Run RMYMALA

print(' ')
print('BEGINNING OF A SMALL WARM-UP MYULA PERIOD')

nSamples_warm = 10
X = (y+beta)/torch.mean(y+beta)*lambda_mean  # Initialization of the chain

for i in range(0,nSamples_warm):

    X,_ = MYULA_refl(X,L_Phi,grad_Phi,device)

print(' ')
print('BEGINNING OF THE SHORT RUN SAMPLING')

grad_Xk = grad_Phi(X)                 # Gradient of initial condition
logPiTrace_Xk = logPi(X)              # logPi value of initial condition
stepsize = 1/L_Phi                    # stepsize
accept_rate = 0                       # Initial acceptance rate

post_meanvar_burnin = welford(X)
NRMSE_trace_sr = np.append(NRMSE_trace_sr, NRMSE(post_meanvar_burnin.get_mean().cpu().numpy()))
logPiTrace_sr = np.append(logPiTrace_sr,logPi(X).cpu().numpy())
PSNR_trace_sr = np.append(PSNR_trace_sr, PSNR(post_meanvar_burnin.get_mean().cpu().numpy()))
SSIM_trace_sr = np.append(SSIM_trace_sr, SSIM(post_meanvar_burnin.get_mean().cpu().numpy()))

text_file = open(path + '/results.txt', "a")
text_file.write('Iteration [{}/{}] NRMSE : {:.3e}, PSNR: {:.2f}, SSIM : {:.2f} \n'.format(1, nSamples_burnin+nSamples_sr, NRMSE_trace_sr[-1], PSNR_trace_sr[-1], SSIM_trace_sr[-1]))
text_file.close()

start_time_burnin = time.time()
i_x_sr = 0

for i_x in tqdm(range(1,int(nSamples_burnin+nSamples_sr))):

    X, grad_Xk, logPiTrace_Xk, alpha, min_values, accept_sample = MYMALA_refl(X,grad_Xk,logPiTrace_Xk,stepsize,grad_Phi,logPi,device)

    if  accept_sample == 1:
        accept_rate = accept_rate + 1
        if i_x%100 == 0:
            print(" Acceptance rate : ", round(accept_rate/(i_x+1),3))

    # Update statistics
    post_meanvar_burnin.update(X)

    # save log posterior trace of the new sample
    logPiTrace_sr = np.append(logPiTrace_sr,logPiTrace_Xk.cpu().numpy())
    
    # save NRMSE
    NRMSE_trace_sr = np.append(NRMSE_trace_sr, NRMSE(post_meanvar_burnin.get_mean().cpu().numpy()))

    # save PSNR
    PSNR_trace_sr = np.append(PSNR_trace_sr, PSNR(post_meanvar_burnin.get_mean().cpu().numpy()))

    # save SSIM
    SSIM_trace_sr = np.append(SSIM_trace_sr, SSIM(post_meanvar_burnin.get_mean().cpu().numpy()))

    # save minimum value of sample X before reflection
    min_values_trace = np.append(min_values_trace, min_values.cpu().numpy())

    # save acceptance probability
    alpha_trace = np.append(alpha_trace, alpha.cpu().numpy())

    if (i_x+1)%500==0 or (i_x+1)==(nSamples_burnin+nSamples_sr):

        text_file = open(path + '/results.txt', "a")
        text_file.write('Iteration [{}/{}] NRMSE : {:.3e}, PSNR: {:.2f}, SSIM : {:.2f} \n'.format(i_x + 1, nSamples_burnin+nSamples_sr, NRMSE_trace_sr[-1], PSNR_trace_sr[-1], SSIM_trace_sr[-1]))
        text_file.close()

    if i_x > nSamples_burnin-1 :

        if i_x_sr == 0:
            # Initialise recording of sample summary statistics after burnin
            post_meanvar = welford(X)
        else:
            post_meanvar.update(X)

        if (i_x_sr+1)%(int((nSamples_sr)/store_samples))==0:
            
            RMYMALA_trace[k] = X
            k=k+1

        i_x_sr = i_x_sr + 1

    if (i_x+1)%10==0 and verbose_lr == 1:
        
        # save  log posterior trace of the new sample (long run)
        logPiTrace_lr = np.append(logPiTrace_lr, logPiTrace_sr[-1])
        # save NRMSE (long run)
        NRMSE_trace_lr = np.append(NRMSE_trace_lr, NRMSE_trace_sr[-1])
        # save negative values of the new sample

torch.cuda.synchronize() 
end_time_burnin = time.time()
elapsed_burnin = end_time_burnin - start_time_burnin       

print('\nEND OF SHORT-RUN PERIOD')

RMYMALA_results = {
    "theta": theta,
    "acceptance_ratio_sr": round(accept_rate/(i_x+nSamples_burnin+1),3),
    "NRMSE_trace_RMYMALA": NRMSE_trace_sr,
    "NRMSE_RMYMALA": NRMSE(post_meanvar.get_mean().cpu().numpy()),
    "RSNR_RMYMALA": RSNR(post_meanvar.get_mean().cpu().numpy()),
    "PSNR_RMYMALA": PSNR(post_meanvar.get_mean().cpu().numpy()),
    "SSIM_RMYMALA": SSIM(post_meanvar.get_mean().cpu().numpy()),
    "logPiTrace_RMYMALA" : logPiTrace_sr,
    "neg_values_RMYMALA": min_values_trace,
    "meanSamples_RMYMALA": post_meanvar.get_mean().cpu().numpy(),
    "variance_RMYMALA": post_meanvar.get_var().cpu().numpy()}

RMYMALA_MC_chain={"RMYMALA_MC_chain": RMYMALA_trace.cpu().numpy()}

savemat(path + "/Poisson_deblur_MIV_1_RMYMALA_results.mat", RMYMALA_results)
savemat(path + "/Poisson_deblur_MIV_1_RMYMALA_MC_chain.mat", RMYMALA_MC_chain)

img_type = "png"
plots(im,y,post_meanvar, NRMSE_trace_sr, logPiTrace_sr, min_values_trace, path, img_type, alpha_trace)
method_str = "RMYMALA"
lags = 100
ac_var_plots(RMYMALA_trace.cpu().numpy(), lags, method_str, path, img_type)

if verbose_lr == 1:
    print(' ')
    print('BEGINNING OF THE LONG-RUN PERIOD with theta= ',theta)

    start_time = time.time()

    for i_x in tqdm(range(nSamples_lr)):

        # Update X

        X, grad_Xk, logPiTrace_Xk, alpha, neg_values, accept_sample = MYMALA_refl(X,grad_Xk,logPiTrace_Xk,stepsize,grad_Phi,logPi,device)

        if  accept_sample == 1:
            accept_rate = accept_rate + 1
            if i_x%100==0:
                print(" Acceptance rate : ", round(accept_rate/(i_x + nSamples_sr + nSamples_burnin + 1),3))
        
        post_meanvar_burnin.update(X)

        if (i_x+1)%10==0:
            
            logPiTrace_lr = np.append(logPiTrace_lr, logPiTrace_Xk.cpu().numpy())
            NRMSE_trace_lr = np.append(NRMSE_trace_lr, NRMSE(post_meanvar_burnin.get_mean().cpu().numpy()))

    end_time = time.time()
    elapsed = end_time - start_time       

    #%% Results

    RMYMALA_results_lr = {
        "theta": theta,
        "acceptance_ratio_lr": round(accept_rate/(i_x+nSamples_sr+nSamples_burnin+1),3),
        "NRMSE_trace_MYMALA_pois_lr": NRMSE_trace_lr,
        "NRMSE_MYMALA_pois_lr": NRMSE_trace_lr[-1],
        "RSNR_MYMALA_pois_lr": RSNR(post_meanvar_burnin.get_mean().cpu().numpy()),
        "PSNR_MYMALA_pois_lr": PSNR(post_meanvar_burnin.get_mean().cpu().numpy()),
        "SSIM_MYMALA_pois_lr": SSIM(post_meanvar_burnin.get_mean().cpu().numpy()),
        "logPiTrace_MYMALA_pois_lr" : logPiTrace_lr,
        "meanSamples_MYMALA_pois_lr": post_meanvar_burnin.get_mean().cpu().numpy()
    }

    savemat(path + "/Poisson_deblur_MIV_1_RMYMALA_results_lr.mat", RMYMALA_results)
