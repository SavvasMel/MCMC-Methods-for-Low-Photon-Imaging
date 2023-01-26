"""
IMAGE DENOISING EXPERIMENT UNDER BINOMIAL NOISE

We implement the Reflected MY-MALA algorithm described in: "Efficient Bayesian Computation 
for low-photon imaging problems", Savvas Melidonis, Paul Dobson, Yoann Altmann, 
Marcelo Pereyra, Konstantinos C. Zygalakis, arXiv https://arxiv.org/abs/2206.05350 , June 2022

@author: Savvas Melidonis

"""

#%% Setting the necessary packages.

import numpy as np
import torch
import math
from methods.RMYMALA import RMYMALA
from methods.RMYULA import RMYULA
from scipy.io import savemat

from functions.norml12 import tv
from functions.Grad_Image import Grad_Image
from functions.plots import plots
from functions.ac_and_variance_plots import ac_var_plots
from functions.welford import welford
from functions.chambolle_prox_TV import chambolle_prox_TV

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

path = '../bin_denoising_MIV_1_t10_RMYMALA'
if not os.path.exists(path):
  # Create a new directory because it does not exist 
  os.makedirs(path)
  print("The new directory is created!")

#%% Load the true image

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
#%% Setup experiment

h = torch.ones([nx,ny]).to(device).detach().clone()
b = (torch.zeros([nx,ny]) + lambda_mean*0.01).to(device).detach().clone()
t = (torch.zeros([nx,ny], dtype=int) + 10).to(device).detach().clone()

# Create synthetic data

y_np = np.random.binomial(t.cpu().numpy().astype(np.int32),1-np.exp(-(h.cpu().numpy()*im_np)))
y = torch.from_numpy(y_np.astype(np.float64)).cuda(device).detach().clone() 

fig, ax = plt.subplots()
ax.set_title("Noisy image")
plt.imshow(y.cpu().numpy(),cmap="gray")

#%% Set up parameters and gradients

# Lipschitz constants and algorithm parameters

L_F = torch.max(y)*(torch.max(h)**2)*torch.exp(-torch.min(b))/((1-torch.exp(-torch.min(b)))**2)   # Lipshcitz constant

lambda_prox = 1/L_F 	 # MY enveloppe parameter 
theta = 6.4               # hyperparameter of the prior

L_G = 1/lambda_prox
L_Phi = L_F+L_G

F=lambda z: torch.sum(-y*torch.log(1-torch.exp(-(h*z+b)))+h*(t-y)*z)    # Negative log-likelihood   
G=lambda z: theta*tv(Grad_Image(z,device))                              # Regularization term

logPi = lambda z:  (- F(z) - G(z))                             # Log of posterior distribution

def gradF(z):
    inv= 1-torch.exp(-(h*z+b))
    return  -y*h*(torch.exp(-(h*z+b))/inv) + h*(t-y)

varagrin = {'lambda' : theta*lambda_prox, 'MaxIter' : 25}
proxG = lambda x: chambolle_prox_TV(x,varagrin,device)

gradG = lambda x: (x -proxG(x))/lambda_prox 			 # gradient of the prior
    
grad_Phi = lambda x: gradF(x) + gradG(x) 				 # gradient of the model

#%% Metrics
MSE = lambda z: (1/dim)*np.linalg.norm((z-im_np).ravel(),2)**2
NRMSE = lambda z: np.linalg.norm((z-im_np).ravel(),2)/np.linalg.norm(im_np.ravel(),2)  # NRMSE (Normalized RMSE)
RSNR = lambda z: 10*np.log10(np.sum(im_np.ravel()**2)/np.sum((z-im_np).ravel()**2))    # Reconstructed signal-Noise ratio
PSNR = lambda z: 20*math.log10(np.max(im_np))-10*math.log10(MSE(z))                    # Reconstructed signal-Noise ratio
SSIM = lambda z: ssim(im_np, z, data_range=float(np.max(im_np)-np.min(im_np)))

#%% Setting the experiment

nSamples_sr = int(1e6)                     # number of samples to produce in the sampling stage (long run)
nSamples_burnin = int(nSamples_sr*0.05)    # number of samples to produce in the burnin stage
X = y/torch.mean(y)*lambda_mean            # Initialization of the chain

# save the trace of the chain
store_samples= int(5000)    # thinning samples
RMYMALA_trace=torch.zeros([store_samples,nx,ny])
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

#%% Run RMYMALA

nSamples_warm = 10

print(' ')
print('BEGINNING OF A SMALL WARM-UP MYULA PERIOD')

for i in range(0,nSamples_warm):

    X,_ = RMYULA(X,L_Phi,grad_Phi,device)

print(' ')
print('BEGINNING OF THE BURN-IN PERIOD')

grad_Xk = grad_Phi(X)                 # Gradient of initial condition
logPiTrace_Xk = logPi(X)              # logPi value of initial condition
stepsize = 1/L_Phi                    # Initial stepsize
accept_rate = 0                       # Initial acceptance rate

start_time_burnin = time.time()

post_meanvar_burnin = welford(X)
NRMSE_trace_sr = np.append(NRMSE_trace_sr, NRMSE(post_meanvar_burnin.get_mean().cpu().numpy()))
logPiTrace_sr = np.append(logPiTrace_sr,logPi(X).cpu().numpy())
PSNR_trace_sr = np.append(PSNR_trace_sr, PSNR(post_meanvar_burnin.get_mean().cpu().numpy()))
SSIM_trace_sr = np.append(SSIM_trace_sr, SSIM(post_meanvar_burnin.get_mean().cpu().numpy()))
i_x_sr =0

text_file = open(path + '/results.txt', "a")
text_file.write('Iteration [{}/{}] NRMSE : {:.3e}, PSNR: {:.2f}, SSIM : {:.2f} \n'.format(1, nSamples_burnin+nSamples_sr, NRMSE_trace_sr[-1], PSNR_trace_sr[-1], SSIM_trace_sr[-1]))
text_file.close()

for i_x in tqdm(range(1,int(nSamples_burnin+nSamples_sr))):

    X, grad_Xk, logPiTrace_Xk, alpha, min_values, accept_sample = RMYMALA(X,grad_Xk,logPiTrace_Xk,stepsize,grad_Phi,logPi,device)

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

print('\nEND OF SAMPLING PERIOD')

#%% Results

print("NRMSE (from burn-in): ", NRMSE_trace_sr[-1])
print("NRMSE: ", NRMSE(post_meanvar.get_mean().cpu().numpy()))
print("RSNR of noisy image: ", PSNR((y/torch.mean(y)*lambda_mean).cpu().numpy()))
print("RSNR of posterior mean: ", PSNR(post_meanvar.get_mean().cpu().numpy()))

RMYMALA_results = {
    "theta": theta,
    "acceptance_ratio_sr": round(accept_rate/(i_x+nSamples_burnin+1),3),
    "NRMSE_trace_MYMALA": NRMSE_trace_sr,
    "NRMSE_MYMALA": NRMSE(post_meanvar.get_mean().cpu().numpy()),
    "RSNR_MYMALA": RSNR(post_meanvar.get_mean().cpu().numpy()),
    "PSNR_MYMALA": PSNR(post_meanvar.get_mean().cpu().numpy()),
    "SSIM_MYMALA": SSIM(post_meanvar.get_mean().cpu().numpy()),
    "logPiTrace_MYMALA" : logPiTrace_sr,
    "neg_values_MYMALA": min_values_trace,
    "meanSamples_MYMALA": post_meanvar.get_mean().cpu().numpy(),
    "variance_MYMALA": post_meanvar.get_var().cpu().numpy()}

RMYMALA_MC_chain={"RMYMALA_MC_chain": RMYMALA_trace.cpu().numpy()}

savemat(path + "/bin_denoising_MIV_1_t10_RMYMALA_results.mat", RMYMALA_results)
savemat(path + "/bin_denoising_MIV_1_t10_RMYMALA_MC_chain.mat", RMYMALA_MC_chain)

img_type = "png"
plots(im,y,post_meanvar, NRMSE_trace_sr, logPiTrace_sr, min_values_trace, path, img_type,alpha_trace)
method_str = "RMYMALA"
ac_var_plots(RMYMALA_trace.cpu().numpy(), method_str, path, img_type)


#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------

if verbose_lr == 1:
    print(' ')
    print('BEGINNING OF THE LONG-RUN PERIOD with theta= ',theta)

    start_time = time.time()

    for i_x in tqdm(range(nSamples_lr)):

        # Update X

        X, grad_Xk, logPiTrace_Xk, alpha, neg_values, accept_sample = RMYMALA(X,grad_Xk,logPiTrace_Xk,stepsize,grad_Phi,logPi,device)

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

    print("NRMSE (long-run): ", NRMSE_trace_lr[-1])
    print("RSNR of posterior mean (long-run): ", RSNR(post_meanvar_burnin.get_mean().cpu().numpy()))

    RMYMALA_results_lr = {
	"theta": theta,
	"acceptance_ratio_lr": round(accept_rate/(i_x+nSamples_sr+nSamples_burnin+1),3),
	"NRMSE_trace_MYMALA_lr": NRMSE_trace_lr,
    "NRMSE_MYMALA_lr": NRMSE_trace_lr[-1],
    "RSNR_MYMALA_lr": RSNR(post_meanvar_burnin.get_mean().cpu().numpy()),
    "PSNR_MYMALA_lr": PSNR(post_meanvar_burnin.get_mean().cpu().numpy()),
    "SSIM_MYMALA_lr": SSIM(post_meanvar_burnin.get_mean().cpu().numpy()),
	"logPiTrace_MYMALA_lr" : logPiTrace_lr,
	"meanSamples_MYMALA_lr": post_meanvar_burnin.get_mean()}

    savemat(path + "/bin_denoising_MIV_1_t10_RMYMALA_results_lr.mat", RMYMALA_results_lr)
