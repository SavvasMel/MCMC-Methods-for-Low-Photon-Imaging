"""
IMAGE INPAINTING EXPERIMENT UNDER GEOMETRIC NOISE

We implement the Reflected MYUULA algorithm described in: "Efficient Bayesian Computation 
for low-photon imaging problems", Savvas Melidonis, Paul Dobson, Yoann Altmann, 
Marcelo Pereyra, Konstantinos C. Zygalakis, arXiv https://arxiv.org/abs/2206.05350 , June 2022

@author: Savvas Melidonis

"""
#%% Setting the necessary packages.

import numpy as np
import torch
import math
from scipy.io import savemat
from methods.RMYUULA_EE import RMYUULA

from functions.norml12 import tv
from functions.Grad_Image import Grad_Image
from functions.chambolle_prox_TV import chambolle_prox_TV
from functions.plots import plots
from functions.ac_and_variance_plots import ac_var_plots
from functions.welford import welford
from functions.max_eigenval import max_eigenval
from functions.mask_matrix import mask_torch

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

path = '../geo_inpainting_MIV_0.1_RMYUULA'
if not os.path.exists(path):
  # Create a new directory because it does not exist 
  os.makedirs(path)
  print("The new directory is created!")

#%% Setup data

im_np = np.loadtxt('images/cameraman.txt', dtype=float)

# Define dimensions of image
nx = im_np.shape[0]
ny = im_np.shape[1]
dim=nx*ny

# Scale the image to have the desired Mean Intensity Value (M.I.V) 
lambda_mean = 0.1
im_np = (im_np/np.mean(im_np)*lambda_mean)
im = torch.Tensor(im_np.astype(np.float64)).cuda(device)

# Show the true image
fig, ax = plt.subplots()
ax.set_title("Ground truth")
plt.imshow(im.cpu().numpy(), cmap="gray")

#%% Create synthetic data

h = torch.ones([nx,ny]).to(device).detach().clone()
b = (torch.zeros([nx,ny]) + lambda_mean*0.01).to(device).detach().clone()

p = 0.70

Ma = mask_torch(p,nx,ny,device,load=False)

Ma_f = lambda x: torch.matmul(Ma,x.ravel())

MaT_f = lambda x: torch.matmul(torch.transpose(Ma,0,1),x).reshape(nx,ny)

im_t = np.random.geometric(1-np.exp((-Ma_f(h*im+b)).cpu().numpy()))
t = torch.Tensor(im_t.astype(np.float64)).to(device).detach().clone()

fig, ax = plt.subplots()
ax.set_title("Noisy image")
plt.imshow((MaT_f(t)).cpu().numpy(),cmap="gray")

#%% Set up parameters and gradients

Ma_norm= max_eigenval(Ma_f, MaT_f, nx, ny,1e-4,int(1e4),0,device)

L_F = Ma_norm*(torch.max(h)**2)*torch.exp(-torch.min(b))/((1-torch.exp(-torch.min(b)))**2)  

lambda_prox = 1/L_F 	 # MY enveloppe parameter 
theta = 43               # hyperparameter of the prior

L_G = 1/lambda_prox
L_Phi = L_F+L_G

F=lambda z: torch.sum(-torch.log(1-torch.exp(-Ma_f(h*z+b))) +Ma_f(h*z+b)*(t-1))#+Ma_f(h*z)*(t-1).ravel())   # Negative log-likelihood   
G=lambda z: theta*tv(Grad_Image(z,device))                                         # Regularization term

logPi = lambda z:  (- F(z) - G(z))                # Log of posterior distribution

def gradF(z):
    inv= 1-torch.exp(-Ma_f(h*z+b))
    return  - h* (MaT_f(torch.exp(-Ma_f(h*z+b))/inv) - MaT_f((t-1).ravel())) 

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

nSamples = int(1e6)     						# number of samples to produce in the sampling stage
nSamples_burnin = int(nSamples*0.05)            # number of samples to produce in the burnin stage
X = (torch.ones([nx,ny])/100).to(device)        # Initialization of the chain
V = torch.zeros([nx,ny]).to(device)

# save the trace of the chain
store_samples = int(5000)    # thinning samples
RMYUULA_trace = torch.zeros([store_samples,nx,ny])
k=0

# save the log-pi trace
logPiTrace = np.zeros(0)         

# to save the NRMSE,PSNR, SSIM of the posterior mean in the sampling stage
NRMSE_trace = np.zeros(0)                              
PSNR_trace = np.zeros(0)                              
SSIM_trace = np.zeros(0)                              

# Store the minimum value of each sample to check the range of negative values bfore reflection.
min_values_trace = np.zeros(0)

#%% Run R-MYUULA

print(' ')
print('BEGINNING OF THE SAMPLING PERIOD')

start_time_burnin = time.time()

post_meanvar_burnin = welford(X)
NRMSE_trace = np.append(NRMSE_trace, NRMSE(post_meanvar_burnin.get_mean().cpu().numpy()))
PSNR_trace = np.append(PSNR_trace, PSNR(post_meanvar_burnin.get_mean().cpu().numpy()))
SSIM_trace = np.append(SSIM_trace, SSIM(post_meanvar_burnin.get_mean().cpu().numpy()))
logPiTrace = np.append(logPiTrace,logPi(X).cpu().numpy())
j_x = 0

text_file = open(path + '/results.txt', "a")
text_file.write('Iteration [{}/{}] NRMSE : {:.3e}, PSNR: {:.2f}, SSIM : {:.2f} \n'.format(1, nSamples_burnin+nSamples, NRMSE_trace[-1], PSNR_trace[-1], SSIM_trace[-1]))
text_file.close()

for i_x in tqdm(range(1,int(nSamples_burnin+nSamples))):
    
    X, V, min_value = RMYUULA(X,V,L_Phi,grad_Phi,device)

    # Update statistics
    post_meanvar_burnin.update(X)

    # save log posterior trace of the new sample
    logPiTrace = np.append(logPiTrace,logPi(X).cpu().numpy())

    # save NRMSE
    NRMSE_trace = np.append(NRMSE_trace, NRMSE(post_meanvar_burnin.get_mean().cpu().numpy()))

    # save PSNR
    PSNR_trace = np.append(PSNR_trace, PSNR(post_meanvar_burnin.get_mean().cpu().numpy()))

    # save SSIM
    SSIM_trace = np.append(SSIM_trace, SSIM(post_meanvar_burnin.get_mean().cpu().numpy()))

    # save minimum value of sample X before reflection
    min_values_trace = np.append(min_values_trace, min_value.cpu().numpy())

    if (i_x+1)%500==0 or (i_x+1)==(nSamples_burnin+nSamples):

        text_file = open(path + '/results.txt', "a")
        text_file.write('Iteration [{}/{}] NRMSE : {:.3e}, PSNR: {:.2f}, SSIM : {:.2f} \n'.format(i_x + 1, nSamples_burnin+nSamples, NRMSE_trace[-1], PSNR_trace[-1], SSIM_trace[-1]))
        text_file.close()

    if i_x > nSamples_burnin-1 :

        if j_x == 0:
            # Initialise recording of sample summary statistics after burnin
            post_meanvar = welford(X)
        else:
            post_meanvar.update(X)

        if (j_x+1)%(int((nSamples)/store_samples))==0:
            
            RMYUULA_trace[k] = X
            k=k+1

        j_x = j_x + 1

torch.cuda.synchronize() 
end_time_burnin = time.time()
elapsed_burnin = end_time_burnin - start_time_burnin       

print('\nEND OF SAMPLING PERIOD')

#%% Results

print("NRMSE (from burn-in): ", NRMSE_trace[-1])
print("NRMSE: ", NRMSE(post_meanvar.get_mean().cpu().numpy()))
print("RSNR of posterior mean: ", RSNR(post_meanvar.get_mean().cpu().numpy()))

RMYUULA_results = {
	"theta": theta,
	"NRMSE_trace_RMYUULA": NRMSE_trace,
	"NRMSE_RMYUULA": NRMSE(post_meanvar.get_mean().cpu().numpy()),
	"RSNR_RMYUULA": RSNR(post_meanvar.get_mean().cpu().numpy()),
    "PSNR_RMYUULA": PSNR(post_meanvar.get_mean().cpu().numpy()),
    "SSIM_RMYUULA": SSIM(post_meanvar.get_mean().cpu().numpy()),
	"logPiTrace_RMYUULA" : logPiTrace,
	"neg_values_RMYUULA": min_values_trace,
	"meanSamples_RMYUULA": post_meanvar.get_mean().cpu().numpy(),
	"variance_RMYUULA": post_meanvar.get_var().cpu().numpy()}

RMYUULA_MC_chain={"RMYUULA_MC_chain": RMYUULA_trace.cpu().numpy()}

savemat(path + "/geo_inpainting_MIV_0.01_RMYUULA_results.mat", RMYUULA_results)
savemat(path + "/geo_inpainting_MIV_0.01_RMYUULA_MC_chain.mat", RMYUULA_MC_chain)
    
# Plots

img_type = "png"
plots(im, MaT_f, t, post_meanvar, NRMSE_trace, logPiTrace, min_values_trace, path, img_type)
method_str = "RMYUULA"
lags = 100
ac_var_plots(RMYUULA_trace.cpu().numpy(), lags, method_str, path, img_type)
