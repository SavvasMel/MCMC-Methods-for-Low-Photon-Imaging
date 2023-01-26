'''
This code estimates the regularization parameter theta for the experiments
in [1]. We estimate theta by maximising the marginal likelihood p(y|theta) by adapting
the code presented in [2] (we make use of the introduced Reflected SDEs and respective
discretization in [1] instead of the usual Langevin SDEs).

The original code can be found in https://github.com/anafvidal/research-code.

[1] Savvas Melidonis, Paul Dobson, Yoann Altmann, Marcelo Pereyra, Konstantinos C. Zygalakis
Efficient Bayesian Computation for low-photon imaging problems.
arXiv preprint https://arxiv.org/abs/2206.05350, June 2022

[2] A. F. Vidal, V. De Bortoli, M. Pereyra, and A. Durmus (2020). 
Maximum Likelihood Estimation of Regularization Parameters in High-Dimensional Inverse Problems: 
An Empirical Bayesian Approach Part I: Methodology and Experiments. 
SIAM Journal on Imaging Sciences, 13(4), 1945-1989.

'''

#%% Setting the necessary packages.

import numpy as np
import torch

from PIL import Image

import os
import pickle as pl
import matplotlib
import matplotlib.pyplot as plt
from scipy.io import savemat

from functions.norml12 import tv
from functions.Grad_Image import Grad_Image
from functions.chambolle_prox_TV import chambolle_prox_TV
from functions.SAPG import SAPG
from functions.blur_operators import blur_operators

from tqdm.auto import tqdm

#%% Set a seed for the random number generation, choose cuda device, create path for results.
SEED = 666

np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device('cuda')
torch.set_default_tensor_type(torch.FloatTensor)
print('Using device:', device)


path = '../results/SAPG'
isExist = os.path.exists(path) # Check whether the specified path exists or not
if not isExist:
  os.makedirs(path)            # Create a new directory because it does not exist 
  print("The new directory is created!")

#%% Import and show the image

N = 256  #image dimension
dim=N*N  #vector dimension

im = np.loadtxt('images/cameraman.txt', dtype=float)

N = im.shape[0]
dim = N*N

lambda_mean = 1  # mean photons per pixel
z_true = torch.Tensor(im.astype(np.float64))
z_true = (z_true/torch.mean(z_true)*lambda_mean).cuda(device).detach().clone()

#%% Create the blurring operator and create synthetic data
kernel_len = [5,5]
size = [im.shape[0],im.shape[1]]
type_blur = "uniform"
K, KT, K_norm = blur_operators(kernel_len, size, type_blur, device)

y=np.array(np.random.poisson(K(z_true)),np.float)

#%% Algorithm settings

# Algorithm parameters
lambdaMax = 2
beta = lambda_mean * 0.01

# Lipschitz Constants

L_F = (torch.max(y)/beta**2)*(K_norm**2)            # Lipshcitz constant
lambda_prox =  min(1/L_F,lambdaMax)      # regularization parameter

F=lambda z:  torch.sum(K(z)+beta-torch.log(K(z)+beta)*y)            # Negative log-likelihood
g=lambda z: tv(Grad_Image(z,device))
G=lambda z,theta: theta*g(z)                   # Regularization term

logPi = lambda z,theta:  (- F(z) - G(z,theta))  # Log of posterior distribution

def gradF(z):
    Kz= K(z)
    inv_Kz= 1/(Kz+ beta)
    return  1 - KT(y*inv_Kz) 

proxG = lambda x,k: chambolle_prox_TV(x,{'lambda' : k, 'MaxIter' : 25},device)

gradG = lambda x,k: (x -proxG(x,k))/lambda_prox   # gradient of the prior

#%% Run the experiment

X_init= y/torch.mean(y)*lambda_mean
gamma = 0.98*1/(L_F+(1/lambda_prox))
th_init=45
min_th=0.01
max_th=100
d_scale= 10  
d_exp=0.8
warmupSteps=1000
burnIn=25
total_iter=100
                        
theta_EB,last_theta,thetas,logPiTraceX,gXTrace,last_samp= SAPG(y,X_init,gamma,lambda_prox,th_init,min_th,max_th, d_scale,d_exp,
                                                                gradF,gradG,logPi,g,
                                                                warmupSteps,burnIn,total_iter,device) 

print("Estimated value of theta ",theta_EB)

data= {
	"th_init": th_init,
	"theta_EB": theta_EB,
	"last_theta": last_theta,
	"thetas": thetas,
	"logPiTraceX": logPiTraceX,
	"gXTrace": gXTrace,
	"last_samp": last_samp
}

savemat(path + "/data_SAPG.mat", data)

matplotlib.rc('font', size=20)
matplotlib.rc('font', family='serif')
matplotlib.rc('figure', figsize=(14, 8))
matplotlib.rc('lines', linewidth=2.5,linestyle="-.")
matplotlib.rc('lines', markersize=10)
matplotlib.rc('figure.subplot', hspace=.4)

plot1 = plt.figure()
plt.plot(thetas[:],linestyle="-",label="$\\theta_{n}$")
plt.xlabel("$Iterations$")
plt.ylabel("$\\theta$")
plt.savefig(path + "/trace_theta.png")
plt.show(block=False)
plt.close()

plot1 = plt.figure()
plt.plot(logPiTraceX,linestyle="-",label="$\\theta_{n}$")
plt.xlabel("$Iterations$")
plt.ylabel("$log(p(x|y))$")
plt.savefig(path + "/trace_logPi.png")
plt.show(block=False)
plt.close()

plot1 = plt.figure()
plt.plot(gXTrace[:],linestyle="-",label="$\\theta_{n}$")
plt.xlabel("$Iterations$")
plt.ylabel("$G(x)$")
plt.savefig(path + "/trace_G.png")
plt.show(block=False)
plt.close()

plot1 = plt.figure()
plt.plot(gXTrace[10:],linestyle="-",label="$G(X^{(n)})$")
plt.plot(256*256/thetas[10:],linestyle="-",label="$d/\\theta^{(n)}$")
plt.xlabel("$Iterations$")
plt.ylabel("$G(x)$")
plt.legend()
plt.savefig(path + "/trace_G_validity.png")
plt.show(block=False)
plt.close()