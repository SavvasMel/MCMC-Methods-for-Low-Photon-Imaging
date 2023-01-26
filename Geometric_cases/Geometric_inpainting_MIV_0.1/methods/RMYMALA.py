"""
Reflected MYMALA SAMPLING METHOD                        

This function samples the distribution \pi(x) = exp(-F(x)-G(x)) under non-negativity
constraint thanks to a proximal MCMC algorithm called Reflected MYMALA (Reflected MYULA
+ MH correction step) (see "Efficient Bayesian Computation for low-photon imaging problems", Savvas
Melidonis, Paul Dobson, Yoann Altmann, Marcelo Pereyra, Konstantinos C. Zygalakis, arXiv, June 2022).

    INPUTS:
        X: current MCMC iterate (2D-array)
        grad X: gradient of current MCMC iterate (2D-array)
        grad_Phi: handle function that computes the gradient of the potential F(x)+G(x)
        logPi: log-posterior probability of current MCMC iterate
        stepsize: stepsize of discretization
        device : user-defined cuda device

    OUTPUTS:

        Xk_new: new value for X (2D-array).
        grad_Xknew: gradient of new MCMC iterate (2D-array)
        logPi_Xknew: log-posterior probability of new MCMC iterate
        alpha: log of acceptance probability
        min_value: minimum value of the new MCMC iterate (2D-array) before reflection
        accept_sample: flag if the proposal got accepted or not (0 or 1) 

@author: Savvas Melidonis
"""

import torch 
from functions.Metropolis_Hastings import Metropolis_Hastings

def RMYMALA(X,grad_X,logPi_X,stepsize,grad_Phi,logPi,device):
        
    Q = torch.sqrt(2*stepsize)*torch.randn_like(X).to(device) # diffusion term
                                    
    # MYULA sample
    
    X_step = X - stepsize*grad_X + Q
    X_proposal = torch.abs(X_step)
    grad_X_proposal = grad_Phi(X_proposal) 
    logPi_proposal = logPi(X_proposal)
    min_value = torch.min(X_step)
    Xknew, grad_Xknew, logPi_Xknew, alpha, accept_sample = Metropolis_Hastings(X_proposal, X, grad_X_proposal, grad_X, logPi_proposal, logPi_X, stepsize, device)
   
    return Xknew, grad_Xknew, logPi_Xknew, alpha, min_value, accept_sample 
