"""
Reflected MYMALA SAMPLING METHOD                        

This function samples the distribution \pi(x) = exp(-F(x)-G(x)) under non-negativity
constraint thanks to a proximal MCMC algorithm called Reflected MYULA
(see "Efficient Bayesian Computation for low-photon imaging problems", Savvas
Melidonis, Paul Dobson, Yoann Altmann, Marcelo Pereyra, Konstantinos C. Zygalakis, arXiv, June 2022).

    INPUTS:
        X: current MCMC iterate (2D-array)
        grad X: gradient of current MCMC iterate (2D-array)
        Lipschitz: user-defined lipschitz constant of the model
        grad_Phi: handle function that computes the gradient of the potential F(x)+G(x)
        device : user-defined cuda device

    OUTPUTS:

        Xk_new: new value for X (2D-array).
        min_value: minimum value of the new MCMC iterate (2D-array) before reflection

@author: Savvas Melidonis
"""

import torch

def RMYULA(X,Lipschitz,grad_Phi,device):
   
    # MYULA step-size
    dtMYULA = 1/(Lipschitz) # step-size

    Q=torch.sqrt(2*dtMYULA)*torch.randn_like(X).cuda(device) # diffusion term

    grad= grad_Phi(X) 
    
    # MYULA sample
    XkMYULA = (X - dtMYULA*grad + Q).detach().clone()
    
    Xnew = torch.abs(XkMYULA)
    min_value = torch.min(XkMYULA)
	 
    return Xnew, min_value   # new sample produced by the MYULA algorithm and the minimum value of this sample.
    