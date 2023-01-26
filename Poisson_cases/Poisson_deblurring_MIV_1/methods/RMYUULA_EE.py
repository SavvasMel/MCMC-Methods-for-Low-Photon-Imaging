"""
Reflected MYUULA SAMPLING METHOD                        

This function samples the distribution \pi(x) = exp(-F(x)-G(x)) under non-negativity
constraint thanks to a proximal MCMC algorithm called Reflected MYUULA
(see "Efficient Bayesian Computation for low-photon imaging problems", Savvas
Melidonis, Paul Dobson, Yoann Altmann, Marcelo Pereyra, Konstantinos C. Zygalakis, arXiv, June 2022).

The algorithm MYUULA is known as Euler Exponential (EE) integrator, 
see http://proceedings.mlr.press/v75/cheng18a.html

    INPUTS:
        X: current MCMC iterate (2D-array)
        V: current velocity (2D-array)
        Lipschitz: user-defined lipschitz constant of the model
        grad_Phi: handle function that computes the gradient of the potential F(x)+G(x)
        device : user-defined cuda device

    OUTPUTS:

        Xk_new: new value for X (2D-array).
        VkMYUULA: new value for velocity (2D-array). 
        min_value: minimum value of the new MCMC iterate (2D-array) before reflection

@author: Savvas Melidonis
"""

import torch

def RMYUULA(X,V,Lipschitz,grad_Phi,device):
           
    # MYUULA step-size
    
    delta = torch.tensor(2).to(device)				

    Q1=torch.randn_like(X).to(device).detach().clone() # noise
    
    Q2=torch.randn_like(X).to(device).detach().clone() # noise
    
    grad= grad_Phi(X) 
    
    gamma = torch.tensor(2).to(device)
  
    # MYUULA sample
    
    var_Vk=  (1/Lipschitz)*(1-torch.exp(-(2*gamma)*delta))
    
    var_Xk=  (1/Lipschitz)*(2*delta/gamma-(1/(gamma**2))*torch.exp(-(2*gamma)*delta)- \
                                (3/(gamma**2))+4*torch.exp(-gamma*delta)/(gamma**2))
    
    cov_Xk_Vk = (1/(gamma*Lipschitz))*(1+torch.exp(-2*gamma*delta)-2*torch.exp(-gamma*delta))
    
    corr= cov_Xk_Vk/(torch.sqrt(var_Vk)*torch.sqrt(var_Xk))
    
    VkMYUULA = V*torch.exp(-gamma*delta) - (1/(gamma*Lipschitz)) \
                                    * (1-torch.exp(-gamma*delta))*grad \
                                    + torch.sqrt(var_Vk)*Q1
    
    XkMYUULA = X + (1/gamma)*(1-torch.exp(-gamma*delta))*V - \
                (1/(gamma*Lipschitz))*(delta-(1/gamma)* \
                                           (1-torch.exp(-gamma*delta)))*grad \
                                            + torch.sqrt(var_Xk)*corr*Q1 \
                                            + torch.sqrt(var_Xk)*torch.sqrt(1-corr**2)*Q2
    
    Xnew = torch.abs(XkMYUULA)
    min_value = torch.min(XkMYUULA)
    
    return Xnew, VkMYUULA, min_value # new sample produced by the MYULA algorithm 
