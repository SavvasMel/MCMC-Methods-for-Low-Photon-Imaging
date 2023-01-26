"""
The Metropolis Hastings (MH) algorithm with a folded normal
distribution as a proposal.

"""
import torch 

def Metropolis_Hastings(X_proposal, X, grad_X_proposal,grad_X, logPi_proposal, logPi_X, dt, device):
    
    def q(y,x,grad,dt):
        a1 = torch.exp(-(1/(4*dt))*(y - (x - dt*grad))**2)
        a2 = torch.exp(-(1/(4*dt))*(y + (x - dt*grad))**2)
        return torch.sum(torch.log(a1 + a2))
    
    log_u = torch.log(torch.rand(1).to(device))
    r = logPi_proposal - logPi_X + q(X, X_proposal, grad_X_proposal, dt) - q(X_proposal, X, grad_X, dt)
    alpha = torch.minimum(torch.zeros(1).to(device), r)    
        
    if log_u < alpha: 
        accept_sample = 1
        return X_proposal, grad_X_proposal, logPi_proposal, alpha, accept_sample
    else:
        accept_sample = 0
        return X,grad_X, logPi_X, alpha, accept_sample   
            
