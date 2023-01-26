"""
The Metropolis Hastings (MH) algorithm with a folded normal
distribution as a proposal.

"""
import torch 

def Metropolis_Hastings(X_proposal,X,\
                        grad_X_proposal,grad_X,\
                        logPi_proposal,logPi_X,dt,device):
    
    q= lambda y,x,grad,dt: torch.sum(torch.log(torch.exp(-(1/(4*dt))*(y-(x- dt*grad))**2) \
                            + torch.exp(-(1/(4*dt))*(y+(x- dt*grad))**2)))
                
    # Produce random number
    
    log_u=torch.log(torch.rand(1).to(device))
    
    alpha = torch.minimum(torch.Tensor([0]).to(device),logPi_proposal-logPi_X \
            +q(X,X_proposal,grad_X_proposal,dt) \
            -q(X_proposal,X,grad_X,dt))    
        
    if log_u<alpha: 
        return X_proposal,grad_X_proposal,logPi_proposal,alpha
    else:
        return X,grad_X,logPi_X,alpha     
            
