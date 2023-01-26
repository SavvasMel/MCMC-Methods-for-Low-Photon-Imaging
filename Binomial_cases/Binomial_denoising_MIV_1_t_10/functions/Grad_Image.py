"""
Calculate the image gradient.

@author: Savvas Meldonis
"""
import numpy as np
import torch

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def Grad_Image(x,device):

    with torch.no_grad():

        x = x.clone()
        x_temp = x[1:, :] - x[0:-1,:]
        dux = torch.cat((x_temp.T,torch.zeros(x_temp.shape[1],1,device=device)),1).to(device)
        dux = dux.T
        x_temp = x[:,1:] - x[:,0:-1]
        duy = torch.cat((x_temp,torch.zeros((x_temp.shape[0],1),device=device)),1).to(device)
        return  torch.cat((dux,duy),dim=0).to(device)
