"""
Calculate isotropic TV norm of image x.

@author: SavvasM
"""

import numpy as np
import torch

def tv(Dx):
    
    with torch.no_grad():

        Dx=Dx.view(-1)
        N = len(Dx)
        Dux = Dx[:int(N/2)]
        Dvx = Dx[int(N/2):N]
        tv = torch.sum(torch.sqrt(Dux**2 + Dvx**2))
        
        return tv