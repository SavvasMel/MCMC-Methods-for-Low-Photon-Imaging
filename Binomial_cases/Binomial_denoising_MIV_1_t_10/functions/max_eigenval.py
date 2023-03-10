"""
Compute maximal singular value of matrix AA*

Based on https://github.com/anafvidal/research-code/blob/master/utils/max_eigenval.m
author: Savvas Melidonis
"""
import numpy as np
from scipy.linalg import norm
import torch

def max_eigenval(A, At, im_size, tol, max_iter, verbose,device):

    with torch.no_grad():

    #computes the maximum eigen value of the compund operator AtA
        
        x = torch.normal(mean=0, std=1,size=(im_size,im_size))[None][None].to(device)
        x = x/torch.norm(torch.ravel(x),2)
        init_val = 1
        
        for k in range(0,max_iter):
            y = A(x)
            x = At(y)
            val = torch.norm(torch.ravel(x),2)
            rel_var = torch.abs(val-init_val)/init_val
            if (verbose > 1):
                print('Iter = {}, norm = {}',k,val)
            
            if (rel_var < tol):
                break
            
            init_val = val
            x = x/val
        
        if (verbose > 0):
            print('Norm = {}', val)
        
        return val