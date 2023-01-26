import numpy as np
import torch
import scipy
from scipy.sparse import csr_matrix

def mask_torch(p,nx,ny,device,load):

    #Mask matrix (sparse matrix in python)

    if load:

        Ma = scipy.sparse.load_npz("images/mask_matrix.npz")
    
    else:

        mask = (np.random.rand(nx,ny) < p).reshape(nx,ny)
        ind = (mask==True).ravel().nonzero()    
        M = ind[0].shape[0]

        Ma = csr_matrix((np.ones(M), (range(0,M),ind[0])), shape=(M, nx*ny))

    Ma = Ma.tocoo()

    row = torch.from_numpy(Ma.row.astype(np.int64)).to(torch.long)
    col = torch.from_numpy(Ma.col.astype(np.int64)).to(torch.long)
    edge_index = torch.stack([row,col],dim=0)

    val = torch.from_numpy(Ma.data.astype(np.float64)).to(torch.float64)

    Ma = torch.sparse_coo_tensor(edge_index, val, torch.Size(Ma.shape)).to(device)
    
    return Ma

def mask_numpy(p,nx,ny,load):

    #Mask matrix 

    if load:

        Ma = scipy.sparse.load_npz("images/mask_matrix.npz")

    else:

        mask = (np.random.rand(nx,ny) < p).reshape(nx,ny)
        ind = (mask==True).ravel().nonzero()    
        M = ind[0].shape[0]

        Ma = csr_matrix((np.ones(M), (range(0,M),ind[0])), shape=(M, nx*ny))
        
        #scipy.sparse.save_npz("mask_matrix.npz", Ma)

    return Ma