import torch
import numpy as np
from functions.max_eigenval import max_eigenval

def blur_operators(kernel_len, size, type_blur, device, var = None):

    nx = size[0]
    ny = size[1]
    if type_blur=='uniform':
        h = torch.zeros(nx,ny).to(device)
        lx = kernel_len[0]
        ly = kernel_len[1]
        h[0:lx,0:ly] = 1/(lx*ly)
        c =  np.ceil((np.array([ly,lx])-1)/2).astype("int64")
    if type_blur=='gaussian' and var != None :
        [x,y] = torch.meshgrid(torch.arange(-ny/2,ny/2),torch.arange(-nx/2,nx/2)).to(device)
        h = torch.exp(-(x**2+y**2)/(2*var))
        h = h/torch.sum(h)
        c = np.ceil(np.array([nx,ny])/2).astype("int64") 

    H_FFT = torch.fft.fft2(torch.roll(h, shifts = (-c[0],-c[1]), dims=(0,1)))
    HC_FFT = torch.conj(H_FFT)

    # A forward operator
    A = lambda x: torch.fft.ifft2(torch.multiply(H_FFT,torch.fft.fft2(x))).real
    # A backward operator
    AT = lambda x: torch.fft.ifft2(torch.multiply(HC_FFT,torch.fft.fft2(x))).real

    AAT_norm = max_eigenval(A, AT, nx, 1e-4, int(1e4), 0, device)

    return A, AT, AAT_norm