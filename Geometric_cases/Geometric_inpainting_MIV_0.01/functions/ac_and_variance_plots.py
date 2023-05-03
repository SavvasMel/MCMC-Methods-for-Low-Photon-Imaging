'''
This code generates autocorrelation and variance plots. The code assumes the existence of a MCMC chain of the
form N x nx x ny, where N are the number of MCMC samples and nx,ny are the dimensions of the image.

@author: Savvas Melidonis
'''

#%% Import necessary modules

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat
from skimage.measure import block_reduce
from statsmodels.graphics.tsaplots import plot_acf
import os
import joblib
import statsmodels.api as sm

#%% 

def ac_var_plots(MC_chain, lags, method_str, path, img_type):

    # Load the data:
    #   chain_location_str : location of the chain data. This should be a mat file.
    #   chain_name_str : We load the mat file as a dictionary, so we need a key that is linked to the actual chain data.
    #   MC_chain : it is an numpy array of dimension N x nx x ny where N is the number
    #                of MCMC simulations and nx, ny are the image dimensions.      

    # MC_chain = loadmat(chain_location_str).get(chain_name_str)
    N_MC = MC_chain.shape[0]
    nx = MC_chain.shape[1]
    ny = MC_chain.shape[2]

    if  MC_chain.shape[0]<lags:
        raise ValueError('ACF lags should be larger than the number of saved samples')
    
    # Put the different scales that we are going to use later in a list. So, let' say
    # your image is 256x256. For scale = 1, you want to downscale your samples (which
    # have dimension 256x256) by 2*scale (i.e. your samples will have dimension 128x128). 
    # For scale = 2, your samples will have dimension 64x64.

    scale = [1,2,4,8]

    # An empty list to save the st. deviations.

    st_deviation_down= []

    # Choose scale by using loops.

    for i in scale:

        # Your data array is an array of N_MC x nx x ny. Now, if e.g. scale = 1 we want
        # to create an array of N_MC x nx/(2*scale) x ny/(2*scale). So, instead of e.g.
        # 10000 X 256 X 256, it should be 10000 X 128 X 128 etc

        downsample_array= np.zeros([N_MC,int(nx/(i*2)),int(ny/(i*2))])

        # Now, it remains to choose the downsampling method! Supposing the image
        # is 4x4 matrix and the scale = 1. This means a downsampling of 2*scale (=2) so the 
        # matrix should be 2x2. The idea is that the 4x4 matrix is splitted in 4 (disjoint) blocks of 4 pixels
        # each. For example if we start from the pixel with position (1,1) in the matrix we take
        # the pixels next to it in the horizontal, vertical and diagonal direction. Then we calculate the mean
        # of the pixels in this block. We repeat for each of the 4 blocks and the desired 2x2 matrix is formulated. 
        # We repeat for all the samples.

        for j in range(N_MC):

            # Τhe 'block_reduce' from skimage.measure is helpful for the task.
                # INPUTS
                #   block_size: the size of blocks that we use to split the image before apply the downsampling.
                #   func: the downsampling method.

            downsample_array[j] = block_reduce(MC_chain[j], block_size=(i*2,i*2), func=np.mean)

        # Now you have your downsampled data. You calculate the sample st.deviation as usual.

        meanSample_down= np.mean(downsample_array,0)
        second_moment_down= np.mean(downsample_array**2,0)

        st_deviation_down.append(np.sqrt(second_moment_down - meanSample_down**2))

    matplotlib.rc('figure', figsize=(40, 8))
    matplotlib.rc('font', size=30)
    fig, ax = plt.subplots(1,4)
    fig.suptitle("Posterior (log-scaled) St. Deviation" + "-" + method_str, fontsize='large')
    im1 = ax[0].imshow(np.log(st_deviation_down[0]),cmap="gray")
    ax[0].get_xaxis().set_visible(False)
    ax[0].get_yaxis().set_visible(False)
    cbar = fig.colorbar(im1,ax=ax[0])
    im2=ax[1].imshow(np.log(st_deviation_down[1]),cmap="gray")
    ax[1].get_xaxis().set_visible(False)
    ax[1].get_yaxis().set_visible(False)
    cbar = fig.colorbar(im2,ax=ax[1])
    im3=ax[2].imshow(np.log(st_deviation_down[2]),cmap="gray")
    ax[2].get_xaxis().set_visible(False)
    ax[2].get_yaxis().set_visible(False)
    cbar = fig.colorbar(im3,ax=ax[2])
    im4=ax[3].imshow(np.log(st_deviation_down[3]),cmap="gray")
    ax[3].get_xaxis().set_visible(False)
    ax[3].get_yaxis().set_visible(False)
    cbar = fig.colorbar(im4,ax=ax[3])
    fig.tight_layout()
    plt.savefig(path + "/st_deviation_scales_" + method_str + "." + img_type,bbox_inches='tight',dpi=300)

    # Autocorrelation plots of the chain elements with the slowest, median and fastest variance.

    variance_array = np.var(MC_chain, axis=0)

    ind_min_variance = np.argmin(variance_array.ravel()) 
    chain_elem_min_variance = MC_chain.reshape(MC_chain.shape[0],-1)[:,ind_min_variance]

    ind_max_variance = np.argmax(variance_array.ravel()) 
    chain_elem_max_variance = MC_chain.reshape(MC_chain.shape[0],-1)[:,ind_max_variance]
    
    ind_median_variance = np.argsort(variance_array.ravel())[len(variance_array.ravel())//2]
    chain_elem_median_variance = MC_chain.reshape(MC_chain.shape[0],-1)[:,ind_median_variance]

    matplotlib.rc('figure', figsize=(15, 15))
    matplotlib.rc('font', size=30)
    fig,ax = plt.subplots()
    plot_acf(chain_elem_median_variance,ax=ax,label='Median-speed component',alpha=None,lags=100)
    plot_acf(chain_elem_max_variance,ax=ax,label='Slowest component',alpha=None,lags=100)
    plot_acf(chain_elem_min_variance,ax=ax,label='Fastest component',alpha=None,lags=100)
    handles, labels= ax.get_legend_handles_labels()
    handles=handles[1::2]
    labels =labels[1::2]
    ax.set_title(method_str)
    ax.set_ylabel("ACF")
    ax.set_xlabel("lags")
    ax.legend(handles=handles, labels=labels,loc='best',shadow=True, numpoints=1)
    plt.savefig(path + "/acr_" + method_str + "." + img_type, bbox_inches='tight',dpi=300)

    plt.show()
