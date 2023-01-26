"""

This script is used for multiresolution pixel-wise standard deviation.

"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat,loadmat

from skimage.measure import block_reduce

#%% 

def downsampling_variance(data_METHOD_trace_str,METHOD_trace_str,output_str,method_str):

    ############################################################################
    ############################################################################

    # Load the data:
    #   data_METHOD_trace_str : it is the pathway(string) of my data.
    #   data_trace : it is an numpy array of dimension N x nx x ny where N is the number
    #                of MCMC simulations and nx x ny are the image dimensions.      

    MC_chain = loadmat(data_METHOD_trace_str).get(METHOD_trace_str)
    N_MC = data_trace.shape[0]
    nx = data_trace.shape[0]
    ny = data_trace.shape[1]

    # with open(data_METHOD_trace_str,'rb') as f:
    #     data_trace= joblib.load(data_METHOD_trace_str)

    ############################################################################
    ############################################################################

    # Reshaping the data so that instead of being of dimension Nx(d^2), it should
    # be Nxdxd.

    data_trace_array=data_trace[METHOD_trace_str].reshape([N_MC,nx,ny])

    del data_trace

    # Put the different scales that we are going to use later in a list. So, let' say
    # your image is 256x256. For scale = 1, you want to downscale your samples (which
    # have dimension 256x256) by 2*scale (i.e. your samples will have dimension 128x128). 
    # For scale = 2, your samples will have dimension 64x64.

    scale = [1,2,4,8]
    
    # An empty list to save the st. deviations.

    st_deviation_down= []

    # Choose scale by using loops.

    for i in scale:

        # Your data array is an array of N_MC x d x d. Now, if e.g. scale =1 we want
        # to create an array of N_MC x d/(2*scale) x d/(2*scale). So, instead of e.g.
        # 10000 X 256 X 256, it should be 10000 X 128 X 128 etc

        downsample_array= np.zeros([N_MC,int(nx/(i*2)),int(ny/(i*2))])

        # Now, it remains to choose how are we going to downsample! So, let's say your image
        # is 4x4 matrix and your scale = 1. This means a downsampling of 2*scale (=2) so your 
        # matrix should be 2x2. The idea is that you split your 4x4 matrix in 4 (disjoint!!) blocks of 4 pixels
        # each. For example if you start from the pixel with position (1,1) in the matrix you take
        # the pixels next to it in the horizontal, vertical and diagonal direction. Then you take the mean
        # of the pixels in this block and you have now 1 pixel. You do this for each of the 4 blocks and you
        # have your 2x2 matrix. You are doing the same for all the samples.

        for j in range(N_MC):

            # I am using the 'block_reduce' from skimage.measure for that. You need to give how you want
            # to downsample (e.g. by taking the mean of the blocks) and the block size mainly.

            downsample_array[j]= block_reduce(data_trace_array[j], block_size=(i*2,i*2), func=np.mean)

        # Now you have your downsampled data. You calculate the sample st.deviation as usual.

        meanSample_down= np.mean(downsample_array,0)
        second_moment_down= np.mean(downsample_array**2,0)
        
        # So the first element of the 'st_deviation_down' list will be the downsampled
        # st. deviation of scale = 1. The next element will be the downsampled
        # st. deviation of scale = 2 etc.

        st_deviation_down.append(np.sqrt(second_moment_down - meanSample_down**2))

    # Save the data.

    savemat(output_str, st_deviation_down)

    # with open(output_str, 'wb') as f:
    #     # Pickle the 'data' dictionary using the highest protocol available.
    #     pickle.dump(st_deviation_down, f)
    
    # Plot everything with colorbars.

    fig, ax = plt.subplots()
    im=ax.imshow(st_deviation_down[0],cmap="gray")
    ax.set_title(method_str)
    plt.colorbar(im)
    
    fig, ax = plt.subplots()
    im=ax.imshow(st_deviation_down[1],cmap="gray")
    ax.set_title(method_str)
    plt.colorbar(im)
    
    fig, ax = plt.subplots()
    im=ax.imshow(st_deviation_down[2],cmap="gray")
    ax.set_title(method_str)
    plt.colorbar(im)
    
    fig, ax = plt.subplots()
    im=ax.imshow(st_deviation_down[3],cmap="gray")
    ax.set_title(method_str)
    plt.colorbar(im)
    

######################################################################################
######################################################################################
######################################################################################

## MYULA

# This is what I use to call the data.

# Pathway
data_METHOD_trace_str= "../../poisson_mean_1_refMYULA/RMYULA_chain_pois.mat"

# The data array is a dictionary so I give the key (the name of the array).
METHOD_trace_str= "RMYULA_MC_chain"

# Name to save the st. deviations.

output_str= 'st_deviation_down_MYULA_pois.pickle'

# The name of the MCMC method (this is just for the plots title).
method_str= "Posterior St. Deviation (MYULA)"

downsampling_variance(data_METHOD_trace_str,METHOD_trace_str,output_str,method_str)

plt.show()
