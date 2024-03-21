'''
This code estimates the regularization parameter theta for the experiments
in [1]. We estimate theta by maximising the marginal likelihood p(y|theta) by adapting
the code presented in [2] (we make use of the introduced Reflected SDEs and respective
discretization in [1] instead of the usual Langevin SDEs).

The original code can be found in https://github.com/anafvidal/research-code.

[1] Savvas Melidonis, Paul Dobson, Yoann Altmann, Marcelo Pereyra, Konstantinos C. Zygalakis
Efficient Bayesian Computation for low-photon imaging problems.
arXiv preprint https://arxiv.org/abs/2206.05350, June 2022

[2] A. F. Vidal, V. De Bortoli, M. Pereyra, and A. Durmus (2020). 
Maximum Likelihood Estimation of Regularization Parameters in High-Dimensional Inverse Problems: 
An Empirical Bayesian Approach Part I: Methodology and Experiments. 
SIAM Journal on Imaging Sciences, 13(4), 1945-1989.

'''

import numpy as np
import torch
import timeit
import math

#%% SAPG algorithm function

def SAPG(y,X_init,gamma,lambda_prox, \
                   th_init,min_th,max_th,
                   d_scale,d_exp,
                   gradF,gradG,logPi,g,\
                   warmupSteps,burnIn,total_iter,device):
    
    
    # Stop criteria (relative change tolerance)
    
    stopTol=1e-4
    
    # Image dimension
    nx,ny=[X_init.shape[0],X_init.shape[1]]
    dimX= nx*ny

    #%% MYULA Warm-up
    
    X_wu = X_init.to(device).detach().clone()
    
    #Run MYULA sampler with fix theta to warm up the markov chain
    
    fix_theta = th_init
    logPiTrace_WU = torch.zeros(warmupSteps)
    
    
    gradGX_wu = gradG(X_wu,lambda_prox*fix_theta).to(device).detach().clone()
    print('Running Warm up     \n')
    
    for k in range(1,warmupSteps):
        
        X_wu =  torch.abs(X_wu - gamma*gradGX_wu - gamma*gradF(X_wu) + \
                torch.sqrt(2*gamma)*torch.randn_like(X_wu))
                
        gradGX_wu = gradG(X_wu,lambda_prox*fix_theta)
    
        logPiTrace_WU[k] = logPi(X_wu,fix_theta)
    
    
    #%% Run SAPG algorithm 1 to estimate theta
    
    theta_trace = torch.zeros(total_iter)
    theta_trace[0] = th_init
    
    # We work on a logarithmic scale, so we define an axiliary variable 
    #eta such that theta=exp{eta}. 
    
    eta_init = math.log(th_init)
    min_eta = math.log(min_th)
    max_eta = math.log(max_th)
    
    # delta(i) steps for SAPG algorithm 
    delta = lambda i: d_scale*( (i**(-d_exp)) / dimX )

    eta_trace = torch.zeros(total_iter)
    eta_trace[0] = eta_init
    
    
    logPiTraceX = torch.zeros(total_iter)  # to monitor convergence
    gTraceX = torch.zeros(total_iter)          # to monitor how the regularisation function evolves

    X = X_wu.to(device).detach().clone()    # start MYULA markov chain from last sample after warmup
    
    start = timeit.default_timer()

    print('\nRunning SAPG algorithm     \n')

    for k in range(total_iter): 
    
        # Number of samples
        
        m=5000
        
        gTraceX_mcmc=torch.zeros(m)
        
        # Sample from posterior with MYULA:
        
        for ii in range(0,m):
                
            gradGX = gradG(X,lambda_prox*theta_trace[k])
            X =  torch.abs(X - gamma*gradGX -gamma*gradF(X) + math.sqrt(2*gamma)*torch.randn_like(X))

            gTraceX_mcmc[ii]= g(X)

        #Save current state to monitor convergence

        logPiTraceX[k] = logPi(X,theta_trace[k])
        gTraceX[k] = gTraceX_mcmc[-1]

        #Update theta   
        etak = eta_trace[k] + (delta(k+1)/m)*torch.sum(dimX/theta_trace[k]-gTraceX_mcmc)*torch.exp(eta_trace[k]) 

        eta_trace[k+1] = min(max(etak,min_eta),max_eta)
        theta_trace[k+1]=torch.exp(eta_trace[k+1])
        print('Current theta in trace: ', float(theta_trace[k+1]))

       
        if (k+1) % round(total_iter/100)==0:
            print("\n",round((k+1)/(total_iter)*100))

        if k>burnIn:

            #Check stop criteria. If relative error is smaller than op.stopTol stop
            relErrTh1=torch.abs(torch.exp(torch.mean(eta_trace[burnIn:(k+1)]))-
            torch.exp(torch.mean(eta_trace[burnIn:k]))) /torch.exp(torch.mean(eta_trace[burnIn:k]))

            if (relErrTh1<stopTol):
                break
           

    end = timeit.default_timer()

    last_samp = k
    
    logPiTraceX = logPiTraceX[:last_samp+1].detach().cpu().numpy()
    gTraceX = gTraceX[:last_samp+1].detach().cpu().numpy()
    
    theta_EB = torch.exp(torch.mean(eta_trace[burnIn:last_samp+1])).detach().cpu().numpy()
    
    last_theta = theta_trace[last_samp].detach().cpu().numpy()
    
    thetas = theta_trace[:last_samp+1].detach().cpu().numpy()
    
    print("Elapsed time of SAPG phase: ",(end-start)/60)
    
    return theta_EB,last_theta,thetas, \
            logPiTraceX,gTraceX,last_samp

