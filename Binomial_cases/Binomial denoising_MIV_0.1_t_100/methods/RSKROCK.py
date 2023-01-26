"""
Reflected SKROCK SAMPLING METHOD                        

This function samples the distribution \pi(x) = exp(-F(x)-G(x)) under non-negativity
constraints thanks to a proximal MCMC algorithm called Reflected SKROCK (Reflected SKROCK)(see 
"Efficient Bayesian Computation for low-photon imaging problems", Savvas Melidonis, Paul Dobson, 
Yoann Altmann, Marcelo Pereyra, Konstantinos C. Zygalakis, arXiv, June 2022 and "Accelerating 
Proximal Markov Chain Monte Carlo by Using an Explicit Stabilized Method", Marcelo Pereyra, 
Luis Vargas Mieles, and Konstantinos C. Zygalakis, SIAM Journal on Imaging Sciences, 2020).

    INPUTS:
        X: current MCMC iterate (2D-array)
        Lipschitz: user-defined lipschitz constant of the model
        nStages: the number of internal stages of the SK-ROCK iterations
        dt_perc: the fraction of the max. stepsize to be used
        grad_Phi: function that computes the gradient of the potential F(x)+G(x)
        
    OUTPUTS:
        X_new: new value for X (2D-array).
        min_value: minimum value of the new MCMC iterate (2D-array) before reflection

Based on https://github.com/luisvargasmieles/2020-SIIMS-AcceleratingMCMCmethods (Luis Vargas Mieles)
Edited: Savvas Melidonis
"""

import torch
import numpy as np
import math

def RSKROCK(X: torch.Tensor,Lipschitz_U,nStages: int,eta_sk,dt_perc,gradU):

    # SK-ROCK parameters

    # First kind Chebyshev function

    T_s = lambda s,x: np.cosh(s*np.arccosh(x))

    # First derivative Chebyshev polynomial first kind

    T_prime_s = lambda s,x: s*np.sinh(s*np.arccosh(x))/np.sqrt(x**2 -1)

    # computing SK-ROCK stepsize given a number of stages

    # and parameters needed in the algorithm

    denNStag=(2-(4/3)*eta_sk)

    rhoSKROCK = ((nStages - 0.5)**2) * denNStag - 1.5 # stiffness ratio

    dtSKROCK = dt_perc*rhoSKROCK/Lipschitz_U # step-size

    w0=1 + eta_sk/(nStages**2) # parameter \omega_0

    w1=T_s(nStages,w0)/T_prime_s(nStages,w0) # parameter \omega_1

    mu1 = w1/w0 # parameter \mu_1

    nu1=nStages*w1/2 # parameter \nu_1

    kappa1=nStages*(w1/w0) # parameter \kappa_1

    # Sampling the variable X (SKROCK)

    Q=math.sqrt(2*dtSKROCK)*torch.randn_like(X) # diffusion term

    # SKROCK

    # SKROCK first internal iteration (s=1)

    XtsMinus2 = X.clone()

    Xts= torch.abs(X.clone() - mu1*dtSKROCK*gradU(torch.abs(X + nu1*Q)) + kappa1*Q)

    for js in range(2,nStages+1): # s=2,...,nStages SK-ROCK internal iterations

        XprevSMinus2 = Xts.clone()

        mu=2*w1*T_s(js-1,w0)/T_s(js,w0) # parameter \mu_js

        nu=2*w0*T_s(js-1,w0)/T_s(js,w0) # parameter \nu_js

        kappa=1-nu # parameter \kappa_js

        Xts= -mu*dtSKROCK*gradU(Xts) + nu*Xts + kappa*XtsMinus2

        if js==nStages:

            min_value = torch.min(Xts)

        Xts = torch.abs(Xts)

        XtsMinus2=XprevSMinus2

    return Xts, min_value # new sample produced by the SK-ROCK algorithm