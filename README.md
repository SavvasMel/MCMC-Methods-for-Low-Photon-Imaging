# MCMC for Low Photon Imaging

**Abstract:** In this work, we present a  new  and  highly  efficient  Markov  chain  Monte  Carlo  (MCMC)  methodology to perform Bayesian inference in low-photon imaging problems, with particular attention given to situations involving observation noise processes that deviate significantly from Gaussian noise, such as binomial, geometric, and low-intensity Poisson noise. These problems are challenging for many reasons.  From an inferential viewpoint, low-photon numbers lead to severe identifiability issues, poor stability, and high uncertainty about the solution.  Moreover, low-photon models often exhibit poor regularity  properties  that  make  efficient  Bayesian  computation  difficult,  e.g.,  hard  nonnegativity constraints, nonsmooth priors, and log-likelihood terms with exploding gradients.

**We address** these difficulties by proposing an MCMC methodology based on a reflected and regularized Langevin SDE, which is shown to bewell-posed and exponentially ergodic under mild and easily verifiable conditions. For more information, you can find our published paper here https://epubs.siam.org/doi/10.1137/22M1502240. 

**In this repository,** you can find Pytorch codes of our suggested reflected proximal Langevin MCMC algorithms to perform Bayesian computationin low-photon imaging problems. The proposed approach is demonstrated with a range of experiments related to image deblurring, denoising, and inpainting under binomial, geometric, and Poisson noise.

## Examples

### Poisson Deconvolution 

Below, we present an experiment under severe Poisson noise which leads to low photon count in the data. The first reconstruction is the MMSE calculated by our suggested algorithm **R-SKROCK**. The second reconstuction is the MAP calculated by PIDAL [1], an ADMM algorithm adapted for Poisson noise. We used TV convex regularization for these recontructions (however data-driven convex regularizations can also be considered).

|<img src="https://github.com/SavvasMel/MCMC-Methods-for-Low-Photon-Imaging/assets/79579567/038700ad-03cb-406c-a36d-6e82b44cfd92" width="256" height="256">|<img src="https://github.com/SavvasMel/MCMC-Methods-for-Low-Photon-Imaging/assets/79579567/163c8db2-3944-4828-8f94-d6c656765005" width="256" height="256" />|<img src="https://github.com/SavvasMel/MCMC-Methods-for-Low-Photon-Imaging/assets/79579567/264adab3-20c7-4a56-afd6-a2983cc06636" width="256" height="260">|<img src="https://github.com/SavvasMel/MCMC-Methods-for-Low-Photon-Imaging/assets/79579567/f80468d3-14a4-4206-8a09-d36fd4c3d0c2" width="256" height="256">|
|:-:|:-:|:-:|:-:|
|Ground truth|Noisy image (Poisson)|MAP (PSNR 19.21 dB)|MMSE (PSNR 20.53 dB)|

The use of **MCMC methods** allows for **uncertainty quantification** (not usually possible when optimization methods or end-to-end networks are used). Below, we present uncertainty visualisation plots presenting the marginal standard deviation of pixels at different resolutions. For more information check our paper.

![st_deviation_SKROCK](https://github.com/SavvasMel/MCMC-Methods-for-Low-Photon-Imaging/assets/79579567/84066dd7-4f1a-4f79-9347-99126ef74100)

Below, we illustrate the convergence speed of the different MCMC methods considered in our paper. More precisely, we plot the evolution of the NRMSE estimation for the MMSE solutions as a function of the number of gradient evaluations (in log-scale). We see that R-SKROCK has the highest convergence speed followed by SPA [2].

<p align="center">
<img src="https://github.com/SavvasMel/MCMC-Methods-for-Low-Photon-Imaging/assets/79579567/22877ed8-fbc2-400a-b22c-b8a15d2628b6" width="256" height="256">
</p>

<!---
![data_PIDAL_miv_1_b_0 01_mu_non_adaptive_5 65](https://github.com/SavvasMel/MCMC-Methods-for-Low-Photon-Imaging/assets/79579567/f80468d3-14a4-4206-8a09-d36fd4c3d0c2)
![mean_pois_skrock](https://github.com/SavvasMel/MCMC-Methods-for-Low-Photon-Imaging/assets/79579567/264adab3-20c7-4a56-afd6-a2983cc06636)
-->
<!---
<figure class="image" style="text-align:center">
  <img src="https://github.com/SavvasMel/MCMC-Methods-for-Low-Photon-Imaging/assets/79579567/038700ad-03cb-406c-a36d-6e82b44cfd92" width="256" height="256">
  <figcaption>Title</figcaption>
</figure>
-->
  
<!---
<img src="https://github.com/SavvasMel/MCMC-Methods-for-Low-Photon-Imaging/assets/79579567/038700ad-03cb-406c-a36d-6e82b44cfd92" width="256" height="256" />
<img src="https://github.com/SavvasMel/MCMC-Methods-for-Low-Photon-Imaging/assets/79579567/163c8db2-3944-4828-8f94-d6c656765005" width="256" height="256" />
-->

<!---
![ground_truth](https://github.com/SavvasMel/MCMC-Methods-for-Low-Photon-Imaging/assets/79579567/038700ad-03cb-406c-a36d-6e82b44cfd92) 
![ground_noisy](https://github.com/SavvasMel/MCMC-Methods-for-Low-Photon-Imaging/assets/79579567/163c8db2-3944-4828-8f94-d6c656765005)
-->

## References

[1] M. A. T. Figueiredo and J. M. Bioucas-Dias, Restoration of Poissonian Images using Alternating Direction Optimization, IEEE Transactions on Image Processing, 19 (2010), pp. 3133–3145.

[2] M. Vono, N. Dobigeon, and P. Chainais, Bayesian image restoration under Poisson noise and log-concave prior, in ICASSP 2019 - 2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2019, pp. 1712–1716.
