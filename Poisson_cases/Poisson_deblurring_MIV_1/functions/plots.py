import numpy as np
import matplotlib
matplotlib.rc('font', size=20)
matplotlib.rc('font', family='serif')
matplotlib.rc('figure', figsize=(14, 8))
matplotlib.rc('lines', linewidth=2.5,linestyle="-.")
matplotlib.rc('lines', markersize=10)
matplotlib.rc('figure.subplot', hspace=.4)
matplotlib.rc('text' ,  usetex=False)
import matplotlib.pyplot as plt

def plots(z_true, y, post_meanvar, NRMSE_trace, logPiTrace, min_val_trace, path, img_type, alpha_trace = None):

    fig, ax = plt.subplots()
    plt.imshow(z_true.cpu().numpy(), cmap="gray")      
    plt.title("Ground truth")  
    plt.savefig(path + "/ground_truth.png", bbox_inches="tight", dpi=300)
    plt.close()

    fig, ax = plt.subplots()
    ax.set_title("Posterior mean")
    plt.imshow(post_meanvar.get_mean().cpu().numpy(), cmap="gray")
    plt.savefig( path + "/posterior_mean" + "." + img_type, bbox_inches="tight", dpi=300)
    plt.close()

    fig, ax = plt.subplots()
    im = ax.imshow(np.log(np.sqrt(post_meanvar.get_var().cpu().numpy())), cmap="gray")
    ax.set_title("Posterior variance")
    plt.colorbar(im)
    plt.savefig( path + "/log_st_dev" + "." + img_type, bbox_inches="tight", dpi=300)
    plt.close()

    fig, ax = plt.subplots()
    ax.plot(NRMSE_trace,linestyle="-")
    ax.set_xlabel("$Iterations$")
    ax.set_ylabel("$NRMSE$")
    plt.savefig( path + "/NRMSE" + "." + img_type, bbox_inches="tight", dpi=300)
    plt.close()

    fig, ax = plt.subplots()
    ax.plot(logPiTrace, linestyle="-")
    ax.set_xlabel("$Iterations$")
    ax.set_ylabel("$log(p(z|y))$")
    plt.savefig( path + "/logPitrace" + "." + img_type, bbox_inches="tight", dpi=300)
    plt.close()

    fig, ax = plt.subplots()
    ax.plot(min_val_trace,linestyle="-")
    ax.set_xlabel("$Iterations$")
    ax.set_ylabel("$min(X_{n})$")
    plt.savefig( path + "/minXtrace" + "." + img_type, bbox_inches="tight", dpi=300)
    plt.close()

    if type(alpha_trace) == np.dtype:
        fig, ax = plt.subplots()
        ax.plot(alpha_trace,linestyle="-")
        ax.set_xlabel("$Iterations$")
        ax.set_ylabel("$alpha\,\,(log-scale)$")
        plt.savefig( path + "/accept_rate_trace" + "." + img_type, bbox_inches="tight", dpi=300)
        plt.close()

