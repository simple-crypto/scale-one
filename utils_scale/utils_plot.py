import matplotlib.pyplot as plt
import numpy as np
from utils_scale import utils_ta

def close_all():
    plt.close('all')

def is_labels_in_figs(fig):
    labels_in_fig = True
    for ax in fig.axes:
        if not ax.get_legend_handles_labels() == ([], []) : break
    else:
        labels_in_fig = False
    return labels_in_fig

def plot_traces(traces, color=None, hfig=None, hold=False, label=None, **kwargs):
    """
    Simple plot a bunch of power traces. 

    traces: power traces as ndarray of shape (ntraces, nsamples)
    """
    # Create the figure and plot the loaded traces
    if hfig is None:
        f = plt.figure()
        ax = f.add_subplot(1,1,1)
    else:
        (f, ax) = hfig
    ax.plot(traces.T,color=color,label=label, **kwargs)
    if not hold:
        ax.set_xlabel("Time index")
        ax.set_ylabel("Power")
        f.tight_layout()
        if is_labels_in_figs(f):
            plt.legend()
        plt.show()
    else:
        return (f, ax)

def plot_metric(trace, metric, metric_label=None):
    """
    Simple figure, with two subplot
    - an examplary trace
    - the metric to plot

    traces: power traces as ndarray of shape (ntrace, nsamples)
    metric: power traces as ndarray of shape (ntraces, nsamples)
    """
    # Create the figure and plot the loaded traces
    f = plt.figure()
    ax0 = f.add_subplot(2,1,1)
    ax1 = f.add_subplot(2,1,2)
    ax0.plot(trace)
    ax1.plot(metric,color="black")
    ax0.set_ylabel("Power")
    ax1.set_ylabel(metric_label)
    ax1.set_xlabel("Time index")
    f.tight_layout()
    plt.show()

def display_correlation(correlation, example_trace, index):
    # Plot
    f = plt.figure()
    ax_traces = f.add_subplot(2,1,1)
    ax_corr = f.add_subplot(2,1,2)
    ax_traces.plot(example_trace)
    ax_corr.plot(correlation)
    ax_traces.set_ylabel("Power")
    ax_corr.set_ylabel(r'$r_{xy}$')
    ax_corr.set_xlabel("Time samples")
    ax_traces.set_title(r'Correlation for $y_{{{}}}=\text{{Sbox}}[p_{{{}}} \oplus k_{{{}}}]$'.format(index, index, index))
    print("Maximum of correlation ({:.03f}) spotted at time sample index {}.".format(np.max(correlation),np.argmax(correlation)))



def display_two_correlations(corr_noisy, corr_origin, trs_noisy, trs_origin, key, bindex, title_other, ntraces=10, title_ref="Origin"):
    f = plt.figure(figsize=(10,8))
    axe_tr_origin = f.add_subplot(2,2,1)
    axe_tr_noisy = f.add_subplot(2,2,2)
    axe_corr_origin = f.add_subplot(2,2,3)
    axe_corr_noisy = f.add_subplot(2,2,4)

    axe_tr_origin.plot(trs_origin[:ntraces].T)
    axe_tr_noisy.plot(trs_noisy[:ntraces].T)

    axe_corr_origin.plot(corr_origin[bindex,key[bindex]])
    axe_corr_noisy.plot(corr_noisy[bindex,key[bindex]])

    axe_tr_origin.set_title(title_ref)
    axe_tr_noisy.set_title(title_other)
    
    for a in [axe_tr_origin, axe_tr_noisy]:
        a.set_ylabel("Power")

    axe_corr_origin.set_ylabel(r"$\rho(SB_{{{}}}] ; L)$".format(bindex))
    axe_corr_noisy.set_ylabel(r"$\rho(SB_{{{}}}] ; L)$".format(bindex))

    f.tight_layout()
    plt.show()

def display_noisy_correlation(corr_noisy, corr_origin, trs_noisy, trs_origin, key, bindex, std, ntraces=10):
    otitle =r"Noisy ($\sigma={}$)".format(std)
    display_two_correlations(corr_noisy, corr_origin, trs_noisy, trs_origin, key, bindex, otitle, ntraces=10)
    

def display_clipped_correlation(corr_noisy, corr_origin, trs_noisy, trs_origin, key, bindex, clip, ntraces=10):
    otitle=f"Clipped {clip}mV"
    display_two_correlations(corr_noisy, corr_origin, trs_noisy, trs_origin, key, bindex, otitle, ntraces=10)

def display_misaligned_correlation(corr_noisy, corr_origin, trs_noisy, trs_origin, key, bindex, std, ntraces=10):
    otitle = f"Misaligned (std: {std})"
    display_two_correlations(corr_noisy, corr_origin, trs_noisy, trs_origin, key, bindex, otitle, ntraces=10)

def display_HW_emp_corr(tr, corrhw, corremp):
    f = plt.figure(figsize=(8,5))
    axes = [f.add_subplot(3,1,i+1) for i in range(3)]
    axes[0].plot(tr)
    axes[1].plot(corrhw)
    axes[2].plot(corremp)
    axes[0].set_ylabel("Power")
    axes[2].set_xlabel("Time")
    axes[1].set_title("HW")
    axes[1].set_ylabel("Correlation")
    axes[2].set_title("Empirical")
    axes[2].set_ylabel("Correlation")
    f.tight_layout()
    plt.show()

def plot_temporal_average(traces_set, labels, plot_mean=False):
    assert len(traces_set)==len(labels)
    assert len(traces_set)==2
    f = plt.figure()
    ax = f.add_subplot(1,1,1)
    gmean = []
    for i, t in enumerate(traces_set):
        temp_mean = np.mean(t,axis=1)
        ax.plot(temp_mean, label=labels[i])
        if plot_mean:
            gmean += [np.mean(temp_mean)]
        ax.set_ylabel("Temporal mean")
        ax.set_xlabel("Trace index")
    for i, gm in enumerate(gmean):
        style = "solid" if i==0 else "dotted"
        ax.plot(len(temp_mean)*[gm],color="black",linestyle=style)
    plt.legend()
    plt.show()

def plot_empirical_vs_model_dist(ds, classes, pois, models, bindex, vclass):
    # Get the data use to build the model
    raw = ds['traces'][classes[:,bindex]==vclass,pois[bindex,0]]
    xdist = np.linspace(min(raw), max(raw), 1000)
    ydist = utils_ta.gaussian_pdf(xdist, models[0][bindex,vclass], models[1][bindex,vclass])
    
    f = plt.figure()
    ax1 = f.add_subplot(1,2,1)
    ax1.hist(raw, bins='auto',density=True, label="Empirical PDF")
    ax1.plot(xdist, ydist, color='red', label="Model PDF")
    ax1.set_title("Histogram vs theoritical model")
    ax1.set_ylabel("Density")
    ax1.set_xlabel("Leakage")
    
    ax2 = f.add_subplot(1,2,2)
    ax2.set_title("Model vs empirical")
    rvs = np.random.normal(loc=models[0][bindex,vclass], scale=models[1][bindex,vclass],size=5*len(raw))
    ax2.violinplot([raw, rvs])
    ax2.set_ylabel("Leakage")
    plt.tight_layout()

    ax1.legend()
    plt.show()

def plot_empirical_vs_model_dist_several(ds, classes, pois, models, bindex, vclasses, pdfcolor=False):
    f = plt.figure(figsize=(8,5))
    ax1 = f.add_subplot(1,2,1)
    ax1.set_title("Histogram vs theoritical model")
    ax1.set_ylabel("Density")
    ax1.set_xlabel("Leakage")

    # Get the data use to build the model
    for vclass in vclasses:
        mstd = models[1][bindex,vclass]
        raw = ds['traces'][classes[:,bindex]==vclass,pois[bindex,0]]
        xdist = np.linspace(min(raw)-2*mstd, max(raw)+2*mstd, 1000)
        ydist = utils_ta.gaussian_pdf(xdist, models[0][bindex,vclass], models[1][bindex,vclass])
        
        la = ax1.hist(raw, bins='auto',density=True, label=f"{vclass}")
        if pdfcolor:
            col = la[-1][-1].get_facecolor()
        else:
            col = "black"
        ax1.plot(xdist, ydist, color=col, )
    
    ax1.legend()
    plt.tight_layout()
