import numpy as np
from utils_scale import test_scale

import tqdm

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, NoNorm
from matplotlib.ticker import LogFormatterSciNotation, LogLocator
from matplotlib.tri import Triangulation
from matplotlib.patches import Rectangle

def bridge_TA_res_probs_25_to_24(res25):
    (nds, nqas, nvars, nclasses) = res25.shape
    probs = np.zeros([nds, nvars, nclasses, nqas], dtype=res25.dtype)
    for dsi in range(nds):
        for nvi in range(nvars):
            for ci in range(nclasses):
                for qi in range(nqas):
                    probs[dsi, nvi, ci, qi] = res25[dsi, qi, nvi, ci]
    return probs


# CAUTION: crappy bridge between 2025 and 2024 edition ...
def display_rank_esti_full_key(uni_TA_results, multi_TA_results, byte_indexes, key_rank_approximation_scalib):
    # Unpack the results
    (uTA_probs, _, _, uTAqas, uTAkbytes) = uni_TA_results
    (mTA_probs, _, _, mTAqas, mTAkbytes) = multi_TA_results

    # Here,
    # *_probs of shape (n_ds_validation, n_qas, nvars, 256)
    # *qas of shape (nqas, )
    # *kbytes of shape (n_ds_validation, nvars)

    test_scale.display_rank_esti_full_key(
            (bridge_TA_res_probs_25_to_24(uTA_probs), uTAkbytes, np.array(list(uTAqas))),
            (bridge_TA_res_probs_25_to_24(mTA_probs), mTAkbytes, np.array(list(mTAqas))),
            byte_indexes,
            key_rank_approximation_scalib
            )

def explore_params(pi_LDA_multi_TA, ti_LDA_multi_TA, traces, labels, ntraces_pi, npois, ndims, qp=None):
    # Some size fetching
    ntraces, nvars = labels.shape
    l_npois = len(npois)
    l_ndims = len(ndims)

    # Allocate memory for results
    pis = np.zeros([nvars, l_npois, l_ndims])
    tis = np.zeros(pis.shape)

    # Amount of traces used for the training
    if qp is None:
        ntr_train = ntraces - ntraces_pi
    else:
        ntr_train = qp


    # First, we have to compute all the computations.
    with tqdm.tqdm(total=len(npois)*len(ndims), desc="progress") as pbar:
        for i_npois, e_npois in enumerate(npois):
            for i_ndims, e_ndims in enumerate(ndims):
                # PI
                wrap_pi_LDA_multi_TA = lambda a,b,c,d: pi_LDA_multi_TA(a, b, c, d, e_npois, e_ndims)
                results_pi_multi = test_scale.compute_pi_estimations(
                    traces[:ntr_train], 
                    labels[:ntr_train], 
                    traces[-ntraces_pi:], 
                    labels[-ntraces_pi:], 
                    wrap_pi_LDA_multi_TA, 
                    [ntr_train]
                )
                # TI
                wrap_ti_LDA_multi_TA = lambda a,b: ti_LDA_multi_TA(a, b, e_npois, e_ndims)            
                results_ti_multi = test_scale.compute_ti_estimations(
                    traces[:ntr_train], 
                    labels[:ntr_train], 
                    wrap_ti_LDA_multi_TA, 
                    [ntr_train]
                )
                # Store data
                pis[:,i_npois,i_ndims] = results_pi_multi['it'][:,0]
                tis[:,i_npois,i_ndims] = results_ti_multi['it'][:,0]
                pbar.update(1)
    # Return
    return (npois, ndims, pis, tis)
    
def make_single_heatmap(ax, npois, pndims, byte, pi, ti, cmap, norm, show_xaxis, show_yaxis):
    # Display the title
    axtitle = "Byte {}".format(byte)
    #ax.set_title(axtitle)
    
    m = pi.shape[1]
    n = pi.shape[0]
    x = np.arange(m + 1)
    y = np.arange(n + 1)
    xs, ys = np.meshgrid(x, y)
    squares = [
        (
            i + j * (m + 1),
            i + 1 + j * (m + 1),
            i + 1 + (j + 1) * (m + 1),
            i + (j + 1) * (m + 1),
        )
        for j in range(n)
        for i in range(m)
    ]
    tri_pi = [(bxby, bxty, txty) for (bxby, txby, txty, bxty) in squares]
    tri_ti = [(bxby, txty, txby) for (bxby, txby, txty, bxty) in squares]
    imgs = []
    for tri, z in [(tri_pi, pi), (tri_ti, ti)]:
        tri = Triangulation(xs.ravel(), ys.ravel(), tri)
        imgs.append(ax.tripcolor(tri, z.ravel(), cmap=cmap, norm=norm))
    
    # Draw square around max pi
    (max_y, max_x) = np.unravel_index(np.argmax(pi), pi.shape)
    ax.add_patch(Rectangle((max_x, max_y), 1, 1, edgecolor="r", facecolor="none"))
    
    # Axis configuration
    ax.invert_yaxis()
    ax.margins(0)
    if show_xaxis:
        xlabels = [str(e) for e in npois]
        ax.set_xticks(0.5 + np.arange(m), labels=xlabels, rotation=45)
        # ax.set_xlabel("POIs amount")
    else:
        ax.set_xticks([], labels=[])
    if show_yaxis:
        ylabels = [str(e) for e in pndims]
        ax.set_yticks(0.5 + np.arange(n), labels=ylabels)
        # ax.set_ylabel(r"$p$")
    else:
        ax.set_yticks([], labels=[])
    return imgs


def make_heatmap(res_explo):
    (npois, ndims, pi, ti) = res_explo

    # Create the colormap
    cmap_im = mpl.colormaps.get_cmap("viridis")
    cmap_im.set_bad(color="red")

    # Skip all-nan results.
    if np.isnan(pi).all() and np.isnan(ti).all():
        print('ok')
    # Recover bounds for valid data
    maxv = np.nanmax([pi, ti])
    minv = np.nanmin([pi, ti])
    minv = np.nanmax(
        [minv, maxv / 100]
    )  # Below some threshold, PI might as well just be 0.
    norm_im = LogNorm(vmin=minv, vmax=maxv)

    # Create figure
    f = plt.figure(figsize=(7, 5))
    axes = f.subplots(4, 4)

    # Some global configuration
    imgs = []

    # Enumerate over all the vaiables
    for byte in range(16):
        # Create the ax for the variable index
        ax_y = byte % 4
        ax_x = byte // 4
        # ax = f.add_subplot(spec[ax_y, ax_x])
        ax = axes[ax_y][ax_x]
        img = make_single_heatmap(
            ax,
            npois,
            ndims,
            byte,
            pi[byte],
            ti[byte],
            cmap_im,
            norm_im,
            show_xaxis=(byte % 4 == 3),
            show_yaxis=byte < 4,
        )
        # Append objects for post-processing
        imgs.append(img)

    f.supxlabel("Number of POIs")
    f.supylabel("$p$")

    # Colorbar
    cbar_axes = f.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar_axes.set_label(r"$\text{log}_{2}[PI]$")
    cbar = f.colorbar(
        imgs[0][0],
        cax=cbar_axes,
        ticks=LogLocator(base=2),
        format=LogFormatterSciNotation(base=2.0),
    )
    #cbar_axes.yaxis.minorticks_off()

    f.subplots_adjust(
        left=0.08, right=0.90, bottom=0.13, top=0.99, wspace=0.04, hspace=0.04
    )
