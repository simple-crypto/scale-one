import numpy as np
from utils_scale import utils_files, utils_aes
from utils_scale.test_scale import ref_snr_scalib
import matplotlib.pyplot as plt
from scalib.modeling import Lda, LdaAcc

def gaussian_pdf(xs, mean, std):
    coef = 1/(std*np.sqrt((2*np.pi)))
    return coef * np.exp(-(xs-mean)**2 / (2*(std**2)))

def POI_selection_SNR(traces, classes, nclasses):
    snrs = ref_snr_scalib(traces, classes, nclasses)
    return np.flip(np.argsort(snrs,axis=1),axis=1)

def univariate_gaussian_models(traces, classes, pois):
    # Allocate return value
    us = np.zeros([classes.shape[1],256])
    ss = np.zeros([classes.shape[1],256])
    
    # Iterate over every variables
    for si in range(classes.shape[1]):
        # Allocate memory squared traces accumulator (used for pooled variance computation)
        sum_sq_center_traces = 0

        # Iterate over all the possible bytes values
        for b in range(256):
            # Compute class mean
            us[si,b] = np.mean(traces[classes[:,si] == b, pois[si]])
            
            # Compute the centered traces
            ctraces = traces[classes[:,si] == b, pois[si]] - us[si,b]
            
            # Accumulate the squared centered traces
            sum_sq_center_traces += np.sum(np.square(ctraces),axis=0)
            
        # Second, compute the pooled variance
        ss[si,:] = np.sqrt((sum_sq_center_traces / (traces.shape[0]-1)))
    # Return
    return (us, ss)
    ###ANSWER_STOP

def log2Pr_class(traces, models):
    # Allocate log-probabilities matrix
    probas = np.zeros([models[0].shape[0],traces.shape[0],models[0].shape[1]])
    for vi, (vtrs, vus, vss) in enumerate(zip(traces.T, models[0], models[1])):
        # Compute the raw density values
        for ci in range(models[0].shape[1]):
            probas[vi, :, ci] = gaussian_pdf(vtrs, vus[ci], vss[ci])
        # Normalize
        probas[vi] = probas[vi] / np.sum(probas[vi],axis=1)[:,np.newaxis]
    return np.log2(probas)

def maximum_likelihood(pts, log2pr_sb):
    lprobas = np.zeros([pts.shape[1], log2pr_sb.shape[2]])
    for vi, (vpt, logpr) in enumerate(zip(pts.T, log2pr_sb)):
        for ki in range(log2pr_sb.shape[2]):
            ist = utils_aes.Sbox[vpt ^ ki]
            lprobas[vi, ki] = np.sum(logpr[np.arange(pts.shape[0]), ist])
    # Normalization
    # Scaling to avoid numerical instabilities
    max_log = np.max(lprobas,axis=1)[:,np.newaxis]
    lprobas = lprobas - max_log 
    # Compute the sum of probas for each key guess
    sum_probas = np.sum(np.exp2(lprobas),axis=1)[:,np.newaxis]
    # Compute the normalized probabilities
    return np.exp2(lprobas - np.log2(sum_probas))

def univariate_TA(traces, pts, pois, models):
    traces_poi = traces[:, pois]
    log2pr_sb = log2Pr_class(traces_poi, models)
    lprobas = maximum_likelihood(pts, log2pr_sb)
    return lprobas

def explore_TA_univariate(dspath_train, dspaths_valid, qp, qas, clean_dataset=False, fn_prof=None):
    ### TRAINING phase
    # First train using the training dataset 
    ds = utils_files.load_dataset(dspath_train, seed_shuffle=0, remove_first=clean_dataset)
    # Fetch all dataset if no training complexity provided
    if qp is None:
        qpu = ds['traces'].shape[0]
    else:
        qpu = qp
    # Compute intermediate states 
    classes = utils_aes.Sbox[ds["pts"][:qpu] ^ ds["ks"][:qpu]]
    # Compute the POIs based on your function
    pois = POI_selection_SNR(ds['traces'][:qpu], classes, 256)

    # Compute the models
    if fn_prof is None:
        models = univariate_gaussian_models(ds['traces'][:qpu], classes[:qpu], pois[:,0])
    else:
        models = fn_prof(ds['traces'][:qpu], classes[:qpu], pois[:,0])
    
    ### ONLINE Phase
    # Allocate memory for the results
    corrprobs = np.zeros([
        len(dspaths_valid), 
        len(qas),
        ds['pts'].shape[1],
        ])
    allprobs = np.zeros([
        len(dspaths_valid), 
        len(qas),
        ds['pts'].shape[1],
        256
        ])
    correct_kbytes = np.zeros([len(dspaths_valid),ds["pts"].shape[1]],dtype=np.uint8)

    for dsi, dsp in enumerate(dspaths_valid):
        # Load the dataset
        ds = utils_files.load_dataset(dsp, seed_shuffle=0, remove_first=clean_dataset)
        correct_kbytes[dsi] = ds['ks'][0]
        for qavi, q_a in enumerate(qas):
            # Performs the template
            probas = univariate_TA(ds['traces'][:q_a], ds['pts'][:q_a], np.array([pois[:,0]]), models)
            allprobs[dsi, qavi] = probas.copy()
            for vi, kc in enumerate(correct_kbytes[dsi]):
                corrprobs[dsi, qavi, vi] = probas[vi,kc]

    return (allprobs, corrprobs, qpu, qas, correct_kbytes)

def multivariate_gaussian_models(traces, classes, pois, ndim):
    lda_acc = LdaAcc(nc=256, pois=pois.tolist())
    lda_acc.fit_u(np.round(traces).astype(np.int16), classes.astype(np.uint16))
    return Lda(lda_acc, p=ndim)

def multivariate_LDA_TA(traces, pts, models):
    pr_sb = models.predict_proba(np.round(traces).astype(np.int16))
    # Finally, perform the ML
    lprobas = maximum_likelihood(pts, np.log2(pr_sb))
    # return lprobas
    return lprobas

def explore_TA_multivariate(dspath_train, dspaths_valid, qp, qas, npois, ndim, pois=None, fn_prof=None, fn_TA=None, clean_dataset=False):
    # TODO: 
   ### TRAINING phase
    # First train using the training dataset 
    ds = utils_files.load_dataset(dspath_train, seed_shuffle=0, remove_first=clean_dataset)
    # Fetch all dataset if no training complexity provided
    if qp is None:
        qpu = ds['traces'].shape[0]
    else:
        qpu = qp
    # Compute intermediate states 
    classes = utils_aes.Sbox[ds["pts"][:qpu] ^ ds["ks"][:qpu]]
    # Compute the POIs based on your function
    if pois is None:
        poisu = POI_selection_SNR(ds['traces'][:qpu], classes, 256)[:,:npois]
    else:
        poisu = pois
    # Compute the models
    if fn_prof is None:
        models = multivariate_gaussian_models(ds['traces'][:qpu], classes[:qpu], poisu, ndim)
    else:
        models = fn_prof(ds['traces'][:qpu], classes[:qpu], poisu, ndim)

    
    ### ONLINE Phase
    # Allocate memory for the results
    corrprobs = np.zeros([
        len(dspaths_valid), 
        len(qas),
        ds['pts'].shape[1],
        ])
    allprobs = np.zeros([
        len(dspaths_valid), 
        len(qas),
        ds['pts'].shape[1],
        256
        ])
    correct_kbytes = np.zeros([len(dspaths_valid),ds["pts"].shape[1]],dtype=np.uint8)

    for dsi, dsp in enumerate(dspaths_valid):
        # Load the dataset
        ds = utils_files.load_dataset(dsp, seed_shuffle=0, remove_first=clean_dataset)
        correct_kbytes[dsi] = ds['ks'][0]
        for qavi, q_a in enumerate(qas):
            # Performs the template
            if fn_TA is None:
                probas = multivariate_LDA_TA(ds['traces'][:q_a], ds['pts'][:q_a], models )
            else:
                probas = fn_TA(ds['traces'][:q_a], ds['pts'][:q_a], models )
                
            allprobs[dsi, qavi, :, :] = probas
            for vi, kc in enumerate(correct_kbytes[dsi]):
                corrprobs[dsi, qavi, vi] = probas[vi,kc]

    return (allprobs, corrprobs, qpu, qas, correct_kbytes)


MY_COLORS = [
         "xkcd:blue",
         "xkcd:green",
         "xkcd:red",
         "xkcd:orange",
         "xkcd:pink",
         ]

def tipping_point(prs, correct):
    runner = prs.shape[0]-1
    while np.argmax(prs[runner])==correct:
        if runner==0:
            break
        else:
            runner -= 1
    return runner+1

def display_explore_TA_univariate_result(res, use_colors=False):
    # Unpack
    (allprobs, corrprobs, qp, qas, correct_kbytes) = res

    # Print some stats for the first results
    for vi in range(allprobs.shape[2]):
        tp = tipping_point(allprobs[0,:,vi,:],correct_kbytes[0,vi])
        print(f'Byte {vi}: {tp} traces required')

    # Plot the res
    scale=0.6
    ax_sx_inch = 5*scale
    ax_sy_inch = 3*scale
    figsize=(4*ax_sx_inch, 4*ax_sy_inch)
    f = plt.figure(figsize=figsize)

    amval = corrprobs.shape[0] 
    nc = allprobs.shape[3]
    axes=[]
    print(allprobs.shape)
    for i in range(corrprobs.shape[2]):
        axes.append(f.add_subplot(4,4,i+1))
        # Plot the wrong guess only for the first to easu visualization
        for dsvi in range(1):
            for b in range(nc): 
                axes[i].plot(qas, allprobs[dsvi, :, i, b], color="xkcd:light grey")
        # Plot the valid guess
        for dsvi in range(amval):
            udsvi = amval-1-dsvi
            if use_colors:
                color_c = MY_COLORS[udsvi % len(MY_COLORS)]
                txt_label = "Set {}".format(udsvi)
            else:
                if udsvi==0:
                    color_c = "xkcd:red"
                else:
                    color_c = "black"
                txt_label = None
            plt.plot(qas, corrprobs[udsvi, :, i], color=color_c, label=txt_label)
        axes[i].set_xlabel("attack data complexity")
        axes[i].set_ylabel(r'Pr($k_{}^* | \boldsymbol{{l}}$)'.format(i))
        axes[i].set_title("Byte {}".format(i))      
        if use_colors:
            axes[i].legend()
    plt.show()


        

