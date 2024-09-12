import numpy as np


def load_datasets_profiled_setting(fp_train, fp_validation):
    # Open the training file and load all the data fields from it. 
    with open(fp_train, 'rb') as f:
        coll = np.load(f, allow_pickle=True)
        training_traces = coll["traces"].astype(np.single)
        training_pts = coll["pts"]
        training_ks = coll["ks"]
    
    # Open the validation file and load all the data fields from it. 
    with open(fp_validation, 'rb') as f:
        coll = np.load(f, allow_pickle=True)
        validation_traces = coll["traces"].astype(np.single)
        validation_pts = coll["pts"]
        validation_ks = coll["ks"]

    # Return the data
    return training_traces, training_pts, training_ks, validation_traces, validation_pts, validation_ks

def filter_cst_vs_random(data, reference):
    Is0 = []
    Is1 = []
    for ri,r in enumerate(data):
        if (r==reference).all():
            Is0.append(ri)
        else:
            Is1.append(ri)
    # Return set of indexes
    return Is0, Is1

def load_ttest_dataset(filepath):
    with open(filepath,"rb") as f:
        coll = np.load(f, allow_pickle=True)
        traces_tt = coll["traces"]
        pts_tt = coll["pts"]
        ks_tt = coll["ks"]
    # Re-identify the class of each traces
    # Look for proper indexes
    vp, cp = np.unique(pts_tt,axis=0, return_counts=True)
    vk, ck = np.unique(ks_tt, axis=0, return_counts=True)
    
    # Switch based on the max value
    if max(cp)>max(ck):
        # filter based on the key
        Is0, Is1 = filter_cst_vs_random(ks_tt, vk[np.argmax(ck)])
    elif max(cp)<max(ck):
        # filter based on the plaintext
        Is0, Is1 = filter_cst_vs_random(pts_tt, vp[np.argmax(cp)])
    else:
        raise ValueError("Not able to distinguish filtering argument")

    labels = np.zeros(pts_tt.shape[0], dtype=np.uint16)
    for i in Is1:
        labels[i]=1

    return traces_tt, pts_tt, ks_tt, labels

def apply_permutation_dataset(traces, pts, ks, seed=0):
    N = traces.shape[0]
    np.random.seed(seed)
    rp = np.random.permutation(np.arange(N))
    np.random.seed(None)
    traces[:,:] = traces[rp]
    pts[:,:] = pts[rp]
    ks[:,:] = ks[rp]
