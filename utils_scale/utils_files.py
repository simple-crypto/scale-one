import numpy as np
from pathlib import Path


# Version 2025
RELATIVE_DIR_FILES_T="scale-dataset-25"
VALIDATION_DS=[f"./{RELATIVE_DIR_FILES_T}/validation{i}/data.npz" for i in range(5)]
TRAINING_DS=[f"./{RELATIVE_DIR_FILES_T}/training{i}/data.npz" for i in range(1)]
TTEST_KF_DS=f"./{RELATIVE_DIR_FILES_T}/ttest-kf/data.npz"
TTEST_PF_DS=f"./{RELATIVE_DIR_FILES_T}/ttest-pf/data.npz"

def I2keep(traces):
    tmpm = np.mean(traces,axis=1)
    mtmpm = np.mean(tmpm)
    th = np.std(tmpm)
    dst = np.abs(tmpm-mtmpm)
    hig = dst>4*th
    kept = np.where(np.logical_not(hig))[0]
    return np.array(kept)

def load_dataset(datafile_path, traces_dtype=np.float64, cropping=None, seed_shuffle=None, remove_first=False):
    """
    Open to file and load all the data fields from it. 
    """
    with open(datafile_path, 'rb') as f:
        coll = np.load(f, allow_pickle=True)
        traces = coll["traces"].astype(traces_dtype)
        pts = coll["pts"]
        ks = coll["ks"]
        cts = coll["cts"]
    if cropping is not None:
        [start, end] = cropping
        traces = traces[:,start:end]
    if remove_first:
        idx = I2keep(traces)
        traces=traces[idx]
        pts=pts[idx]
        ks=ks[idx]
        cts=cts[idx]
    ds = dict(
        traces=traces,
        pts=pts,
        ks=ks,
        cts=cts
    )
    if seed_shuffle is not None:
        apply_permutation_dataset_inplace(ds, seed=seed_shuffle)
    return ds

def apply_permutation_dataset_inplace(dataset, seed=0):
    N = dataset['traces'].shape[0]
    np.random.seed(seed)
    rp = np.random.permutation(np.arange(N))
    np.random.seed(None)
    for f in dataset.keys():
        dataset[f][:,:] = dataset[f][rp]

def assert_file_exists(f):
    fp = Path(f)
    exp_path = "{}/{}".format(Path.cwd(), f)
    assert fp.is_file(), f"The file '{exp_path}' does not exist. Please verify the location of the dataset." 

#############
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
