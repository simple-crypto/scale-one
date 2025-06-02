from utils_scale import utils_files
from utils_scale.utils_obs import RELATIVE_DIR_FILES_T, load_dataset
import numpy as np
import tqdm

from utils_scale.utils_aes import HW, Sbox
from scalib.attacks import Cpa
import matplotlib.pyplot as plt
from utils_scale.test_scale import ref_pearson_corr

def load_noisy_dataset(df_f, std=0.0, seed=0):
    dataset = load_dataset(df_f)
    # Add noise
    tr = dataset["traces"]
    rng = np.random.default_rng(seed=seed)
    noise = rng.normal(loc=0,scale=std,size=tr.shape)
    dataset['traces']+=noise
    return dataset

def load_clipped_dataset(df_f, mv_ref=50, mv_clip=20, res_int=16):
    assert mv_clip<50
    # QnD way to simulate clipping with float traces
    max_v = 2**(res_int-1)
    min_v = -max_v + 1
    dataset = load_dataset(df_f)
    scaling = mv_ref/mv_clip
    trs = dataset["traces"]
    trs *= scaling
    trs[trs>=max_v] = max_v
    trs[trs<=min_v] = min_v
    dataset["traces"]=trs
    return dataset

WINDOW_CROP_MIS=1000
def load_misalign_dataset_ref(df_f):
    dataset = load_dataset(df_f)
    dataset['traces']=dataset['traces'][:,WINDOW_CROP_MIS:-WINDOW_CROP_MIS]
    return dataset

def load_misalign_dataset(df_f, std_mis=10, seed=0):
    dataset = load_dataset(df_f)
    trs = dataset['traces']
    L = trs.shape[1]
    # Generate random shift
    rng = np.random.default_rng(seed=seed)
    shifts = np.round(rng.normal(loc=0,scale=std_mis,size=trs.shape[0]))
    # verify that the shift does not exceed the crop window 
    shifts[shifts>=WINDOW_CROP_MIS] = WINDOW_CROP_MIS
    shifts[shifts<=-WINDOW_CROP_MIS] = -WINDOW_CROP_MIS
    for i,sh in enumerate(shifts):
        abs_sh=int(np.abs(sh))
        if sh>0: 
            trs[i,abs_sh:] = trs[i,:L-abs_sh]
        elif sh<0:
            trs[i,:L-abs_sh] = trs[i,abs_sh:]
    dataset['traces'] = trs[:,WINDOW_CROP_MIS:-WINDOW_CROP_MIS]
    return dataset


def _DCcharge_simu(duration, nt, T, vmax):
    ts = np.linspace(0,duration,nt)
    tau= T/(2*np.pi)
    vctr = vmax*(1-np.exp(-ts/tau))
    return vctr

def load_DCshift_dataset(df_f, shift_mv=10, T=1, mv_ref=50, res_int=16):
    dataset = load_dataset(df_f)
    # QnD way to simulate behavior similar to DC effect
    max_v = 2**(res_int-1)
    min_v = -max_v + 1
    res_mv = mv_ref / max_v
    res_shift = shift_mv / res_mv 
    offsets = _DCcharge_simu(1, dataset['traces'].shape[0], T, 1)*res_shift
    dataset['traces'] += offsets[:,np.newaxis]
    return dataset



def scalib_corr_traces(traces, pts, models):
    """ 
    Compute the HW correlation, for several variables.

    traces: the traces, as an array of shape (nexec, nsamples)
    pts: the plaintexts used, as an array of shape (nexec, nvars)
    models: array of shape (nvars, nclasses, nsamples), 

    Returns: a NumPy array of shape (nvars,nclasses,nsamples), with the correlation values associated to each of the independent subkey candidate
    """
    cpa = Cpa(nc=models.shape[1], kind=Cpa.Xor)
    cpa.fit_u(np.round(traces).astype(np.int16),pts.astype(np.uint16))
    return cpa.get_correlation(models)

def model_HW_outSB(nvars, nsamples):
    """
    Compute the HW models for SCALib CPA, for all the Sboxes output variables.

    To speed up computation, the intermediate considered in SCALib CPA is the result of an configurable 
    operation (e.g. XOR) between the key and the states provided during the fitting phase. 
    In our case, the class label of the models are the one correspondign to the result of the XOR between 
    the plaintext byte and the candidate key byte, or the input of the Sbox. 
    
    """
    return np.tile(HW[Sbox][np.newaxis,:,np.newaxis], (nvars, 1, nsamples)).astype(np.float64)

def scalib_complete_cpa_out_sbox(traces, pts):
    """ 
    Perform a CPA against the full key.

    traces: the traces, as an array of shape (nexec, nsamples)
    pts: the plaintexts used, as an array of shape (nexec, 16)

    Returns: a NumPy array of shape (16,), with the best key candidate 
    """
    models = model_HW_outSB(pts.shape[1], traces.shape[1])
    correlation_abs = np.abs(scalib_corr_traces(traces,pts, models))
    return np.argmax(np.max(correlation_abs, axis=2), axis=1)


def cpa_with_exploration(dataset_file, cpa_func, std=[0], qa=[10,100,1000,2000,3000]):
    # Allocate recovered byte for each cases
    res = np.zeros([len(std), len(qa)])
    # Run the cases
    with tqdm.tqdm(total=len(std), desc="Progress") as pbar:
        for stdi, stde in enumerate(std):
            # Load the dataset
            dataset = load_noisy_dataset(dataset_file, std=stde)
            # Perform the cpa for the given qa
            for qai, qae in enumerate(qa):
                kg = cpa_func(dataset["traces"][:qae],dataset["pts"][:qae])
                res[stdi, qai] = np.sum(dataset["ks"][0]==kg)
            # Update pbar
            pbar.update(1)
    # PLot the results
    f = plt.figure()
    ax = f.add_subplot(1,1,1)
    for stdi, stde in enumerate(std):
        ainst = ax.plot(qa, res[stdi], label=r"$\sigma = {}$".format(stde))
        ax.plot(qa, res[stdi], 'x', color=ainst[-1].get_color())
    ax.set_xscale("log", base=10)
    ax.set_ylabel("Key bytes recovered")
    ax.set_xlabel(r"$q_a$")
    f.legend()
    plt.show()

def std(v):
    return np.std(v, axis=0)

def covariance(l, m):
    lc = l - np.mean(l, axis=0)
    mc = m - np.mean(m)
    return np.mean(lc*mc[:,np.newaxis], axis=0)

def _cpa_parts_exploration_compute(dataset_file, bindex, std=[0]):
    # 0 noise used as reference for time sample inference
    ostd = np.sort(std).tolist()
    covs = np.zeros(len(std))
    corr = np.zeros(len(std))
    std_m = np.zeros(len(std))
    std_t = np.zeros(len(std))
 
    # Compute reference correlation, identify poi
    dataset = load_noisy_dataset(dataset_file, std=0)
    pts = dataset['pts'][:,bindex][:,np.newaxis]
    k = dataset["ks"][0,bindex]
    m = HW[Sbox[pts ^ k]]
    models = model_HW_outSB(1, dataset['traces'].shape[1])
    corrref = scalib_corr_traces(dataset['traces'], dataset['pts'][:,bindex][:,np.newaxis], models)
    poi = np.argmax(np.abs(corrref[0,k]))
    maxcorr = np.max(np.abs(corrref[0,k]))
    # Compute metrics
    model_loop = model_HW_outSB(1, 1)
    for i, stde in enumerate(ostd):
        dataset = load_noisy_dataset(dataset_file, std=stde, seed=0)
        trs = dataset['traces'][:,poi][:,np.newaxis]
        corr[i] = scalib_corr_traces(trs, pts, model_loop)[0,k,0]
        covs[i] = covariance(trs, m[:,0])[0]
        std_m[i] = np.std(m)
        std_t[i] = np.std(trs)
    return dict(
        poi=poi,
        stds=std,
        corr=corr,
        covs=covs,
        std_m=std_m,
        std_t=std_t,
        bindex=bindex,
        maxcorr=maxcorr
    )

def print_cpa_parts_report(res):
    print(f'Using time index {res["poi"]} [corr. max {res["maxcorr"]}]')
    for i,stde in enumerate(res['stds']):
        print(f'---> For std_noise = {stde}')
        print(f'correlation:{res["corr"][i]}')
        print(f'covariance:{res["covs"][i]}')
        print(f'std_model:{res["std_m"][i]}')
        print(f'std_traces:{res["std_t"][i]}')
        print()

def cpa_parts_exploration(dataset_file, bindex, std=[0], plot=False):
    res = _cpa_parts_exploration_compute(dataset_file, bindex, std=std)
    print_cpa_parts_report(res)
    if plot:
        f = plt.figure()
        axes = [f.add_subplot(4,1,i+1) for i in range(4)]
        axes[0].plot(res['stds'],res['corr'])
        axes[0].set_xscale("log", base=10)
        axes[0].set_title(r"$\rho (HW[SB_{{{}}}] ; L_{{{}}})$".format(bindex, res['poi']))
    
        axes[1].plot(res['stds'],res['covs'])
        axes[1].set_xscale("log", base=10)
        axes[1].set_title(r"$\mathsf{{cov}}(HW[SB_{{{}}}] ; L_{{{}}})$".format(bindex, res['poi']))
    
        axes[2].plot(res['stds'],res['std_m'])
        axes[2].set_xscale("log", base=10)
        axes[2].set_title(r"$\mathsf{{std}}(HW[SB_{{{}}}])$".format(bindex))

        axes[3].plot(res['stds'],res['std_t'])
        axes[3].set_xscale("log", base=10)
        axes[3].set_title(r"$\mathsf{{std}}(L_{{{}}})$".format(res['poi']))
        axes[3].set_xlabel(r"$\sigma_{{noise}}$".format(res['poi']))

        f.tight_layout()
        plt.show()


def scatter_HW_mean(dataset, bindex):
    pts = dataset["pts"][:,bindex]
    ks = dataset["ks"][:,bindex]
    trs = dataset["traces"]

    models = model_HW_outSB(1, trs.shape[1])
    corr = scalib_corr_traces(trs, pts[:,np.newaxis], models)
    poi = np.argmax(np.abs(corr[0,ks[0]]))

    istate = Sbox[pts ^ ks]
    hwm = HW[istate]

    f = plt.figure()
    ax0 = f.add_subplot(1,1,1)

    m = np.zeros(256)
    for i in range(256):
        m[i] = np.mean(trs[istate==i,poi])

    ax0.set_title(r"Scatter for time index {} ;  $SB_{{{}}}$".format(poi,bindex))
    ax0.set_ylabel("HW")
    ax0.set_xlabel("Averaged trace value")
    plt.scatter(m,HW)
    plt.show()

