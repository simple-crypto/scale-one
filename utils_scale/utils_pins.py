import numpy as np
from cryptography.hazmat.primitives import hashes
import matplotlib.pyplot as plt
from utils_scale import utils_files

SAMPLES_PER_TRACE=400

# Path to file
RELATIVE_DIR_FILES_T="scale-dataset-25/pin"

# Expected SHA-256 PIN value
EXP_PIN_HASH=b"K'\xfeK\xba\xa3\x0c.\xc2\xf9\xd6\xe1z_\x89Ra\x08\xd9\x7f\xc1\xb8\x98\xb1~Ka]\x9c\xfe\xb9\x03"


def assert_valid_pin(tpin):
    assert len(tpin)==4, "The try MUST be a 4-digit PIN"
    for t in tpin:
        assert t>=0 and t<10, "Each digit MUST be in the range 0-9"

def plot_pins_traces(traces, pins):
    assert len(traces)==len(pins), "The traces and tries provided must be of same length"
    plt.figure()
    for i in range(len(pins)):
        plt.plot(+traces[i], linewidth=1, label='{}'.format([int(x) for x in pins[i]]))
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

def acq_short(tries, f):
    for t in tries:
        for i in range(1,4):
            assert t[i] == 0, "PIN value not supported by pre-recorded traces."
    utils_files.assert_file_exists(f)
    traces_unfiltered = np.load(f)
    traces = np.zeros((len(tries),SAMPLES_PER_TRACE))
    for (i, t) in enumerate(tries):
        assert_valid_pin(t)
        traces[i,:] = traces_unfiltered[t[0]]
    return traces

def acquire_traces1(tries):
    return acq_short(tries, f"{RELATIVE_DIR_FILES_T}/task1_1.npy")

def acquire_traces2(tries):
    traces_unfiltered = np.load(f"{RELATIVE_DIR_FILES_T}/task1_2.npy")
    traces = np.zeros((len(tries),SAMPLES_PER_TRACE))
    for (i, t) in enumerate(tries):
        assert_valid_pin(t)
        t_idx = int("".join(str(_) for _ in t))
        traces[i] = traces_unfiltered[t_idx]
    return traces

def verify_pin(tpin):
    assert_valid_pin(tpin)
    digest = hashes.Hash(hashes.SHA256())
    digest.update(tpin)
    hash_pin = digest.finalize()
    if hash_pin==EXP_PIN_HASH:
        print("CONGRATS! You found it.")
    else:
        print(f"SORRY, {tpin} is not the correct PIN. Try again!")
    
def acquire_traces3(tries):
    return acq_short(tries, f"{RELATIVE_DIR_FILES_T}/task1_3.npy")

