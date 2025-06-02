from utils_scale import utils_files
from cryptography.hazmat.primitives import hashes
import functools as ft
import numpy as np

RELATIVE_DIR_FILES_T=utils_files.RELATIVE_DIR_FILES_T

CASE_DATASET_P0K0="fixed-p0k0"
CASE_DATASET_P1K0="fixed-p1k0"
CASE_DATASET_P0K1="fixed-p0k1"

def load_dataset(f):
    utils_files.assert_file_exists(f)
    return utils_files.load_dataset(f)

def load_dataset_p0k0():
    return load_dataset(f"{RELATIVE_DIR_FILES_T}/{CASE_DATASET_P0K0}/data.npz")
    
def load_dataset_p1k0():
    return load_dataset(f"{RELATIVE_DIR_FILES_T}/{CASE_DATASET_P1K0}/data.npz")

def load_dataset_p0k1():
    return load_dataset(f"{RELATIVE_DIR_FILES_T}/{CASE_DATASET_P0K1}/data.npz")

def load_dataset_all_random():
    return load_dataset(f"{RELATIVE_DIR_FILES_T}/training0/data.npz")

def bvalue_from_bits(arr):
    assert len(arr)<9, "Only 8-bit words are supported"
    return sum([e*2**i for i,e in enumerate(arr)])

def load_traces_eid1():
    fe1 = f"{RELATIVE_DIR_FILES_T}/obs_ex1/data.npz"
    utils_files.assert_file_exists(fe1)
    with open(fe1,'rb') as f:
        coll = np.load(f, allow_pickle=True)
        traces = coll["traces"]
    return traces


def verify_order(order, ref, labels):
    for e in order:
        assert e in labels, "Only the labels {} are supported (you provided '{}')".format(labels,e)
    digest = hashes.Hash(hashes.SHA256())
    digest.update(bytes(order))
    hash_order = digest.finalize()
    return hash_order == ref

def verify_eid1(order):
    assert len(order)==5, "Wrong amount of id provided (must be equal to 5)"
    fe1 = f"{RELATIVE_DIR_FILES_T}/obs_ex1/data.npz"
    utils_files.assert_file_exists(fe1)
    with open(fe1,'rb') as f:
        coll = np.load(f, allow_pickle=True)
        hash = coll["hash"]
        labels= coll["labels"]
    print("Your guess is as follows:")
    for i, e in enumerate(order):
        print(f"t{i} corresponds to p{e}")
    vstatus = verify_order(order, hash, labels)
    if vstatus:
        print("Good job! :)")
    else:
        print("Sadly, you did not succeeded in the plaintexts used correctly.")


def load_traces_eid2():
    fe1 = f"{RELATIVE_DIR_FILES_T}/obs_ex2/data.npz"
    utils_files.assert_file_exists(fe1)
    with open(fe1,'rb') as f:
        coll = np.load(f, allow_pickle=True)
        traces = coll["traces"]
        labels = coll["labels"]
        bindex = coll["bindex"]

    return traces, labels, bindex

def verify_eid2(order):
    assert len(order)==5, "Wrong amount of id provided (must be equal to 5)"
    fe1 = f"{RELATIVE_DIR_FILES_T}/obs_ex2/data.npz"
    utils_files.assert_file_exists(fe1)
    with open(fe1,'rb') as f:
        coll = np.load(f, allow_pickle=True)
        hash = coll["hash"]
        labels= coll["labels"]
    print("Your guess is as follows:")
    for i, e in enumerate(order):
        print(f"t{i} corresponds to the class {e}")
    vstatus = verify_order(order, hash, labels)
    if vstatus:
        print("Good job! :)")
    else:
        print("Sadly, you did not succeeded in the plaintexts used correctly.")
