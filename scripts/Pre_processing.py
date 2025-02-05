# Library importation
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io
from scipy.signal import spectrogram
import torch
import pickle
import pandas as pd
from scipy.signal import butter, filtfilt
import h5py
from datetime import datetime



dt=1/100
dx=10 #values for La Palma

def process_H(H):
    print(">> Processing DAS data...")
    remove_coherent_noise(H, method='fit')
    fs = 1/dt
    nyquist = 0.5 * fs
    low = 1 / nyquist
    high = 49 / nyquist
    b, a = butter(4, [low, high], btype='band')
    H[:, :] = filtfilt(b, a, H, axis=1)  
    remove_trend(H)
    print(">> Processing complete")
    return H


def remove_coherent_noise(H, method='median'):
    # Remove coherent synchronous noise from all the traces
    md = np.median(H, 0)

    if method == 'simple':
        for i in range(H.shape[1]):
            H[:, i] = H[:, i] - md[i]

    elif method == 'fit':
        den = np.sum(md * md)
        for i in range(H.shape[0]):
            dd = H[i, :]
            am = np.sum(dd * md) / den
            H[i, :] = dd - am * md


def remove_trend(H):
    # Remove linear trend along individual traces through LSQR fit
    xx = np.arange(H.shape[1])
    for i in range(H.shape[0]):
        dr = H[i, :]
        try:
            po = np.polyfit(xx, dr, 1)
        except:
            print(dr)
            exit(1)
        mo = np.polyval(po, xx)
        H[i, :] = dr - mo