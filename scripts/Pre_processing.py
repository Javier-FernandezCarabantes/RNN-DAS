import numpy as np
from scipy.signal import butter, filtfilt
import os
import h5py

def load_h5(name_id, path="H5_data"):
    """
    Load the data from an HDF5 file.

    Args:
        name_id (str): The ID used to find the corresponding H5 file.
        path (str): The directory where the H5 file is located. Default is "H5_data".

    Returns:
        data (numpy.ndarray): The data stored in the H5 file, should be a channels x time_steps 2D matrix .
        dt_s (float): The time step in seconds (from the "dt_s" attribute).
        dx_m (float): The spatial step in meters (from the "dx_m" attribute).
        begin_date (str): The begin time stored in the "begin_time" attribute (ISO 8601 format).
    """
    # Construct the full path to the H5 file
    file_path = os.path.join(path, f"{name_id}.h5")
    
    # Open the H5 file and read the data and attributes
    with h5py.File(file_path, "r") as fp:
        data = np.array(fp["data"])
        dt_s = fp["data"].attrs["dt_s"]
        dx_m = fp["data"].attrs["dx_m"]
        begin_date = fp["data"].attrs["begin_time"]
    
    # Return the loaded data and attributes
    return data, dt_s, dx_m, begin_date


def read_fileids(file_path):
    """
    Reads a list of file identifiers from a text file.

    Args:
        file_path (str): Path to the text file containing file IDs.

    Returns:
        list: List of file IDs.
    """
    with open(file_path, "r") as f:
        fileids = [line.strip() for line in f.readlines()]
    return fileids

def process_H(H, dt):
    print(">> Processing DAS data...")
    H_processed=H.copy()
    remove_trend(H_processed)
    remove_coherent_noise(H_processed, method='simple')
    fs = 1/dt
    nyquist = 0.5 * fs
    low = 1 / nyquist
    high = 49 / nyquist
    b, a = butter(4, [low, high], btype='band')
    H_processed[:, :] = filtfilt(b, a, H_processed, axis=1)  
    print(">> Processing complete")
    return H_processed


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