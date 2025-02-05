import numpy as np
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from python_speech_features import logfbank, delta  
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
#from HELPER_FUNCTIONS_RNN_LSTM import read_features, Create_Tensor
import os
from datetime import datetime, timedelta
import sys
from joblib import Parallel, delayed
import re
from tqdm import tqdm
from scipy.signal import butter, filtfilt
import pickle
import pandas as pd

def split_channels_features(n_channels, num_processes):
    """
    Divides the number of channels into groups based on the number of available processes.
    """
    groups = []
    channels_per_process = n_channels // num_processes
    for i in range(num_processes):
        start = i * channels_per_process
        end = start + channels_per_process if i < num_processes - 1 else n_channels  # Last group of channels
        groups.append((start, end))
    return groups

def create_features(H, window_duration, overlap_duration, mean, std):
    """
    Creates features using parallel computation for multiple channels simultaneously.
    """
    n, m = H.shape
    num_processes = os.cpu_count()  # Number of available CPU cores
    channel_groups = split_channels_features(n, num_processes)

    # Process all channel groups simultaneously
    print(">> Processing the signal...")
    results = Parallel(n_jobs=num_processes)( 
        delayed(process_channel_group_features)(H, 100, window_duration, overlap_duration, start, end, mean, std) 
        for start, end in channel_groups
    )
    results = [res.numpy() for res in results]
    # Concatenate the results from all processes
    features = np.concatenate(results, axis=0)
    #print(features.shape)
    #print(features.shape) #[batch, time_step, features]
    return features


def calculate_LFB(da, srate, nsamp, sp, window_duration=6, overlap_duration=1.2, hop_length=1024, n_LFB=16):
    """
    Calculates an LFB feature matrix and its derivatives.
    
    Parameters:
    - da: Input data array (2D, where each row corresponds to a channel)
    - srate: Sampling rate (Hz)
    - nsamp: Total number of samples in the data
    - sp: Index of the channel of interest
    - window_duration: Duration of each window (in seconds)
    - overlap_duration: Duration of overlap between windows (in seconds)
    - hop_length: Hop length for the FFT
    - n_LFB: Number of log-Mel filters to use
    
    Returns:
    - LFB_matrix_vals: LFB feature matrix (n_LFB x n_frames)
    - deltas_LFB: Matrix of the first derivatives of LFB
    - deltas_deltas_LFB: Matrix of the second derivatives of LFB
    """
    # Calculate the number of points per window and the overlap
    points_per_frame = int(window_duration * srate)
    overlap_points = int(overlap_duration * srate)
    step_points = int(points_per_frame - overlap_points)
    step_duration = window_duration - overlap_duration

    num_frames = (nsamp - points_per_frame) // step_points + 1
    # Initialize the matrix to store LFB coefficients
    LFB_matrix_vals = np.zeros((num_frames, n_LFB))

    # Iterate over the frames and calculate LFB coefficients
    for i, index in enumerate(range(0, nsamp - points_per_frame + 1, step_points)):
        start_index = index
        end_index = start_index + points_per_frame
        frame_data = da[:, start_index:end_index]

        # Calculate LFB coefficients for the current frame
        if overlap_duration == 0:
            overlap_duration = window_duration
        LFB = np.float32(logfbank(1 + frame_data[sp], samplerate=srate, 
                                  winlen=window_duration, winstep=step_duration, 
                                  nfilt=n_LFB, nfft=hop_length, lowfreq=0, 
                                  highfreq=None, preemph=0.97))
        # Store the coefficients in the matrix
        LFB_matrix_vals[i, :] = LFB
    LFB_matrix_vals = np.array(LFB_matrix_vals)
    # Calculate the derivatives of LFB coefficients
    deltas_LFB = delta(LFB_matrix_vals, 2)
    deltas_deltas_LFB = delta(deltas_LFB, 2)
    # Transpose the matrices so that dimensions are n_LFB x n_frames
    LFB_matrix_vals = LFB_matrix_vals.T
    deltas_LFB = deltas_LFB.T
    deltas_deltas_LFB = deltas_deltas_LFB.T
    return LFB_matrix_vals, deltas_LFB, deltas_deltas_LFB


def process_channel_group_features(H, srate, window_duration, overlap_duration, start, end, mean, std):
    """
    Processes a group of channels and returns the features.
    
    Parameters:
    - H: Data matrix (2D, [n_channels, n_samples])
    - srate: Sampling rate (Hz)
    - stime: Initial time (datetime.datetime)
    - dt: Temporal resolution between samples (seconds)
    - window_duration: Duration of each window (seconds)
    - overlap_duration: Duration of overlap between windows (seconds)
    - start: Index of the first channel to process
    - end: Index of the last channel to process (exclusive)
    - mean: Mean tensor for normalization
    - std: Standard deviation tensor for normalization

    Returns:
    - features: Tensor with the features (core, time_step, features)
    """
    features = []

    # Ensure mean and std are in tensor format
    mean = mean.clone().detach() if isinstance(mean, torch.Tensor) else torch.tensor(mean, dtype=torch.float32)
    std = std.clone().detach() if isinstance(std, torch.Tensor) else torch.tensor(std, dtype=torch.float32)
    mean = mean.unsqueeze(dim=0)  # [1, 144] to avoid issues with normalization
    std = std.unsqueeze(dim=0)  # [1, 144]
    print(std, mean)
    n, m = H.shape  # Data dimensions: n_channels x n_samples

    for sp in range(start, end):
        print(sp)
        # Process the previous channel (if it's not the first channel)
        if sp == 0:
            lfb_previous, lfb_delta_previous, lfb_delta_delta_previous = calculate_LFB(
                da=H, srate=srate, nsamp=m, sp=sp, window_duration=window_duration, overlap_duration=overlap_duration
            ) 
        else:
            lfb_previous, lfb_delta_previous, lfb_delta_delta_previous = calculate_LFB(
                da=H, srate=srate, nsamp=m, sp=sp-1, window_duration=window_duration, overlap_duration=overlap_duration
            )

        # Process the current channel
        lfb_current, lfb_delta_current, lfb_delta_delta_current = calculate_LFB(
            da=H, srate=srate, nsamp=m, sp=sp, window_duration=window_duration, overlap_duration=overlap_duration
        )

        # Process the next channel (if it's not the last channel)
        if sp == n - 1:
            lfb_next, lfb_delta_next, lfb_delta_delta_next = calculate_LFB(
                da=H, srate=srate, nsamp=m, sp=sp, window_duration=window_duration, overlap_duration=overlap_duration
            )
        else:
            lfb_next, lfb_delta_next, lfb_delta_delta_next = calculate_LFB(
                da=H, srate=srate, nsamp=m, sp=sp+1, window_duration=window_duration, overlap_duration=overlap_duration
            )

        # Concatenate features (previous, current, next)
        feature = np.concatenate((
            lfb_previous.T, lfb_delta_previous.T, lfb_delta_delta_previous.T,
            lfb_current.T, lfb_delta_current.T, lfb_delta_delta_current.T,
            lfb_next.T, lfb_delta_next.T, lfb_delta_delta_next.T
        ), axis=1)
        print(feature.shape)
        # Normalize the features
        feature = torch.tensor(feature, dtype=torch.float32)
        feature = (feature - mean) / std

        # Store the features for the channel
        features.append(feature)

    # Convert the list of tensors into a single tensor
    features = torch.stack(features)  # Dimensions: [core, time_step, features]

    return features

def create_prediction_dataloader(data, batch_size=256):
    tensor_data = torch.tensor(data, dtype=torch.float32) 
    dataset = TensorDataset(tensor_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader


def read_mean_std_from_text(file_path):
    """
    Reads means and standard deviations from a text file and returns them as tensors.
    
    Args:
        file_path (str): Path to the text file containing mean and std values for each column.

    Returns:
        mean (torch.Tensor): Tensor containing the mean values for each feature.
        std (torch.Tensor): Tensor containing the standard deviations for each feature.
    """
    means = []
    stds = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(', ')
            mean_value = float(parts[0].split('=')[1])
            std_value = float(parts[1].split('=')[1])
            means.append(mean_value)
            stds.append(std_value)
    
    mean = torch.tensor(means)
    std = torch.tensor(stds)
    
    return mean, std


def features(data, filepath):
    """
    Function to execute the RNN-DAS model with the provided data.
    
    Args:
        data: The input data to be processed.
        filepath: The path of the mean/std values file to normalize.
    
    Returns:
        normalized_dataloader: normalized features of the DAS data.
    """
    
    mean, std = read_mean_std_from_text(filepath)
    features=create_features(data, 6, 1.2, mean, std)
    dataloader = create_prediction_dataloader(features)
    print("Done")
    return dataloader
