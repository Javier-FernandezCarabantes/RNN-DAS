import numpy as np
import torch
import os
from python_speech_features import logfbank, delta  
from torch.utils.data import DataLoader, TensorDataset
from joblib import Parallel, delayed

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



def create_features(H, window_duration, overlap_duration, mean, std, num_processes):
    """
    Creates features using parallel computation for multiple channels simultaneously.
    """
    n, m = H.shape 
    channel_groups = split_channels_features(n, num_processes)  # Number of available CPU cores

    # Compute all LFB, LFB' and LFB'' features for all channels in parallel
    print(">> Computing energy coefficients...")
    lfb_results = Parallel(n_jobs=num_processes)(
        delayed(calculate_all_LFB)(H, 100, window_duration, overlap_duration, sp)
        for sp in range(n)
    )

    # Process all channel groups simultaneously
    print(">> Computing features...")
    results = Parallel(n_jobs=num_processes)(
        delayed(process_channel_group_features)(lfb_results, start, end, mean, std)
        for start, end in channel_groups
    )
    results = [res.numpy() for res in results]
    # Concatenate the results from all processes
    features = np.concatenate(results, axis=0)
    #print(features.shape) #[batch, time_step, features]
    return features

def calculate_all_LFB(H, srate, window_duration, overlap_duration, sp):
    """
    Calculate LFB, LFB' and LFB'' features for a specific channel.
    """
    lfb, lfb_delta, lfb_delta_delta = calculate_LFB(
        da=H, srate=srate, nsamp=H.shape[1], sp=sp, window_duration=window_duration, overlap_duration=overlap_duration
    )
    return lfb.T, lfb_delta.T, lfb_delta_delta.T

def process_channel_group_features(lfb_results, start, end, mean, std):
    """
    Processes a group of channels and returns the features.
    
    Parameters:
    - lfb_results: List of LFB features for all channels
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

    for sp in range(start, end):
        # Process the previous, current, and next channels
        lfb_previous, lfb_delta_previous, lfb_delta_delta_previous = lfb_results[sp-1] if sp > 0 else lfb_results[0]
        lfb_current, lfb_delta_current, lfb_delta_delta_current = lfb_results[sp]
        lfb_next, lfb_delta_next, lfb_delta_delta_next = lfb_results[sp+1] if sp < len(lfb_results)-1 else lfb_results[-1]

        # Concatenate features (previous, current, next)
        feature = np.concatenate((
            lfb_previous, lfb_delta_previous, lfb_delta_delta_previous,
            lfb_current, lfb_delta_current, lfb_delta_delta_current,
            lfb_next, lfb_delta_next, lfb_delta_delta_next
        ), axis=1)
        #print(feature.shape)
        # Normalize the features
        feature = torch.tensor(feature, dtype=torch.float32)
        feature = (feature - mean) / std

        # Store the features for the channel
        features.append(feature)

    # Convert the list of tensors into a single tensor
    features = torch.stack(features)  # Dimensions: [core, time_step, features]

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


def features(data, filepath, num_processes):
    """
    Function to execute the RNN-DAS model with the provided data.
    
    Args:
        data: The input data to be processed.
        filepath: The path of the mean/std values file to normalize.
        num_processes: The number of cpu cores to be employed in parallelization
    
    Returns:
        normalized_dataloader: normalized features of the DAS data.
    """
    
    mean, std = read_mean_std_from_text(filepath)
    features=create_features(data, 6, 1.2, mean, std, num_processes)
    dataloader = create_prediction_dataloader(features)
    print("Done")
    return dataloader
