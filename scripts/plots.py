import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from scipy.signal import spectrogram



def plot_das_bi(data, probabilities, window_duration=6, overlap_duration=1.2, threshold=0.9, fsamp=100, channel_idx=0):
    """
    Plots a visualization of DAS data and the RNN-DAS model predictions.
    
    Parameters:
    data (numpy.ndarray): A 2D array of shape (num_channels, time_steps) representing the DAS data.
    probabilities (numpy.ndarray): A 3D array of shape (num_channels, num_windows, num_classes) representing the predicted probabilities for each window and channel.
    window_duration (float): Duration of each window in seconds (default is 6 seconds).
    overlap_duration (float): Duration of overlap between windows in seconds (default is 1.2 seconds).
    threshold (float): Threshold probability for classifying a window (default is 0.9).
    fsamp (int): Sampling frequency (default is 100 Hz).
    channel_idx (int): Index of the channel to plot in the time-domain (default is 0).
    
    Output:
    Displays a figure with subplots:
        - The first subplot shows the normalized DAS data.
        - The second subplot overlays the RNN-DAS predictions on top of the DAS data.
        - The second row displays the time-domain signal of the selected channel and highlights windows where the RNN predicts a class with probability above the threshold.
        - The third row displays the spectrogram of the selected signal.
    """

    # Number of channels and time steps in the data
    num_channels = data.shape[0]
    time_steps = data.shape[1]
    time_step_duration = window_duration - overlap_duration

    # Select all channels
    selected_channels = np.arange(0, num_channels, 1)
    data_selected = data[selected_channels, :]

    # Create a figure with 3x3 subplots
    fig, axs = plt.subplots(3, 2, figsize=(12, 8),
                             gridspec_kw={'height_ratios': [5, 1, 1], 'wspace': 0.05, 'hspace': 0.0},
                             sharex=True, sharey='row')

    # Normalize data
    normalize = lambda x: (x - np.mean(x, axis=-1, keepdims=True)) / np.std(x, axis=-1, keepdims=True)
    min_max_normalize = lambda x: 2 * (x - np.min(x)) / (np.max(x) - np.min(x)) - 1
    data_normalized = normalize(data_selected)
    channel_signal = min_max_normalize(data[channel_idx, :].reshape(1, -1)).flatten()

    # Custom colormap for class labels
    base_colors = [
        (1, 1, 1, 0),  # -1: transparent
        "red",         # 0: red
        "gray",        # 1: gray
        "green"        # 2: green
    ]

    # Create a prediction grid
    num_windows = probabilities.shape[1]
    pred_grid = np.full((len(selected_channels), num_windows), -1)
    for idx, channel in enumerate(selected_channels):
        for w in range(num_windows):
            max_prob = np.max(probabilities[channel, w, :])
            if max_prob >= threshold:
                dominant_class = np.argmax(probabilities[channel, w, :])
                pred_grid[idx, w] = dominant_class

    # Filtered colors for unique predictions
    unique_values = np.unique(pred_grid)
    filtered_colors = [base_colors[i] for i in unique_values + 1]
    custom_cmap = ListedColormap(filtered_colors)

    # Plot 1: Original DAS Data
    axs[0, 0].imshow(data_normalized, cmap="seismic", vmin=-1, vmax=1, aspect="auto",
                     extent=[0, time_steps / fsamp, selected_channels[-1], selected_channels[0]],
                     interpolation="none")
    axs[0, 0].set_ylabel('Spatial channels')
    axs[0, 0].set_title('Original DAS', fontweight='bold')

    # Plot 2: RNN-DAS Predictions
    axs[0, 1].imshow(data_normalized, cmap="seismic", vmin=-1, vmax=1, aspect="auto",
                     extent=[0, time_steps / fsamp, selected_channels[-1], selected_channels[0]],
                     interpolation="none")
    time_extent = [0, time_steps / fsamp, num_channels, 0]
    axs[0, 1].imshow(pred_grid, cmap=custom_cmap, aspect="auto", extent=time_extent, interpolation="none")
    axs[0, 1].set_title('RNN-DAS', fontweight='bold')

    # Plot signal of selected channel in time-domain
    for j, ax in enumerate(axs[1, :]):
        ax.plot(np.arange(time_steps) / fsamp, channel_signal, color='black', lw=1)
        ax.set_xlabel('Time (s)')
        if j == 1:  # RNN-DAS probabilities for the selected channel
            channel_probabilities = probabilities[channel_idx, :, 2]
            above_threshold = channel_probabilities >= threshold
            for w in range(len(above_threshold)):
                if above_threshold[w]:
                    start = w * time_step_duration
                    end = start + window_duration
                    ax.axvspan(start, end, color='green', linestyle='-', linewidth=1, alpha=0.3)

    # Add color bar for normalized strain rate
    cbar = fig.colorbar(axs[0, 0].images[0], ax=axs[:2, :].ravel().tolist(), orientation='vertical', fraction=0.02, pad=0.02, extend='both')
    cbar.set_label('Normalized Strain Rate')


    # Compute and plot spectrogram for the third row of subplots
    frequencies, times, Sxx = spectrogram(channel_signal, fs=fsamp, nfft=1024, nperseg=600, noverlap=200)

    for j in range(2):
        im2=axs[2, j].imshow(
            10 * np.log10(Sxx),
            cmap='jet',
            aspect='auto',
            extent=[0, time_steps / fsamp, frequencies[-1], frequencies[0]], 
            interpolation='none',
            vmin=-100,
            vmax=0
        )
        axs[2, j].set_ylim(1, 20)
        axs[2, j].set_yticks([5, 15])
        axs[2, j].set_xlabel('Time (s)')
    
    axs[2, 0].set_ylabel('Frequency\n(Hz)')
    cbar2 = fig.colorbar(im2, ax=axs[2, :].ravel().tolist(), orientation='vertical', fraction=0.02, pad=0.02, aspect=3)
    cbar2.set_label('dB')

    # Adjust layout and show figure
    plt.show()



def plot_das_grammar(data, probabilities, probabilities_grammar, threshold=2/3, fsamp=100, window_duration=6, overlap_duration=1.2, channel_idx=0):
    """
    Plots DAS data and RNN-DAS predictions (with and without grammar).
    
    Parameters:
    data (numpy.ndarray): A 2D array representing the DAS data (shape: num_channels x time_steps).
    probabilities (numpy.ndarray): A 3D array of predicted probabilities for each channel, window, and class (shape: num_channels x num_windows x num_classes).
    probabilities_grammar (numpy.ndarray): A 3D array of predicted probabilities with grammar for each channel, window, and class (shape: num_channels x num_windows x num_classes).
    threshold (float): The probability threshold above which a class is considered as predicted (default is 2/3).
    fsamp (int): The sampling frequency in Hz (default is 100 Hz).
    window_duration (float): The duration of each window in seconds (default is 6 seconds).
    overlap_duration (float): The overlap duration between windows in seconds (default is 1.2 seconds).
    channel_idx (int): Index of the channel to plot in the time-domain (default is 0).

    Output:
    Displays a figure with:
        - The first row of subplots displaying:
            1. The normalized DAS data.
            2. The RNN-DAS predictions without grammar.
            3. The RNN-DAS predictions with grammar.
        - The second row with the time-domain signal for a selected channel.
        - The third row displaying spectrograms of the selected signal.
    """

    # Normalize functions
    normalize = lambda x: (x - np.mean(x, axis=-1, keepdims=True)) / np.std(x, axis=-1, keepdims=True)
    min_max_normalize = lambda x: 2 * (x - np.min(x)) / (np.max(x) - np.min(x)) - 1
    time_step_duration = window_duration - overlap_duration
    # Base colors for visualization
    base_colors = [
        (1, 1, 1, 0),  # Transparent color
        "red",          # Class 0: red
        "gray",         # Class 1: gray
        "green"         # Class 2: green
    ]
    
    # Function to create the prediction grid
    def create_pred_grid(probabilities, selected_channels_length, num_windows, threshold):
        pred_grid = np.full((selected_channels_length, num_windows), -1) 
        for idx in range(selected_channels_length):
            for w in range(num_windows):
                max_prob = np.max(probabilities[idx, w, :])  
                if max_prob >= threshold:
                    dominant_class = np.argmax(probabilities[idx, w, :])
                    pred_grid[idx, w] = dominant_class
        return pred_grid

    # Function to create custom colormap
    def create_custom_cmap(pred_grid):
        unique_values = np.unique(pred_grid)  
        filtered_colors = [base_colors[i] for i in unique_values + 1] 
        custom_cmap = ListedColormap(filtered_colors)
        return custom_cmap

    fig, axs = plt.subplots(3, 3, figsize=(10, 5), 
                             gridspec_kw={'height_ratios': [5, 1, 1], 'wspace': 0.05, 'hspace': 0.0},
                             sharex=True, sharey='row')
    
    # Extracting data dimensions and initializing variables
    num_windows = probabilities.shape[1]
    num_channels = data.shape[0]
    time_steps = data.shape[1]
    selected_channels_length = num_channels
    window_step_duration = window_duration - overlap_duration

    # Data normalization
    data_normalized = normalize(data)
    mean_signal = min_max_normalize(data_normalized[channel_idx, :])

    # Create prediction grids and colormaps
    pred_grid_no_grammar = create_pred_grid(probabilities, selected_channels_length, num_windows, threshold)
    custom_cmap_no_grammar = create_custom_cmap(pred_grid_no_grammar)
    pred_grid_grammar = create_pred_grid(probabilities_grammar, selected_channels_length, num_windows, threshold)
    custom_cmap_grammar = create_custom_cmap(pred_grid_grammar)

    # Compute the spectrogram of the mean signal
    frequencies, times, Sxx = spectrogram(mean_signal, fs=fsamp, nfft=1024, nperseg=600, noverlap=200)

    # Plotting the data and predictions
    im1 = axs[0, 0].imshow(
        data_normalized,
        cmap="seismic",
        vmin=-1,
        vmax=1,
        aspect="auto",
        extent=[0, time_steps / fsamp, num_channels, 0],
        interpolation="none",
    )
    axs[0, 0].set_ylabel('Channels')
    axs[0, 0].set_title("DAS record", fontsize=12, fontweight='bold')

    axs[0, 1].imshow(
        data_normalized,
        cmap="seismic",
        aspect="auto",
        extent=[0, time_steps / fsamp, num_channels, 0],
        interpolation="none"
    )
    time_extent = [0, time_steps / fsamp, num_channels, 0]
    axs[0, 1].imshow(pred_grid_no_grammar, cmap=custom_cmap_no_grammar, aspect="auto", extent=time_extent, interpolation="none", alpha=0.7)
    axs[0, 1].set_title("No grammar", fontsize=12, fontweight='bold')

    axs[0, 2].imshow(
        data_normalized,
        cmap="seismic",
        aspect="auto",
        extent=[0, time_steps / fsamp, num_channels, 0],
        interpolation="none",
    )
    axs[0, 2].imshow(pred_grid_grammar, cmap=custom_cmap_grammar, aspect="auto", extent=time_extent, interpolation="none", alpha=0.7)
    axs[0, 2].set_title("Grammar", fontsize=12, fontweight='bold')

    # Plot time-domain signals
    for j in range(3):
        axs[1, j].plot(np.arange(time_steps) / fsamp, mean_signal, color='black', lw=1)
        axs[1, j].set_xlabel('Time (s)')
    
        # Plot spectrograms
        im2 = axs[2, j].imshow(
            10 * np.log10(Sxx),
            cmap='jet',
            aspect='auto',
            extent=[0, time_steps / fsamp, frequencies[-1], frequencies[0]], 
            interpolation='none',
            vmin=-100,
            vmax=0
        )
        axs[2, j].set_ylim(1, 20)
        axs[2, j].set_yticks([5, 15])

    axs[2, 0].set_ylabel('Frequency\n(Hz)')

    # Highlight regions where probability exceeds threshold
    # avg_prob = (probabilities[channel_idx, :, :])
    # above_threshold = avg_prob[:, 2] >= threshold
    # start_indices, end_indices = [], []
    # is_in_region = False
    # for j in range(len(above_threshold)):
    #     if above_threshold[j] and not is_in_region:
    #         start_indices.append(j)
    #         is_in_region = True
    #     elif not above_threshold[j] and is_in_region:
    #         end_indices.append(j - 1)
    #         is_in_region = False
    # if is_in_region:
    #     end_indices.append(len(above_threshold) - 1)
    
    # # Highlight in the plots
    # for w in range(len(above_threshold)):
    #     if above_threshold[w]:
    #         start = w * window_step_duration
    #         end = start + window_duration
    #         axs[1, 1].axvspan(start, end, color='green', alpha=0.3)
    channel_probabilities = probabilities[channel_idx, :, 2]
    above_threshold = channel_probabilities >= threshold
    for w in range(len(above_threshold)):
        if above_threshold[w]:
            start = w * time_step_duration
            end = start + window_duration
            axs[1, 1].axvspan(start, end, color='green', linestyle='-', linewidth=1, alpha=0.3)
    # avg_prob_grammar = np.mean(probabilities_grammar, axis=0)
    # above_threshold_grammar = avg_prob_grammar[:, 2] >= threshold
    # start_indices_grammar, end_indices_grammar = [], []
    # is_in_region_grammar = False
    # for j in range(len(above_threshold_grammar)):
    #     if above_threshold_grammar[j] and not is_in_region_grammar:
    #         start_indices_grammar.append(j)
    #         is_in_region_grammar = True
    #     elif not above_threshold_grammar[j] and is_in_region_grammar:
    #         end_indices_grammar.append(j - 1)
    #         is_in_region_grammar = False
    # if is_in_region_grammar:
    #     end_indices_grammar.append(len(above_threshold_grammar) - 1)
    
    # for w in range(len(above_threshold_grammar)):
    #     if above_threshold_grammar[w]:
    #         start = w * window_step_duration
    #         end = start + window_duration
    #         axs[1, 2].axvspan(start, end, color='green', alpha=0.3)

    channel_probabilities = probabilities_grammar[channel_idx, :, 2]
    above_threshold = channel_probabilities >= threshold
    for w in range(len(above_threshold)):
        if above_threshold[w]:
            start = w * time_step_duration
            end = start + window_duration
            axs[1, 2].axvspan(start, end, color='green', linestyle='-', linewidth=1, alpha=0.3)

    # Add colorbars
    cbar1 = fig.colorbar(im1, ax=axs[:2], orientation='vertical', fraction=0.02, pad=0.02, extend='both')
    cbar1.set_label('Normalized Strain Rate')

    cbar2 = fig.colorbar(im2, ax=axs[2], orientation='vertical', fraction=0.02, pad=0.02, aspect=3)
    cbar2.set_label('dB')

    # Adjust layout and show the figure
    #plt.tight_layout()
    plt.show()

def plot_stacked_traces_normalized(stream):
    """
    Plots all traces from an ObsPy Stream in a single figure, normalizing the amplitude between 0 and 1.
    
    Args:
        stream (obspy.Stream): ObsPy Stream object containing multiple traces.
    """
    plt.figure(figsize=(8, 12))  # Increase figure size
    
    colors = ['b', 'r', 'g']  # Alternating colors (blue, red, green)
    offset = 0  
    
    for i, trace in enumerate(stream):
        # Get data and timestamps
        data = trace.data
        times = np.linspace(0, len(data) / trace.stats.sampling_rate, len(data))
        
        # Normalize amplitude between 0 and 1
        data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
        
        # Plot with alternating colors and offset to avoid overlap
        plt.plot(times, data_norm + offset, color=colors[i % 3], linewidth=1.5) 
        
        # Increase offset for the next trace
        offset += 0.25  # Adjust spacing for better visibility
    
    # Configure plot
    plt.xlabel("Time (s)", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks([])  # Remove y-axis ticks
    plt.title(f"Stacked Normalized Traces - Event at {stream[0].stats.starttime}", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)  # Light grid for better readability
    
    plt.show()

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

def plot_das_argparse(data, event_id='plot', output_folder='./plots', fsamp=100, channel_idx=0):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Z-score and min-max normalization functions
    z_score_normalize = lambda x: (x - np.mean(x, axis=-1, keepdims=True)) / np.std(x, axis=-1, keepdims=True)
    min_max_normalize = lambda x: 2 * (x - np.min(x, axis=-1, keepdims=True)) / (np.max(x, axis=-1, keepdims=True) - np.min(x, axis=-1, keepdims=True)) - 1

    # Data selection
    data_selected = data[:, :]
    num_channels = data_selected.shape[0]
    time_steps = data_selected.shape[1]

    # Normalize data
    data_selected = z_score_normalize(data_selected)
    signal = min_max_normalize(data_selected[channel_idx, :])

    # Compute the spectrogram
    frequencies, times, Sxx = spectrogram(signal, fs=fsamp, nfft=1024, nperseg=600, noverlap=200)

    # Create the figure
    fig, axs = plt.subplots(
        3, 1, figsize=(10, 8),
        gridspec_kw={'height_ratios': [5, 1, 1], 'hspace': 0},
        sharex=True
    )

    # Plot the normalized data
    im = axs[0].imshow(
        data_selected,
        cmap="seismic",
        vmin=-1,
        vmax=1,
        aspect="auto",
        extent=[0, time_steps / fsamp, num_channels, 0],
        interpolation="none"
    )
    axs[0].set_ylabel('Channels')
    axs[0].tick_params(labelbottom=False)

    # Plot the signal
    axs[1].plot(np.arange(time_steps) / fsamp, signal, color='black', lw=1)
    axs[1].set_ylabel('Normalized\nStrain Rate')

    # Plot the spectrogram
    im2 = axs[2].imshow(
        10 * np.log10(Sxx),
        cmap='jet',
        aspect='auto',
        extent=[0, time_steps / fsamp, frequencies[-1], frequencies[0]],
        interpolation='none',
        vmin=-100,
        vmax=10
    )
    axs[2].set_ylabel('Frequency\n(Hz)')
    axs[2].set_ylim(1, 20)
    axs[2].set_yticks([5, 15])
    axs[2].set_xlabel('Time (s)')

    # Set x-axis limits
    axs[2].set_xlim(axs[0].get_xlim())

    # Add colorbars
    cbar1 = fig.colorbar(im, ax=axs[:2], orientation='vertical', fraction=0.02, pad=0.02, extend='both')
    cbar1.set_label('Normalized Strain Rate')
    cbar2 = fig.colorbar(im2, ax=axs[2], orientation='vertical', fraction=0.02, pad=0.02, aspect=5.5, shrink=0.8)
    cbar2.set_label('dB')

    # Save the figure
    plot_filename = os.path.join(output_folder, f'{event_id}.png')
    plt.savefig(plot_filename)

    print(f"Plot saved at: {plot_filename}")


def plot_stacked_traces_normalized_argparse(stream, output_folder='./plots', event_id='plot'):
    """
    Plots all traces from an ObsPy Stream in a single figure, normalizing the amplitude between 0 and 1,
    and saves the figure to the specified folder.
    
    Args:
        stream (obspy.Stream): ObsPy Stream object containing multiple traces.
        output_folder (str): Folder where the figure will be saved (default is './plots').
    """
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    plt.figure(figsize=(8, 12))  # Increase figure size
    
    colors = ['b', 'r', 'g']  # Alternating colors (blue, red, green)
    offset = 0  
    
    for i, trace in enumerate(stream):
        # Get data and timestamps
        data = trace.data
        times = np.linspace(0, len(data) / trace.stats.sampling_rate, len(data))
        
        # Normalize amplitude between 0 and 1
        data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
        
        # Plot with alternating colors and offset to avoid overlap
        plt.plot(times, data_norm + offset, color=colors[i % 3], linewidth=1.5) 
        
        # Increase offset for the next trace
        offset += 0.25  # Adjust spacing for better visibility
    
    # Configure plot
    plt.xlabel("Time (s)", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks([])  # Remove y-axis ticks
    plt.title(f"Stacked Normalized Traces - Event at {stream[0].stats.starttime}", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)  # Light grid for better readability
    
    # Save the figure
    plot_filename = os.path.join(output_folder, f'{event_id}_traces.png')
    plt.savefig(plot_filename)
    
    print(f"Plot saved at: {plot_filename}")


def plot_das_grammar_argparse(data, probabilities, probabilities_grammar, event_id='plot', threshold=2/3, fsamp=100, window_duration=6, overlap_duration=1.2, channel_idx=0, output_folder='./plots'):
    """
    Plots DAS data and RNN-DAS predictions (with and without grammar) and saves the figure in the specified folder.

    Parameters:
    data (numpy.ndarray): A 2D array representing the DAS data (shape: num_channels x time_steps).
    probabilities (numpy.ndarray): A 3D array of predicted probabilities for each channel, window, and class (shape: num_channels x num_windows x num_classes).
    probabilities_grammar (numpy.ndarray): A 3D array of predicted probabilities with grammar for each channel, window, and class (shape: num_channels x num_windows x num_classes).
    threshold (float): The probability threshold above which a class is considered as predicted (default is 2/3).
    fsamp (int): The sampling frequency in Hz (default is 100 Hz).
    window_duration (float): The duration of each window in seconds (default is 6 seconds).
    overlap_duration (float): The overlap duration between windows in seconds (default is 1.2 seconds).
    channel_idx (int): Index of the channel to plot in the time-domain (default is 0).
    output_folder (str): Folder to save the plot (default is './plots').

    Output:
    Saves the figure in the specified output folder.
    """

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Normalize functions
    normalize = lambda x: (x - np.mean(x, axis=-1, keepdims=True)) / np.std(x, axis=-1, keepdims=True)
    min_max_normalize = lambda x: 2 * (x - np.min(x)) / (np.max(x) - np.min(x)) - 1

    # Base colors for visualization
    base_colors = [
        (1, 1, 1, 0),  # Transparent color
        "red",          # Class 0: red
        "gray",         # Class 1: gray
        "green"         # Class 2: green
    ]
    
    # Function to create the prediction grid
    def create_pred_grid(probabilities, selected_channels_length, num_windows, threshold):
        pred_grid = np.full((selected_channels_length, num_windows), -1) 
        for idx in range(selected_channels_length):
            for w in range(num_windows):
                max_prob = np.max(probabilities[idx, w, :])  
                if max_prob >= threshold:
                    dominant_class = np.argmax(probabilities[idx, w, :])
                    pred_grid[idx, w] = dominant_class
        return pred_grid

    # Function to create custom colormap
    def create_custom_cmap(pred_grid):
        unique_values = np.unique(pred_grid)  
        filtered_colors = [base_colors[i] for i in unique_values + 1] 
        custom_cmap = ListedColormap(filtered_colors)
        return custom_cmap

    fig, axs = plt.subplots(3, 3, figsize=(10, 5), 
                             gridspec_kw={'height_ratios': [5, 1, 1], 'wspace': 0.05, 'hspace': 0.0},
                             sharex=True, sharey='row')
    
    # Extracting data dimensions and initializing variables
    num_windows = probabilities.shape[1]
    num_channels = data.shape[0]
    time_steps = data.shape[1]
    selected_channels_length = num_channels
    window_step_duration = window_duration - overlap_duration

    # Data normalization
    data_normalized = normalize(data)
    mean_signal = min_max_normalize(data_normalized[channel_idx, :])

    # Create prediction grids and colormaps
    pred_grid_no_grammar = create_pred_grid(probabilities, selected_channels_length, num_windows, threshold)
    custom_cmap_no_grammar = create_custom_cmap(pred_grid_no_grammar)
    pred_grid_grammar = create_pred_grid(probabilities_grammar, selected_channels_length, num_windows, threshold)
    custom_cmap_grammar = create_custom_cmap(pred_grid_grammar)

    # Compute the spectrogram of the mean signal
    frequencies, times, Sxx = spectrogram(mean_signal, fs=fsamp, nfft=1024, nperseg=600, noverlap=200)

    # Plotting the data and predictions
    im1 = axs[0, 0].imshow(
        data_normalized,
        cmap="seismic",
        vmin=-1,
        vmax=1,
        aspect="auto",
        extent=[0, time_steps / fsamp, num_channels, 0],
        interpolation="none",
    )
    axs[0, 0].set_ylabel('Channels')
    axs[0, 0].set_title("DAS record", fontsize=12, fontweight='bold')

    axs[0, 1].imshow(
        data_normalized,
        cmap="seismic",
        aspect="auto",
        extent=[0, time_steps / fsamp, num_channels, 0],
        interpolation="none"
    )
    time_extent = [0, time_steps / fsamp, num_channels, 0]
    axs[0, 1].imshow(pred_grid_no_grammar, cmap=custom_cmap_no_grammar, aspect="auto", extent=time_extent, interpolation="none", alpha=0.7)
    axs[0, 1].set_title("No grammar", fontsize=12, fontweight='bold')

    axs[0, 2].imshow(
        data_normalized,
        cmap="seismic",
        aspect="auto",
        extent=[0, time_steps / fsamp, num_channels, 0],
        interpolation="none",
    )
    axs[0, 2].imshow(pred_grid_grammar, cmap=custom_cmap_grammar, aspect="auto", extent=time_extent, interpolation="none", alpha=0.7)
    axs[0, 2].set_title("Grammar", fontsize=12, fontweight='bold')

    # Plot time-domain signals
    for j in range(3):
        axs[1, j].plot(np.arange(time_steps) / fsamp, mean_signal, color='black', lw=1)
        axs[1, j].set_xlabel('Time (s)')
    
        # Plot spectrograms
        im2 = axs[2, j].imshow(
            10 * np.log10(Sxx),
            cmap='jet',
            aspect='auto',
            extent=[0, time_steps / fsamp, frequencies[-1], frequencies[0]], 
            interpolation='none',
            vmin=-100,
            vmax=0
        )
        axs[2, j].set_ylim(1, 20)
        axs[2, j].set_yticks([5, 15])

    axs[2, 0].set_ylabel('Frequency\n(Hz)')

    # Highlight regions where probability exceeds threshold
    avg_prob = (probabilities[channel_idx, :, :])
    above_threshold = avg_prob[:, 2] >= threshold
    start_indices, end_indices = [], []
    is_in_region = False
    for j in range(len(above_threshold)):
        if above_threshold[j] and not is_in_region:
            start_indices.append(j)
            is_in_region = True
        elif not above_threshold[j] and is_in_region:
            end_indices.append(j - 1)
            is_in_region = False
    if is_in_region:
        end_indices.append(len(above_threshold) - 1)
    
    # Highlight in the plots
    for w in range(len(above_threshold)):
        if above_threshold[w]:
            start = w * window_step_duration
            end = start + window_duration
            axs[1, 1].axvspan(start, end, color='green', alpha=0.3)
    
    avg_prob_grammar = np.mean(probabilities_grammar, axis=0)
    above_threshold_grammar = avg_prob_grammar[:, 2] >= threshold
    start_indices_grammar, end_indices_grammar = [], []
    is_in_region_grammar = False
    for j in range(len(above_threshold_grammar)):
        if above_threshold_grammar[j] and not is_in_region_grammar:
            start_indices_grammar.append(j)
            is_in_region_grammar = True
        elif not above_threshold_grammar[j] and is_in_region_grammar:
            end_indices_grammar.append(j - 1)
            is_in_region_grammar = False
    if is_in_region_grammar:
        end_indices_grammar.append(len(above_threshold_grammar) - 1)
    
    for w in range(len(above_threshold_grammar)):
        if above_threshold_grammar[w]:
            start = w * window_step_duration
            end = start + window_duration
            axs[1, 2].axvspan(start, end, color='green', alpha=0.3)

    # Add colorbars
    cbar1 = fig.colorbar(im1, ax=axs[:2], orientation='vertical', fraction=0.02, pad=0.02, extend='both')
    cbar1.set_label('Normalized Strain Rate')

    cbar2 = fig.colorbar(im2, ax=axs[2], orientation='vertical', fraction=0.02, pad=0.02, aspect=3)
    cbar2.set_label('dB')

    # Save the figure
    plot_filename = os.path.join(output_folder, f'{event_id}_grammar_vs.png')
    plt.savefig(plot_filename)

    print(f"Plot saved at: {plot_filename}")


def plot_das_bi_argparse(data, probabilities, event_id='plot', save_path='./plots', window_duration=6, overlap_duration=1.2, threshold=0.9, fsamp=100, channel_idx=0):
    """
    Plots a visualization of DAS data and the RNN-DAS model predictions and saves the plot to a given directory.
    
    Parameters:
    data (numpy.ndarray): A 2D array of shape (num_channels, time_steps) representing the DAS data.
    probabilities (numpy.ndarray): A 3D array of shape (num_channels, num_windows, num_classes) representing the predicted probabilities for each window and channel.
    save_path (str): Directory path to save the plot (default is './plots').
    window_duration (float): Duration of each window in seconds (default is 6 seconds).
    overlap_duration (float): Duration of overlap between windows in seconds (default is 1.2 seconds).
    threshold (float): Threshold probability for classifying a window (default is 0.9).
    fsamp (int): Sampling frequency (default is 100 Hz).
    channel_idx (int): Index of the channel to plot in the time-domain (default is 0).
    
    Output:
    Saves the figure to the specified directory.
    """

    # Number of channels and time steps in the data
    num_channels = data.shape[0]
    time_steps = data.shape[1]
    time_step_duration = window_duration - overlap_duration

    # Select all channels
    selected_channels = np.arange(0, num_channels, 1)
    data_selected = data[selected_channels, :]

    # Create the specified directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # Create a figure with 3x3 subplots
    fig, axs = plt.subplots(3, 2, figsize=(12, 8),
                             gridspec_kw={'height_ratios': [5, 1, 1], 'wspace': 0.05, 'hspace': 0.0},
                             sharex=True, sharey='row')

    # Normalize data
    normalize = lambda x: (x - np.mean(x, axis=-1, keepdims=True)) / np.std(x, axis=-1, keepdims=True)
    min_max_normalize = lambda x: 2 * (x - np.min(x)) / (np.max(x) - np.min(x)) - 1
    data_normalized = normalize(data_selected)
    channel_signal = min_max_normalize(data[channel_idx, :].reshape(1, -1)).flatten()

    # Custom colormap for class labels
    base_colors = [
        (1, 1, 1, 0),  # -1: transparent
        "red",         # 0: red
        "gray",        # 1: gray
        "green"        # 2: green
    ]

    # Create a prediction grid
    num_windows = probabilities.shape[1]
    pred_grid = np.full((len(selected_channels), num_windows), -1)
    for idx, channel in enumerate(selected_channels):
        for w in range(num_windows):
            max_prob = np.max(probabilities[channel, w, :])
            if max_prob >= threshold:
                dominant_class = np.argmax(probabilities[channel, w, :])
                pred_grid[idx, w] = dominant_class

    # Filtered colors for unique predictions
    unique_values = np.unique(pred_grid)
    filtered_colors = [base_colors[i] for i in unique_values + 1]
    custom_cmap = ListedColormap(filtered_colors)

    # Plot 1: Original DAS Data
    axs[0, 0].imshow(data_normalized, cmap="seismic", vmin=-1, vmax=1, aspect="auto",
                     extent=[0, time_steps / fsamp, selected_channels[-1], selected_channels[0]],
                     interpolation="none")
    axs[0, 0].set_ylabel('Spatial channels')
    axs[0, 0].set_title('Original DAS', fontweight='bold')

    # Plot 2: RNN-DAS Predictions
    axs[0, 1].imshow(data_normalized, cmap="seismic", vmin=-1, vmax=1, aspect="auto",
                     extent=[0, time_steps / fsamp, selected_channels[-1], selected_channels[0]],
                     interpolation="none")
    time_extent = [0, time_steps / fsamp, num_channels, 0]
    axs[0, 1].imshow(pred_grid, cmap=custom_cmap, aspect="auto", extent=time_extent, interpolation="none")
    axs[0, 1].set_title('RNN-DAS', fontweight='bold')

    # Plot signal of selected channel in time-domain
    for j, ax in enumerate(axs[1, :]):
        ax.plot(np.arange(time_steps) / fsamp, channel_signal, color='black', lw=1)
        ax.set_xlabel('Time (s)')
        if j == 1:  # RNN-DAS probabilities for the selected channel
            channel_probabilities = probabilities[channel_idx, :, 2]
            above_threshold = channel_probabilities >= threshold
            for w in range(len(above_threshold)):
                if above_threshold[w]:
                    start = w * time_step_duration
                    end = start + window_duration
                    ax.axvspan(start, end, color='green', linestyle='-', linewidth=1, alpha=0.3)

    # Add color bar for normalized strain rate

    cbar = fig.colorbar(axs[0, 0].images[0], ax=axs[:2, :].ravel().tolist(), orientation='vertical', fraction=0.02, pad=0.02, extend='both')
    cbar.set_label('Normalized Strain Rate')
    # Compute and plot spectrogram for the third row of subplots
    frequencies, times, Sxx = spectrogram(channel_signal, fs=fsamp, nfft=1024, nperseg=600, noverlap=200)

    for j in range(2):
        im2=axs[2, j].imshow(
            10 * np.log10(Sxx),
            cmap='jet',
            aspect='auto',
            extent=[0, time_steps / fsamp, frequencies[-1], frequencies[0]], 
            interpolation='none',
            vmin=-100,
            vmax=0
        )
        axs[2, j].set_ylim(1, 20)
        axs[2, j].set_yticks([5, 15])
        axs[2, j].set_xlabel('Time (s)')
    
    axs[2, 0].set_ylabel('Frequency\n(Hz)')
    cbar2 = fig.colorbar(im2, ax=axs[2, :].ravel().tolist(), orientation='vertical', fraction=0.02, pad=0.02, aspect=3)
    cbar2.set_label('dB')

    # Save the figure to the specified path
    plot_filename = os.path.join(save_path, f'{event_id}_bi.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')  # Save the plot with high quality
    print(f"Plot saved at: {plot_filename}")

