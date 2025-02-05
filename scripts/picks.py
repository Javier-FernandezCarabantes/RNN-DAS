import numpy as np
import os
import pickle



def write_pickle(data, filename):
    # Check if the directory exists, if not, create it
    dir_name = os.path.dirname(filename)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    # Save the data to the pickle file
    with open(filename, 'wb') as file:
        data = np.round(data.reshape(-1, data.shape[-1]), decimals=3)       
        pickle.dump(data, file)

def detect_phases(output_csv_name, start_date, probabilities, fsamp=100, probability_threshold=0.6, window_duration=6, overlap_duration=1.2):
    """
    Detects events in signals based on probabilities and save the results in a CSV file.
    Parameters:
    - output_csv_name (str): Name of the output CSV file.
    - start_date (str): Start date and time in string format.
    - probabilities (np.ndarray): Array containing probability values for event detection.
    - fsamp (int, optional): Sampling frequency of the signal in Hz (default is 100).
    - probability_threshold (float, optional): Threshold for detecting an event (default is 0.6).
    - window_duration (float, optional): Duration of the analysis window in seconds (default is 6).
    - overlap_duration (float, optional): Overlap duration between windows in seconds (default is 1.2).
    Returns:
    None
        This function does not return anything but saves the detected phase picks in a CSV file in the `RNN-DAS_picks` directory.
    """

    import os
    import pandas as pd
    import numpy as np
    from datetime import timedelta

    step_duration = window_duration - overlap_duration  # Step duration between windows

    num_channels, num_windows, num_classes = probabilities.shape
    predictions = np.argmax(probabilities, axis=2)
    results = []

    for channel_index in range(num_channels):
        current_window = 0
        last_event_end = -float('inf')  # Index of the end of the last detected event for the channel

        while current_window < num_windows:
            # Look for the first window where class 2 (VT) is detected for this channel
            if (predictions[channel_index, current_window] == 2 and
                probabilities[channel_index, current_window, 2] >= probability_threshold):
                start_window = current_window

                # Verify that subsequent windows also belong to class 2
                while (current_window < num_windows and
                       predictions[channel_index, current_window] == 2 and
                       probabilities[channel_index, current_window, 2] >= probability_threshold):
                    current_window += 1

                # Save the information for the first window of the detected group
                if start_window > last_event_end + 2:  # At least two windows of separation from the last event
                    phase_time_index = start_window * step_duration * fsamp
                    phase_time = start_date + timedelta(seconds=start_window * step_duration)

                    # Detect the last window of the group (Coda)
                    end_window = current_window - 1
                    coda_time_index = end_window * step_duration * fsamp + step_duration * fsamp

                    # Calculate the average phase_score for the event
                    phase_scores = probabilities[channel_index, start_window:end_window + 1, 2]
                    average_phase_score = np.median(phase_scores)

                    results.append({
                        "channel_index": channel_index,
                        "event_index": int(phase_time_index),
                        "event_time": phase_time.isoformat(),  
                        "event_score": round(average_phase_score, 3),
                        "coda_index": int(coda_time_index),
                        "coda_time": (phase_time + timedelta(seconds=((coda_time_index-phase_time_index) / fsamp))).isoformat(),  
                        "windows": list(range(start_window, end_window + 1)),
                    })

                    # Update the index of the last detected event
                    last_event_end = end_window

            # Move to the next group
            current_window += 1

    # Create output folder if it does not exist
    output_folder = "RNN-DAS_picks"
    os.makedirs(output_folder, exist_ok=True)

    # Create the DataFrame and save to CSV
    output_path = os.path.join(output_folder, output_csv_name)
    df = pd.DataFrame(results, columns=["channel_index", "event_index", "event_time", "event_score", "coda_index", "coda_time", "windows"])
    df.to_csv(f"{output_path}_RNN-DAS.csv", index=False)

    print(f"Events detected and saved in: {output_path}")
    print(f"Total number of events detected: {len(df)}")


from obspy import Stream, Trace
import numpy as np

def extract_events_to_mseed(H, df, fsamp, threshold=0.9, output_file="event_data.mseed",
                            network_code="XX", station_prefix="DAS", location_code="00"):
    """
    Extracts events from the data matrix H based on detected events in df and saves them in MiniSEED format.

    Parameters:
        H : numpy.ndarray
            Data matrix with dimensions (channels, time).
        df : pandas.DataFrame
            DataFrame containing detected events (columns: channel_index, event_index, coda_index, event_score, event_time).
        fsamp : int
            Sampling frequency (Hz).
        threshold : float, optional
            Minimum event_score threshold to consider an event. Default is 0.9.
        output_file : str, optional
            Name of the output MiniSEED file. Default is "event_data.mseed".
        network_code : str, optional
            Network code for MiniSEED metadata. Default is "XX".
        station_prefix : str, optional
            Prefix for station names (e.g., "DAS"). Default is "DAS".
        location_code : str, optional
            Location code for MiniSEED metadata. Default is "00".

    Returns:
        obspy.Stream
            Stream object containing all extracted events as individual traces.
    """

    stream = Stream()

    for _, row in df.iterrows():
        if row["event_score"] >= threshold:
            channel_index = row["channel_index"]
            start_sample = int(row["event_index"])
            end_sample = int(row["coda_index"])

            # Extract the corresponding waveform
            data_segment = H[channel_index, start_sample:end_sample]

            # Create an ObsPy Trace object
            trace = Trace()
            trace.data = np.array(data_segment, dtype=np.float32)  # Ensure correct data type
            trace.stats.sampling_rate = fsamp
            trace.stats.network = network_code
            trace.stats.station = f"{station_prefix}_{channel_index}"  # Unique station name per channel
            trace.stats.location = location_code
            trace.stats.channel = f"{channel_index:04d}"  # DAS-specific channel code (DAS0001, DAS0002, etc.)
            trace.stats.starttime = row["event_time"]  # Start time of the event

            stream.append(trace)

    # Save all traces in a single MiniSEED file
    folder_path = "RNN-DAS_waveforms"
    os.makedirs(folder_path, exist_ok=True) 
    output_path = os.path.join(folder_path, f"{output_file}.mseed")
    stream.write(output_path, format="MSEED")
    print(f"Saved extracted events to {output_file}")

    return stream