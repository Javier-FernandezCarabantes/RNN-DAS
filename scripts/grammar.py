
import numpy as np

def grammar(probabilities, threshold=2/3, threshold_channels=0.5, interval_size=100, threshold_trigger_on=0.9, threshold_trigger_off=0.1):
    """
    Adjusts probabilities based on the predominant class in channel intervals,
    window by window, ignoring channels where noise_2 (class 1) is predominant.
    Also, activates a "trigger" for the VT class if a threshold is exceeded and applies VT class propagation.

    Parameters:
    probabilities : numpy.ndarray
        Probability matrix with dimensions (num_channels, num_windows, num_classes).
    threshold : float
        Minimum fraction of channels in the interval that must predict the same class to apply the adjustment.
    threshold_channels : float
        Threshold of channels in each interval that must predict the same class to change the probabilities.
    interval_size : int
        Size of each channel interval (default 100).
    threshold_trigger_on : float
        Threshold to activate the VT class "trigger" (default 0.9).
    threshold_trigger_off : float
        Threshold to deactivate the VT class "trigger" (default 0.1).

    Returns:
    numpy.ndarray
        Probability matrix adjusted according to the grammar.
    """
    # Create a copy of probabilities to avoid modifying the original
    probabilities = probabilities.copy()
    
    num_channels, num_windows, num_classes = probabilities.shape
    num_intervals = int(np.ceil(num_channels / interval_size))

    for interval in range(num_intervals):  # Iterate over blocks of channels
        start_channel = interval * interval_size
        end_channel = min((interval + 1) * interval_size, num_channels)

        for channel in range(start_channel, end_channel):
            trigger_active = False  # Variable to control if the "trigger" is active

            # Find the first window where the VT class exceeds the trigger_on threshold
            for w in range(num_windows):
                if probabilities[channel, w, 2] >= threshold_trigger_on:  # If the VT class exceeds the trigger_on threshold
                    trigger_active = True
                    trigger_on_window = w  # Save the activation window
                    break  # Only need the first window that exceeds the threshold

            if trigger_active:
                # Propagate VT and noise_2 in the following windows until the trigger_off threshold is met
                for future_w in range(trigger_on_window, num_windows):
                    if probabilities[channel, future_w, 2] < threshold_trigger_off:
                        # Stop propagation when the VT probability falls below the trigger_off threshold
                        trigger_active = False
                        break

                    # Change to dominant VT if it is not already
                    current_prediction = np.argmax(probabilities[channel, future_w, :])
                    if current_prediction != 2:  # It's not VT
                        probabilities[channel, future_w, 2] = threshold  # Change to dominant VT
                        
                        # Redistribute the remaining probabilities among the other classes
                        remaining_classes = np.arange(num_classes) != 2  # VT is class 2
                        remaining_sum = 1 - threshold  # The remaining to redistribute
                        remaining_probabilities_sum = probabilities[channel, future_w, remaining_classes].sum()
                        if remaining_probabilities_sum > 0:
                            probabilities[channel, future_w, remaining_classes] = probabilities[channel, future_w, remaining_classes] / remaining_probabilities_sum * remaining_sum
                # Once trigger_off is met, restart the search for trigger_on
                if not trigger_active:
                    continue
                
    # Second part: applying the original grammar (without changes to noise_2)
    for interval in range(num_intervals):  # Iterate over blocks of channels
        start_channel = interval * interval_size
        end_channel = min((interval + 1) * interval_size, num_channels)

        for w in range(num_windows):  # Iterate over each time window
            # Extract the probabilities and predictions from the current block for this window
            block_probabilities = probabilities[start_channel:end_channel, w, :]
            block_predictions = np.argmax(block_probabilities, axis=1)
            # Identify channels where noise_2 (class 1) is predominant
            is_ruido_2 = block_predictions == 1

            # Exclude noise_2 from the predominance analysis
            valid_predictions = block_predictions[~is_ruido_2]

            if len(valid_predictions) == 0:
                # If only noise_2 exists in this block for this window, move to the next window
                continue  

            # Count the classes noise_1 (0) and VT (2) in the interval
            class_counts = np.bincount(valid_predictions, minlength=num_classes)

            # Evaluate if a class exceeds the threshold
            for target_class in [0, 2]:  # Only noise_1 and VT
                if class_counts[target_class] >= np.round(threshold_channels * len(valid_predictions), 0):
                    # Predominant class identified
                    for channel in range(start_channel, end_channel):
                        if is_ruido_2[channel - start_channel]:
                            # Do not modify channels dominated by noise_2
                            continue
                        
                        # Extract the current probabilities of the channel for this window
                        current_probabilities = probabilities[channel, w, :]
                        current_prediction = np.argmax(current_probabilities)

                        if current_prediction != target_class:
                            # Adjust the probability of the target_class to the threshold
                            probabilities[channel, w, target_class] = threshold
                            
                            # Redistribute the remaining probabilities among the other classes
                            remaining_classes = np.arange(num_classes) != target_class
                            remaining_sum = 1 - threshold  # The remaining to redistribute
                            # Avoid division by zero
                            remaining_probabilities_sum = probabilities[channel, w, remaining_classes].sum()
                            if remaining_probabilities_sum > 0:
                                probabilities[channel, w, remaining_classes] = probabilities[channel, w, remaining_classes] / remaining_probabilities_sum * remaining_sum
                        elif current_probabilities[target_class] < threshold:
                            # Adjust the probability of the predominant class to the threshold
                            diff = threshold - current_probabilities[target_class]
                            probabilities[channel, w, target_class] = threshold

                            # Reduce the other probabilities proportionally to maintain the sum at 1
                            remaining_classes = np.arange(num_classes) != target_class
                            remaining_probabilities_sum = probabilities[channel, w, remaining_classes].sum()
                            if remaining_probabilities_sum > 0:
                                probabilities[channel, w, remaining_classes] -= probabilities[channel, w, remaining_classes] / remaining_probabilities_sum * diff
                    break  # No need to keep searching once the criterion is met
    return probabilities
