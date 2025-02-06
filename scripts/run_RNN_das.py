import torch
import torch.nn as nn
import numpy as np
import os
import h5py

def load_h5(name_id, path="H5_data"):
    file_path = os.path.join(path, f"{name_id}.h5")
    
    with h5py.File(file_path, "r") as fp:
        data = np.array(fp["data"])
        dt_s = fp["data"].attrs["dt_s"]
        dx_m = fp["data"].attrs["dx_m"]
    return data, dt_s, dx_m

def predict(model, dataloader):
    all_predictions = []
    all_probabilities = []

    softmax = nn.Softmax(dim=2)  # Convert logits to probabilities

    with torch.no_grad():  # Disable gradient computation
        for batch in dataloader:
            # Unpack the batch
            batch_x = batch[0]  
            # Move batch to GPU if available
            batch_x = batch_x.cuda() if torch.cuda.is_available() else batch_x
            
            # Make predictions
            outputs, _ = model(batch_x)
            # Process outputs to obtain probabilities and predictions
            probabilities = torch.softmax(outputs, dim=2)  
            predictions = torch.argmax(probabilities, dim=2)

            all_predictions.append(predictions.cpu().numpy())  # Move predictions to CPU and convert to numpy
            all_probabilities.append(probabilities.cpu().numpy())  # Move probabilities to CPU and convert to numpy
            
    all_predictions = np.concatenate(all_predictions)
    all_probabilities = np.concatenate(all_probabilities)
    
    return all_predictions, all_probabilities

def run_model(model, data):
    """
    Function to execute the RNN-DAS model with the provided data.
    
    Args:
        model: The trained RNN-DAS model.
        data: The input dataloader to be processed.
    
    Returns:
        predictions, probabilities: Model outputs.
    """
    predictions, probabilities = predict(model, data)  
    return predictions, probabilities

def run_batch_model(model, data, fileids):
    """
    Function to execute the RNN-DAS model with the provided data for various samples.
    
    Args:
        model: The trained RNN-DAS model.
        data: The input data to be processed.
        fileids: List of events ids
    
    Returns:
        predictions, probabilities: Model outputs.
    """
    for fileid in fileids:
        predictions, probabilities = predict(model, data)  
        return predictions, probabilities


