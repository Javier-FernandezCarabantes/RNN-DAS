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

    softmax = nn.Softmax(dim=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    with torch.no_grad():
        for batch in dataloader:
            batch_x = batch[0]
            batch_x = batch_x.to(device)
            
            outputs, _ = model(batch_x)
            probabilities = torch.softmax(outputs, dim=2)
            predictions = torch.argmax(probabilities, dim=2)

            all_predictions.append(predictions.cpu().numpy())
            all_probabilities.append(probabilities.cpu().numpy())
            
    all_predictions = np.concatenate(all_predictions)
    all_probabilities = np.concatenate(all_probabilities)
    
    return all_predictions, all_probabilities

def run_model(model, data):
    predictions, probabilities = predict(model, data)  
    return predictions, probabilities

def run_batch_model(model, data, fileids):
    for fileid in fileids:
        predictions, probabilities = predict(model, data)  
        return predictions, probabilities

