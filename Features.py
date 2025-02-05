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
    Crea los features usando cálculo paralelo para múltiples canales simultáneamente.
    """
    n, m = H.shape
    num_procesos = os.cpu_count()  # Número de núcleos disponibles
    grupos_canales = split_channels_features(n, num_procesos)

    # Procesar todos los grupos de canales a la vez
    print(">> Processing the signal...")
    results = Parallel(n_jobs=num_procesos)( 
        delayed(process_channel_group_features)(H,100,datetime(2020,9,22,19,34,16),1/100, window_duration, overlap_duration, inicio, fin, mean, std) 
        for inicio, fin in grupos_canales
    )
    results = [res.numpy() for res in results]
    # Concatenar los resultados de todos los procesos
    features = np.concatenate(results, axis=0)
    #print(features.shape)
    #print(features.shape) #[batch, time_step, features]
    return features


def calculate_LFB(da, srate, nsamp, stime, dt, sp, window_duration=4, overlap_duration=3.5, hop_length=1024, n_LFB=16):
    """
    Calcula una matriz de características LFB y sus derivadas.
    
    Parámetros:
    - da: Array de datos de entrada (2D, donde cada fila corresponde a un canal)
    - srate: Frecuencia de muestreo (Hz)
    - nsamp: Número total de muestras en los datos
    - stime: Tiempo inicial (datetime.datetime)
    - dt: Resolución temporal entre muestras (segundos)
    - sp: Índice del canal de interés
    - window_duration: Duración de cada ventana (segundos)
    - overlap_duration: Duración del solapamiento entre ventanas (segundos)
    - hop_length: Longitud del salto para la FFT
    - n_LFB: Número de filtros log-Mel a usar
    
    Retorna:
    - LFB_matrix_vals: Matriz de características LFB (n_LFB x n_frames)
    - deltas_LFB: Matriz de las primeras derivadas de LFB
    - deltas_deltas_LFB: Matriz de las segundas derivadas de LFB
    """
    # Calcular el número de puntos en cada ventana y el solapamiento
    points_per_frame = int(window_duration * srate)
    overlap_points = int(overlap_duration * srate)
    if overlap_points == 0:
        overlap_points = points_per_frame

    # Crear un array de tiempos para el eje x
    times = np.arange(0, nsamp - points_per_frame + 1, overlap_points) * dt
    times = [stime + timedelta(seconds=t) for t in times]  # Convertir tiempos al formato datetime

    # Inicializar la matriz para almacenar los coeficientes LFB
    LFB_matrix_vals = np.zeros((len(times), n_LFB))

    # Iterar sobre las ventanas y calcular los coeficientes LFB
    for i, index in enumerate(range(0, nsamp - points_per_frame + 1, overlap_points)):
        start_index = index
        end_index = start_index + points_per_frame
        frame_data = da[:, start_index:end_index]

        # Calcular los coeficientes LFB para la ventana actual
        if overlap_duration == 0:
            overlap_duration = window_duration
        LFB = np.float32(logfbank(1 + frame_data[sp], samplerate=srate, 
                                  winlen=window_duration, winstep=overlap_duration, 
                                  nfilt=n_LFB, nfft=hop_length, lowfreq=0, 
                                  highfreq=None, preemph=0.97))
        
        # Almacenar los coeficientes en la matriz
        LFB_matrix_vals[i, :] = LFB
    LFB_matrix_vals=np.array(LFB_matrix_vals)

    # Calcular las derivadas de los coeficientes LFB
    deltas_LFB = delta(LFB_matrix_vals, 2)
    deltas_deltas_LFB = delta(deltas_LFB, 2)

    # Transponer las matrices para que las dimensiones sean n_LFB x n_frames
    LFB_matrix_vals = LFB_matrix_vals.T
    deltas_LFB = deltas_LFB.T
    deltas_deltas_LFB = deltas_deltas_LFB.T

    return LFB_matrix_vals, deltas_LFB, deltas_deltas_LFB



def process_channel_group_features(H, srate, stime, dt, window_duration, overlap_duration, inicio, fin, mean, std):
    """
    Procesa un grupo de canales y devuelve las características.
    
    Parámetros:
    - H: Matriz de datos (2D, [n_canales, n_muestras])
    - srate: Frecuencia de muestreo (Hz)
    - stime: Tiempo inicial (datetime.datetime)
    - dt: Resolución temporal entre muestras (segundos)
    - window_duration: Duración de cada ventana (segundos)
    - overlap_duration: Duración del solapamiento entre ventanas (segundos)
    - inicio: Índice del primer canal a procesar
    - fin: Índice del último canal a procesar (exclusivo)
    - mean: Tensor de medias para normalización
    - std: Tensor de desviaciones estándar para normalización

    Retorna:
    - features: Tensor con las características (core, time_step, features)
    """
    overlap_duration_codigo_HDAS = window_duration - overlap_duration
    features = []

    # Asegurar que mean y std están en formato tensor
    mean = mean.clone().detach() if isinstance(mean, torch.Tensor) else torch.tensor(mean, dtype=torch.float32)
    std = std.clone().detach() if isinstance(std, torch.Tensor) else torch.tensor(std, dtype=torch.float32)
    mean = mean.unsqueeze(dim=0)  # [1, 144] para evitar problemas con la normalización
    std = std.unsqueeze(dim=0)  # [1, 144]
    print(std, mean)
    n, m = H.shape  # Dimensiones de los datos: n_canales x n_muestras

    for sp in range(inicio, fin):
        print(sp)
        # Procesar el canal anterior (si no es el primer canal)
        if sp == 0:
            lfb_anterior, lfb_delta_anterior, lfb_delta_delta_anterior = calculate_LFB(
                da=H, srate=srate, nsamp=m, stime=stime, dt=dt, sp=sp, window_duration=window_duration, overlap_duration=overlap_duration_codigo_HDAS
            ) 
        else:
            lfb_anterior, lfb_delta_anterior, lfb_delta_delta_anterior = calculate_LFB(
                da=H, srate=srate, nsamp=m, stime=stime, dt=dt, sp=sp-1, window_duration=window_duration, overlap_duration=overlap_duration_codigo_HDAS
            )

        # Procesar el canal actual
        lfb_actual, lfb_delta_actual, lfb_delta_delta_actual = calculate_LFB(
            da=H, srate=srate, nsamp=m, stime=stime, dt=dt, sp=sp, window_duration=window_duration, overlap_duration=overlap_duration_codigo_HDAS
        )

        # Procesar el canal siguiente (si no es el último canal)
        if sp == n - 1:
            lfb_siguiente, lfb_delta_siguiente, lfb_delta_delta_siguiente = calculate_LFB(
                da=H, srate=srate, nsamp=m, stime=stime, dt=dt, sp=sp, window_duration=window_duration, overlap_duration=overlap_duration_codigo_HDAS
            )
        else:
            lfb_siguiente, lfb_delta_siguiente, lfb_delta_delta_siguiente = calculate_LFB(
                da=H, srate=srate, nsamp=m, stime=stime, dt=dt, sp=sp+1, window_duration=window_duration, overlap_duration=overlap_duration_codigo_HDAS
            )

        # Concatenar características (anterior, actual, siguiente)
        feature = np.concatenate((
            lfb_anterior.T, lfb_delta_anterior.T, lfb_delta_delta_anterior.T,
            lfb_actual.T, lfb_delta_actual.T, lfb_delta_delta_actual.T,
            lfb_siguiente.T, lfb_delta_siguiente.T, lfb_delta_delta_siguiente.T
        ), axis=1)
        print(feature.shape)
        # Normalizar las características
        feature = torch.tensor(feature, dtype=torch.float32)
        feature = (feature - mean) / std

        # Almacenar las características del canal
        features.append(feature)

    # Convertir la lista de tensores en un único tensor
    features = torch.stack(features)  # Dimensiones: [core, time_step, features]

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
