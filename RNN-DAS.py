import os
import numpy as np
import pandas as pd
import h5py
from datetime import datetime
import obspy
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import sys
scripts_path = os.path.abspath(os.path.join(".", "scripts"))
sys.path.append(scripts_path)
from Pre_processing import read_fileids, process_H, load_h5
from model_RNN_DAS import load_model
from Features import features
from run_RNN_das import run_model, run_batch_model
from grammar import grammar
from plots import plot_das_argparse, plot_das_bi_argparse, plot_das_grammar_argparse, plot_stacked_traces_normalized_argparse
from picks import write_pickle, detect_phases
from picks import extract_events_to_mseed
import argparse

def parse_arguments():
    """
    Parses command-line arguments for running the RNN-DAS model.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Run the RNN-DAS model with specified configurations.")
    
    # Model and normalization paths
    parser.add_argument("--model_path", type=str, default="./model/RNN-DAS_1150", 
                        help="Path to the trained RNN-DAS model.")
    parser.add_argument("--normalization_path", type=str, default="./model/Normalization_RNN-DAS_1150.txt",
                        help="Path to the normalization file.")
    
    # Input files
    parser.add_argument("--files_id", type=str, default="./files.txt",
                        help="Path to the text file containing the list of file IDs.")
    parser.add_argument("--data_folder", type=str, default="./data_to_predict",
                    help="Path to the data files folder.")

    # Pre-processing
    parser.add_argument("--pre_processing", type=bool, default=True,
                        help="Apply pre-processing to the data before running the model (recommended).")

    parser.add_argument("--dt", type=float, default=0.01,
                        help="Sampling time (in seconds)")
    parser.add_argument("--dx", type=float, default=10,
                        help="Sampling spacing (in meters)")
    # Plot options
    parser.add_argument("--plot_das", type=bool, default=False,
                        help="Plot DAS data before running the model.")
    parser.add_argument("--plot_das_bi", type=bool, default=True,
                        help="Plot DAS data with the grammar-based predictions.")
    parser.add_argument("--plot_das_grammar", type=bool, default=False,
                        help="Plot grammar-based vs raw DAS predictions.")
    parser.add_argument("--plot_stream_stack", type=bool, default=False,
                        help="Plot stacked stream traces.")
    parser.add_argument("--plot_threshold", type=float, default=2/3,
                        help="Threshold value to consider a predominant probability class per frame when plotting.")
    parser.add_argument("--plot_channel", type=int, default=50,
                       help="DAS channel to plot in the straingram and spectrogram subplots.")
    
    # Grammar processing
    parser.add_argument("--grammar", type=bool, default=True,
                        help="Enable grammar-based event detection.")
    
    # Saving options
    parser.add_argument("--predictions_saved", type=bool, default=False,
                        help="Save model predictions.")
    parser.add_argument("--probabilities_saved", type=bool, default=False,
                        help="Save prediction probabilities.")
    parser.add_argument("--save_results_csv", type=bool, default=True,
                        help="Save results in CSV format.")
    parser.add_argument("--save_results_mseed", type=bool, default=True,
                        help="Save waveform results in MiniSEED format.")
    parser.add_argument("--grammar_save", type=bool, default=True,
                        help="Save grammar results.")

    # Grammar parameters
    parser.add_argument("--grammar_parameters_threshold", type=float, default=2/3,
                        help="Threshold for grammar-based event detection.")
    parser.add_argument("--grammar_parameter_threshold_channels", type=float, default=0.5,
                        help="Channel-based grammar threshold.")
    parser.add_argument("--grammar_parameter_interval_size", type=int, default=10,
                        help="Interval size for grammar-based analysis.")
    parser.add_argument("--grammar_parameter_trigger_on", type=float, default=0.9,
                        help="Trigger-on threshold for grammar-based detection.")
    parser.add_argument("--grammar_parameter_trigger_off", type=float, default=0.05,
                        help="Trigger-off threshold for grammar-based detection.")
    
    # MSEED parameters
    parser.add_argument("--threshold_mseed", type=float, default=0.9, 
                        help="Threshold for trace detection")
    parser.add_argument("--network_code", type=str, default="LP", 
                        help="Network code (default: 'LP')")
    parser.add_argument("--station_prefix", type=str, default="DAS", 
                        help="Station prefix (default: 'DAS')")
    parser.add_argument("--location_code", type=str, default="XX", 
                        help="Location code (default: 'XX')")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    print(args)  # Print parsed arguments for debugging
    print("\n")
    print("-" * 50)
    print("Loading the RNN-DAS model...")
    # Loading the model
    n_inputs = 144
    n_hidden = 256
    n_layers = 1
    n_classes = 3
    cell_type = "LSTM"
    filepath_normalization= args.normalization_path
    model_RNN_DAS=load_model(args.model_path, n_inputs, n_hidden, n_layers, n_classes, cell_type)
    print("Done")
    #Reading the data 
    filesids_path=args.files_id
    filesids=read_fileids(filesids_path)
    print("-" * 50)
    print("List of files to analyze:\n")
    print(filesids)
    print("-" * 50)

    for fileid in filesids:
        print("*" * 50)
        print(f"Running RNN-DAS on {fileid}")
        print("Reading the DAS data...")
        H, dt, dx, start_time = load_h5(fileid, f'{args.data_folder}')
        print("Done")
        #Preprocessing
        if args.pre_processing:
            print("Beginning preprocessing...")
            H=process_H(H[:, :], dt) 
        print("Done")
        #Ploting the DAS?
        if args.plot_das:
            print("Plotting the DAS record...")
            plot_das_argparse(data=H, event_id=fileid, channel_idx=args.plot_channel, fsamp=1/dt)
        # Computing the features of the data
        features_dataloader=features(H[:, :], filepath_normalization) # normalization values
        # Making the predictions and raw probabilities
        predictions, probabilities = run_model(model_RNN_DAS, features_dataloader)
        #Saving the predictions or raw probabilities?
        if args.predictions_saved:
            write_pickle(predictions, os.path.join("RNN-DAS_predictions", f"{fileid}_predictions"))
        if args.probabilities_saved:
            write_pickle(predictions, os.path.join("RNN-DAS_predictions", f"{fileid}_probabilities"))
        #Computing the grammar function
        if args.grammar:
            print("Applying the grammar function to raw probabilities...")
            probabilities_grammar = grammar(probabilities=probabilities, threshold=args.grammar_parameters_threshold, 
                                            threshold_channels=args.grammar_parameter_threshold_channels, 
                                            interval_size=args.grammar_parameter_interval_size, 
                                            threshold_trigger_on=args.grammar_parameter_trigger_on, 
                                            threshold_trigger_off=args.grammar_parameter_trigger_off)
            print("Done")
        #Saving the grammar probabilities?
        if args.grammar_save:
            write_pickle(probabilities_grammar, os.path.join("RNN-DAS_predictions", f"{fileid}_probabilities_grammar"))       
        #Plot das_bi?
        if args.plot_das_bi:
            print("Plotting the DAS record (PLOT_DAS_BI)...")
            plot_das_bi_argparse(data=H, probabilities=probabilities_grammar, event_id=fileid, threshold=args.plot_threshold, channel_idx=args.plot_channel, fsamp=1/dt)
        #Plot grammar_vs_no_grammar?
        if args.plot_das_grammar:
            print("Plotting the DAS record (PLOT_DAS_GRAMMAR_VS)...")
            plot_das_grammar_argparse(data=H, probabilities=probabilities, event_id=fileid, probabilities_grammar=probabilities_grammar, threshold=args.plot_threshold, channel_idx=args.plot_channel, fsamp=1/dt)
        #Creating the csv of detected events
        if args.save_results_csv:
            print("Saving the detected events in a csv...")
            detect_phases(output_csv_name=fileid, start_date=start_time, probabilities=probabilities_grammar, fsamp=1/dt, probability_threshold=args.plot_threshold)
            results=pd.read_csv(os.path.join(".", "RNN-DAS_picks", f"{fileid}_RNN-DAS.csv"))
        #Creating the mseed file with the data to predict
        if args.save_results_mseed:
            print("Saving the detected events in a mseed...")
            stream = extract_events_to_mseed(H=H, df=results, fsamp=1/dt, threshold=args.threshold_mseed, output_file=fileid, network_code=args.network_code, station_prefix=args.station_prefix, location_code=args.location_code)
        #Stack plot of selected traces?
        if args.plot_stream_stack:
            print("Plotting the DAS record (mseed traces)...")
            plot_stacked_traces_normalized_argparse(stream=stream, event_id=fileid)











