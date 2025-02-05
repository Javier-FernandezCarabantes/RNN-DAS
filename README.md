# RNN-DAS: A New Deep Learning Approach for Detection and Real-Time Monitoring of Volcano-Tectonic Events Using Distributed Acoustic Sensing

## Overview
RNN-DAS is a novel Deep Learning model based on Recurrent Neural Networks (RNNs) with Long Short-Term Memory (LSTM) cells, designed for real-time Volcano-Seismic Signal Recognition (VSR) using Distributed Acoustic Sensing (DAS) measurements. The model was trained on a comprehensive dataset of Volcano-Tectonic (VT) events from the 2021 La Palma eruption, recorded by a High-fidelity submarine Distributed Acoustic Sensing array (HDAS) near the eruption site.

RNN-DAS detects VT events, analyzes their temporal evolution, and classifies their waveforms with approximately 97% accuracy. The model has demonstrated strong generalization capabilities for different time intervals and volcanoes, enabling continuous real-time seismic monitoring with minimal computational resources and retraining requirements.

## Features
- Detection and classification of VT events from DAS data.
- Utilizes frequency-based signal energy features to enhance spatial and temporal contextual information.
- High accuracy in detecting and classifying complete VT waveforms.
- Real-time processing capabilities for continuous monitoring.
- Generalizable to different volcanic environments with minimal retraining.

## Installation and Requirements
To run RNN-DAS, install the required dependencies:

```bash
pip install -r requirements.txt
```

Ensure you have the necessary Deep Learning libraries such as TensorFlow or PyTorch (depending on the implementation) and signal processing tools.

## Running the Model

### Command-Line Execution with Argparse
To run the model using command-line arguments, use the provided script with `argparse`. The script allows users to specify input parameters such as the dataset location, model configuration, and output options.

Example usage:

```bash
python run_model.py --input_data data/example_signal.mseed --output results/predictions.txt --model_checkpoint models/rnn_das_checkpoint.pth
```

For more details on available arguments, run:

```bash
python run_model.py --help
```

### Example Notebook: Running RNN-DAS on La Palma Data
An example Jupyter Notebook is provided to demonstrate how the model operates on real data from the 2021 La Palma eruption. This notebook walks through loading the DAS dataset, preprocessing signals, and running the trained model on a sample 3-minute VT event with a magnitude of Ml=4.04.

To explore the example, open the notebook:

```bash
jupyter notebook examples/RNN-DAS_LaPalma.ipynb
```

## Citation
If you use this model or any part of this repository, please cite the corresponding article:

> [RNN-DAS: A New Deep Learning Approach for Detection and Real-Time Monitoring of Volcano-Tectonic Events Using Distributed Acoustic Sensing](URL_to_the_article)

## License
This repository and its contents are subject to the same terms and conditions as specified in the accompanying publication. Future updates may refine the model or add new functionalities.

For inquiries or contributions, feel free to open an issue or submit a pull request.

