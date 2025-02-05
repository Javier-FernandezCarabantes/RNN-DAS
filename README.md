# RNN-DAS: A New Deep Learning Approach for Detection and Real-Time Monitoring of Volcano-Tectonic Events Using Distributed Acoustic Sensing

## Overview
RNN-DAS is an innovative Deep Learning model based on Recurrent Neural Networks (RNNs) with Long Short-Term Memory (LSTM) cells, developed for real-time Volcano-Seismic Signal Recognition (VSR) using Distributed Acoustic Sensing (DAS) measurements. The model was trained on a comprehensive dataset of Volcano-Tectonic (VT) events from the 2021 La Palma eruption, recorded by a High-fidelity submarine Distributed Acoustic Sensing array (HDAS) located near the eruption site.

RNN-DAS is capable of detecting VT events, analyzing their temporal evolution, and classifying their waveforms with approximately 97% accuracy. The model has demonstrated excellent generalization capabilities for different time intervals and volcanoes, facilitating continuous, real-time seismic monitoring with minimal computational resources and retraining requirements.

This software was developed as part of the DigiVolCan project - A digital infrastructure for forecasting volcanic eruptions in the Canary Islands.
The results from the RNN-DAS model are the outcome of collaboration between the University of Granada, the Canary Islands Volcanological Institute (INVOLCAN), the Institute of Technological and Renewable Energies (ITER), the University of La Laguna, and Aragón Photonics.

![Project Screenshot](./logo.png)

The project was funded by the Ministry of Science, Innovation, and Universities / State Research Agency (MICIU/AEI) of Spain, and the European Union through the Recovery, Transformation, and Resilience Plan, Next Generation EU Funds.
Project: PLEC2022-009271 funded by MICIU/AEI /10.13039/501100011033 and by the European Union Next GenerationEU/ PRTR.

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
 
Alternatively, you can install the provided Conda environment `RNN_DAS`.  

The model has been **trained and tested** for functionality in:  

- **Python** 3.7  
- **GPU**: NVIDIA GeForce GTX 1080  
- **CUDA**: version 9.0  

Therefore, the model should work without issues on these versions and later.

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
An example Jupyter Notebook is provided to demonstrate how the model operates on real data from the 2021 La Palma eruption. This notebook walks through loading the DAS dataset, preprocessing signals, and running the trained model on a sample 3-minute VT event with a magnitude of Ml=4.04. The provided test data corresponds only to the first 100 channels of the DAS array due to repository size limitations. 
To explore the example, open the notebook:

```bash
jupyter notebook examples/RNN-DAS_LaPalma.ipynb
```

To apply the model to the full dataset in the Jupyter Notebook, simply download the data from [this link] and use it within the notebook.  




## Citation
If you use this model or any part of this repository, please cite the corresponding article:

> [RNN-DAS: A New Deep Learning Approach for Detection and Real-Time Monitoring of Volcano-Tectonic Events Using Distributed Acoustic Sensing](URL_to_the_article)

## License
This repository and its contents are subject to the same terms and conditions as specified in the accompanying publication. Future updates may refine the model or add new functionalities.

For inquiries or contributions, feel free to open an issue or submit a pull request.

