To clone this directory with the RNN paper submodule, clone using the "git clone --recurse-submodules" command.

# Functionality
## Create H5 files
Use ROOTToH5 to cluster leptons and tracks, and convert from ROOT to H5.
## Split H5 files
The first time you run Loader from Isolator, it will create pkl files if they don't exist, separated into signal and bkg.
## Run RNN
Isolator runs RNN analysis on the H5 files.
### Default Options
The dictionary stored in default options specifies the parameters used by the rnn including the type of rnn being used.

# Code Structure
## DataStructures
Stores the datastructures used for handling the lepton data
## Architectures
Holds the code for the Neural Networks
## HEP
Helper code for high energy calculation
## Loader
Data Converters for easy use with neural networks
## Options
Serves as configuration files for the neural network
## Paper
A submodule containing the RNN paper

# Run Instructions
## RNN
Create an options file in Isolator/Options/. Use default_options.py as a template.
In Isolator/isolator.py, on the first line of the main function, load in your options file.
Run the RNN with "python isolator.py". Use Python 3.
