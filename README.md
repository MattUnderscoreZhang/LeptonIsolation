[![CodeFactor](https://www.codefactor.io/repository/github/particularlypythonicbs/leptonisolation/badge)](https://www.codefactor.io/repository/github/particularlypythonicbs/leptonisolation)

## Overview

This package is a lepton-isolation classification tool, to be used for analysis of particle collision data at CERN.

The tool is able to take leptons from collision events, feed the surrounding tracks and calorimeter depositions into a neural net, and provide a number between 0 and 1 indicating likelihood of the lepton being prompt, as opposed to coming from a heavy-flavor decay.

## SamplePrep

Data production is the first step of the training pipeline. In this step, we take collision data in the form of DAOD or AOD ROOT files, perform event and object filtering, and produce trees with information on each lepton and its surrounding objects. Tracks and calorimeter clusters which were used in a lepton's own reconstruction are not included in its list of associated objects.

Run this package on lxplus for access to the relevant libraries. Simply edit and source make_samples.sh in the SamplePrep/ folder in order to produce samples.

## Training

Train a recurrent neural net to perform lepton isolation. Simply edit and run isolator.py.

The code in the Training/ folder can also perform hyperparameter tuning, plot production, model saving and loading, isolation variable sanity checks, and final analysis.

## MiscScripts

These tools perform various diagnostic tests and checks.

## Outputs

Training outputs are by default stored in /public/data/RNN/runs/, though this can be changed in the Python code.

How to display outputs when using SSH (e.g. on UIUC Skynet):
* On local computer, map Tensorboard port 6006 on Skynet to local port 16006:
ssh -N -f -L localhost:16006:skynet:6006 <user>@skynet 
* tensorboard --logdir /public/data/RNN/runs
* Go to http://localhost:16006 on local browser.
