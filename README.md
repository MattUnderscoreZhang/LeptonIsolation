## SamplePrep

Run on lxplus in order to convert ROOT files to a data format suitable for training in Python.

## Training

Train an RNN to perform lepton isolation.

## Outputs

The results of training and plotting.

How to display outputs when using SSH (e.g. on UIUC Skynet):
* On local computer, map Tensorboard port 6006 on Skynet to local port 16006:
ssh -N -f -L localhost:16006:localhost:6006 <user>@skynet 
* tensorboard --logdir /public/data/RNN/runs
* Go to http://localhost:16006 on local browser.
