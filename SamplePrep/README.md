These scripts have to be run on lxplus due to the relevant CERN-related libraries.

The ROOTToH5 package takes a ROOT file, extracts lepton and track data, and saves the result as a H5 file.
The package is based on code from https://gitlab.cern.ch/atlas-flavor-tagging-tools/training-dataset-dumper/tree/master and https://github.com/dguest/mlbnn.

H5ToPkl is in charge of taking this H5 file, grouping leptons and tracks, applying object selection cuts, adding various features, and outputting a pkl file that is useable by our training framework.

I'm not very happy about using pkl format, or about having to split data prep functionality across two packages, but I haven't found a way to solve these problems yet.

------

To make samples, simply edit the paths in make\_samples.sh, and then source it.
