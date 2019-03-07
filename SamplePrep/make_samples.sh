#!/bin/bash

ROOT_path="/eos/user/m/mazhang/LepIso/ROOT/MUON5/mc16_13TeV.410470.PhPy8EG_A14_ttbar_hdamp258p75_nonallhad.deriv.DAOD_MUON5.e6337_e5984_s3126_r10201_r10210_p3584"
sample_path="/eos/user/m/mazhang/LepIso/samples"
plot_path="../Outputs/SampleDiagnosisPlots"

cd ROOTToH5
mkdir build/
cd build/
source ../dumpxAOD/setup.sh
cmake ../dumpxAOD
make
cd ..
source make_all.sh ${ROOT_path} ${sample_path}
cd ..

python H5ToPkl/h5_to_pkl.py ${sample_path}

python H5ToPkl/make_diagnostic_plots.py ${sample_path} ${plot_path}
