#!/bin/bash

ROOT_path="/eos/user/m/mazhang/LepIso/ROOT/MUON5/mc16_13TeV.410470.PhPy8EG_A14_ttbar_hdamp258p75_nonallhad.deriv.DAOD_MUON5.e6337_e5984_s3126_r10201_r10210_p3584"
sample_path="/eos/user/m/mazhang/LepIso/samples"
plot_path="../Outputs/SampleDiagnosisPlots"

cd ROOTToH5
source dumpxAOD/setup.sh
mkdir build/
cd build/
cmake ../dumpxAOD
make
# temporary fix for library linking issue
ln -s /cvmfs/atlas.cern.ch/repo/sw/software/21.2/AnalysisBaseExternals/21.2.29/InstallArea/x86_64-slc6-gcc62-opt/lib/libhdf5.so libhdf5.so.6
export LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH
# Test command:
#./build/x86_64-slc6-gcc62-opt/bin/dump-xaod /eos/user/m/mazhang/LepIso/ROOT/MUON5/mc16_13TeV.410470.PhPy8EG_A14_ttbar_hdamp258p75_nonallhad.deriv.DAOD_MUON5.e6337_e5984_s3126_r10201_r10210_p3584/DAOD_MUON5.14537525._000001.pool.root.1
cd ..
source make_all.sh ${ROOT_path} ${sample_path}
cd ..


python H5ToPkl/h5_to_pkl.py ${sample_path}

python H5ToPkl/make_diagnostic_plots.py ${sample_path} ${plot_path}
