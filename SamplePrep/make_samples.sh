#!/bin/bash

# set paths
ROOT_path="/eos/user/m/mazhang/LepIso/ROOT/MUON5/mc16_13TeV.410470.PhPy8EG_A14_ttbar_hdamp258p75_nonallhad.deriv.DAOD_MUON5.e6337_e5984_s3126_r10201_r10210_p3584"
sample_path="/eos/user/m/mazhang/LepIso/samples"
plot_path="../Outputs/SampleDiagnosisPlots"

# set up environment
echo "=== running setupATLAS ==="
export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh - q
echo "=== running asetup ==="
asetup AnalysisBase,21.2.29

# compile SampleMaker
cd SampleMaker
mkdir build/
cd build/
rm -rf x86*
cmake ..
make
# temporary fix for library linking issue
ln -s /cvmfs/atlas.cern.ch/repo/sw/software/21.2/AnalysisBaseExternals/21.2.29/InstallArea/x86_64-slc6-gcc62-opt/lib/libhdf5.so libhdf5.so.6
export LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH
# Test command:
#./x86_64-centos7-gcc62-opt/bin/SampleMaker /eos/user/m/mazhang/LepIso/ROOT/MUON5/mc16_13TeV.410470.PhPy8EG_A14_ttbar_hdamp258p75_nonallhad.deriv.DAOD_MUON5.e6337_e5984_s3126_r10201_r10210_p3584/DAOD_MUON5.14537525._000001.pool.root.1
cd ../..

# create samples
mkdir -p ${sample_path}/H5
for i in {1..30}
do
    printf -v padded_i "%06d" ${i}
    ./SampleMaker/build/x86_64-centos7-gcc62-opt/bin/SampleMaker ${ROOT_path}/DAOD_MUON5.14537525._${padded_i}.pool.root.1
    mv output.h5 ${sample_path}/H5/output_${i}.h5
done
#python H5ToPkl/h5_to_pkl.py ${sample_path}
#python H5ToPkl/make_diagnostic_plots.py ${sample_path} ${plot_path}
