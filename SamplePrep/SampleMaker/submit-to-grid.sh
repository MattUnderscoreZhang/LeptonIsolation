#!/usr/bin/env bash

# To run this script:
# source submit-to-grid.sh
# --osMatching --athenaTag=22.0.6\

source setup_env.sh
source build/x86_64-centos7-gcc8-opt/setup.sh

GRID_NAME=${RUCIO_ACCOUNT-${USER}}
JOB_TAG=$(date +%F-%H-%M)

INPUT_DATASETS=(
    mc16_13TeV.361106.PowhegPythia8EvtGen_AZNLOCTEQ6L1_Zee.deriv.DAOD_SUSY2.e3601_e5984_s3126_r10724_r10726_p3895
    mc16_13TeV.308093.Sherpa_221_NNPDF30NNLO_Zmm2jets_Min_N_TChannel.deriv.DAOD_SUSY2.e5767_e5984_s3126_r10724_r10726_p3875
)

lsetup panda

for IN_DS in ${INPUT_DATASETS[*]}
do
    DSID=$(sed -r 's/[^\.]*\.([0-9]{6,8})\..*/\1/' <<< ${IN_DS})
    OUT_DS=user.${GRID_NAME}.RNN.${DSID}.${JOB_TAG}
    prun --exec "./build/x*/bin/SampleMaker %IN"\
        --athenaTag=AnalysisBase,21.2.97\
        --inDS ${IN_DS} --outDS ${OUT_DS}\
        --noEmail > ${OUT_DS}.log 2>&1
done
