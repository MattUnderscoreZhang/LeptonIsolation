#!/usr/bin/env bash
#
# Example grid submit script
#
# Short example to submit the dumpxAOD job to the grid. Uses prun to
# create a tarball, then loops over a list of datasets and submits one
# job for each.

# To run this script:
# lsetup panda
# ./submit-to-grid.sh

# This script should not be sourced, we don't need anything in here to
# propigate to the surrounding environment.
if [[ $- == *i* ]] ; then
    echo "Don't source me!" >&2
    return 1
else
    # set the shell to exit if there's an error (-e), and to error if
    # there's an unset variable (-u)
    set -eu
fi

##########################
# Real things start here #
##########################

###################################################
# Part 1: variables you you _might_ need to change
###################################################
#
# Users's grid name
GRID_NAME=${RUCIO_ACCOUNT-${USER}}
#
# This job's tag (the current expression is something random)
BATCH_TAG=$(date +%F-%H-%M)


######################################################
# Part 2: variables you probably don't have to change
######################################################
#
# Build a zip of the files we're going to submit
ZIP=job.tgz
#
# This is the subdirectory we submit from
SUBMIT_DIR=submit


###################################################
# Part 3: prep the submit area
###################################################
#
echo "preping submit area"
if [[ -d ${SUBMIT_DIR} ]]; then
    echo "removing old submit directory"
    rm -rf ${SUBMIT_DIR}
fi
mkdir ${SUBMIT_DIR}
cd ${SUBMIT_DIR}


###########################################
# Part 4: build a tarball of the job
###########################################
# The --outTarBall, --noSubmit, and --useAthenaPackages arguments are
# important. The --outDS and --exec don't matter at all here, they are
# just placeholders to keep panda from complianing.
prun --outTarBall=${ZIP} --noSubmit --useAthenaPackages --outDS user.${GRID_NAME}.x --exec "ls"


##########################################
# Part 5: loop over datasets and submit
##########################################
#
# Get a list of input datasets
INPUT_DATASETS=(
    mc16_13TeV.361106.PowhegPythia8EvtGen_AZNLOCTEQ6L1_Zee.deriv.DAOD_SUSY2.e3601_e5984_s3126_r10724_r10726_p3895
    mc16_13TeV.308093.Sherpa_221_NNPDF30NNLO_Zmm2jets_Min_N_TChannel.deriv.DAOD_SUSY2.e5767_e5984_s3126_r10724_r10726_p3875
)
#
# Loop over all inputs
for DS in ${INPUT_DATASETS[*]}
do
    # This regex extracts the DSID from the input dataset name, so
    # that we can give the output dataset a unique name. It's not
    # pretty: ideally we'd just suffix our input dataset name with
    # another tag. But thanks to insanely long job options names we
    # use in the generation stage we're running out of space for
    # everything else.
    DSID=$(sed -r 's/[^\.]*\.([0-9]{6,8})\..*/\1/' <<< ${DS})
    #
    # Build the full output dataset name
    OUT_DS=user.${GRID_NAME}.RNN.${DSID}.${BATCH_TAG}
    #
    # Now submit. The script we're running expects one argument per
    # input dataset, whereas %IN gives us comma separated files, so we
    # have to run it through `tr`.
    #
    # SAD HACK Part 2: since we're hacking in a library by copying it
    # into the submit directory, we also have to include the working
    # directory in the LD_LIBRARY_PATH.
    echo "Submitting for ${GRID_NAME} on ${DS} -> ${OUT_DS}"
    prun --exec 'LD_LIBRARY_PATH=$(pwd):$LD_LIBRARY_PATH dump-xaod $(echo %IN | tr "," " ")'\
         --outDS ${OUT_DS} --inDS ${DS}\
         --useAthenaPackages --inTarBall=${ZIP}\
         --outputs output.h5\
         --noEmail > ${OUT_DS}.log 2>&1
done

