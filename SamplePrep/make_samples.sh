#!/bin/bash

# set up environment
cd SampleMaker
source setup_env.sh
source compile.sh

# Test command:
./x*/bin/SampleMaker /eos/user/m/mazhang/LepIso/ROOT/SUSY2/Zee/DAOD_SUSY2.18586363._000001.pool.root.1
./x*/bin/SampleMaker /eos/user/m/mazhang/LepIso/ROOT/SUSY2/Zmm/DAOD_SUSY2.18255207._000001.pool.root.1
./x*/bin/SampleMaker /eos/user/m/mazhang/LepIso/ROOT/MUON5/mc16_13TeV.410470.PhPy8EG_A14_ttbar_hdamp258p75_nonallhad.deriv.DAOD_MUON5.e6337_e5984_s3126_r10201_r10210_p3584/DAOD_MUON5.14537525._000001.pool.root.1

# run on grid
cd ..
source submit-to-grid.sh
