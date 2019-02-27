#!/bin/bash
# source make_all.sh {ROOT_path} {sample_path}

mkdir -p $2/H5

for i in {1..30}
do
    printf -v padded_i "%06d" ${i}
    ./build/x86_64-slc6-gcc62-opt/bin/dump-xaod $1/DAOD_MUON5.14537525._${padded_i}.pool.root.1
    mv output.h5 $2/H5/output_${i}.h5
done
