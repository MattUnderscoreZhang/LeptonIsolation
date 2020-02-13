#!/bin/bash
# compile SampleMaker
lsetup cmake
mkdir build/
cd build/
cmake ..
make
