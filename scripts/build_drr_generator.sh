#!/usr/bin/bash
# compiles DRRGenerator inside ./external subdirectory and adds to path

mkdir external
cd external || exit
wget https://github.com/InsightSoftwareConsortium/ITK/archive/refs/tags/v5.3.0.tar.gz
tar -xzvf v5.3.0.tar.gz
cd ITK-5.3.0 || exit
mkdir build
cd build || exit
cmake .. -DModule_TwoProjectionRegistration=ON -DBUILD_TESTING=ON -DBUILD_EXAMPLES=ON 
make -j 20