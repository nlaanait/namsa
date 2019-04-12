#!/bin/bash -l
BUILDS=${PROJWORK}/lrn001/nl/builds
PYTHON=${BUILDS}/miniconda3
export PATH=$PYTHON/bin:$PATH
#conda install --name "tf1p12" matplotlib
source activate "tf1p12" 
cd namsa
python -u setup.py install

