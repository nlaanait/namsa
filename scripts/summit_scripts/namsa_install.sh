#!/bin/bash -l
HOME="/gpfs/wolf/gen113/scratch/nl7/work"
PYTHON=${HOME}/anaconda3 
cd ${HOME}/MSA/namsa
${PYTHON}/bin/python -u setup.py install

