#!/bin/bash
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -q ee
export CC=g++
export CXX=g++
 
cd $PBS_O_WORKDIR
sh "/home/eegroup/eemsrl03/miniconda3/etc/profile.d/conda.sh"
module load cuda/cuda-10.2/x86_64
module load compiler/GCC/6.3.1/x86_64
python train.py 
