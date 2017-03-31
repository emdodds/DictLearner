#!/bin/bash
#
# Partition:
#SBATCH --partition=cortex
#
# Wall clock limit:
#SBATCH --time=48:00:00
#
# Memory:
#SBATCH --mem-per-cpu=8G
#
# Constraint:
#SBATCH --gres=gpu:1
cd /global/home/users/edodds/DictLearner
export MODULEPATH=/global/software/sl-6.x64_64/modfiles/apps:$MODULEPATH
module load ml/tensorflow/0.11.0rc0:q

python scripts/tf_lca_script.py