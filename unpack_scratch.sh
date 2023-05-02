#!/bin/bash
#SBATCH --account=def-bengioy
#SBATCH -J unpack-scratch
#SBATCH -o /home/anmadc/logs/%x.%j.out
#SBATCH -e /home/anmadc/logs/%x.%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=24:00:00

cp $HOME/project/ecoroar.tar.gz $SCRATCH/ecoroar.tar.gz
tar -mxvf $SCRATCH/ecoroar.tar.gz -C /
echo "completed"
