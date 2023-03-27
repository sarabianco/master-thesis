#!/bin/bash
#SBATCH --time=0:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=blade,ib 
#SBATCH --mem=2000 
#SBATCH --job-name="test constraint"
#SBATCH --output=/home/STUDENTI/sara.bianco8/master_thesis/output/a%j.out
#SBATCH --error=/home/STUDENTI/sara.bianco8/master_thesis/errors/a%j.err
#SBATCH --constraint=blade
#SBATCH --mail-user=sara.bianco8@studio.unibo.it
#SBATCH --mail-type=ALL
cd /home/STUDENTI/sara.bianco8/master_thesis
python3 DecayLengths.py