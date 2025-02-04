#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem=2GB
#SBATCH --job-name=test
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_outputs/%x/%j.out

module purge

singularity exec --nv \
	    --overlay /scratch/mr7401/projects/pytorch-example/overlay-50G-10M.ext3:ro \
	    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif\
	    /bin/bash -c "source /ext3/env.sh; source activate test || conda activate test; python /scratch/mr7401/projects/meta_comp/tests/torch-test.py"
