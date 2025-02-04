#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=00:50:00
#SBATCH --mem=100GB
#SBATCH --job-name=downloads
#SBATCH --output=slurm_outputs/%j_%x.out

module purge

singularity exec --nv \
	    --overlay /scratch/mr7401/projects/pytorch-example/overlay-50G-10M.ext3:ro \
	    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif\
	    /bin/bash -c "source /ext3/env.sh; source activate test || conda activate test; echo 'h'; python /scratch/mr7401/projects/meta_comp/download_models.py"
