#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --mem=10GB
#SBATCH --job-name=meta_learning
#SBATCH --gres=gpu:0
#SBATCH --output=/scratch/mr7401/outputs/%x/%j.out

# module purge
# latent_dim=$LATENT_DIM
# echo "Running with latent_dim: ${latent_dim}"
# singularity exec --nv \
# 	    --overlay /scratch/mr7401/projects/pytorch-example/overlay-50G-10M.ext3:ro \
# 	    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif\
# 	    /bin/bash -c "source /ext3/env.sh; source activate test || conda activate test; python /scratch/mr7401/projects/meta_comp/celeba/train.py --latent-dim ${latent_dim}"

# module purge
# lr=$LEARNING_RATE

# echo "Running with learning_rate: ${lr}"

# singularity exec --nv \
# 	    --overlay /scratch/mr7401/projects/pytorch-example/overlay-50G-10M.ext3:ro \
# 	    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif\
# 	    /bin/bash -c "source /ext3/env.sh; source activate test || conda activate test; python /scratch/mr7401/projects/meta_comp/celeba/train_meta.py --lr ${lr} --training_seed 2"


module purge
lr=$LEARNING_RATE

echo "Running with learning_rate: ${lr}"

singularity exec --nv \
	    --overlay /scratch/mr7401/projects/pytorch-example/overlay-50G-10M.ext3:ro \
	    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif\
	    /bin/bash -c "source /ext3/env.sh; source activate test || conda activate test; python /scratch/mr7401/projects/meta_comp/celeba/train_meta.py --lr ${lr} --training_seed 2 --loss_fn mse_dir"
