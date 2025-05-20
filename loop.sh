# for latent_dim in 784 1384; do #2 4 8 16 32 64 128 256; do 
#   sbatch --export=LATENT_DIM=$latent_dim simple_launch.sbatch
# done

# for latent_dim in 2 4 8 16 32 64 128 256 512 1024 2048 718 748 1384; do 
#   sbatch --export=LATENT_DIM=$latent_dim simple_launch.sbatch
# done

# for latent_dim in 1384 2048; do 
#   sbatch --export=LATENT_DIM=$latent_dim simple_launch.sbatch
# done

for lr in 1e-2 5e-2 1e-3 1e-4 1e-5 1e-6 1e-7 1e-8 1e-9; do 
  sbatch --export=LEARNING_RATE=$lr simple_launch.sbatch
done

