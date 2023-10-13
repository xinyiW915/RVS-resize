#!/bin/sh
#SBATCH --account=eeng028284
#SBATCH --job-name=Saliency
##SBATCH --nodes=2
##SBATCH --gres=gpu:2
#SBATCH --time=6:0:00
#SBATCH --partition gpu_veryshort
#SBATCH -o /user/work/um20242/RVS-resize/log/matlab_Saliency.out
#SBATCH --mem=100GB

# Load modules required for runtime e.g.##
module add apps/matlab/2018a
module load apps/ffmpeg/4.3
module load languages/anaconda3/2020.02-tflow-1.15
module load CUDA
# Activate virtualenv
source activate reproducibleresearch

cd $SLURM_SUBMIT_DIR

# Activate virtualenv
CUDA_VISIBLE_DEVICES=0
matlab -nodisplay -nodesktop -nosplash -singleCompThread < compute_Saliency_feats.m

## Deactivate virtualenv
conda deactivate
