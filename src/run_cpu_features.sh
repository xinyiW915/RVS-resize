#!/bin/bash
#SBATCH --account=eeng028284
#SBATCH --job-name=vsfacnn_saliency
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -o /user/work/um20242/RVS-resize/log/vsfacnn_saliency_runtime.out
#SBATCH --mem-per-cpu=100G

# Load modules required for runtime e.g.##
module load apps/ffmpeg/4.3
module add apps/matlab/2018a
module load languages/anaconda3/2020.02-tflow-1.15
module load CUDA

# Activate virtualenv
source activate reproducibleresearch

cd $SLURM_SUBMIT_DIR

# Run Python script
# CNN features extraction
#CUDA_VISIBLE_DEVICES=2 python CNNfeats_Saliency_KONVID.py --database=KoNViD_1k --frame_batch_size=32
CUDA_VISIBLE_DEVICES=2 python CNNfeats_Saliency.py --database=YOUTUBE_UGC_1080P_test --frame_batch_size=32

## Deactivate virtualenv
conda deactivate

