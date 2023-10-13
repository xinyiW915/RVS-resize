#!/bin/sh
#SBATCH --account=eeng028284
#SBATCH --job-name=info
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=100G

# Load modules required for runtime e.g.##
module add apps/matlab/2018a
module load apps/ffmpeg/4.3
module load languages/anaconda3/2020.02-tflow-1.15
module load CUDA
# Activate virtualenv
source activate reproducibleresearch

cd $SLURM_SUBMIT_DIR

# Activate virtualenv
matlab -nodisplay -nodesktop -nosplash -singleCompThread < data_info_maker.m

## Deactivate virtualenv
conda deactivate
