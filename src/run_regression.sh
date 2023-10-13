#!/bin/bash
#SBATCH --account=eeng028284
#SBATCH --job-name=regreugc
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -o /user/work/um20242/RVS-resize/log/regression.out
#SBATCH --mem-per-cpu=100G

cd "${SLURM_SUBMIT_DIR}"

echo Running on host "$(hostname)"
echo Time is "$(date)"
echo Directory is "$(pwd)"
echo Slurm job ID is "${SLURM_JOBID}"
echo This jobs runs on the following machines:
echo "${SLURM_JOB_NODELIST}"

module load languages/anaconda3/2020.02-tflow-1.15
#conda create -n reproducibleresearch pip python=3.6

# Activate virtualenv
source activate reproducibleresearch
#pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# Run Python script
#python evaluate_bvqa_features_regression.py \
#  --model_name RAPIQUENSS_Saliency \
#  --dataset_name KONVID_1K \
#  --feature_file ../feat_file/KONVID_1K_RAPIQUENSS_Saliency_feats.mat \
#  --mos_file ../mos_file/KONVID_1K_metadata.save_csv \
#  --out_file ../result/KONVID_1K_RAPIQUENSS_Saliency_SVR_corr.mat \
#  --log_file ../log/KONVID_1K_RAPIQUENSS_Saliency_SVR.log \
#  --use_parallel

#python < demo_eval_BVQA_feats_all_combined.py
python < demo_eval_BVQA_feats_one_dataset.py


## Deactivate virtualenv
conda deactivate