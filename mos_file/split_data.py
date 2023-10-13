import pandas as pd
import os


metadate = '/mnt/storage/home/um20242/scratch/RAPIQUE-VSFA-Saliency/mos_file/YOUTUBE_UGC_480P_metadata.csv'
df = pd.read_csv(metadate)
print(len(df))
split1 = df[0: 50]
split2 = df[50: 100]
split3 = df[80: 120]
split4 = df[120: 160]
split5 = df[160: 200]
split6 = df[200: 240]
split7 = df[240: 271]

# metadata_path = '/mnt/storage/home/um20242/scratch/ugc-dataset/data_filter/'
# df.to_csv(metadata_path + 'YOUTUBE_UGC_metadata.csv')
