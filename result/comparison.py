import numpy as np
import pandas as pd

csv_file_resize = '../result/YOUTUBE_UGC_1080P_RS_resize_Predicted_Score.csv'
# csv_file_resize = '../result/YOUTUBE_UGC_1080P_VS_resize_Predicted_Score.csv'
# csv_file_resize = '../result/YOUTUBE_UGC_1080P_RVS_resize_Predicted_Score.csv'

csv_file_old = '/mnt/storage/home/um20242/scratch/BVQA/results/YOUTUBE_UGC_1080P_RAPIQUE_SALIENCY_Predicted_Score.csv'
# csv_file_old = '/mnt/storage/home/um20242/scratch/BVQA/results/YOUTUBE_UGC_1080P_VSFACNN_SALIENCY_Predicted_Score.csv'
# csv_file_old = '/mnt/storage/home/um20242/scratch/BVQA/results/YOUTUBE_UGC_1080P_RAPIQUE_VSFACNN_SALIENCY_Predicted_Score.csv'

data_name = 'YOUTUBE_UGC_1080P'
model_name = 'RS_resize'
# model_name = 'VS_resize'
# model_name = 'RVS_resize'

d_resize = pd.read_csv(csv_file_resize)
score_resize = d_resize['Predicted Score']
name_resize = d_resize['Video_name']
mos = d_resize['MOS']
name_resize = name_resize.values.tolist()
# print(score1)

d_old = pd.read_csv(csv_file_old)
score_old = d_old['Predicted Score']
name_old = d_old['Video_name']
name_old = name_old.values.tolist()
# print(score_old)

if name_resize == name_old:
    print('same')

dif_resize = score_resize - mos
pos_resize = [i for i in dif_resize if i > 0]
print("pos_resize: %s" % len(pos_resize))
nag_resize = [i for i in dif_resize if i < 0]
print("nag_resize: %s" % len(nag_resize))

dif_old = score_old -mos
pos_old = [i for i in dif_old if i > 0]
print("pos_old: %s" % len(pos_old))
nag_old = [i for i in dif_old if i < 0]
print("nag_old: %s" % len(nag_old))

# print(abs(dif_resize))
# print(abs(dif_old))
plus = 0
minus = 0
eq = 0
for i in range(len(dif_resize)):
    if abs(dif_resize[i]) > abs(dif_old[i]):
        plus += 1
    elif abs(dif_resize[i]) < abs(dif_old[i]):
        minus += 1
    elif abs(dif_resize[i]) == abs(dif_old[i]):
        eq += 1
    else:
        eq += 1

print(plus)
print(minus)
print(eq)
comparison = {'Video_name': name_resize,
              'MOS': mos,
              'Predicted Score_resize': score_resize,
              'Predicted Score_old': score_old,
              'Difference_resize': dif_resize,
              'Difference_old': dif_old}

result = pd.DataFrame(comparison)
result['Positive_resize'] = ''
result['Positive_resize'][0] = len(pos_resize)
result['Positive_old'] = ''
result['Positive_old'][0] = len(pos_old)

result['Negative_resize'] = ''
result['Negative_resize'][0] = len(nag_resize)
result['Negative_old'] = ''
result['Negative_old'][0] = len(nag_old)

result['abs(dif_resize) > abs(dif_old)'] = ''
result['abs(dif_resize) > abs(dif_old)'][0] = plus
result['abs(dif_resize) < abs(dif_old)'] = ''
result['abs(dif_resize) < abs(dif_old)'][0] = minus

print(result)
result_path = '../result/' + data_name + '_' + model_name +'_Predicted_Score_Comparison.csv'
result.to_csv(result_path, index=False, header=True)