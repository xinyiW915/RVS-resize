"""Extracting Content-Aware Perceptual Features using Pre-Trained ResNet-50"""
# Modify: Xinyi Wang
# Date: 2021/10/05

import skvideo
# skvideo.setFFmpegPath('/mnt/storage/software/apps/ffmpeg-4.3/bin/ffmpeg')

import torch
from torchvision import transforms, models
import torch.nn as nn
from torch.utils.data import Dataset
import skvideo.io
from PIL import Image
import os
import h5py
import numpy as np
import random
from argparse import ArgumentParser

from func.frame_saliency import fsam
from utils import *
from cam.scorecam import *
# import torch.nn.functional as F
import cv2
import subprocess

class VideoDataset(Dataset):
    """Read data from the original dataset for feature extraction"""
    def __init__(self, videos_dir, video_names, score, width, height, framerate, video_format='RGB'):

        super(VideoDataset, self).__init__()
        self.videos_dir = videos_dir
        self.video_names = video_names
        self.score = score
        self.format = video_format
        self.width = width
        self.height = height
        self.framerate = framerate

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        # video_name = self.video_names[idx] + '.mkv'
        video_name = self.video_names[idx]# konvid
        video_width = int(self.width[idx])
        video_height = int(self.height[idx])
        video_framerate = int(self.framerate[idx])
        print('framerate:', video_framerate)

        # Down sampling the video
        resolution = str(video_width) + 'x' + str(video_height)
        resolutionNew = str(224) + 'x' + str(224)
        filterName = 'lanczos'
        out_yuv = '../tmp/temp_saliency/' + self.video_names[idx].replace('.mp4', '') + '.yuv'
        tmp_yuv = '../tmp/temp_saliency/' + 'tmp.yuv'
        out_avi = '../tmp/temp_saliency/' + self.video_names[idx].replace('.mp4', '') +'.avi'
        print(videos_dir + video_name)

        cmd1 = f'ffmpeg -loglevel error -y -i {videos_dir + video_name} -pix_fmt yuv420p -vsync 0 {out_yuv}'
        cmd2 = f'ffmpeg -f rawvideo -s {resolution} -pix_fmt yuv420p -i {out_yuv} -vf scale={resolutionNew}:flags={filterName} -c:v rawvideo -pix_fmt yuv420p {tmp_yuv}'
        cmd3 = f'ffmpeg -f rawvideo -s {resolutionNew} -pix_fmt yuv420p -i {tmp_yuv} -c:v rawvideo {out_avi}'
        subprocess.run(cmd1, encoding="utf-8", shell=True)
        subprocess.run(cmd2, encoding="utf-8", shell=True)
        subprocess.run(cmd3, encoding="utf-8", shell=True)
        os.remove(out_yuv)
        os.remove(tmp_yuv)

        assert self.format == 'YUV420' or self.format == 'RGB'
        if self.format == 'YUV420':
            video_data = skvideo.io.vread(out_avi, video_height, video_width, inputdict={'-pix_fmt':'yuvj420p'})
            # print(video_data)
        else:
            video_data = skvideo.io.vread(out_avi)
            # print(video_data)
        video_score = self.score[idx]

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        video_length = video_data.shape[0]
        video_channel = video_data.shape[3]
        video_height = video_data.shape[1]
        video_width = video_data.shape[2]
        video_sample = int(video_length/video_framerate)
        print('video length: ', video_length)
        print('video sample: ', video_sample)
        # transformed_video = torch.zeros([video_sample, video_channel, video_height, video_width])
        transformed_video = torch.zeros([video_sample, video_channel, 224, 224])
        sample_idx = 0

        torch.cuda.empty_cache()
        # vgg score saliency map
        vgg = models.vgg16(pretrained=True).eval()
        vgg_model_dict = dict(type='vgg16', arch=vgg, layer_name='features_29', input_size=(224, 224))
        vgg_scorecam = ScoreCAM(vgg_model_dict)

        for frame_idx in range(0, video_framerate*video_sample, video_framerate):
            # print(frame_idx)
            frame = video_data[frame_idx]
            # frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_CUBIC) # resize video frames

            # saliency map
            image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)            # RGBtoBGR meets the opencv display format
            fileName = '../tmp/image_temp/' + str(frame_idx) + '.jpg'  # Temporary storage path
            cv2.imwrite(fileName, image, [cv2.IMWRITE_JPEG_QUALITY, 100])
            input_image = load_image(fileName)
            input_ = apply_transforms(input_image)
            if torch.cuda.is_available():
                input_ = input_.cuda()
            predicted_class = vgg(input_).max(1)[-1]

            scorecam_map = vgg_scorecam(input_)
            t = scorecam_map.cpu()
            t = torch.squeeze(t)
            # print(t)
            # print(t.shape)
            # smap = t.numpy()  # tensor 2 numpy

            # Clear temp data
            os.remove(fileName)

            frame = Image.fromarray(frame)
            frame = transform(frame)
            # print(frame)
            # print(frame.shape)
            attention = frame + t
            # print(attention)
            # print(attention.shape)
            # print(sample_idx)
            transformed_video[sample_idx] = attention
            sample_idx += 1

        sample = {'video': transformed_video,
                  'score': video_score}

        os.remove(out_avi)

        return sample


class ResNet50(torch.nn.Module):
    """Modified ResNet50 for feature extraction"""
    def __init__(self):
        super(ResNet50, self).__init__()
        self.features = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-2])
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x):
        # features@: 7->res5c
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii == 7:
                features_mean = nn.functional.adaptive_avg_pool2d(x, 1)
                features_std = global_std_pool2d(x)
                return features_mean, features_std


def global_std_pool2d(x):
    """2D global standard variation pooling"""
    return torch.std(x.view(x.size()[0], x.size()[1], -1, 1),
                     dim=2, keepdim=True)


def get_features(video_data, frame_batch_size=32, device='cuda'):
    """feature extraction"""
    extractor = ResNet50().to(device)
    video_length = video_data.shape[0]
    frame_start = 0
    frame_end = frame_start + frame_batch_size
    output1 = torch.Tensor().to(device)
    output2 = torch.Tensor().to(device)
    extractor.eval()
    with torch.no_grad():
	    while frame_end < video_length:
	        batch = video_data[frame_start:frame_end].to(device)
	        features_mean, features_std = extractor(batch)
	        output1 = torch.cat((output1, features_mean), 0)
	        output2 = torch.cat((output2, features_std), 0)
	        frame_end += frame_batch_size
	        frame_start += frame_batch_size

	    last_batch = video_data[frame_start:video_length].to(device)
	    features_mean, features_std = extractor(last_batch)
	    output1 = torch.cat((output1, features_mean), 0)
	    output2 = torch.cat((output2, features_std), 0)
	    output = torch.cat((output1, output2), 1).squeeze()

    return output


if __name__ == "__main__":
    parser = ArgumentParser(description='"Extracting Content-Aware Perceptual Features using Pre-Trained ResNet-50')
    parser.add_argument("--seed", type=int, default=19920517)
    parser.add_argument('--database', default='YOUTUBE_UGC_TEST', type=str,
                        help='database name (default: KoNViD-1k)')
    parser.add_argument('--frame_batch_size', type=int, default=64,
                        help='frame batch size for feature extraction (default: 64)')

    parser.add_argument('--disable_gpu', action='store_true',
                        help='flag whether to disable GPU')
    args = parser.parse_args()

    torch.manual_seed(args.seed)  #
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.utils.backcompat.broadcast_warning.enabled = True

    if args.database == 'KoNViD_1k':
        videos_dir = '/mnt/storage/home/um20242/scratch/dataset/KoNViD_1k/KoNViD_1k_videos/'
        features_dir = '/mnt/storage/home/um20242/scratch/RVS-resize/CNN_features/CNN_features_KoNVid/'
        datainfo = '/mnt/storage/home/um20242/scratch/RVS-resize/mos_file/data_info/KONVID_1K_info.mat'

    if not os.path.exists(features_dir):
        os.makedirs(features_dir)

    device = torch.device("cuda" if not args.disable_gpu and torch.cuda.is_available() else "cpu")

    Info = h5py.File(datainfo, 'r')
    video_names = [Info[Info['video_names'][0, :][i]][()].tobytes()[::2].decode() for i in range(len(Info['video_names'][0, :]))]
    scores = Info['scores'][0, :]
    video_format = 'RGB'
    # video_format = Info['video_format'][()].tobytes()[::2].decode()
    print('.................')
    print(video_format)
    width = Info['width'][0, :]
    height = Info['height'][0, :]
    framerate = Info['framerate'][0, :]

    video_list = []
    scores_list = []
    width_list = []
    height_list = []
    framerate_list = []

    for j in range(len(video_names)):
        # video = videos_dir + video_names[j] + '.mkv'
        video = videos_dir + video_names[j]# konvid
        if os.path.isfile(video):
            video_list.append(video_names[j])
            scores_list.append(scores[j])
            width_list.append(width[j])
            height_list.append(height[j])
            framerate_list.append(framerate[j])
    dataset = VideoDataset(videos_dir, video_list, scores_list, width_list, height_list, framerate_list, video_format)

    for i in range(len(dataset)):
        current_data = dataset[i]
        # print(current_data)
        current_video = current_data['video']
        current_score = current_data['score']
        print('Video {}: length {}'.format(i, current_video.shape[0]))
        features = get_features(current_video, args.frame_batch_size, device)
        np.save(features_dir + str(i) + '_resnet-50_res5c', features.to('cpu').numpy())
        np.save(features_dir + str(i) + '_score', current_score)
