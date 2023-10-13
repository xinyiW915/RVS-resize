# Author: Xinyi Wang
# Date: 2021/10/05

import cv2
from PIL import Image
import numpy as np
import pandas as pd
import os
import torch
import torch.nn.functional as F
import torchvision.models as models

import scipy.io as io

from utils import *
from cam.scorecam import *

def fsam(video_name, framerate):
    
    torch.cuda.empty_cache()
    # vgg score saliency map
    vgg = models.vgg16(pretrained=True).eval()
    vgg_model_dict = dict(type='vgg16', arch=vgg, layer_name='features_29', input_size=(224, 224))
    torch.cuda.empty_cache()
    vgg_scorecam = ScoreCAM(vgg_model_dict)
    torch.cuda.empty_cache()

    # open the video
    cap = cv2.VideoCapture(video_name)  # Get the video object
    isOpened = cap.isOpened  # Determine if it is open
    # Video information acquisition
    fps = cap.get(cv2.CAP_PROP_FPS)
    torch.cuda.empty_cache()

    imageNum = 0
    sum = 0
    # timef = 30  # Save a picture every 30 frames
    print('framerate:', framerate)

    sum_samp = 0
    while (isOpened):

        sum += 1
        (frameState, frame) = cap.read()  # Recording of each frame and acquisition status
        torch.cuda.empty_cache()
        if frameState == True and (sum % framerate == 0):

            # Format transformation, BGRtoRGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to Image
            frame = Image.fromarray(np.uint8(frame))
            frame = np.array(frame)
            # RGBtoBGR meets the opencv display format
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            torch.cuda.empty_cache()
            imageNum = imageNum + 1
            fileName = '../tmp/video_image_temp/' + str(
                imageNum) + '.jpg'  # Temporary storage path
            cv2.imwrite(fileName, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
            # print(fileName + " successfully write in")  # Output storage status
            input_image = load_image(fileName)
            input_ = apply_transforms(input_image)
            if torch.cuda.is_available():
                input_ = input_.cuda()
            predicted_class = vgg(input_).max(1)[-1]

            scorecam_map = vgg_scorecam(input_)
            # print(input_)
            t = scorecam_map.cpu()
            smap = t.numpy()  # tensor 2 numpy
            # print(scorecam_map)
            print(smap)
            sum_samp += smap

            # Clear temp data
            os.remove(fileName)
            torch.cuda.empty_cache()

        elif frameState == False:
            break
    print('Complete the extraction of video frames!')
    mean_samp = sum_samp / imageNum
    # print(mean_samp)
    mean_samp = np.squeeze(mean_samp)

    torch.cuda.empty_cache()

    return mean_samp

if __name__ == "__main__":
    fsam('/mnt/storage/home/um20242/scratch/ugc-dataset/360P/Animation_360P-08c9.mkv', 30)
