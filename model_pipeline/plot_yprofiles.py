import numpy as np
import pandas as pd
from pandas import read_csv
import math
import matplotlib.pyplot as plt
import glob
from copy import deepcopy
import sys
import os
np.set_printoptions(threshold=sys.maxsize)
sys.path.insert(1, '../pretrain-data-prep')
# https://github.com/smart-pix/pretrain-data-prep/tree/main
from dataset_utils import quantize_manual, add_noise

original_atHighThresh = np.load('./tmp_16x16_centeredIncidence/1000_1600_2400/yprofiles.npy')
yprofile_atHighThresh = read_csv('/mnt/local/CMSPIX28/data/ChipVersion1_ChipID23_SuperPix1/2026.01.09_22.52.47_DNN_vth0-0.047_vth1-0.082_vth2-0.129/yprofiles.csv', header=None).values

original_atLowThresh = np.load('./tmp_16x16_centeredIncidence/400_1600_2400/yprofiles.npy')
yprofile_atLowThresh = read_csv('/mnt/local/CMSPIX28/data/ChipVersion1_ChipID23_SuperPix1/2026.01.29_07.16.26_DNN_vth0-0.012_vth1-0.082_vth2-0.129/yprofiles.csv', header=None).values

labels = read_csv('./tmp_16x16_centeredIncidence/1000_1600_2400/clslabels.csv', header=None).values
# Find indices of one event from each class (0, 1, 2)
num_classes = 3
class_indices = {}
for class_id in range(num_classes):
    indices = [i for i, label in enumerate(labels) if label[0] == class_id and i < 150_000] # ASIC data were taken only for 150k events 
    if indices:
        class_indices[class_id] = np.random.choice(indices)
    else:
        class_indices[class_id] = None  # No event for this class

print(class_indices)

event_indices = list(class_indices.values())

# Plotting
for array, savename in zip([original_atHighThresh, yprofile_atHighThresh, original_atLowThresh, yprofile_atLowThresh], ['1000e-', '1000e-_DNN', '400e-', '400e-_DNN']):
    selected_profiles = np.stack([array[idx] for idx in event_indices])
    for idx in event_indices:
        thresh = savename.split('_')[0]
        yprofile = np.array(array[idx]).reshape(1, -1)  # shape (1, 16)
        plt.figure(figsize=(8, 1.5))
        im = plt.imshow(yprofile, aspect='auto', cmap='viridis', vmin=0, vmax=36)
        plt.xlabel('Matrix row number')
        plt.title(f'Y Profile: event {idx} ({thresh} threshold, Class {labels[idx][0]})')
        cbar = plt.colorbar(im, label='Quantized charge value', shrink=1.2, aspect=15, pad=0.02)
        cbar.set_ticks(np.arange(0, 37, 5))  # Show ticks at 0, 5, 10, ..., 45
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label('Quantized charge Value', fontsize=8)
        plt.xticks(np.arange(0,16,1))
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(f'yprofile_{savename}_class{labels[idx][0]}.png')