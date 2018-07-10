# handler for ECCV HANDS 2018 Workshop Synthom Dataset

import torch
import numpy as np
from scipy import misc
from torch.utils.data.dataset import Dataset
import os

class Synthom_dataset(Dataset):
    root_folder = ''
    img_res = (640, 480)
    depth_prefix = 'depth_'
    rgb_prefix = 'rgb_'
    file_ext = 'png'
    frame_nums = []
    filepaths_depth = {}

    def fill_filepaths(self):
        self.filepaths_rgb = {}
        self.filepaths_depth = {}
        for root, dirs, files in os.walk(self.root_folder, topdown=True):
            for filename in sorted(files):
                curr_file_ext = filename.split('.')[1]
                if curr_file_ext == self.file_ext:
                    frame_num = int(filename.split('_')[1].split('.')[0])
                    self.frame_nums.append(frame_num)
                    if filename[0:3] == 'rgb':
                        self.filepaths_rgb[frame_num] = filename
                    if filename[0:5] == 'depth':
                        self.filepaths_depth[frame_num] = filename
        self.frame_nums = sorted(self.frame_nums)
        self.length = len(self.frame_nums)

    def load_rgbd(self, idx):
        # read rgb image
        rgb_filepath = self.root_folder + self.rgb_prefix + str(self.frame_nums[idx]) + '.' + self.file_ext
        rgbd_image = misc.imread(rgb_filepath)
        rgbd_image = rgbd_image.swapaxes(0, 1)
        # read depth image
        depth_filepath = self.root_folder + self.depth_prefix + str(self.frame_nums[idx]) + '.' + self.file_ext
        depth_image = misc.imread(depth_filepath)
        depth_image = depth_image.swapaxes(0, 1)
        depth = np.zeros((self.img_res[0], self.img_res[1]))
        for depth_channel in range(2):
            depth += depth_image[:, :, depth_channel] / np.power(255, depth_channel + 1)
        rgbd_image[:, :, 3] = depth
        return rgbd_image

    def __init__(self, root_folder):
        self.root_folder = root_folder
        self.fill_filepaths()

    def __getitem__(self, idx):
        rgbd = self.load_rgbd(idx)
        return rgbd, 0

    def __len__(self):
        return self.length



