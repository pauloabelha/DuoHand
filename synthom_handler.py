# handler for ECCV HANDS 2018 Workshop Synthom Dataset

import torch
import numpy as np
from scipy import misc
from torch.utils.data.dataset import Dataset
import os

def change_res_image(image, new_res):
    image = misc.imresize(image, new_res)
    return image

def get_crop_coords(joints_uv, image_rgbd):
    min_u = min(joints_uv[:, 0]) - 10
    min_v = min(joints_uv[:, 1]) - 10
    max_u = max(joints_uv[:, 0]) + 10
    max_v = max(joints_uv[:, 1]) + 10
    u0 = int(max(min_u, 0))
    v0 = int(max(min_v, 0))
    u1 = int(min(max_u, image_rgbd.shape[1]))
    v1 = int(min(max_v, image_rgbd.shape[2]))
    # get coords
    coords = [u0, v0, u1, v1]
    return coords

def get_labels_cropped_heatmaps(labels_colorspace, joint_ixs, crop_coords, heatmap_res):
    res_transf_u = (heatmap_res[0] / (crop_coords[2] - crop_coords[0]))
    res_transf_v = (heatmap_res[1] / (crop_coords[3] - crop_coords[1]))
    labels_ix = 0
    labels_heatmaps = np.zeros((len(joint_ixs), heatmap_res[0], heatmap_res[1]))
    labels_colorspace_mapped = np.copy(labels_colorspace)
    for joint_ix in joint_ixs:
        label_crop_local_u = labels_colorspace[joint_ix, 0] - crop_coords[0]
        label_crop_local_v = labels_colorspace[joint_ix, 1] - crop_coords[1]
        label_u = int(label_crop_local_u * res_transf_u)
        label_v = int(label_crop_local_v * res_transf_v)
        labels_colorspace_mapped[joint_ix, 0] = label_u
        labels_colorspace_mapped[joint_ix, 1] = label_v
        label = conv.color_space_label_to_heatmap(labels_colorspace_mapped[joint_ix, :], heatmap_res,
                                                     orig_img_res=heatmap_res)
        label = label.astype(float)
        labels_heatmaps[labels_ix, :, :] = label
        labels_ix += 1
    return labels_heatmaps, labels_colorspace_mapped

def crop_hand_rgbd(joints_uv, image_rgbd, crop_res):
    crop_coords = get_crop_coords(joints_uv, image_rgbd)
    # crop hand
    crop = image_rgbd[:, crop_coords[0]:crop_coords[2], crop_coords[1]:crop_coords[3]]
    crop = crop.swapaxes(0, 1)
    crop = crop.swapaxes(1, 2)
    crop_rgb = change_res_image(crop[:, :, 0:3], crop_res)
    crop_depth = change_res_image(crop[:, :, 3], crop_res)
    # normalize depth
    crop_depth = np.divide(crop_depth, np.max(crop_depth))
    crop_depth = crop_depth.reshape(crop_depth.shape[0], crop_depth.shape[1], 1)
    crop_rgbd = np.append(crop_rgb, crop_depth, axis=2)
    crop_rgbd = crop_rgbd.swapaxes(1, 2)
    crop_rgbd = crop_rgbd.swapaxes(0, 1)
    return crop_rgbd, crop_coords

def crop_image_get_labels(data, labels_colorspace, joint_ixs=range(16), crop_res=(128, 128)):
    data, crop_coords = crop_hand_rgbd(labels_colorspace, data, crop_res=crop_res)
    labels_heatmaps, labels_colorspace =\
        get_labels_cropped_heatmaps(labels_colorspace, joint_ixs, crop_coords, heatmap_res=crop_res)
    return data, crop_coords, labels_heatmaps, labels_colorspace

class Synthom_dataset(Dataset):
    root_folder = ''
    img_res = (640, 480)
    num_objs = 4
    num_hand_bones = 16
    depth_prefix = 'depth_'
    rgb_prefix = 'rgb_'
    image_ext = 'png'
    obj_file_prefix = 'obj_pose_conv'
    hand_file_prefix = 'right_hand_conv'
    frame_nums = []
    filepaths_depth = {}

    obj_id_of_frame = {}
    obj_pose_of_frame = {}
    hand_pose_of_frame = {}

    def fill_filepaths(self):
        self.filepaths_rgb = {}
        self.filepaths_depth = {}
        for root, dirs, files in os.walk(self.root_folder, topdown=True):
            for filename in sorted(files):
                curr_file_ext = filename.split('.')[1]
                if curr_file_ext == self.image_ext:
                    frame_num = int(filename.split('_')[1].split('.')[0])
                    self.frame_nums.append(frame_num)
                    if filename[0:3] == 'rgb':
                        self.filepaths_rgb[frame_num] = filename
                    if filename[0:5] == 'depth':
                        self.filepaths_depth[frame_num] = filename
        self.frame_nums = sorted(self.frame_nums)
        self.length = len(self.frame_nums)

    def load_rgbd(self, idx):
        frame_num = self.frame_nums[idx]
        # read rgb image
        rgb_filepath = self.root_folder + self.rgb_prefix + str(frame_num) + '.' + self.image_ext
        rgbd_image = misc.imread(rgb_filepath)
        rgbd_image = rgbd_image.swapaxes(0, 1)
        # read depth image
        depth_filepath = self.root_folder + self.depth_prefix + str(frame_num) + '.' + self.image_ext
        depth_image = misc.imread(depth_filepath)
        depth_image = depth_image.swapaxes(0, 1)
        depth = np.zeros((self.img_res[0], self.img_res[1]))
        for depth_channel in range(2):
            depth += depth_image[:, :, depth_channel] / np.power(255, depth_channel + 1)
        rgbd_image[:, :, 3] = depth
        return rgbd_image

    def fill_obj_poses(self):
        obj_filepath = self.root_folder + self.obj_file_prefix + '.txt'
        with open(obj_filepath) as f:
            next(f)
            next(f)
            next(f)
            for line in f:
                line_split = line.split(',')
                frame_num = int(line_split[0])
                obj_id = int(line_split[1])
                obj_pose = np.array([float(i) for i in line_split[2:]])
                self.obj_id_of_frame[frame_num] = obj_id
                self.obj_pose_of_frame[frame_num] = obj_pose

    def fill_hand_poses(self):
        hand_filepath = self.root_folder + self.hand_file_prefix + '.txt'

        with open(hand_filepath) as f:
            next(f)
            next(f)
            next(f)
            line = '.'
            while line != '':
                hand_pose = np.zeros((self.num_hand_bones, 3))
                for i in range(self.num_hand_bones - 1):
                    line = f.readline()
                    if line == '':
                        break
                    line_split = line.split(',')
                    frame_num = int(line_split[0])
                    bone_idx = int(line_split[1])
                    hand_pose[bone_idx, :] = [float(i) for i in line_split[2:]]
                self.hand_pose_of_frame[frame_num] = hand_pose

    def __init__(self, root_folder):
        self.root_folder = root_folder
        self.fill_filepaths()
        self.fill_obj_poses()
        self.fill_hand_poses()

    def __getitem__(self, idx):
        rgbd = self.load_rgbd(idx)
        rgbd = change_res_image(rgbd, new_res=(128, 128))
        #rgbd, _, _, _ = crop_image_get_labels(rgbd)

        rgbd = rgbd.swapaxes(1, 2).swapaxes(0, 1)
        rgbd = torch.from_numpy(rgbd).float()
        frame_num = self.frame_nums[idx]
        obj_id = self.obj_id_of_frame[frame_num]
        obj_id_prob = np.zeros((self.num_objs, 1))
        obj_id_prob[obj_id] = 1.0
        obj_id_prob = torch.from_numpy(obj_id_prob).float()
        obj_pose = self.obj_pose_of_frame[frame_num]
        obj_pose = torch.from_numpy(obj_pose).float()
        hand_pose = self.hand_pose_of_frame[frame_num]
        hand_pose = torch.from_numpy(hand_pose).float()
        return (rgbd, obj_id_prob, obj_pose), hand_pose

    def __len__(self):
        return self.length



