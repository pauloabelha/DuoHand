# handler for ECCV HANDS 2018 Workshop Synthom Dataset

import torch
import numpy as np
from scipy import misc
from torch.utils.data.dataset import Dataset
import os
from matplotlib import pyplot as plt

def numpy_to_plottable_rgb(numpy_img):
    img = numpy_img
    if len(numpy_img.shape) == 3:
        channel_axis = 0
        for i in numpy_img.shape:
            if i == 3 or i == 4:
                break
            channel_axis += 1
        if channel_axis == 0:
            img = numpy_img.swapaxes(0, 1)
            img = img.swapaxes(1, 2)
        elif channel_axis == 1:
            img = numpy_img.swapaxes(1, 2)
        elif channel_axis == 2:
            img = numpy_img
        else:
            return None
        img = img[:, :, 0:3]
    img = img.swapaxes(0, 1)
    return img.astype(int)

def plot_image(data, title='', fig=None):
    if fig is None:
        fig = plt.figure()
    data_img_RGB = numpy_to_plottable_rgb(data)
    plt.imshow(data_img_RGB)
    if not title == '':
        plt.title(title)
    return fig

def change_res_image(image, new_res):
    image = misc.imresize(image, new_res)
    return image

def get_crop_coords(joints_uv, image_rgbd, pixel_mult=1.5):
    min_u = min(joints_uv[:, 0])
    min_v = min(joints_uv[:, 1])
    max_u = max(joints_uv[:, 0])
    max_v = max(joints_uv[:, 1])
    u_size = max_u - min_u
    v_size = max_v - min_v
    u_pixel_add = u_size * (pixel_mult - 1)
    v_pixel_add = v_size * (pixel_mult - 1)
    min_u -= u_pixel_add
    min_v -= v_pixel_add
    max_u += u_pixel_add
    max_v += v_pixel_add
    u0 = int(max(min_u, 0))
    v0 = int(max(min_v, 0))
    u1 = int(min(max_u, image_rgbd.shape[1]))
    v1 = int(min(max_v, image_rgbd.shape[2]))
    # get coords
    coords = [u0, v0, u1, v1]
    return coords


def convert_labels_2D_new_res(color_space_label, orig_img_res, heatmap_res):
    new_ix_res1 = int(color_space_label[0] /
                      (orig_img_res[0] / heatmap_res[0]))
    new_ix_res2 = int(color_space_label[1] /
                      (orig_img_res[1] / heatmap_res[1]))
    return np.array([new_ix_res1, new_ix_res2])

def color_space_label_to_heatmap(color_space_label, heatmap_res, orig_img_res=(640, 480)):
    '''
    Convert a (u,v) color-space label into a heatmap
    In this case, the heat map has only one value set to 1
    That is, the value (u,v)
    :param color_space_label: a pair (u,v) of color space joint position
    :param image_res: a pair (U, V) with the values for image resolution
    :return: numpy array of dimensions image_res with one position set to 1
    '''
    SMALL_PROB = 0.0
    heatmap = np.zeros(heatmap_res) + SMALL_PROB
    new_label_res = convert_labels_2D_new_res(color_space_label, orig_img_res, heatmap_res)
    heatmap[new_label_res[0], new_label_res[1]] = 1 - (SMALL_PROB * heatmap.size)
    return heatmap

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
        label = color_space_label_to_heatmap(labels_colorspace_mapped[joint_ix, :], heatmap_res,
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
    #plot_image(data)
    #plt.show()
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

    get_rand_idx = []

    obj_id_of_idx = {}
    obj_pose_of_idx = {}
    obj_uv_of_idx = {}
    hand_pose_of_idx = {}
    hand_uv_of_idx = {}

    idx_to_filestruct = {}
    filestruct_to_idx = {}

    def fill_idx_to_filestruct(self):
        self.filepaths_rgb = {}
        self.filepaths_depth = {}
        elem_ix = 0
        for root, dirs, files in os.walk(self.root_folder, topdown=True):
            for filename in sorted(files):
                curr_file_ext = filename.split('.')[1]
                if curr_file_ext == self.image_ext:
                    scene = filename.split('_')[0]
                    image_type = filename.split('_')[1]
                    frame_num = filename.split('_')[2].split('.')[0]
                    if image_type == 'depth':
                        filestruct = '_'.join([root, scene, frame_num])
                        self.idx_to_filestruct[elem_ix] = filestruct
                        self.filestruct_to_idx[filestruct] = elem_ix
                        elem_ix += 1
        self.length = elem_ix
        self.get_rand_idx = np.array(list(range(self.length)))
        np.random.shuffle(self.get_rand_idx)

    def load_rgbd(self, idx):
        filestruct = self.idx_to_filestruct[idx]
        filestruct_split = filestruct.split('_')
        root_folder = filestruct_split[0]
        scene_obj = filestruct_split[1]
        frame_num = filestruct_split[2]
        # read rgb image
        rgb_filepath = root_folder + '/' + scene_obj + '_' + self.rgb_prefix + frame_num + '.' + self.image_ext
        rgbd_image = misc.imread(rgb_filepath)
        rgbd_image = rgbd_image.swapaxes(0, 1).astype(float)
        # read depth image
        depth_filepath = root_folder + '/' + scene_obj + '_' + self.depth_prefix + frame_num + '.' + self.image_ext
        depth_image = misc.imread(depth_filepath)
        depth_image = depth_image.swapaxes(0, 1)
        depth = np.zeros((self.img_res[0], self.img_res[1])).astype(float)
        for depth_channel in range(2):
            depth += depth_image[:, :, depth_channel] / np.power(255, depth_channel + 1)
        rgbd_image[:, :, 3] = depth
        return rgbd_image

    def fill_obj_poses(self):
        for root, dirs, files in os.walk(self.root_folder, topdown=True):
            for filename in sorted(files):
                curr_file_ext = filename.split('.')[1]
                if curr_file_ext == 'txt' and 'obj_pose_conv' in filename:
                    scene = filename.split('_')[0]
                    obj_filepath = root + '/' + filename
                    with open(obj_filepath) as f:
                        next(f)
                        next(f)
                        next(f)
                        for line in f:
                            line_split = line.split(',')
                            frame_num = int(line_split[0])
                            filestruct = '_'.join([root, scene, str(frame_num)])
                            idx = self.filestruct_to_idx[filestruct]
                            obj_id = int(line_split[1])
                            obj_pose = np.array([float(i) for i in line_split[2:8]])
                            obj_uv = np.array([float(i) for i in line_split[8:10]])
                            self.obj_id_of_idx[idx] = obj_id
                            self.obj_pose_of_idx[idx] = obj_pose
                            self.obj_uv_of_idx[idx] = obj_uv

    def fill_hand_poses(self):
        for root, dirs, files in os.walk(self.root_folder, topdown=True):
            for filename in sorted(files):
                curr_file_ext = filename.split('.')[1]
                if curr_file_ext == 'txt' and 'right_hand_conv' in filename:
                    scene = filename.split('_')[0]
                    hand_filepath = root + '/' + filename
                    with open(hand_filepath) as f:
                        next(f)
                        next(f)
                        next(f)
                        line = '.'
                        while line != '':
                            hand_pose = np.zeros((self.num_hand_bones, 3))
                            hand_uv = np.zeros((self.num_hand_bones, 2))
                            for i in range(self.num_hand_bones):
                                line = f.readline()
                                if line == '':
                                    break
                                line_split = line.split(',')
                                frame_num = int(line_split[0])
                                filestruct = '_'.join([root, scene, str(frame_num)])
                                idx = self.filestruct_to_idx[filestruct]
                                bone_idx = int(line_split[1])
                                hand_pose[bone_idx, :] = [float(j) for j in line_split[2:5]]
                                hand_uv[bone_idx, 0] = int(line_split[5].strip())
                                hand_uv[bone_idx, 1] = int(line_split[6].strip())
                            if not line == '':
                                self.hand_pose_of_idx[idx] = hand_pose
                                self.hand_uv_of_idx[idx] = hand_uv
        a = 0

    def __init__(self, root_folder):
        self.root_folder = root_folder
        self.fill_idx_to_filestruct()
        self.fill_obj_poses()
        self.fill_hand_poses()

    def __getitem__(self, idx):
        #idx = self.get_rand_idx[idx]

        obj_id = self.obj_id_of_idx[idx]
        obj_id_prob = np.zeros((self.num_objs,))
        obj_id_prob[obj_id] = 1.0
        obj_id_prob = torch.from_numpy(obj_id_prob).float()
        obj_orient = self.obj_pose_of_idx[idx][3:].reshape((3,))
        obj_orient = torch.from_numpy(obj_orient).float()
        obj_uv = self.obj_uv_of_idx[idx].reshape((2,))
        obj_uv = torch.from_numpy(obj_uv).float()
        obj_position = self.obj_pose_of_idx[idx][0:3].reshape((3,))
        obj_position = torch.from_numpy(obj_position).float()
        obj_pose = torch.cat((obj_position, obj_uv, obj_orient), 0)

        hand_pose = self.hand_pose_of_idx[idx]
        hand_pose = torch.from_numpy(hand_pose).float()
        target_hand_pose = hand_pose.reshape((hand_pose.shape[0] * 3,))

        rgbd = self.load_rgbd(idx)
        hand_uv = self.hand_uv_of_idx[idx]
        # plot_image(rgbd, title=str(idx))
        # plt.show()
        # rgbd = change_res_image(rgbd, new_res=(128, 128))
        rgbd = rgbd.swapaxes(1, 2).swapaxes(0, 1)
        rgbd, crop_coords, target_heatmaps, _ = crop_image_get_labels(rgbd, hand_uv)
        rgbd = torch.from_numpy(rgbd).float()
        target_heatmaps = torch.from_numpy(target_heatmaps).float()

        return (rgbd, obj_id_prob, obj_pose), (target_hand_pose, target_heatmaps)

    def __len__(self):
        return self.length



