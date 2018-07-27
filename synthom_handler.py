# handler for ECCV HANDS 2018 Workshop Synthom Dataset

import pickle
from scipy import misc
from torch.utils.data.dataset import Dataset
import os
from util import *

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
    new_label_res[0] = max(0, new_label_res[0])
    new_label_res[1] = max(0, new_label_res[1])
    new_label_res[0] = min(heatmap_res[0] - 1, new_label_res[0])
    new_label_res[1] = min(heatmap_res[1] - 1, new_label_res[1])
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

    min_obj_pose = 0
    max_obj_pose = 0

    get_rand_idx = []

    obj_gt_of_idx = {}
    hand_gt_of_idx = {}
    rgb_filepath_of_idx = {}
    depth_filepath_of_idx = {}

    train_split_size = 0

    def fillall(self):
        self.filepaths_rgb = {}
        self.filepaths_depth = {}
        ix = 0
        idx_elem = 0
        min_obj_pose = [10000] * 6
        max_obj_pose = [-10000] * 6
        for root, dirs, files in os.walk(self.root_folder, topdown=True):
            if ix == 0:
                ix += 1
                continue
            folder_name = root.split('/')[-1]
            idx_num = 0
            for char_ in folder_name:
                if char_.isdigit():
                    break
                idx_num += 1
            aa = len(folder_name)-idx_num
            obj_name = folder_name[:-aa]
            hand_gt_filepath = root + '/' + obj_name + '_right_hand_conv.txt'
            obj_gt_filepath = root + '/' + obj_name + '_obj_pose_conv.txt'
            with open(hand_gt_filepath) as hand_file:
                print(hand_gt_filepath)
                with open(obj_gt_filepath) as obj_file:
                    print(obj_gt_filepath)
                    next(hand_file)
                    next(hand_file)
                    next(hand_file)
                    next(obj_file)
                    next(obj_file)
                    next(obj_file)

                    obj_line = obj_file.readline()
                    while not obj_line == '':
                        line_split = obj_line.split(',')
                        obj_frame_num = int(line_split[0])

                        self.rgb_filepath_of_idx[idx_elem] = root + '/' + obj_name +\
                                                             '_rgb_' + str(obj_frame_num) + '.png'
                        self.depth_filepath_of_idx[idx_elem] = root + '/' + obj_name +\
                                                             '_depth_' + str(obj_frame_num) + '.png'


                        obj_id = int(line_split[1])
                        obj_pose = np.array([float(i) for i in line_split[2:8]])

                        obj_uv = np.array([float(i) for i in line_split[8:10]])
                        if obj_pose[0] < 0 or obj_pose[1] < 0 or obj_pose[2] < 270 \
                                or obj_uv[0] < 0 or obj_uv[1] < 0:
                            obj_line = obj_file.readline()

                            for i in range(self.num_hand_bones):
                                hand_line = hand_file.readline()
                            continue



                        hand_pose = np.zeros((self.num_hand_bones, 3))
                        hand_uv = np.zeros((self.num_hand_bones, 2))
                        for i in range(self.num_hand_bones):
                            hand_line = hand_file.readline()
                            if hand_line == '':
                                raise Exception('Reached end of hand file before end of object file')
                            line_split = hand_line.split(',')
                            hand_frame_num = int(line_split[0])
                            if not hand_frame_num == obj_frame_num:
                                obj_line = obj_file.readline()
                                for i in range(self.num_hand_bones):
                                    hand_file.readline()
                                #raise Exception('Hand and object files are inconsistent!\n{}\n{}\n{}\n{}'.
                                #                format(hand_gt_filepath, obj_gt_filepath, obj_line, hand_line))
                            bone_idx = int(line_split[1])
                            hand_pose[bone_idx, :] = [float(j) for j in line_split[2:5]]
                            hand_uv[bone_idx, 0] = int(line_split[5].strip())
                            hand_uv[bone_idx, 1] = int(line_split[6].strip())
                        hand_root = np.copy(hand_pose[0, :])
                        hand_pose -= hand_root
                        hand_pose = hand_pose[1:, :]
                        hand_root *= 10
                        hand_pose *= 10
                        self.hand_gt_of_idx[idx_elem] = (hand_root, hand_pose, hand_uv)
                        obj_line = obj_file.readline()

                        obj_pose[0:3] = (obj_pose[0:3] * 10) - hand_root
                        for i in range(3):
                            if obj_pose[i] < min_obj_pose[i]:
                                min_obj_pose[i] = obj_pose[i]
                            if obj_pose[i] > max_obj_pose[i]:
                                max_obj_pose[i] = obj_pose[i]

                        for i in range(3):
                            idx = i + 3
                            if obj_pose[idx] < min_obj_pose[idx]:
                                min_obj_pose[idx] = obj_pose[idx]
                            if obj_pose[idx] > max_obj_pose[idx]:
                                max_obj_pose[idx] = obj_pose[idx]


                        self.obj_gt_of_idx[idx_elem] = (obj_id, obj_pose, obj_uv)

                        idx_elem += 1
        self.min_obj_pose = np.array(min_obj_pose)
        self.max_obj_pose = np.array(max_obj_pose)
        self.length = idx_elem - 1
        self.get_rand_idx = list(range(self.length))
        np.random.shuffle(self.get_rand_idx)

    def load_rgbd(self, idx):
        # read rgb image
        rgb_filepath = self.rgb_filepath_of_idx[idx]
        rgbd_image = misc.imread(rgb_filepath)
        rgbd_image = rgbd_image.swapaxes(0, 1).astype(float)
        # read depth image
        depth_filepath = self.depth_filepath_of_idx[idx]
        depth_image = misc.imread(depth_filepath)
        depth_image = depth_image.swapaxes(0, 1)
        depth = np.zeros((self.img_res[0], self.img_res[1])).astype(float)
        for depth_channel in range(2):
            depth += depth_image[:, :, depth_channel] / np.power(255, depth_channel + 1)
        rgbd_image[:, :, 3] = depth
        return rgbd_image

    def __init__(self, root_folder, load=True, type='train', train_split_prop=0.9):
        self.root_folder = root_folder
        if type == 'test':
            print('Loading test set from file')
            with open(root_folder + 'dataset.pkl', 'rb') as handle:
                dataset_dict = pickle.load(handle)
                self.train_split_size = dataset_dict['train_split_size']
                self.get_rand_idx = dataset_dict['get_rand_idx'][self.train_split_size:]
                self.length = len(self.get_rand_idx)
                self.obj_gt_of_idx = dataset_dict['obj_gt_of_idx']
                self.hand_gt_of_idx = dataset_dict['hand_gt_of_idx']
                self.rgb_filepath_of_idx = dataset_dict['rgb_filepath_of_idx']
                self.depth_filepath_of_idx = dataset_dict['depth_filepath_of_idx']
                self.max_obj_pose = dataset_dict['max_obj_pose']
                self.min_obj_pose = dataset_dict['min_obj_pose']
        elif type == 'train':
            if load:
                print('Loading dataset from file')
                with open(root_folder + 'dataset.pkl', 'rb') as handle:
                    dataset_dict = pickle.load(handle)
                    self.train_split_size = dataset_dict['train_split_size']
                    self.get_rand_idx = dataset_dict['get_rand_idx'][0:self.train_split_size]
                    self.length = len(self.get_rand_idx)
                    self.obj_gt_of_idx = dataset_dict['obj_gt_of_idx']
                    self.hand_gt_of_idx = dataset_dict['hand_gt_of_idx']
                    self.rgb_filepath_of_idx = dataset_dict['rgb_filepath_of_idx']
                    self.depth_filepath_of_idx = dataset_dict['depth_filepath_of_idx']
                    self.max_obj_pose = dataset_dict['max_obj_pose']
                    self.min_obj_pose = dataset_dict['min_obj_pose']
            else:
                print('Creating new dataset split')
                self.fillall()
                self.train_split_size = int(self.length * train_split_prop)
                get_rand_idx_tot = self.get_rand_idx
                self.get_rand_idx = get_rand_idx_tot[0:self.train_split_size]
                self.length = len(self.get_rand_idx)
                with open(root_folder + 'dataset.pkl', 'wb') as handle:
                    dataset_dict = {'get_rand_idx': get_rand_idx_tot,
                                  'train_split_size': self.train_split_size,
                                  'obj_gt_of_idx': self.obj_gt_of_idx,
                                  'hand_gt_of_idx': self.hand_gt_of_idx,
                                  'rgb_filepath_of_idx': self.rgb_filepath_of_idx,
                                  'depth_filepath_of_idx': self.depth_filepath_of_idx,
                                    'min_obj_pose': self.min_obj_pose,
                                    'max_obj_pose': self.max_obj_pose}
                    pickle.dump(dataset_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def __getitem__(self, idx):
        idx = self.get_rand_idx[idx]

        hand_gt = self.hand_gt_of_idx[idx]
        hand_root, hand_pose, hand_uv = hand_gt
        hand_pose = torch.from_numpy(hand_pose).float()
        target_hand_pose = hand_pose.reshape((hand_pose.shape[0] * 3,))

        obj_gt = self.obj_gt_of_idx[idx]
        obj_id, obj_pose, obj_uv = obj_gt
        obj_id_prob = np.zeros((self.num_objs,))
        obj_id_prob[obj_id] = 1.0
        obj_id_prob = torch.from_numpy(obj_id_prob).float()
        obj_orient = obj_pose[3:].reshape((3,))
        #obj_orient = (obj_orient + 180.) / 360.
        obj_orient = torch.from_numpy(obj_orient).float()
        obj_uv[0] = obj_uv[0] / 640
        obj_uv[1] = obj_uv[1] / 480
        obj_uv = torch.from_numpy(obj_uv).float()
        obj_position = obj_pose[0:3].reshape((3,))
        #print(self.min_obj_pose)
        #print(self.max_obj_pose)
        obj_position = (obj_position - self.min_obj_pose[0:3]) / (self.max_obj_pose[0:3] - self.min_obj_pose[0:3])
        obj_position = torch.from_numpy(obj_position).float()
        obj_pose = torch.cat((obj_position, obj_orient), 0)

        rgbd = self.load_rgbd(idx)
        #plot_image(rgbd, title=str(idx))
        #plt.show()
        # rgbd = change_res_image(rgbd, new_res=(128, 128))
        rgbd = rgbd.swapaxes(1, 2).swapaxes(0, 1)
        rgbd, crop_coords, target_heatmaps, _ = crop_image_get_labels(rgbd, hand_uv)

        #plot_image(rgbd, title=str(idx))
        #plt.show()
        #plot_3D_joints(target_hand_pose)
        #show()

        rgbd = torch.from_numpy(rgbd).float()
        target_heatmaps = torch.from_numpy(target_heatmaps).float()

        return (rgbd, obj_id_prob, obj_pose, hand_root), (target_hand_pose, target_heatmaps)

    def __len__(self):
        return self.length



