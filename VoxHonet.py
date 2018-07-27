import VargoNet
import torch
from VargoNet import VargoNet as VargoNet_class
import torch.nn as nn
from VargoNet import cudafy
import numpy as np
from pointcloud import PointCloud
from pyvox import voxelize
import my_linalg

class VoxHonet(VargoNet_class):
    max_obj_size = 0.3 # in m
    innerprod1_size = 256 * 16 * 16
    crop_res = (128, 128)
    size_obj_input = 4 + 3 + 3
    voxel_grid_side = 50
    pcl_names = ['crackers', 'mustard', 'orange', 'woodblock']
    pcls = []



    def load_pcls(self, dataset_folder):
        self.pcls = []
        max_size = 0.22
        for i in range(len(self.pcl_names)):
            filepath = dataset_folder + 'pcls/' + self.pcl_names[i] + '.ply'
            self.pcls.append(PointCloud.from_file(filepath))

    def map_out_to_loss(self, innerprod1_size):
        return cudafy(nn.Linear(in_features=innerprod1_size, out_features=200), self.use_cuda)

    def map_out_conv(self, in_channels):
        return cudafy(VargoNet.VargoNetConvBlock(
            kernel_size=3, stride=1, filters=21, in_channels=in_channels, padding=1),
            self.use_cuda)

    def __init__(self, params_dict):
        super(VoxHonet, self).__init__(params_dict)

        self.voxel_grid_side = params_dict['voxel_grid_side']
        self.load_pcls(params_dict['dataset_folder'])

        self.num_joints = 16
        self.main_loss_conv = cudafy(VargoNet.VargoNetConvBlock(
                kernel_size=3, stride=1, filters=21, in_channels=256, padding=1),
            self.use_cuda)
        self.main_loss_deconv1 = cudafy(nn.Upsample(size=self.crop_res, mode='bilinear'), self.use_cuda)
        if self.cross_entropy:
            self.softmax_final = cudafy(VargoNet.SoftmaxLogProbability2D(), self.use_cuda)
        self.innerproduct1_joint1 = cudafy(
            nn.Linear(in_features=524288, out_features=200), self.use_cuda)
        self.innerproduct2_joint1 = cudafy(
            nn.Linear(in_features=200, out_features=(self.num_joints - 1) * 3), self.use_cuda)

        self.innerproduct1_joint2 = cudafy(
            nn.Linear(in_features=262144, out_features=200), self.use_cuda)
        self.innerproduct2_joint2 = cudafy(
            nn.Linear(in_features=200, out_features=(self.num_joints - 1) * 3), self.use_cuda)

        self.innerproduct1_joint3 = cudafy(
            nn.Linear(in_features=131072, out_features=200), self.use_cuda)
        self.innerproduct2_joint3 = cudafy(
            nn.Linear(in_features=200, out_features=(self.num_joints - 1) * 3), self.use_cuda)

        self.innerproduct1_joint_main = cudafy(
            nn.Linear(in_features=65536, out_features=200), self.use_cuda)
        self.innerproduct2_join_main = cudafy(
            nn.Linear(in_features=200, out_features=200), self.use_cuda)

        self.obj_voxel_in = cudafy(nn.Conv3d(in_channels=1, out_channels=1, kernel_size=6, stride=2), self.use_cuda)
        self.obj_conv3d0 = cudafy(nn.Conv3d(in_channels=1, out_channels=1, kernel_size=5, stride=2), self.use_cuda)
        self.obj_conv3d1 = cudafy(nn.Conv3d(in_channels=1, out_channels=1, kernel_size=5), self.use_cuda)

        self.funnel_in0 = cudafy(
            nn.Linear(in_features=200+216, out_features=100), self.use_cuda)
        self.funnel_in1 = cudafy(
            nn.Linear(in_features=100, out_features=100), self.use_cuda)
        self.funnel_in2 = cudafy(
            nn.Linear(in_features=100, out_features=(self.num_joints - 1) * 3), self.use_cuda)


    def get_obj_voxel(self, obj_id, obj_pose):
        voxel_grid = np.zeros((obj_id.shape[0], 1, self.voxel_grid_side, self.voxel_grid_side, self.voxel_grid_side))
        for i in range(obj_id.shape[0]):
            pcl = self.pcls[np.argmax(obj_id[i])]
            rot_mtx = my_linalg.get_eul_rot_mtx(obj_pose[i][3:], axes_order='xyz')
            pcl.vertices = np.dot(pcl.vertices, rot_mtx)
            voxel_grid[i, 0] = voxelize(pcl.vertices, max_size=self.max_obj_size, voxel_grid_side=self.voxel_grid_side)
        return voxel_grid

    def forward(self, x):
        (rgbd, obj_id, obj_pose) = x

        voxel_grid = self.get_obj_voxel(obj_id, obj_pose)
        voxel_grid = torch.from_numpy(voxel_grid).float()
        if self.use_cuda:
            voxel_grid = voxel_grid.cuda()
        out_obj = self.obj_voxel_in(voxel_grid)
        out_obj = self.obj_conv3d0(out_obj)
        out_obj = self.obj_conv3d1(out_obj)
        out_obj_size = out_obj.shape[1] * out_obj.shape[2] * out_obj.shape[3] * out_obj.shape[4]
        out_obj_out = out_obj.view(-1, out_obj_size)

        out_intermed_hm1, out_intermed_hm2, out_intermed_hm3, conv4fout, \
        res3aout, res4aout, conv4eout = self.forward_subnet(rgbd)
        out_intermed_hm_main = self.forward_main_loss(conv4fout)
        innerprod1_size = res3aout.shape[1] * res3aout.shape[2] * res3aout.shape[3]
        out_intermed_j1 = res3aout.view(-1, innerprod1_size)
        out_intermed_j1 = self.innerproduct1_joint1(out_intermed_j1)
        out_intermed_j1 = self.innerproduct2_joint1(out_intermed_j1)

        innerprod1_size = res4aout.shape[1] * res4aout.shape[2] * res4aout.shape[3]
        out_intermed_j2 = res4aout.view(-1, innerprod1_size)
        out_intermed_j2 = self.innerproduct1_joint2(out_intermed_j2)
        out_intermed_j2 = self.innerproduct2_joint2(out_intermed_j2)

        innerprod1_size = conv4eout.shape[1] * conv4eout.shape[2] * conv4eout.shape[3]
        out_intermed_j3 = conv4eout.view(-1, innerprod1_size)
        out_intermed_j3 = self.innerproduct1_joint3(out_intermed_j3)
        out_intermed_j3 = self.innerproduct2_joint3(out_intermed_j3)

        innerprod1_size = conv4fout.shape[1] * conv4fout.shape[2] * conv4fout.shape[3]
        out_intermed_j_main = conv4fout.view(-1, innerprod1_size)
        out_intermed_j_main = self.innerproduct1_joint_main(out_intermed_j_main)

        out_hand_obj = torch.cat((out_obj_out, out_intermed_j_main), 1)
        joints_out = self.funnel_in0(out_hand_obj)
        joints_out = self.funnel_in1(joints_out)
        joints_out = self.funnel_in2(joints_out)

        return out_intermed_hm1, out_intermed_hm2, out_intermed_hm3, out_intermed_hm_main,\
               out_intermed_j1, out_intermed_j2, out_intermed_j3, joints_out
