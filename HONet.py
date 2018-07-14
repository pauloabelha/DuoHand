import VargoNet
import torch
from VargoNet import VargoNet as VargoNet_class
import torch.nn as nn
from VargoNet import cudafy

class HONet(VargoNet_class):
    innerprod1_size = 256 * 16 * 16
    crop_res = (128, 128)
    size_obj_input = 4 + 3 + 2 + 3
    #innerprod1_size = 65536

    def map_out_to_loss(self, innerprod1_size):
        return cudafy(nn.Linear(in_features=innerprod1_size, out_features=200), self.use_cuda)

    def map_out_conv(self, in_channels):
        return cudafy(VargoNet.VargoNetConvBlock(
            kernel_size=3, stride=1, filters=21, in_channels=in_channels, padding=1),
            self.use_cuda)

    def __init__(self, params_dict):
        super(HONet, self).__init__(params_dict)

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
            nn.Linear(in_features=200, out_features=self.num_joints * 3), self.use_cuda)

        self.innerproduct1_joint2 = cudafy(
            nn.Linear(in_features=262144, out_features=200), self.use_cuda)
        self.innerproduct2_joint2 = cudafy(
            nn.Linear(in_features=200, out_features=self.num_joints * 3), self.use_cuda)

        self.innerproduct1_joint3 = cudafy(
            nn.Linear(in_features=131072, out_features=200), self.use_cuda)
        self.innerproduct2_joint3 = cudafy(
            nn.Linear(in_features=200, out_features=self.num_joints * 3), self.use_cuda)

        self.innerproduct1_joint_main = cudafy(
            nn.Linear(in_features=65536, out_features=200), self.use_cuda)
        self.innerproduct2_join_main = cudafy(
            nn.Linear(in_features=200, out_features=200), self.use_cuda)

        self.obj0 = cudafy(
            nn.Linear(in_features=self.size_obj_input, out_features=self.size_obj_input), self.use_cuda)
        self.obj1 = cudafy(
            nn.Linear(in_features=self.size_obj_input, out_features=self.size_obj_input), self.use_cuda)
        self.obj2 = cudafy(
            nn.Linear(in_features=self.size_obj_input, out_features=self.size_obj_input), self.use_cuda)

        self.merge_hand_obj = cudafy(
            nn.Linear(in_features=200+self.size_obj_input, out_features=200+self.size_obj_input), self.use_cuda)
        self.hand_obj0 = cudafy(
            nn.Linear(in_features=200+self.size_obj_input, out_features=100), self.use_cuda)
        self.hand_obj1 = cudafy(
            nn.Linear(in_features=100, out_features=48), self.use_cuda)



    def forward(self, x):
        (rgbd, obj_id, obj_pose) = x

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
        #out_intermed_j_main = self.innerproduct2_join_main(out_intermed_j_main)

        # put object in the net
        obj_in = torch.cat((obj_id, obj_pose), 1)
        obj_out = self.obj0(obj_in)
        #obj_out = self.obj1(obj_out)
        #obj_out = self.obj2(obj_out)

        # merge hand and object
        obj_hand_out = torch.cat((out_intermed_j_main, obj_out), 1)
        #obj_hand_out = self.merge_hand_obj(obj_hand_out)
        obj_hand_out = self.hand_obj0(obj_hand_out)
        joints_out = self.hand_obj1(obj_hand_out)

        return out_intermed_hm1, out_intermed_hm2, out_intermed_hm3, out_intermed_hm_main,\
               out_intermed_j1, out_intermed_j2, out_intermed_j3, joints_out
