import torch.nn as nn
import torch.nn.functional as F

def cudafy(object, use_cuda):
    if use_cuda:
        return object.cuda()
    else:
        return object

class HORNet(nn.Module):

    use_cuda = False                     # 16 joints excluding root
    num_joints = 15
    size_input = 55


    def __init__(self, params_dict):
        super(HORNet, self).__init__()

        self.use_cuda = params_dict['use_cuda']
        self.num_joints = params_dict['num_joints']
        self.size_input = ((self.num_joints - 1) * 3) + 10  # 3D joints + obj info (obj softmax vec (4), pos (3) and rot (3))

        self.layer_in = cudafy(
            nn.Linear(in_features=self.size_input, out_features=self.size_input * 2), self.use_cuda)
        self.layer2 = cudafy(
            nn.Linear(in_features=self.size_input * 2, out_features=self.size_input * 2), self.use_cuda)
        self.layer3 = cudafy(
            nn.Linear(in_features=self.size_input * 2, out_features=self.size_input * 2), self.use_cuda)

        self.layer_out = cudafy(
            nn.Linear(in_features=self.size_input * 2, out_features=(self.num_joints - 1) * 3), self.use_cuda)

    def forward(self, x):
        out = F.relu(self.layer_in(x))
        out = F.relu(self.layer2(out))
        out = F.relu(self.layer3(out))
        out = F.relu(self.layer_out(out))

        return out