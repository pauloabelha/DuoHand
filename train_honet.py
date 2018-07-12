import torch
import synthom_handler
from torch.autograd import Variable
from HONet import HONet

def euclidean_loss(output, target):
    batch_size = output.data.shape[0]
    return (output - target).abs().sum() / batch_size

def cross_entropy_loss_p_logq(torchvar_p, torchvar_logq, eps=1e-9):
    batch_size = torchvar_p.data.shape[0]
    return (-((torchvar_p + eps) * torchvar_logq + eps).sum(dim=1).sum(dim=1)).sum() / batch_size

def calculate_subloss_JORNet(loss_func, output_hm, output_j, target_heatmaps, target_joints,
                             joint_ixs, weight_heatmaps_loss, weight_joints_loss, iter_size):
    loss_heatmaps = 0
    for joint_ix in joint_ixs:
        loss_heatmaps += loss_func(output_hm[:, joint_ix, :, :], target_heatmaps[:, joint_ix, :, :])
    loss_heatmaps /= iter_size
    loss_joints = euclidean_loss(output_j, target_joints)
    loss_joints /= iter_size
    loss = (weight_heatmaps_loss * loss_heatmaps) + (weight_joints_loss * loss_joints)
    return loss, loss_heatmaps, loss_joints

def calculate_loss_JORNet(loss_func, output, target_heatmaps, target_joints, joint_ixs,
                          weights_heatmaps_loss, weights_joints_loss, iter_size):
    loss = 0
    loss_heatmaps = 0
    loss_joints = 0
    for loss_ix in range(4):
        loss_sub, loss_heatmaps_sub, loss_joints_sub =\
            calculate_subloss_JORNet(loss_func, output[loss_ix], output[loss_ix+4],
                                     target_heatmaps, target_joints, joint_ixs,
                                     weights_heatmaps_loss[loss_ix], weights_joints_loss[loss_ix],
                                     iter_size)
        loss += loss_sub
        loss_heatmaps += loss_heatmaps_sub
        loss_joints += loss_joints_sub
    return loss, loss_heatmaps, loss_joints

def get_loss_weights(curr_iter):
    weights_heatmaps_loss = [0.5, 0.5, 0.5, 1.0]
    weights_joints_loss = [1250, 1250, 1250, 2500]
    if curr_iter > 45000:
        weights_heatmaps_loss = [0.1, 0.1, 0.1, 1.0]
        weights_joints_loss = [250, 250, 250, 2500]
    return weights_heatmaps_loss, weights_joints_loss

root_folder = '/home/paulo/MockDataset1/'

synthom_dataset = synthom_handler.Synthom_dataset(root_folder)
synthom_loader = torch.utils.data.DataLoader(
                                            synthom_dataset,
                                            batch_size=1,
                                            shuffle=False)
honet_params = {'num_joints': 16}
honet = HONet(honet_params)

for batch_idx, (data, target) in enumerate(synthom_loader):
    (rgbd, obj_id, obj_pose) = data
    target_joints, target_heatmaps = target
    rgbd, obj_id, obj_pose, target_joints, target_heatmaps = Variable(rgbd), Variable(obj_id),\
                                     Variable(obj_pose), Variable(target_joints), \
                                     Variable(target_heatmaps)
    output = honet(data)
    weights_heatmaps_loss, weights_joints_loss = get_loss_weights(batch_idx)
    loss_func = cross_entropy_loss_p_logq
    loss, loss_heatmaps, loss_joints = calculate_loss_JORNet(
        loss_func, output, target_heatmaps, target_joints, range(16),
        weights_heatmaps_loss, weights_joints_loss, 1)
    

    a = 0





