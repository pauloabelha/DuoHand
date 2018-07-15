import torch
import numpy as np
import torch.optim as optim
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization!

ADADELTA_LEARNING_RATE = 0.05
ADADELTA_MOMENTUM = 0.9
ADADELTA_WEIGHT_DECAY = 0.005

def get_adadelta(halnet,
                        momentum=ADADELTA_MOMENTUM,
                        weight_decay=ADADELTA_WEIGHT_DECAY,
                        learning_rate=ADADELTA_LEARNING_RATE):
    return optim.Adadelta(halnet.parameters(),
                               rho=momentum,
                               weight_decay=weight_decay,
                               lr=learning_rate)

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
    return loss, loss_heatmaps, loss_joints, loss_joints_sub

def get_loss_weights(curr_iter):
    weights_heatmaps_loss = [0.5, 0.5, 0.5, 1.0]
    weights_joints_loss = [1250, 1250, 1250, 2500]
    if curr_iter > 45000:
        weights_heatmaps_loss = [0.1, 0.1, 0.1, 1.0]
        weights_joints_loss = [250, 250, 250, 2500]
    return weights_heatmaps_loss, weights_joints_loss

def load_checkpoint(filename, NetworkClass, params_dict, use_cuda=False):
    print('Loading model from: {}'.format(filename))
    torch_file = torch.load(filename, map_location=lambda storage, loc: storage)
    '''
    if use_cuda:
        try:
            torch_file = torch.load(filename)
        except:
            torch_file = torch.load(filename, map_location=lambda storage, loc: storage)
            use_cuda = False
    else:
        torch_file = torch.load(filename, map_location=lambda storage, loc: storage)
    '''
    model_state_dict = torch_file['model_state_dict']
    params_dict['use_cuda'] = use_cuda
    if not use_cuda:
        params_dict['use_cuda'] = False
    model = NetworkClass(params_dict)
    model.load_state_dict(model_state_dict)
    if use_cuda:
        model = model.cuda()
    optimizer_state_dict = torch_file['optimizer_state_dict']
    optimizer = torch.optim.Adadelta(model.parameters())
    optimizer.load_state_dict(optimizer_state_dict)
    del optimizer_state_dict, model_state_dict
    start_batch_idx = torch_file['batch_idx'] + 1
    return model, optimizer, start_batch_idx

def calc_avg_joint_loss(output_main, target_joints, num_joints=16):
    loss = 0.
    range_ = target_joints.shape[0]
    for i in range(range_):
        target_joints_ = target_joints[i].reshape((num_joints, 3))
        output_joints = output_main[i].reshape((num_joints, 3))
        dist = np.sqrt(np.sum(np.square(target_joints_[:, np.newaxis, :] - output_joints), axis=2))
        loss += float(np.sum(np.diag(dist)) / num_joints)
    loss /= range_
    return loss

def plot_3D_joints(joints_vec, num_joints=16, title='', fig=None, ax=None, color=None):
    if fig is None:
        fig = plt.figure()
        ax = Axes3D(fig)
    if joints_vec.shape[0] == (num_joints - 1) * 3:
        joints_vec = joints_vec.reshape((num_joints - 1, 3))
        joints_vec = np.vstack([np.zeros((1, 3)), joints_vec])
    else:
        joints_vec = joints_vec.reshape((num_joints, 3))
    for i in range(5):
        idx = (i * 3) + 1
        if color is None:
            curr_color = 'C0'
        else:
            curr_color = color
        ax.plot([joints_vec[0, 1], joints_vec[idx, 1]],
                [joints_vec[0, 0], joints_vec[idx, 0]],
                [joints_vec[0, 2], joints_vec[idx, 2]],
                label='',
                color=curr_color)
    for j in range(5):
        idx = (j * 3) + 1
        for i in range(2):
            if color is None:
                curr_color = 'C' + str(j+1)
            else:
                curr_color = color
            ax.plot([joints_vec[idx, 1], joints_vec[idx + 1, 1]],
                    [joints_vec[idx, 0], joints_vec[idx + 1, 0]],
                    [joints_vec[idx, 2], joints_vec[idx + 1, 2]],
                    label='',
                    color=curr_color)
            idx += 1
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    #ax.set_xlim(0, 640)
    #ax.set_ylim(0, 480)
    #ax.set_zlim(0, 500)
    ax.view_init(azim=0, elev=180)
    ax.set_title(title)
    return fig, ax

def show():
    plt.show()