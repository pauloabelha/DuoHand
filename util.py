import torch
import numpy as np
import torch.optim as optim
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization!
import resnet

ADADELTA_LEARNING_RATE = 0.05
ADADELTA_MOMENTUM = 0.9
ADADELTA_WEIGHT_DECAY = 0.005

def get_adadelta(net,
                        momentum=ADADELTA_MOMENTUM,
                        weight_decay=ADADELTA_WEIGHT_DECAY,
                        learning_rate=ADADELTA_LEARNING_RATE):
    return optim.Adadelta(net.parameters(),
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
    start_epoch = torch_file['curr_epoch']
    train_ix = torch_file['train_ix']
    return model, optimizer, start_batch_idx, start_epoch, train_ix

# num_joints should be one less (no hand root)
def calc_avg_joint_loss(output_main, target_joints, num_joints=15):
    loss = 0.
    range_ = target_joints.shape[0]
    avg_loss_per_joint = np.zeros((15,))
    for i in range(range_):
        target_joints_ = target_joints[i].reshape((num_joints, 3))
        output_joints = output_main[i].reshape((num_joints, 3))
        dist = np.sqrt(np.sum(np.square(target_joints_[:, np.newaxis, :] - output_joints), axis=2))
        loss_per_joint = np.diag(dist)
        avg_loss_per_joint += loss_per_joint
        loss += float(np.sum(loss_per_joint) / num_joints)
    loss /= range_
    avg_loss_per_joint /= range_
    return loss, avg_loss_per_joint

def plot_3D_joints(joints_vec, num_joints=16, title='', fig=None, ax=None, color=None):
    num_bones_per_finger = int((num_joints - 1) / 5)
    if fig is None:
        fig = plt.figure()
        ax = Axes3D(fig)
    if joints_vec.shape[0] == (num_joints - 1) * 3:
        joints_vec = joints_vec.reshape((num_joints - 1, 3))
        joints_vec = np.vstack([np.zeros((1, 3)), joints_vec])
    else:
        joints_vec = joints_vec.reshape((num_joints, 3))
    for i in range(5):
        idx = (i * num_bones_per_finger) + 1
        if color is None:
            curr_color = 'C0'
        else:
            curr_color = color
        ax.plot([joints_vec[0, 0], joints_vec[idx, 0]],
                [joints_vec[0, 2], joints_vec[idx, 2]],
                [joints_vec[0, 1], joints_vec[idx, 1]],
                label='',
                color=curr_color)
    for j in range(5):
        idx = (j * num_bones_per_finger) + 1
        for i in range(2):
            if color is None:
                curr_color = 'C' + str(j+1)
            else:
                curr_color = color
            ax.plot([joints_vec[idx, 0], joints_vec[idx + 1, 0]],
                    [joints_vec[idx, 2], joints_vec[idx + 1, 2]],
                    [joints_vec[idx, 1], joints_vec[idx + 1, 1]],
                    label='',
                    color=curr_color)
            idx += 1
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    #ax.set_xlim(0, 640)
    #ax.set_ylim(0, 480)
    #ax.set_zlim(0, 500)
    ax.view_init(azim=90, elev=0)
    ax.set_title(title)
    return fig, ax

def plot_3D_joints_bighand(joints_vec):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(azim=270, elev=45)
    for i in range(5):
        ax.plot([joints_vec[0, 0], joints_vec[i + 1, 0]],
                [joints_vec[0, 1], joints_vec[i + 1, 1]],
                [joints_vec[0, 2], joints_vec[i + 1, 2]],
                label='',
                color='C' + str(i))

    for finger_idx in range(5):
        b_idx = finger_idx + 1
        for i in range(3):
            a_idx = b_idx
            color = 'C' + str(finger_idx)
            b_idx = i + (finger_idx + 2) * 3
            ax.plot([joints_vec[a_idx, 0], joints_vec[b_idx, 0]],
                    [joints_vec[a_idx, 1], joints_vec[b_idx, 1]],
                    [joints_vec[a_idx, 2], joints_vec[b_idx, 2]],
                    label='',
                    color=color)
    plt.show()

def conv_bighand_joint_ix_to_canonical():
    canonical_ixs = [0, 1, 6, 7, 8, 2, 9, 10, 11, 3, 12, 13, 14, 4, 15, 16, 17, 5, 18, 19, 20]
    canonical_ixs = np.array(canonical_ixs).astype(int)
    return canonical_ixs



def show():
    plt.show()

def print_file(txt, filepath):
    print(txt)
    with open(filepath, "a") as f:
        f.write(txt)
        f.write('\n')
        
def load_resnet_weights_into_net(net, use_rgbd, obj_channel, filepath):
    print_file("Loading RESNet50...", filepath)
    resnet50 = resnet.resnet50(pretrained=True)
    print_file("Done loading RESNet50", filepath)
    # initialize net with RESNet50
    print_file("Initializaing network with RESNet50...", filepath)
    # initialize level 1
    # initialize conv1
    resnet_weight = resnet50.conv1.weight.data.cpu()
    resnet_weight = resnet_weight.numpy()
    n_extra_channels = 0
    if obj_channel:
        n_extra_channels = 10
    if use_rgbd:
        n_extra_channels = 1
    if obj_channel or use_rgbd:
        float_tensor = np.random.normal(np.mean(resnet_weight),
                                        np.std(resnet_weight),
                                        (resnet_weight.shape[0],
                                         n_extra_channels, resnet_weight.shape[2],
                                         resnet_weight.shape[2]))
        resnet_weight = np.concatenate((resnet_weight, float_tensor), axis=1)
    resnet_weight = torch.FloatTensor(resnet_weight)
    net.conv1[0]._parameters['weight'].data.copy_(resnet_weight)
    # initialize level 2
    # initialize res2a
    resnet_weight = resnet50.layer1[0].conv1.weight.data
    net.res2a.right_res[0][0]._parameters['weight'].data.copy_(resnet_weight)
    resnet_weight = resnet50.layer1[0].conv2.weight.data
    net.res2a.right_res[2][0]._parameters['weight'].data.copy_(resnet_weight)
    resnet_weight = resnet50.layer1[0].conv3.weight.data
    net.res2a.right_res[4][0]._parameters['weight'].data.copy_(resnet_weight)
    resnet_weight = resnet50.layer1[0].downsample[0].weight.data
    net.res2a.left_res[0]._parameters['weight'].data.copy_(resnet_weight)
    # initialize res2b
    resnet_weight = resnet50.layer1[1].conv1.weight.data
    net.res2b.right_res[0][0]._parameters['weight'].data.copy_(resnet_weight)
    resnet_weight = resnet50.layer1[1].conv2.weight.data
    net.res2b.right_res[2][0]._parameters['weight'].data.copy_(resnet_weight)
    resnet_weight = resnet50.layer1[1].conv3.weight.data
    net.res2b.right_res[4][0]._parameters['weight'].data.copy_(resnet_weight)
    # initialize res2c
    resnet_weight = resnet50.layer1[2].conv1.weight.data
    net.res2c.right_res[0][0]._parameters['weight'].data.copy_(resnet_weight)
    resnet_weight = resnet50.layer1[2].conv2.weight.data
    net.res2c.right_res[2][0]._parameters['weight'].data.copy_(resnet_weight)
    resnet_weight = resnet50.layer1[2].conv3.weight.data
    net.res2c.right_res[4][0]._parameters['weight'].data.copy_(resnet_weight)
    # initialize res3a
    resnet_weight = resnet50.layer2[0].conv1.weight.data
    net.res3a.right_res[0][0]._parameters['weight'].data.copy_(resnet_weight)
    resnet_weight = resnet50.layer2[0].conv2.weight.data
    net.res3a.right_res[2][0]._parameters['weight'].data.copy_(resnet_weight)
    resnet_weight = resnet50.layer2[0].conv3.weight.data
    net.res3a.right_res[4][0]._parameters['weight'].data.copy_(resnet_weight)
    resnet_weight = resnet50.layer2[0].downsample[0].weight.data
    net.res3a.left_res[0]._parameters['weight'].data.copy_(resnet_weight)
    # initialize res3b
    resnet_weight = resnet50.layer2[1].conv1.weight.data
    net.res3b.right_res[0][0]._parameters['weight'].data.copy_(resnet_weight)
    resnet_weight = resnet50.layer2[1].conv2.weight.data
    net.res3b.right_res[2][0]._parameters['weight'].data.copy_(resnet_weight)
    resnet_weight = resnet50.layer2[1].conv3.weight.data
    net.res3b.right_res[4][0]._parameters['weight'].data.copy_(resnet_weight)
    # initialize res3c
    resnet_weight = resnet50.layer2[2].conv1.weight.data
    net.res3c.right_res[0][0]._parameters['weight'].data.copy_(resnet_weight)
    resnet_weight = resnet50.layer2[2].conv2.weight.data
    net.res3c.right_res[2][0]._parameters['weight'].data.copy_(resnet_weight)
    resnet_weight = resnet50.layer2[2].conv3.weight.data
    net.res3c.right_res[4][0]._parameters['weight'].data.copy_(resnet_weight)
    print_file("Done initializaing network with RESNet50", filepath)
    print_file("Deleting resnet from memory", filepath)
    del resnet50
    return net

