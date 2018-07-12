import torch
import synthom_handler
from torch.autograd import Variable
from HONet import HONet
import torch.optim as optim

root_folder = '/home/paulo/MockDataset1/'
load_filepath = '/home/paulo/DuoHand/trained_honet.pth.tar'
use_cuda = False
batch_size = 4

ADADELTA_LEARNING_RATE = 0.05
ADADELTA_MOMENTUM = 0.9
ADADELTA_WEIGHT_DECAY = 0.005

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

def get_adadelta_halnet(halnet,
                        momentum=ADADELTA_MOMENTUM,
                        weight_decay=ADADELTA_WEIGHT_DECAY,
                        learning_rate=ADADELTA_LEARNING_RATE):
    return optim.Adadelta(halnet.parameters(),
                               rho=momentum,
                               weight_decay=weight_decay,
                               lr=learning_rate)

def load_checkpoint(filename, params_dict, use_cuda=False):
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
    model = HONet(params_dict)
    model.load_state_dict(model_state_dict)
    if use_cuda:
        model = model.cuda()
    optimizer_state_dict = torch_file['optimizer_state_dict']
    optimizer = torch.optim.Adadelta(model.parameters())
    optimizer.load_state_dict(optimizer_state_dict)
    del optimizer_state_dict, model_state_dict
    start_batch_idx = torch_file['batch_idx'] + 1
    print('Starting at batch: {}'.format(start_batch_idx))
    return model, optimizer, start_batch_idx

synthom_dataset = synthom_handler.Synthom_dataset(root_folder)
synthom_loader = torch.utils.data.DataLoader(
                                            synthom_dataset,
                                            batch_size=batch_size,
                                            shuffle=False)

print('Length of dataset: {}'.format(len(synthom_loader) * batch_size))
print('Number of batches: {}'.format(len(synthom_loader)))

honet_params = {'num_joints': 16, 'use_cuda': use_cuda}
if load_filepath == '':
    honet = HONet(honet_params)
    optimizer = get_adadelta_halnet(honet)
    start_batch_idx = 0
else:
    honet, optimizer, start_batch_idx = load_checkpoint(load_filepath, params_dict=honet_params, use_cuda=use_cuda)

for batch_idx, (data, target) in enumerate(synthom_loader):
    if batch_idx < start_batch_idx:
        continue
    (rgbd, obj_id, obj_pose) = data
    target_joints, target_heatmaps = target
    rgbd, obj_id, obj_pose, target_joints, target_heatmaps = Variable(rgbd), Variable(obj_id),\
                                     Variable(obj_pose), Variable(target_joints), \
                                     Variable(target_heatmaps)
    output = honet(data)
    weights_heatmaps_loss, weights_joints_loss = get_loss_weights(batch_idx)
    loss_func = cross_entropy_loss_p_logq
    loss, loss_heatmaps, loss_joints, loss_main_joints = calculate_loss_JORNet(
        loss_func, output, target_heatmaps, target_joints, range(16),
        weights_heatmaps_loss, weights_joints_loss, 1)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if batch_idx > 0 and batch_idx % 10 == 0:
        print('-----------------------------------------------------')
        print('Batch idx: {}'.format(batch_idx))
        print('Curr overall loss: {}'.format(loss.item()))
        print('Curr joint loss: {}'.format(loss_main_joints.item()))
        print('Curr average joint loss (per joint): {}'.format(int(loss_main_joints.item() / 48)))
        print('Saving model to disk...')
        checkpoint_dict = {
            'model_state_dict': honet.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'batch_idx': batch_idx,
            'loss': loss
        }
        torch.save(checkpoint_dict, 'trained_honet.pth.tar')
        print('Model saved')
        print('-----------------------------------------------------')





