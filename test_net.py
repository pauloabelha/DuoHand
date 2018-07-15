import synthom_handler
from torch.autograd import Variable
from HNet import HNet
from util import *

NetworkClass = HNet
dataset_folder = '/home/paulo/Output/'
net_filepath = '/home/paulo/DuoHand/trained_hnet.pth.tar'
use_cuda = False
batch_size = 4
num_joints = 16
log_interv = 10

synthom_dataset = synthom_handler.Synthom_dataset(dataset_folder, type='test')
synthom_loader = torch.utils.data.DataLoader(
                                            synthom_dataset,
                                            batch_size=batch_size,
                                            shuffle=False)

print('Length of dataset: {}'.format(len(synthom_loader) * batch_size))
length_dataset = len(synthom_loader)
print('Number of batches: {}'.format(length_dataset))

model_params = {'num_joints': num_joints, 'use_cuda': use_cuda}
model, _, start_batch_idx = load_checkpoint(net_filepath, NetworkClass, params_dict=model_params, use_cuda=use_cuda)

accum_overall_loss = 0.
accum_loss = 0.
accum_tot_loss = 0.
idx_a = 0
for batch_idx, (data, target) in enumerate(synthom_loader):
    (rgbd, obj_id, obj_pose) = data
    target_joints, target_heatmaps = target
    rgbd, obj_id, obj_pose, target_joints, target_heatmaps = Variable(rgbd), Variable(obj_id),\
                                     Variable(obj_pose), Variable(target_joints), \
                                     Variable(target_heatmaps)
    output = model(data)
    weights_heatmaps_loss, weights_joints_loss = get_loss_weights(batch_idx)
    loss_func = cross_entropy_loss_p_logq
    loss, loss_heatmaps, loss_joints, loss_main_joints = calculate_loss_JORNet(
        loss_func, output, target_heatmaps, target_joints, range(num_joints),
        weights_heatmaps_loss, weights_joints_loss, 1)
    report_loss = calc_avg_joint_loss(output[7].data.numpy(), target_joints.data.numpy())
    accum_loss += report_loss
    accum_tot_loss += report_loss
    accum_overall_loss += loss.item()
    if batch_idx > 0 and batch_idx % log_interv == 0:
        idx_a = 0
        avg_loss_joints = accum_loss / log_interv
        accum_loss = 0.
        avg_accum_overall_loss = accum_overall_loss / log_interv
        accum_overall_loss = 0.
        print('-----------------------------------------------------')
        print('Batch idx: {}/{}'.format(batch_idx, length_dataset))
        print('Curr avg loss: {}'.format(int(avg_loss_joints)))
        print('Curr avg total loss: {}'.format(int(accum_tot_loss / batch_idx)))
        print('-----------------------------------------------------')
    idx_a += 1



