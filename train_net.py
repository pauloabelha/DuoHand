import synthom_handler
from torch.autograd import Variable
from HNet import HNet
from HONet import HONet
from util import *

NetworkClass = HNet
dataset_folder = '/home/paulo/Output/'
net_filename = 'trained_' + NetworkClass.__name__ + '.pth.tar'
load_net = False
use_cuda = False
batch_size = 4
num_joints = 16
log_interv = 10
save_file_interv = 50

synthom_dataset = synthom_handler.Synthom_dataset(dataset_folder, type='train')
synthom_loader = torch.utils.data.DataLoader(
                                            synthom_dataset,
                                            batch_size=batch_size,
                                            shuffle=False)

print('Length of dataset: {}'.format(len(synthom_loader) * batch_size))
length_dataset = len(synthom_loader)
print('Number of batches: {}'.format(length_dataset))

net_params = {'num_joints': num_joints, 'use_cuda': use_cuda}
if load_net:
    net, optimizer, start_batch_idx = load_checkpoint(net_filename, NetworkClass,
                                                      params_dict=net_params,
                                                      use_cuda=use_cuda)
else:
    net = NetworkClass(net_params)
    optimizer = get_adadelta(net)
    start_batch_idx = 0


accum_net_loss = 0.
accum_report_loss = 0.
avg_net_loss = 0.
avg_report_loss = 0.
net_loss = 0.
idx_a = 0
net.train()
batch_idx = 0
for batch_idx, (data, target) in enumerate(synthom_loader):
    if batch_idx < start_batch_idx:
        continue
    (rgbd, obj_id, obj_pose) = data
    target_joints, target_heatmaps = target
    rgbd, obj_id, obj_pose, target_joints, target_heatmaps = Variable(rgbd), Variable(obj_id),\
                                     Variable(obj_pose), Variable(target_joints), \
                                     Variable(target_heatmaps)
    output = net(data)
    weights_heatmaps_loss, weights_joints_loss = get_loss_weights(batch_idx)
    loss_func = cross_entropy_loss_p_logq
    loss, loss_heatmaps, loss_joints, loss_main_joints = calculate_loss_JORNet(
        loss_func, output, target_heatmaps, target_joints, range(num_joints),
        weights_heatmaps_loss, weights_joints_loss, 1)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    accum_net_loss += loss.item()
    accum_report_loss += calc_avg_joint_loss(output[7].data.numpy(), target_joints.data.numpy())

    if batch_idx > 0 and batch_idx % log_interv == 0:
        idx_a = 0
        avg_net_loss = int(accum_net_loss / log_interv)
        accum_net_loss = 0.
        avg_report_loss = int(accum_report_loss / log_interv)
        accum_report_loss = 0.
        print('-----------------------------------------------------')
        print('Batch idx: {}/{}'.format(batch_idx, length_dataset))
        print('Curr avg overall network loss: {}'.format(avg_net_loss))
        print('Curr avg per joint loss (mm): {}'.format(avg_report_loss))
        print('-----------------------------------------------------')
    if batch_idx > 0 and batch_idx % save_file_interv == 0:
        print('Saving model to disk...')
        checkpoint_dict = {
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'avg_net_loss': avg_net_loss,
            'avg_report_loss': avg_report_loss,
            'batch_idx': batch_idx
        }
        torch.save(checkpoint_dict, net_filename)
        print('Model saved')
        print('-----------------------------------------------------')

print('Saving final model to disk...')
checkpoint_dict = {
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'avg_net_loss': avg_net_loss,
            'avg_report_loss': avg_report_loss,
            'batch_idx': batch_idx
        }
torch.save(checkpoint_dict, net_filename)
print('Model saved')
print('-----------------------------------------------------')



