import synthom_handler
from torch.autograd import Variable
from HNet import HNet
from HONet import HONet
from util import *
import argparse

NetworkClass = HONet
load_net = False
batch_size = 4
num_joints = 16
log_interv = 10
save_file_interv = 50

parser = argparse.ArgumentParser(description='')
parser.add_argument('-d', dest='dataset_folder', default='/home/paulo/Output/', required=True, help='Root folder for dataset')
parser.add_argument('-e', dest='num_epochs', type=int, required=True, help='Number of epochs to train')
parser.add_argument('-c', dest='net_class', default='', required=True, help='Network class (hnet or honet)')
parser.add_argument('-n', dest='net_filename', default='trained_' + NetworkClass.__name__ + '.pth.tar', help='Network filename')
parser.add_argument('--load_dataset', dest='load_dataset', action='store_true', default=True, help='Whether to use cuda for training')
parser.add_argument('--use_cuda', dest='use_cuda', action='store_true', default=False, help='Whether to use cuda for training')
parser.add_argument('--rgbd', dest='use_rgbd', action='store_true', default=True, help='Whether to use RGB-D (or RGB is false)')
args = parser.parse_args()

if args.net_class == 'hnet':
    NetworkClass = HNet
elif args.net_class == 'honet':
    NetworkClass = HONet
else:
    raise 1
net_filename = 'trained_' + NetworkClass.__name__ + '.pth.tar'

synthom_dataset = synthom_handler.Synthom_dataset(args.dataset_folder, type='train', load=args.load_dataset)
synthom_loader = torch.utils.data.DataLoader(
                                            synthom_dataset,
                                            batch_size=batch_size,
                                            shuffle=False)

print('Number of epochs to train: {}'.format(args.num_epochs))
print('Network class: {}'.format(NetworkClass.__name__))
print('Length of dataset: {}'.format(len(synthom_loader) * batch_size))
length_dataset = len(synthom_loader)
print('Number of batches: {}'.format(length_dataset))

net_params = {'num_joints': num_joints, 'use_cuda': args.use_cuda}
if load_net:
    net, optimizer, start_batch_idx = load_checkpoint(args.net_filename, NetworkClass,
                                                      params_dict=net_params,
                                                      use_cuda=args.use_cuda)
else:
    net = NetworkClass(net_params)
    optimizer = get_adadelta(net)
    start_batch_idx = 0

if args.use_cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

accum_net_loss = 0.
accum_report_loss = 0.
avg_net_loss = 0.
avg_report_loss = 0.
net_loss = 0.
idx_a = 0
net.train()
batch_idx = 0
accum_report_tot_loss = 0.

curr_epoch = 0
for curr_epoch in args.num_epochs:
    for batch_idx, (data, target) in enumerate(synthom_loader):
        if batch_idx < start_batch_idx:
            continue
        (rgbd, obj_id, obj_pose) = data
        if not args.use_rgbd:
            rgbd = rgbd[:, 0:3, :, :]
        target_joints, target_heatmaps = target
        rgbd, obj_id, obj_pose, target_joints, target_heatmaps = Variable(rgbd), Variable(obj_id),\
                                         Variable(obj_pose), Variable(target_joints), \
                                         Variable(target_heatmaps)
        if args.use_cuda:
            rgbd = rgbd.cuda()
            obj_id = obj_id.cuda()
            obj_pose = obj_pose.cuda()
            target_joints = target_joints.cuda()
            target_heatmaps = target_heatmaps.cuda()
            data = (rgbd, obj_id, obj_pose)

        data = (rgbd, obj_id, obj_pose)
        output = net(data)

        if args.use_cuda:
            output_main_np = output[7].cpu().data.numpy()
            target_joints_np = target_joints.cpu().data.numpy()
        else:
            output_main_np = output[7].data.numpy()
            target_joints_np = target_joints.data.numpy()

        weights_heatmaps_loss, weights_joints_loss = get_loss_weights(batch_idx)
        loss_func = cross_entropy_loss_p_logq
        loss, loss_heatmaps, loss_joints, loss_main_joints = calculate_loss_JORNet(
            loss_func, output, target_heatmaps, target_joints, range(num_joints),
            weights_heatmaps_loss, weights_joints_loss, 1)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        accum_net_loss += loss.item()
        report_loss = calc_avg_joint_loss(output_main_np, target_joints_np)
        accum_report_loss += report_loss
        accum_report_tot_loss += report_loss

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
            print('Avg overall network loss: {}'.format(int(accum_report_tot_loss / batch_idx)))
            print('-----------------------------------------------------')
        if batch_idx > 0 and batch_idx % save_file_interv == 0:
            print('Saving model to disk...')
            checkpoint_dict = {
                'args': args,
                'curr_epoch': curr_epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'avg_net_loss': avg_net_loss,
                'avg_report_loss': avg_report_loss,
                'accum_report_tot_loss': accum_report_tot_loss,
                'batch_idx': batch_idx
            }
            torch.save(checkpoint_dict, args.net_filename)
            print('Model saved')
            print('-----------------------------------------------------')

print('Saving final model to disk...')
checkpoint_dict = {
            'args': args,
            'curr_epoch': curr_epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'avg_net_loss': avg_net_loss,
            'avg_report_loss': avg_report_loss,
            'accum_report_tot_loss': accum_report_tot_loss,
            'batch_idx': batch_idx
        }
torch.save(checkpoint_dict, args.net_filename)
print('Model saved')
print('-----------------------------------------------------')



