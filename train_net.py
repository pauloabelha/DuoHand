import synthom_handler
from torch.autograd import Variable
from HNet import HNet
from HONet import HONet
from util import *
import argparse
import os

load_net = False
batch_size = 4
num_joints = 16
log_interv = 10
save_file_interv = 100

parser = argparse.ArgumentParser(description='')
parser.add_argument('-d', dest='dataset_folder', default='/home/paulo/josh16/', required=True, help='Root folder for dataset')
parser.add_argument('-e', dest='num_epochs', type=int, required=True, help='Number of epochs to train')
parser.add_argument('-c', dest='net_class', default='', required=True, help='Network class (hnet or honet)')
parser.add_argument('-o', dest='output_filepath', default='output.txt', required=True, help='Output file path')
parser.add_argument('-n', dest='net_filename', default='', help='Network filename')
parser.add_argument('--load_dataset', dest='load_dataset', action='store_true', default=False, help='Whether to use cuda for training')
parser.add_argument('--use_cuda', dest='use_cuda', action='store_true', default=False, help='Whether to use cuda for training')
parser.add_argument('--rgbd', dest='use_rgbd', action='store_true', default=False, help='Whether to use RGB-D (or RGB is false)')
args = parser.parse_args()

if args.net_class == 'hnet':
    NetworkClass = HNet
elif args.net_class == 'honet':
    NetworkClass = HONet
else:
    raise 1

if args.net_filename == '':
    args.net_filename = 'trained_' + args.net_class + '.pth.tar'

synthom_dataset = synthom_handler.Synthom_dataset(args.dataset_folder, type='train', load=args.load_dataset)
synthom_loader = torch.utils.data.DataLoader(
                                            synthom_dataset,
                                            batch_size=batch_size,
                                            shuffle=False)

print_file('-------------------------------------------------------------', args.output_filepath)
print_file('-------------------------------------------------------------', args.output_filepath)
print_file('Number of epochs to train: ' + str(args.num_epochs), args.output_filepath)
print_file('Network class: ' + str(NetworkClass.__name__), args.output_filepath)
print_file('Length of dataset: ' + str(len(synthom_loader) * batch_size), args.output_filepath)
length_dataset = len(synthom_loader)
print_file('Number of batches: ' + str(length_dataset), args.output_filepath)

net_params = {'num_joints': num_joints, 'use_cuda': args.use_cuda}
start_epoch = 0
if load_net:
    net, optimizer, start_batch_idx, start_epoch, train_ix = load_checkpoint(args.net_filename, NetworkClass,
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
train_ix = 0
for curr_epoch in range(args.num_epochs):
    if curr_epoch < start_epoch:
        continue
    for batch_idx, (data, target) in enumerate(synthom_loader):
        if curr_epoch == start_epoch and batch_idx < start_batch_idx:
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
            print_file('-----------------------------------------------------', args.output_filepath)
            print_file('Epoch idx: ' + str(curr_epoch) + '/' + str(args.num_epochs), args.output_filepath)
            print_file('Batch idx: ' + str(batch_idx) + '/' + str(length_dataset), args.output_filepath)
            print_file('Curr avg overall network loss: ' + str(avg_net_loss), args.output_filepath)
            print_file('Curr avg per joint loss (mm): ' + str(avg_report_loss), args.output_filepath)
            print_file('Avg overall network loss: ' + str(int(accum_report_tot_loss / train_ix)), args.output_filepath)
            print_file('-----------------------------------------------------', args.output_filepath)
        if batch_idx > 0 and batch_idx % save_file_interv == 0:
            print_file('Saving model to disk...', args.output_filepath)
            checkpoint_dict = {
                'args': args,
                'train_ix': train_ix,
                'curr_epoch': curr_epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'avg_net_loss': avg_net_loss,
                'avg_report_loss': avg_report_loss,
                'accum_report_tot_loss': accum_report_tot_loss,
                'batch_idx': batch_idx
            }
            torch.save(checkpoint_dict, args.net_filename)
            print_file('Model saved', args.output_filepath)
            print_file('-----------------------------------------------------', args.output_filepath)
        train_ix += 1
    print_file('Saving model to disk after epoch cycle...', args.output_filepath)
    checkpoint_dict = {
        'args': args,
        'train_ix': train_ix,
        'curr_epoch': curr_epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'avg_net_loss': avg_net_loss,
        'avg_report_loss': avg_report_loss,
        'accum_report_tot_loss': accum_report_tot_loss,
        'batch_idx': batch_idx
    }
    torch.save(checkpoint_dict, args.net_filename)
    print_file('Model saved', args.output_filepath)
    print_file('-----------------------------------------------------', args.output_filepath)

print_file('Saving final model to disk...', args.output_filepath)
checkpoint_dict = {
            'args': args,
            'train_ix': train_ix,
            'curr_epoch': curr_epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'avg_net_loss': avg_net_loss,
            'avg_report_loss': avg_report_loss,
            'accum_report_tot_loss': accum_report_tot_loss,
            'batch_idx': batch_idx
        }
torch.save(checkpoint_dict, args.net_filename)
print_file('Model saved', args.output_filepath)
print_file('-----------------------------------------------------', args.output_filepath)



