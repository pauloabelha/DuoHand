import synthom_handler
from torch.autograd import Variable
from HNet import HNet
from HORNet import HORNet
from util import *
import argparse
import os

load_net = False
batch_size = 16
num_joints = 16
log_interv = 10
save_file_interv = 100
hnet_filepath = 'trained_hnet_rgbd.pth.tar'

parser = argparse.ArgumentParser(description='')
parser.add_argument('-d', dest='dataset_folder', default='/home/paulo/josh16/', required=True, help='Root folder for dataset')
parser.add_argument('-e', dest='num_epochs', type=int, required=True, help='Number of epochs to train')
parser.add_argument('--load_dataset', dest='load_dataset', action='store_true', default=True, help='Whether to use cuda for training')
parser.add_argument('--use_cuda', dest='use_cuda', action='store_true', default=False, help='Whether to use cuda for training')
args = parser.parse_args()


args.net_filename = 'trained_hornet.pth.tar'
args.output_filepath = 'output_hornet.txt'

synthom_dataset = synthom_handler.Synthom_dataset(args.dataset_folder, type='train', load=args.load_dataset)
synthom_loader = torch.utils.data.DataLoader(
                                            synthom_dataset,
                                            batch_size=batch_size,
                                            shuffle=False)

print_file('-------------------------------------------------------------', args.output_filepath)
print_file('-------------------------------------------------------------', args.output_filepath)
print_file('Number of epochs to train: ' + str(args.num_epochs), args.output_filepath)
print_file('Length of dataset: ' + str(len(synthom_loader) * batch_size), args.output_filepath)
print_file('Network file name: ' + args.net_filename, args.output_filepath)
print_file('Output file path: ' + args.output_filepath, args.output_filepath)
length_dataset = len(synthom_loader)
print_file('Number of batches: ' + str(length_dataset), args.output_filepath)

net_params = {'num_joints': num_joints,
              'use_cuda': args.use_cuda,}
start_epoch = 0
hornet = HORNet(net_params)
optimizer = get_adadelta(hornet)
start_batch_idx = 0

if args.use_cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

accum_net_loss = 0.
accum_report_loss = 0.
avg_net_loss = 0.
avg_report_loss = 0.
net_loss = 0.
idx_a = 0
hornet.train()
batch_idx = 0
accum_report_tot_loss = 0.

# load hnet
net_params = {'num_joints': num_joints,
              'use_cuda': args.use_cuda,
              'use_rgbd': True,
              'obj_channel': False}
hnet, optimizer, start_batch_idx, start_epoch, train_ix = load_checkpoint(hnet_filepath, HNet,
                                                      params_dict=net_params,
                                                      use_cuda=args.use_cuda)


curr_epoch = 0
train_ix = 0
print_file('Started training: logging interval is every ' + str(log_interv) + ' batches', args.output_filepath)
start_epoch = 0
start_batch_idx = 0
accum_report_loss_hnet = 0
for curr_epoch in range(args.num_epochs):
    if curr_epoch < start_epoch:
        continue
    for batch_idx, (data, target) in enumerate(synthom_loader):
        if curr_epoch == start_epoch and batch_idx < start_batch_idx:
            continue
        (rgbd, obj_id, obj_pose) = data
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


        # forward hnet
        out_hnet = hnet(data)

        input = torch.cat((out_hnet[7], obj_id, obj_pose), 1)
        output = hornet(input)

        if args.use_cuda:
            output_main_np = output.cpu().data.numpy()
            out_hnet_np = out_hnet[7].cpu().data.numpy()
            target_joints_np = target_joints.cpu().data.numpy()
        else:
            output_main_np = output.data.numpy()
            out_hnet_np = out_hnet[7].data.numpy()
            target_joints_np = target_joints.data.numpy()

        # loss hnet
        report_loss_hnet, _ = calc_avg_joint_loss(out_hnet_np, target_joints_np)
        accum_report_loss_hnet += report_loss_hnet

        weights_heatmaps_loss, weights_joints_loss = get_loss_weights(batch_idx)
        loss = euclidean_loss(output, target_joints)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        accum_net_loss += loss.item()
        report_loss, _ = calc_avg_joint_loss(output_main_np, target_joints_np)
        accum_report_loss += report_loss
        accum_report_tot_loss += report_loss

        if batch_idx > 0 and batch_idx % log_interv == 0:
            idx_a = 0
            avg_net_loss = int(accum_net_loss / log_interv)
            accum_net_loss = 0.
            avg_report_loss = int(accum_report_loss / log_interv)
            avg_report_loss_hnet = int(accum_report_loss_hnet / log_interv)
            accum_report_loss = 0.
            accum_report_loss_hnet = 0.
            print_file('-----------------------------------------------------', args.output_filepath)
            print_file('Epoch idx: ' + str(curr_epoch) + '/' + str(args.num_epochs), args.output_filepath)
            print_file('Batch idx: ' + str(batch_idx) + '/' + str(length_dataset), args.output_filepath)
            print_file('Curr avg overall network loss: ' + str(avg_net_loss), args.output_filepath)
            print_file('Curr avg per joint loss (mm): ' + str(avg_report_loss), args.output_filepath)
            print_file('Curr avg (HNEt) per joint loss (mm): ' + str(avg_report_loss_hnet), args.output_filepath)
            print_file('-----------------------------------------------------', args.output_filepath)
        if batch_idx > 0 and batch_idx % save_file_interv == 0:
            print_file('Saving model to disk...', args.output_filepath)
            checkpoint_dict = {
                'args': args,
                'train_ix': train_ix,
                'curr_epoch': curr_epoch,
                'model_state_dict': hornet.state_dict(),
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
        'model_state_dict': hornet.state_dict(),
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
            'model_state_dict': hornet.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'avg_net_loss': avg_net_loss,
            'avg_report_loss': avg_report_loss,
            'accum_report_tot_loss': accum_report_tot_loss,
            'batch_idx': batch_idx
        }
torch.save(checkpoint_dict, args.net_filename)
print_file('Model saved', args.output_filepath)
print_file('-----------------------------------------------------', args.output_filepath)



