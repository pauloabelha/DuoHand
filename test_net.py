import synthom_handler
from torch.autograd import Variable
from HNet import HNet
from HONet import HONet
from util import *

use_rgbd = False
NetworkClass = HONet
dataset_folder = '/home/paulo/josh16/'
net_filepath = '/home/paulo/DuoHand/' + 'trained_' + NetworkClass.__name__ + '_rgb.pth.tar'
use_cuda = False
batch_size = 1
num_joints = 16
log_interv = 10

synthom_dataset = synthom_handler.Synthom_dataset(dataset_folder, type='test')
synthom_loader = torch.utils.data.DataLoader(
                                            synthom_dataset,
                                            batch_size=batch_size,
                                            shuffle=False)

print('Network class: {}'.format(NetworkClass.__name__))
print('Length of dataset: {}'.format(len(synthom_loader) * batch_size))
length_dataset = len(synthom_loader)
print('Number of batches: {}'.format(length_dataset))

model_params = {'num_joints': num_joints, 'use_cuda': use_cuda}
model, _, _, _, _ = load_checkpoint(net_filepath, NetworkClass, params_dict=model_params, use_cuda=use_cuda)

if use_cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

accum_net_loss = 0.
accum_report_loss = 0.
avg_net_loss = 0.
avg_report_loss = 0.
net_loss = 0.
idx_a = 0
batch_idx = 0
accum_report_tot_loss = 0.
loss_list = []
loss_per_obj = []
for i in range(4):
    loss_per_obj.append([])
joint_losses = np.zeros((len(synthom_loader),))
test_ix = 0
for batch_idx, (data, target) in enumerate(synthom_loader):
    (rgbd, obj_id, obj_pose) = data
    target_joints, target_heatmaps = target
    rgbd, obj_id, obj_pose, target_joints, target_heatmaps = Variable(rgbd), Variable(obj_id),\
                                     Variable(obj_pose), Variable(target_joints), \
                                     Variable(target_heatmaps)
    if not use_rgbd:
        rgbd = rgbd[:, 0:3, :, :]

    if use_cuda:
        rgbd = rgbd.cuda()
        obj_id = obj_id.cuda()
        obj_pose = obj_pose.cuda()
        target_joints = target_joints.cuda()
        target_heatmaps = target_heatmaps.cuda()

    data = (rgbd, obj_id, obj_pose)
    output = model(data)

    if use_cuda:
        output_main_np = output[7].cpu().data.numpy()
        target_joints_np = target_joints.cpu().data.numpy()
    else:
        output_main_np = output[7].data.numpy()
        target_joints_np = target_joints.data.numpy()

    #plot_3D_joints(target_joints_np[0])
    #show()
    #plot_3D_joints(output_main_np[0])
    #show()

    weights_heatmaps_loss, weights_joints_loss = get_loss_weights(batch_idx)
    loss_func = cross_entropy_loss_p_logq
    loss, loss_heatmaps, loss_joints, loss_main_joints = calculate_loss_JORNet(
        loss_func, output, target_heatmaps, target_joints, range(num_joints),
        weights_heatmaps_loss, weights_joints_loss, 1)

    accum_net_loss += loss.item()
    report_loss = calc_avg_joint_loss(output_main_np, target_joints_np)
    loss_list.append(report_loss)
    joint_losses[test_ix] = report_loss
    accum_report_loss += report_loss
    accum_report_tot_loss += report_loss

    aa = int(np.argmax(obj_id.cpu().data.numpy()))
    loss_per_obj[aa].append(report_loss)

    if batch_idx > 0 and batch_idx % log_interv == 0:
        idx_a = 0
        avg_net_loss = int(accum_net_loss / log_interv)
        accum_net_loss = 0.
        avg_report_loss = int(accum_report_loss / log_interv)
        accum_report_loss = 0.
        print('-----------------------------------------------------')
        print('Batch idx: {}/{}'.format(batch_idx, length_dataset))
        print('Mean joint loss (mm): {}'.format(int(np.mean(loss_list))))
        print('Stddev joint loss (mm): {}'.format(int(np.std(loss_list))))
        for i in range(4):
            print('\tMean joint loss per obj (mm) {} : {}'.format(i, int(np.mean(loss_per_obj[i]))))
            print('\tStddev joint loss per obj (mm) {} : {}'.format(i, int(np.std(loss_per_obj[i]))))
        print('-----------------------------------------------------')
    test_ix += 1

max_count = 80
num_count = 80
count_joint_losses = np.zeros((num_count,))
for i in range(joint_losses.shape[0]):
    for j in range(num_count):
        limit = (j + 1) * (max_count / num_count)
        if joint_losses[i] < limit:
            count_joint_losses[j:] += 1
            break

print(count_joint_losses / len(synthom_loader))
plt.plot(count_joint_losses / len(synthom_loader))
plt.show()