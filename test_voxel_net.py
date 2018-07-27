import synthom_handler
from torch.autograd import Variable
from VoxHonet import VoxHonet
from util import *


def get_obj_pose(obj_real_pose):
    for i in range(3):
        obj_real_pose[i] = (obj_real_pose[i] - (-800.41)) / (7632.79 - (-800.41))
    for i in range(3):
        idx = i + 5
        obj_real_pose[idx] = (obj_real_pose[idx] - (-180)) / (180 - (-180))
    return obj_real_pose


def numpy_to_plottable_rgb(numpy_img):
    img = numpy_img
    if len(numpy_img.shape) == 3:
        channel_axis = 0
        for i in numpy_img.shape:
            if i == 3 or i == 4:
                break
            channel_axis += 1
        if channel_axis == 0:
            img = numpy_img.swapaxes(0, 1)
            img = img.swapaxes(1, 2)
        elif channel_axis == 1:
            img = numpy_img.swapaxes(1, 2)
        elif channel_axis == 2:
            img = numpy_img
        else:
            return None
        img = img[:, :, 0:3]
    img = img.swapaxes(0, 1)
    return img.astype(int)

def plot_image(data, title='', fig=None):
    if fig is None:
        fig = plt.figure()
    data_img_RGB = numpy_to_plottable_rgb(data)
    plt.imshow(data_img_RGB)
    if not title == '':
        plt.title(title)
    return fig

voxel_grid_side = 50
use_rgbd = True
obj_channel = True
NetworkClass = VoxHonet
dataset_folder = '/home/paulo/josh17-2/'

if use_rgbd:
    rgbd_str = 'rgbd'
else:
    rgbd_str = 'rgb'

net_name = 'VoxHonet'
net_filepath = 'trained_' + net_name + '_' + rgbd_str + '.pth.tar'

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

model_params = {'num_joints': num_joints,
              'dataset_folder': dataset_folder,
              'use_cuda': use_cuda,
              'voxel_grid_side': voxel_grid_side,
              'use_rgbd': use_rgbd,
              'obj_channel': False}

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
loss_list_per_joint = np.zeros((len(synthom_loader), 15)).astype(float)
loss_per_obj = []
for i in range(4):
    loss_per_obj.append([])
joint_losses = np.zeros((len(synthom_loader),))
test_ix = 0
report_losses = []
avg_loss_accum = []
accum_avg_loss = 0.
for batch_idx, (data, target) in enumerate(synthom_loader):
    (rgbd, obj_id, obj_pose, hand_root) = data

    target_joints, target_heatmaps = target
    rgbd, obj_id, obj_pose, target_joints, target_heatmaps = Variable(rgbd), Variable(obj_id),\
                                     Variable(obj_pose), Variable(target_joints), \
                                     Variable(target_heatmaps)
    if not use_rgbd:
        rgbd = rgbd[:, 0:3, :, :]
    #img = misc.imread('/home/paulo/crackers1_real.JPG')
    #img = image = misc.imresize(img, (128, 128))
    #img = img.swapaxes(0, 1)
    #print(obj_pose)
    #plot_image(img)
    #plt.show()
    #img = img.swapaxes(1, 2).swapaxes(0, 1)
    #rgbd[0] = torch.from_numpy(img).float()

    #obj_pose = get_obj_pose(np.array([-150, -50, 0, 60, 90, -90, 0, 180]).astype(float))
    #obj_pose = torch.from_numpy(obj_pose).float()

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



    weights_heatmaps_loss, weights_joints_loss = get_loss_weights(batch_idx)
    loss_func = cross_entropy_loss_p_logq
    loss, loss_heatmaps, loss_joints, loss_main_joints = calculate_loss_JORNet(
        loss_func, output, target_heatmaps, target_joints, range(num_joints),
        weights_heatmaps_loss, weights_joints_loss, 1)

    accum_net_loss += loss.item()
    report_loss, report_loss_per_joint = calc_avg_joint_loss(output_main_np, target_joints_np)
    accum_avg_loss += report_loss
    loss_list.append(report_loss)
    loss_list_per_joint[batch_idx] = report_loss_per_joint
    joint_losses[test_ix] = report_loss
    accum_report_loss += report_loss
    accum_report_tot_loss += report_loss

    if report_loss > 1000:
        plot_image(rgbd.data.numpy()[0, 0:3, :, :])
        show()
        plt.show()
        fig, ax = plot_3D_joints(target_joints_np[0], color='C0')
        plot_3D_joints(output_main_np[0], fig=fig, ax=ax, title='Avg joint loss (mm) = ' + str(report_loss))
        show()
        a = 0

    report_losses.append(report_loss)
    aa = int(np.argmax(obj_id.cpu().data.numpy()))
    loss_per_obj[aa].append(report_loss)

    idx_a = 0
    avg_net_loss = int(accum_net_loss / log_interv)
    accum_net_loss = 0.
    avg_report_loss = int(accum_report_loss / log_interv)
    accum_report_loss = 0.
    if batch_idx > 0 and batch_idx % log_interv == 0:
        avg_loss_accum.append(accum_avg_loss / log_interv)
        accum_avg_loss = 0.
        print('-----------------------------------------------------')
        print('Average (batch): {}'.format(np.mean(np.array(avg_loss_accum))))
        print('Stddev (batch): {}'.format(np.mean(np.std(avg_loss_accum))))
        print('Batch idx: {}/{}'.format(batch_idx, length_dataset))
        print('Mean joint loss (mm): {}'.format(int(np.mean(loss_list))))
        print('Stddev joint loss (mm): {}'.format(int(np.std(loss_list))))
        print('Per Joint:')
        for i in range(15):
            print('\tJoint {} : Mean joint loss (mm): {}'.format(i+1, int(np.mean(loss_list_per_joint[0:batch_idx, i]))))
            print('\tJoint {} : Stddev joint loss (mm): {}'.format(i+1, int(np.std(loss_list_per_joint[0:batch_idx, i]))))
        print('Per Object:')
        for i in range(4):
            if len(loss_per_obj[i]) == 0:
                continue
            print('\tObj {} : Mean joint loss per obj (mm) : {}'.format(i, int(np.mean(loss_per_obj[i]))))
            print('\tObj {} : Stddev joint loss per obj (mm): {}'.format(i, int(np.std(loss_per_obj[i]))))
        print('-----------------------------------------------------')
    test_ix += 1

plt.hist(report_losses, normed=True, bins=30)
plt.show()

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