import torch
import synthom_handler
from torch.autograd import Variable
from HONet import HONet


root_folder = '/home/paulo/MockDataset1/'

synthom_dataset = synthom_handler.Synthom_dataset(root_folder)
synthom_loader = torch.utils.data.DataLoader(
                                            synthom_dataset,
                                            batch_size=1,
                                            shuffle=False)
honet_params = {'num_joints': 16}
honet = HONet(honet_params)

for batch_idx, (data, target) in enumerate(synthom_loader):
    (rgbd, obj_id, obj_pose) = data
    rgbd, obj_id, obj_pose, target = Variable(rgbd), Variable(obj_id),\
                                     Variable(obj_pose), Variable(target)
    output = honet(data)


