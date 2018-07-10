import torch
import synthom_handler


root_folder = '/home/paulo/Dropbox/postdoc/papers/EECV_Hands/Output/'

synthom_dataset = synthom_handler.Synthom_dataset(root_folder)
synthom_loader = torch.utils.data.DataLoader(
                                            synthom_dataset,
                                            batch_size=3,
                                            shuffle=False)

for batch_idx, (data, target) in enumerate(synthom_loader):
    (rgbd, obj_id, obj_pose) = data
    print(rgbd.shape)
    print(obj_id)
    print(obj_pose)
