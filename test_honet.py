import torch
import synthom_handler


root_folder = '/home/paulo/Dropbox/postdoc/papers/EECV_Hands/Output/'

synthom_dataset = synthom_handler.Synthom_dataset(root_folder)
synthom_loader = torch.utils.data.DataLoader(
                                            synthom_dataset,
                                            batch_size=1,
                                            shuffle=False)

for batch_idx, (data, target) in enumerate(synthom_loader):
    (rgbd, obj_id, obj_pose) = data

