import torch
from torch.autograd import Variable
import muellericcv2017_importer
from VargoNet import VargoNet
import time
import numpy as np
<<<<<<< HEAD

=======
>>>>>>> c6a6a2926c137be4283c525b9d9c867aa16b76df

def print_verbose(str, verbose, n_tabs=0, erase_line=False):
    prefix = '\t' * n_tabs
    msg = prefix + str
    if verbose:
        if erase_line:
            print(msg, end='')
        else:
            print(prefix + str)
    return msg

DEBUG_VISUALLY = False

mueller_modules = muellericcv2017_importer.load_modules(
    module_names=['camera', 'converter', 'io_image', 'dataset_handler',
                  'egodexter_handler', 'synthhands_handler', 'optimizers',
                  'resnet', 'probs', 'visualize',  'debugger', 'trainer', 'losses', 'magic',
                  'validator'])

def validate(valid_loader, model, optimizer, valid_vars, control_vars, verbose=True):
    curr_epoch_iter = 1
    for batch_idx, (data, target) in enumerate(valid_loader):
        control_vars['batch_idx'] = batch_idx
        if batch_idx < control_vars['iter_size']:
            print_verbose("\rPerforming first iteration; current mini-batch: " +
                          str(batch_idx + 1) + "/" + str(control_vars['iter_size']), verbose, n_tabs=0, erase_line=True)
        # start time counter
        start = time.time()
        # get data and targetas cuda variables
        target_heatmaps, target_joints, target_joints_z = target

        data, target_heatmaps = Variable(data), Variable(target_heatmaps)
        if valid_vars['use_cuda']:
            data = data.cuda()
            target_heatmaps = target_heatmaps.cuda()
        # visualize if debugging
        # get model output
        output = model(data)
        # accumulate loss for sub-mini-batch
        if valid_vars['cross_entropy']:
            loss_func = mueller_modules['losses'].cross_entropy_loss_p_logq
        else:
            loss_func = mueller_modules['losses'].euclidean_loss
        loss = mueller_modules['losses'].calculate_loss_HALNet(loss_func,
            output, target_heatmaps, model.heatmap_ixs, model.WEIGHT_LOSS_INTERMED1,
            model.WEIGHT_LOSS_INTERMED2, model.WEIGHT_LOSS_INTERMED3,
            model.WEIGHT_LOSS_MAIN, control_vars['iter_size'])

        if DEBUG_VISUALLY:
            minibatch_ix = 0
            filenamebase_idx = (batch_idx * control_vars['max_mem_batch']) + minibatch_ix
            filenamebase = valid_loader.dataset.get_filenamebase(filenamebase_idx)
<<<<<<< HEAD

            output_heatmap = output[3][minibatch_ix][0].data.numpy()
            output_corner0 = np.unravel_index(np.argmax(output_heatmap), output_heatmap.shape)
            output_heatmap = output[3][minibatch_ix][1].data.numpy()
            output_corner1 = np.unravel_index(np.argmax(output_heatmap), output_heatmap.shape)

            # show bound box
=======
            # get heatmap
            heatmap = output[3][minibatch_ix][0].data.numpy()
            max_output0 = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            heatmap = output[3][minibatch_ix][1].data.numpy()
            max_output1 = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            print(max_output0)
            print(max_output1)
            # show groundtruth bound box
>>>>>>> c6a6a2926c137be4283c525b9d9c867aa16b76df
            fig = mueller_modules['visualize'].plot_image(data[minibatch_ix].data.numpy())
            mueller_modules['visualize'].plot_bound_box_from_coords(target_joints[0],
                                                                    target_joints[1],
                                                                    target_joints[2],
                                                                    target_joints[3],
                                                                    fig=fig)
            mueller_modules['visualize'].show()
<<<<<<< HEAD
            # show groundtruth
            fig = mueller_modules['visualize'].plot_image(data[minibatch_ix].data.numpy())
            mueller_modules['visualize'].plot_bound_box_from_coords(output_corner0[0],
                                                                    output_corner0[1],
                                                                    output_corner1[0],
                                                                    output_corner1[1],
=======
            # show output bound box
            fig = mueller_modules['visualize'].plot_image(data[minibatch_ix].data.numpy())
            mueller_modules['visualize'].plot_bound_box_from_coords(max_output0[0],
                                                                    max_output0[1],
                                                                    max_output1[0],
                                                                    max_output1[1],
>>>>>>> c6a6a2926c137be4283c525b9d9c867aa16b76df
                                                                    fig=fig)
            mueller_modules['visualize'].show()

        #loss.backward()
        valid_vars['total_loss'] += loss
        # accumulate pixel dist loss for sub-mini-batch
        valid_vars['total_pixel_loss'] = mueller_modules['losses'].accumulate_pixel_dist_loss_multiple(
            valid_vars['total_pixel_loss'], output[3], target_heatmaps, control_vars['batch_size'])
        if valid_vars['cross_entropy']:
            valid_vars['total_pixel_loss_sample'] = mueller_modules['losses'].accumulate_pixel_dist_loss_from_sample_multiple(
                valid_vars['total_pixel_loss_sample'], output[3], target_heatmaps, control_vars['batch_size'])
        else:
            valid_vars['total_pixel_loss_sample'] = [-1] * len(model.heatmap_ixs)
        # get boolean variable stating whether a mini-batch has been completed
        minibatch_completed = (batch_idx+1) % control_vars['iter_size'] == 0
        if minibatch_completed:
            # append total loss
            valid_vars['losses'].append(valid_vars['total_loss'].item())
            # erase total loss
            total_loss = valid_vars['total_loss'].item()
            valid_vars['total_loss'] = 0
            # append dist loss
            valid_vars['pixel_losses'].append(valid_vars['total_pixel_loss'])
            # erase pixel dist loss
            valid_vars['total_pixel_loss'] = [0] * len(model.heatmap_ixs)
            # append dist loss of sample from output
            valid_vars['pixel_losses_sample'].append(valid_vars['total_pixel_loss_sample'])
            # erase dist loss of sample from output
            valid_vars['total_pixel_loss_sample'] = [0] * len(model.heatmap_ixs)
            # check if loss is better
            if valid_vars['losses'][-1] < valid_vars['best_loss']:
                valid_vars['best_loss'] = valid_vars['losses'][-1]
                #print_verbose("  This is a best loss found so far: " + str(valid_vars['losses'][-1]), verbose)
            # log checkpoint
            if control_vars['curr_iter'] % control_vars['log_interval'] == 0:
                mueller_modules['trainer'].print_log_info(model, optimizer, 1, total_loss, valid_vars, control_vars)
                model_dict = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'control_vars': control_vars,
                    'train_vars': valid_vars,
                }
                #mueller_modules['trainer'].save_checkpoint(model_dict,
                #                        filename=valid_vars['checkpoint_filenamebase'] +
                #                                 str(control_vars['num_iter']) + '.pth.tar')
            # print time lapse
            prefix = 'Validating (Epoch #' + str(1) + ' ' + str(control_vars['curr_epoch_iter']) + '/' +\
                     str(control_vars['tot_iter']) + ')' + ', (Batch ' + str(control_vars['batch_idx']+1) +\
                     '(' + str(control_vars['iter_size']) + ')' + '/' +\
                     str(control_vars['num_batches']) + ')' + ', (Iter #' + str(control_vars['curr_iter']) +\
                     '(' + str(control_vars['batch_size']) + ')' +\
                     ' - log every ' + str(control_vars['log_interval']) + ' iter): '
            control_vars['tot_toc'] = mueller_modules['magic'].display_est_time_loop(control_vars['tot_toc'] + time.time() - start,
                                                            control_vars['curr_iter'], control_vars['num_iter'],
                                                            prefix=prefix)

            control_vars['curr_iter'] += 1
            control_vars['start_iter'] = control_vars['curr_iter'] + 1
            control_vars['curr_epoch_iter'] += 1


    return valid_vars, control_vars

model, optimizer, control_vars, valid_vars, train_control_vars = mueller_modules['validator'].parse_args(model_class=VargoNet)
if valid_vars['use_cuda']:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

valid_loader = mueller_modules['synthhands_handler'].\
    get_SynthHands_boundbox_loader(root_folder=valid_vars['root_folder'],
                                                             heatmap_res=(320, 240),
                                                             batch_size=1,
                                                             verbose=valid_vars['verbose'],
                                                                 type='valid')

control_vars['num_batches'] = len(valid_loader)
control_vars['n_iter_per_epoch'] = int(len(valid_loader) / control_vars['iter_size'])
control_vars['num_iter'] = len(valid_loader)

control_vars['tot_iter'] = int(len(valid_loader) / control_vars['iter_size'])
control_vars['start_iter_mod'] = control_vars['start_iter'] % control_vars['tot_iter']

mueller_modules['trainer'].print_header_info(model, valid_loader, control_vars)

control_vars['curr_iter'] = 1
control_vars['curr_epoch_iter'] = 1

valid_vars['total_loss'] = 0
valid_vars['total_pixel_loss'] = [0] * len(model.heatmap_ixs)
valid_vars['total_pixel_loss_sample'] = [0] * len(model.heatmap_ixs)

valid_vars, control_vars = validate(valid_loader, model, optimizer, valid_vars, control_vars, control_vars['verbose'])