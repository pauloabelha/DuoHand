import os

scene_name = 'woodblock'
root_folder = '/home/paulo/HOMDataset/'
right_hand_filename = 'right_hand.txt'
right_hand_conv_filename = 'right_hand_' + 'conv' + '.txt'
objpose_file = 'obj_pose.txt'
objpose_conv_file = 'obj_pose_' + 'conv' + '.txt'

Z_handpose_threshold = 200

bone_name_list = ['hand_r',
                  'thumb_r',
                  'thumb_r_001',
                  'thumb_r_002',
                  'index_r',
                  'index_r_001',
                  'index_r_002',
                  'middle_r',
                  'middle_r_001',
                  'middle_r_002',
                  'ring_r',
                  'ring_r_001',
                  'ring_r_002',
                  'pinky_r',
                  'pinky_r_001',
                  'pinky_r_002'
                  ]

obj_name_list = ['crackers', 'mustard', 'orange', 'woodblock']

def map_bone_name_to_index(bone_name):
    if bone_name == 'male_r' or bone_name == 'palm_r':
        return -1
    return bone_name_list.index(bone_name)

def map_obj_name_to_index(obj_name):
    return obj_name_list.index(obj_name)



ix = 0
subfolder_names = []
subfolder_to_convlines = {}
frames_to_disconsider = {}
subfolder_to_objname = {}
num_files_removed = 0
num_files_remaining = 0
num_files_gt_but_no_file = 0
for root, dirs, files in os.walk(root_folder, topdown=True):
    if ix == 0:
        ix += 1
        continue
    subfolder_names.append(root.split('/')[-1])
    print('Processing subfolder: {}'.format(subfolder_names[-1]))
    for filename in sorted(files):
        curr_file_ext = filename.split('.')[1]
        if curr_file_ext == 'txt' and 'right_hand' in filename and not 'conv' in filename:
            with open(root + '/' + filename) as f:
                converted_lines = []
                converted_lines.append('scene: ' + scene_name)
                converted_lines.append('bone_indexes: ' + ','.join(bone_name_list))
                converted_lines.append('frame,bone_index,x_loc,y_loc,z_loc')
                obj_name = filename.split('_')[0]
                for line in f:
                    line_split = line.split(',')
                    frame_num = line_split[0]
                    rgb_filepath = root + '/' + obj_name + '_rgb_' + frame_num + '.png'
                    depth_filepath = root + '/' + obj_name + '_depth_' + frame_num + '.png'
                    bone_name = line_split[1].strip()
                    bone_index = map_bone_name_to_index(bone_name)
                    if bone_index == -1:
                        continue
                    location = line_split[2].strip().split(' ')
                    x_loc = location[0].split('=')[1]
                    y_loc = location[1].split('=')[1]
                    z_loc = location[2].split('=')[1]

                    uv_coords = location = line_split[3].strip().split(' ')
                    u_coord = str(int(float(uv_coords[0].split('=')[1])))
                    v_coord = str(int(float(uv_coords[1].split('=')[1])))

                    if float(z_loc) < Z_handpose_threshold or int(u_coord) > 640 or int(v_coord) > 480:
                        if subfolder_names[-1] == 'woodblock1':
                            print('Frame problem: {}'.format(frame_num))
                        num_files_removed += 1
                        # in this case, this pose and its image should not be considered
                        # that is, we should delete the corresponding RGB and Depth images
                        try:
                            a = frames_to_disconsider[root]
                            frames_to_disconsider[root][frame_num] = 1
                        except:
                            frames_to_disconsider[root] = {}
                        try:
                            #print('File to remove: {}'.format(rgb_filepath))
                            os.remove(rgb_filepath)
                        except:
                            a = 0
                        try:
                            #print('File to remove: {}'.format(depth_filepath))
                            os.remove(depth_filepath)
                        except:
                            a = 0
                    else:
                        # check if files exist in folder
                        rgb_exists = os.path.exists(rgb_filepath)
                        depth_exists = os.path.exists(depth_filepath)
                        if rgb_exists and depth_exists:
                            num_files_remaining += 1
                            converted_line = frame_num + ',' + str(bone_index) + ',' + x_loc + ',' + y_loc + ',' + z_loc + ',' + u_coord + ',' + v_coord
                            converted_lines.append(converted_line)
                        else:
                            frames_to_disconsider[root][frame_num] = 1
                            num_files_gt_but_no_file += 1
                            if subfolder_names[-1] == 'woodblock1':
                                print('Ground truth exists, but RGB-D files are not in folder: {}'.format(rgb_filepath))
                subfolder_to_objname[subfolder_names[-1]] = obj_name
                subfolder_to_convlines[subfolder_names[-1]] = converted_lines
                break

print('Initial number of files: {}'.format(num_files_removed + num_files_remaining + num_files_gt_but_no_file))
print('Number of files removed: {}'.format(num_files_removed))
print('Number of files with GT, but no files in folder: {}'.format(num_files_gt_but_no_file))
print('Number of files remaining: {}'.format(num_files_remaining))

for subfolder_name in subfolder_names:
    filepath = root_folder + subfolder_name + '/' + subfolder_to_objname[subfolder_name] + '_' + right_hand_conv_filename
    with open(filepath, 'w') as f:
        for converted_line in subfolder_to_convlines[subfolder_name]:
            f.write("%s\n" % converted_line)

ix = 0
subfolder_names = []
subfolder_to_convlines = {}
subfolder_to_objname = {}
for root, dirs, files in os.walk(root_folder, topdown=True):
    if ix == 0:
        ix += 1
        continue
    subfolder_names.append(root.split('/')[-1])
    for filename in sorted(files):
        curr_file_ext = filename.split('.')[1]
        if curr_file_ext == 'txt' and 'obj_pose' in filename and not 'conv' in filename:
            with open(root + '/' + filename) as f:
                converted_lines = []
                converted_lines.append('scene: ' + scene_name)
                converted_lines.append('obj name list: ' + ','.join(obj_name_list))
                converted_lines.append('frame,obj_name,x_loc,y_loc,z_loc,roll,pitch,yaw')
                for line in f:
                    line_split = line.split(',')
                    frame_num = line_split[0]
                    obj_name = line_split[1].strip()
                    obj_id = map_obj_name_to_index(obj_name)
                    rotation = line_split[2].strip().split(' ')
                    pitch = rotation[0].split('=')[1]
                    yaw = rotation[1].split('=')[1]
                    roll = rotation[2].split('=')[1]
                    location = line_split[3].strip().split(' ')
                    x_loc = location[0].split('=')[1]
                    y_loc = location[1].split('=')[1]
                    z_loc = location[2].split('=')[1]
                    frames_disc = frames_to_disconsider[root].keys()
                    if frame_num in frames_disc:
                        # this obj pose should not be considered as there is no hand in the image
                        a = 0
                    else:
                        uv_coords = line_split[4].strip().split(' ')
                        u_coord = str(int(float(uv_coords[0].split('=')[1])))
                        v_coord = str(int(float(uv_coords[1].split('=')[1])))
                        converted_lines.append(frame_num + ',' + str(obj_id) +
                                               ',' + x_loc + ',' + y_loc + ',' + z_loc +
                                               ',' + roll + ',' + pitch + ',' + yaw +
                                               ',' + u_coord + ',' + v_coord)
                subfolder_to_convlines[subfolder_names[-1]] = converted_lines
                subfolder_to_objname[subfolder_names[-1]] = obj_name
                break

for subfolder_name in subfolder_names:
    filepath = root_folder + subfolder_name + '/' + subfolder_to_objname[subfolder_name] + '_' + objpose_conv_file
    with open(filepath, 'w') as f:
        for converted_line in subfolder_to_convlines[subfolder_name]:
            f.write("%s\n" % converted_line)
