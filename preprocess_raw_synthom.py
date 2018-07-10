
scene_name = 'woodblock'
root_folder = '/home/paulo/Dropbox/postdoc/papers/EECV_Hands/Output/'
right_hand_filename = 'right_hand.txt'
right_hand_conv_filename = 'right_hand_' + 'conv' + '.txt'
objpose_file = 'obj_pose.txt'
objpose_conv_file = 'obj_pose_' + 'conv' + '.txt'

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

obj_name_list = ['WoodBlock']

def map_bone_name_to_index(bone_name):
    if bone_name == 'male_r' or bone_name == 'palm_r':
        return -1
    return bone_name_list.index(bone_name)

def map_obj_name_to_index(obj_name):
    return obj_name_list.index(obj_name)

with open(root_folder + right_hand_filename) as f:
    converted_lines = []
    converted_lines.append('scene: ' + scene_name)
    converted_lines.append('bone_indexes: ' + ','.join(bone_name_list))
    converted_lines.append('frame,bone_index,x_loc,y_loc,z_loc')
    for line in f:
        line_split = line.split(',')
        frame_num = line_split[0]
        bone_name = line_split[1].strip()
        bone_index = map_bone_name_to_index(bone_name)
        if bone_index == -1:
            continue
        location = line_split[2].strip().split(' ')
        x_loc = location[0].split('=')[1]
        y_loc = location[1].split('=')[1]
        z_loc = location[2].split('=')[1]
        converted_lines.append(frame_num + ',' + str(bone_index) +
                               ',' + x_loc + ',' + y_loc + ',' + z_loc)

with open(root_folder + right_hand_conv_filename, 'w') as f:
    for converted_line in converted_lines:
        f.write("%s\n" % converted_line)


with open(root_folder + objpose_file) as f:
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
        converted_lines.append(frame_num + ',' + str(obj_id) +
                               ',' + x_loc + ',' + y_loc + ',' + z_loc +
                               ',' + roll + ',' + pitch + ',' + yaw)

with open(root_folder + objpose_conv_file, 'w') as f:
    for converted_line in converted_lines:
        f.write("%s\n" % converted_line)
