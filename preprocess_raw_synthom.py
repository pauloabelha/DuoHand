
root_folder = '/home/paulo/Dropbox/postdoc/papers/EECV_Hands/Output/'
right_hand_filename = 'right_hand.txt'

def map_bone_name_to_index(bone_name):
    



with open(root_folder + right_hand_filename) as f:
    for line in f:
        line_split = line.split(',')
        frame_num = int(line_split[0])
        bone_name = line_split[1]
        print(bone_name)