import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pickle
from scipy.spatial.distance import pdist
from util import *

num_joints = 21
root_folder = '/home/paulo/Downloads/'
filename = 'Training_Annotation.txt'
filepath = root_folder + filename

load_from_pickle = True
pickle_filename = 'bighand_joints.p'

load_prior_from_pickle = False
pickle_prior_filename = 'bighands_joint_prior.p'

canonical_ixs = conv_bighand_joint_ix_to_canonical()

if load_from_pickle:
    bighand_joints = pickle.load(open(root_folder + pickle_filename, "rb"))
else:
    with open(filepath) as f:
        lines = f.read().splitlines()

    bighand_joints = np.zeros((len(lines), num_joints, 3))
    idx = 0
    for line in lines:
        line_split = line.split('\t')[1:-1]
        line_split = list(map(str.strip, line_split))
        line_split = list(map(float, line_split))
        joints = np.array(line_split).reshape((num_joints, 3))
        bighand_joints[idx] = joints[canonical_ixs, :]
        idx += 1
    pickle.dump(bighand_joints, open(root_folder + pickle_filename, "wb"))

#joints = bighand_joints[0]
#plot_3D_joints_bighand(joints)
#plt.show()


#joints_canonical = joints[canonical_ixs, :]
#print(joints)
#print(joints_canonical)
#plot_3D_joints(joints_canonical, num_joints=21)
#plt.show()

num_elems = int((num_joints**2 - num_joints) / 2)

if load_prior_from_pickle:
    prior_dict = pickle.load(open(root_folder + pickle_prior_filename, "rb"))
    joints_prior = prior_dict['joints_prior']
    joints_dist_prior = prior_dict['joints_dist_prior']
else:
    max_accepted_dist = 250
    min_dist = 1e10
    max_dist = -1
    joints_dist_prior = np.zeros((num_elems, max_accepted_dist))
    joints_prior = np.zeros((num_joints - 1, max_accepted_dist))
    for idx in range(bighand_joints.shape[0]):
        if idx % 10000 == 0:
            print(str(int(100.0 * idx / bighand_joints.shape[0])) + '%')
        joints = bighand_joints[idx]
        joint_idx = 1
        while joint_idx < num_joints - 1:
            dist_idx = int(np.linalg.norm(joints[joint_idx, :] - joints[0, :]))
            joints_prior[joint_idx, dist_idx] += 1
            joint_idx += 1
        joints_dist = pdist(joints, 'euclidean')
        for elem_idx in range(num_elems):
            dist = joints_dist[elem_idx]
            if dist < min_dist:
                min_dist = dist
                print('Min dist: {}'.format(min_dist))
            if dist > max_dist:
                max_dist = dist
                print('Max dist: {}'.format(max_dist))
            dist_idx = int(np.round(dist))
            joints_dist_prior[elem_idx, dist_idx] += 1
    joints_prior = joints_prior / joints_prior.sum()
    joints_dist_prior = joints_dist_prior / joints_dist_prior.sum()
    prior_dict = {'joints_prior': joints_prior,
                  'joints_dist_prior': joints_dist_prior}
    pickle.dump(prior_dict, open(root_folder + pickle_prior_filename, "wb"))
    print('Min dist: {}'.format(min_dist))
    print('Max dist: {}'.format(max_dist))


plt.imshow(joints_prior, cmap='viridis', interpolation='nearest')
plt.show()

print('Min of prior: {}'.format(np.min(joints_dist_prior)))
print('Max of prior: {}'.format(np.max(joints_dist_prior)))
plt.imshow(joints_dist_prior, cmap='viridis', interpolation='nearest')
plt.show()
