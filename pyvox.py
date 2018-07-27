import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from mpl_toolkits.mplot3d import Axes3D #DO NOT REMOVE THIS IMPORT: it says note use, but we actually need it


def voxelize(vertices_in, max_size, voxel_grid_side, plot=False):
    '''

    :param vertices: N x 3 numpy array of vertices
    :param max_size: maximum side size (x, y, z) of all your point clouds
    :param voxel_grid_side:
    :param centralize:
    :param verbose:
    :return:
    '''
    assert(voxel_grid_side < 100)
    vertices = np.copy(vertices_in)
    # centralize points
    for i in range(3):
        vertices[:, i] -= np.mean(vertices[:, i])
    # get voxel resolution
    voxel_res = max_size / voxel_grid_side
    # convert from point to voxel coordinates

    vertices /= voxel_res
    vertices = np.asarray(vertices, dtype=int)
    #print(np.max(vertices))
    vertices += int(np.floor(voxel_grid_side / 2))
    #print(np.max(vertices))
    #print('-----------------')
    # populate voxel grid
    voxel_grid = np.zeros((voxel_grid_side, voxel_grid_side, voxel_grid_side))
    voxel_grid[vertices[:, 0], vertices[:, 1], vertices[:, 2]] = 1
    # plot voxel grid
    if plot:
        # and plot everything
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.voxels(voxel_grid, edgecolor='k')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.view_init(azim=90, elev=0)
        ax.set_title('Voxel grid')
        plt.show()

    return voxel_grid