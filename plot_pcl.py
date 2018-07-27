from pointcloud import PointCloud
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def set_ax_aspect(ax, X, Y, Z):
    max_range = np.array([np.max(X) - np.min(X), np.max(Y) - np.min(Y), np.max(Z) - np.min(Z)]).max() / 2.0
    mid_x = (np.max(X) + np.min(X)) * 0.5
    mid_y = (np.max(Y) + np.min(Y)) * 0.5
    mid_z = (np.max(Z) + np.min(Z)) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

def plot_pcl(pcl, window_title="", color='k', verbose=False):
    fig = plt.figure()
    fig.canvas.set_window_title(window_title)
    ax = fig.gca(projection='3d')
    ax.scatter(pcl.vertices[:, 0], pcl.vertices[:, 1], pcl.vertices[:, 2], color=color, s=5)
    set_ax_aspect(ax, pcl.vertices[:, 0], pcl.vertices[:, 1], pcl.vertices[:, 2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def plot_pcl_from_file(file_path, verbose=False):
    pcl = PointCloud.from_file(file_path)
    plot_pcl(pcl, window_title=file_path)

