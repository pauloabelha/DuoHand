from pointcloud import PointCloud
from pyvox import voxelize

filepath = "/home/paulo/mustard.ply"
voxel_grid_side = 30
max_size = 0.22
pcl = PointCloud.from_file(filepath)
vertices = pcl.vertices

voxel_grid = voxelize(vertices, max_size=max_size, voxel_grid_side=50, plot=True)

