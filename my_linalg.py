import numpy as np

def signedpow(x, p):
    return np.sign(x) * np.power(np.abs(x), p)

def rot_mtx_2D( theta ):
    rot_mtx = np.identity(2)
    rot_mtx[0, 0] = np.cos(theta)
    rot_mtx[0, 1] = -np.sin(theta)
    rot_mtx[1, 0] = np.sin(theta)
    rot_mtx[1, 1] = np.cos(theta)
    return rot_mtx

def get_3d_rot_mtx(theta, axis_rot):
    R = np.zeros((3, 3))
    if axis_rot == 'x' or axis_rot == 0:
        R[0, 0] = 1
        R[1, 1] = np.cos(theta)
        R[1, 2] = -np.sin(theta)
        R[2, 1] = np.sin(theta)
        R[2, 2] = np.cos(theta)
    elif axis_rot == 'y' or axis_rot == 1:
        R[1, 1] = 1
        R[0, 0] = np.cos(theta)
        R[0, 2] = np.sin(theta)
        R[2, 0] = -np.sin(theta)
        R[2, 2] = np.cos(theta)
    elif axis_rot == 'z' or axis_rot == 2:
        R[0, 0] = np.cos(theta)
        R[0, 1] = -np.sin(theta)
        R[1, 0] = np.sin(theta)
        R[1, 1] = np.cos(theta)
        R[2, 2] = 1
    return R

def get_eul_rot_mtx(eul_angles, axes_order='zyz'):
    eul_rot_mtx = np.identity(3)
    for i in range(2):
        ix = i
        eul_rot_mtx = np.dot(eul_rot_mtx, get_3d_rot_mtx(eul_angles[ix], axes_order[ix]))
    return eul_rot_mtx

def get_rot_transl_mtx(rot_mtx, transl_vec):
    return np.vstack((np.hstack((rot_mtx, transl_vec)), np.array([0, 0, 0, 1])))
