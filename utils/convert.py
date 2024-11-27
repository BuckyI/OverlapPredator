import numpy as np
from scipy.spatial.transform import Rotation as R


def euler2matrix(pitch, yaw, roll, x, y, z):
    trans = np.eye(4)
    trans[:3, 3] = [x, y, z]
    r = R.from_euler("xyz", [pitch, yaw, roll], degrees=True)
    trans[:3, :3] = r.as_matrix()
    return trans


def matrix2euler(trans):
    r = R.from_matrix(trans[:3, :3]).as_euler("xyz", degrees=True)
    t = trans[:3, 3]
    return np.array((r, t)).flatten()


def transform(source: np.ndarray, trans: np.ndarray):
    """
    source: pcd Nx3
    trans: 4x4 transform matrix from source to target
    return: transformed pcd Nx3
    """
    source_homo = np.concatenate((source, np.ones((source.shape[0], 1))), axis=1)
    return (source_homo @ trans.transpose())[..., :3]  # N, 3
