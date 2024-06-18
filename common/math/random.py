"""Functions for random sampling"""
import numpy as np


def uniform_2_sphere(num: int = None):
    """Uniform sampling on a 2-sphere

    Source: https://gist.github.com/andrewbolster/10274979

    Args:
        num: Number of vectors to sample (or None if single)

    Returns:
        Random Vector (np.ndarray) of size (num, 3) with norm 1.
        If num is None returned value will have size (3,)

    """
    if num is not None:
        phi = np.random.uniform(0.0, 2 * np.pi, num)
        cos_theta = np.random.uniform(-1.0, 1.0, num)
    else: # 相当于 num = 1
        phi = np.random.uniform(0.0, 2 * np.pi)
        cos_theta = np.random.uniform(-1.0, 1.0)

    theta = np.arccos(cos_theta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    return np.stack((x, y, z), axis=-1)


if __name__ == '__main__':
    # Visualize sampling
    print("注意直接运行本脚本程序会报错! random.py 名字与 python built-in random 冲突!")
    import matplotlib.pyplot as plt

    rand_2s = uniform_2_sphere(10000)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")  # 创建3D坐标轴
    ax.scatter(rand_2s[:, 0], rand_2s[:, 1], rand_2s[:, 2])  # 绘制点云
    plt.show()
