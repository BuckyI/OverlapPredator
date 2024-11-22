from typing import List

import numpy as np
import pyvista


def transform(source: np.ndarray, trans: np.ndarray):
    """
    source: pcd Nx3
    trans: 4x4 transform matrix from source to target
    return: transformed pcd Nx3
    """
    source_homo = np.concatenate((source, np.ones((source.shape[0], 1))), axis=1)
    return (source_homo @ trans.transpose())[..., :3]  # N, 3


def get_pcd_data(pcd: np.ndarray, value=None):
    """将 array 点云转化为 pyvista 类型的数据，方便可视化
    pcd: Nx3
    value(optional): N-dimensional vector
    return: PolyData
    """
    pdata = pyvista.PolyData(pcd)
    if value is not None:
        pdata["value"] = value
    return pdata


def show_pcd(pcd: np.ndarray, *, title="PCD", point_size=1.0, value=None):
    pdata = pyvista.PolyData(pcd)
    if value is not None:
        pdata["value"] = value
        pdata.plot(point_size=point_size)
    else:
        pdata.plot(color="red", point_size=point_size)  # pdata.plot(cmap="Reds")


def show_pcds(*pcds: List[np.ndarray], title="PCDs", point_size=1.0, export: str = None):
    p = pyvista.Plotter()
    # colors = np.random.randint(0, 256, size=(len(pcds), 3))
    colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
    for pcd, c in zip(pcds, colors):
        p.add_points(pcd, opacity=0.85, color=c, point_size=point_size)
    if export and export.endswith(".html"):
        p.export_html(export)
    else:
        p.show()


def show_pcds_parallel(*pcds: List[np.ndarray], titles: List[str] = None, point_size=1.0, color="red"):
    N = len(pcds)
    titles = titles or ["" for _ in range(N)]
    p = pyvista.Plotter(shape=(1, N))
    for i in range(N):
        p.subplot(0, i)
        p.add_points(pcds[i], color=color, point_size=point_size)
        p.add_title(titles[i], font_size=14)
    p.show()


def show_transformation(source, target, T, *, title="Transformation", point_size=1.0, export: str = None):
    "检查 transformation 是否正确"
    source_homo = np.concatenate((source, np.ones((source.shape[0], 1))), axis=1)
    source_trans = (source_homo @ T.transpose())[..., :3]  # N, 3
    p = pyvista.Plotter()
    p.add_points(source_trans, opacity=0.85, color="red", point_size=point_size)
    p.add_points(target, opacity=0.85, color="blue", point_size=point_size)
    if export and export.endswith(".html"):
        p.export_html(export)
    else:
        p.show()
