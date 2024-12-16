from typing import Callable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
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


def show_pcds(*pcds: List[np.ndarray], title="PCDs", point_size=1.0, export: Optional[str] = None):
    p = pyvista.Plotter()
    # colors = np.random.randint(0, 256, size=(len(pcds), 3))
    colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
    for pcd, c in zip(pcds, colors):
        p.add_points(pcd, opacity=0.85, color=c, point_size=point_size)
    if export and export.endswith(".html"):
        p.export_html(export)
    else:
        p.show()


def show_pcds_parallel(*pcds: List[np.ndarray], titles: Optional[List[str]] = None, point_size=1.0, color="red"):
    N = len(pcds)
    titles = titles or ["" for _ in range(N)]
    p = pyvista.Plotter(shape=(1, N))
    for i in range(N):
        p.subplot(0, i)
        p.add_points(pcds[i], color=color, point_size=point_size)
        p.add_title(titles[i], font_size=14)
    p.show()


def show_pcd_with_keypoints(pcd: np.ndarray, kps: np.ndarray):
    p = pyvista.Plotter()
    p.add_points(pcd, color=[0, 0, 255], point_size=1)
    p.add_points(kps, color=[255, 0, 0], point_size=5)
    p.show()


def show_transformation(source, target, T, *, title="Transformation", point_size=1.0, export: Optional[str] = None):
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


def show_rgbd_image(depth: np.ndarray, color: np.ndarray):
    "display RGB-D image"
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(depth)
    axs[0].axis("off")
    axs[1].imshow(color)
    axs[1].axis("off")
    plt.tight_layout()
    plt.show()


def show_colored_points(points: np.ndarray, colors: np.ndarray, *, point_size=1.0):
    """
    display point cloud with color (RGB-D)
    points: N, 3
    colors: N, 3
    """
    plotter = pyvista.Plotter()
    plotter.add_points(points, scalars=colors, rgb=True, point_size=point_size)
    plotter.show()


def show_pose_graph(graph, cond: Optional[Callable] = None):
    """visualize open3d pose graph
    graph: 需要可视化的 open3d 格式位姿图
    cond: 过滤不需要可视化的边，函数接收 o3d.pipelines.registration.PoseGraphEdge，返回 bool
    注：可视化结果不显示顶点，暂时没找到 workaround
    """
    assert isinstance(graph, o3d.pipelines.registration.PoseGraph)
    cond = cond or (lambda e: True)  # default to display all edges

    nodes = [n.pose[:3, 3] for n in graph.nodes]
    # ref: PyVista documentation https://docs.pyvista.org/examples/00-load/create-truss
    # add 2 to indicate to vtk how many points per edge
    edges = np.array([[2, e.source_node_id, e.target_node_id] for e in graph.edges if cond(e)])
    confidence = np.array([e.confidence for e in graph.edges if cond(e)])

    mesh = pyvista.PolyData(nodes, edges)
    mesh.plot(
        scalars=confidence,
        style="wireframe",
        line_width=1,
        cmap="jet",
        show_scalar_bar=True,
    )
