from typing import Callable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pyvista

# global settings
ADJUST_CAMERA: bool = True


def create_plotter():
    plotter = pyvista.Plotter()
    if ADJUST_CAMERA:
        plotter.camera_position = "yx"
        plotter.camera.roll -= 90
    return plotter


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


def show_pcd(
    pcd: np.ndarray,
    *,
    title="PCD",
    value=None,
    normals=None,
    export: Optional[str] = None,
    **kwargs,
):
    """
    value: N-dimensional scalar vector
    normals: Nx3 vector (could be normals)

    Optional kwargs:
        point_size: point size, 1.0 by default
        cmap: color map for value, "coolwarm" by default
    """
    p = create_plotter()

    kwargs["point_size"] = kwargs.get("point_size", 1.0)
    if value is not None:
        kwargs["cmap"] = kwargs.get("cmap", "coolwarm")
        value = np.asarray(value, dtype=np.float32)  # color map error if integer
        p.add_points(pcd, scalars=value, **kwargs)
    else:
        kwargs["color"] = kwargs.get("color", "red")
        p.add_points(pcd, **kwargs)

    p.add_title(title, font_size=14)

    if normals is not None:
        p.add_arrows(pcd, normals, opacity=0.25, color="black", mag=1)

    if export and export.endswith(".html"):
        p.export_html(export)
    else:
        p.show()


def show_pcds(
    *pcds: List[np.ndarray],
    title="PCDs",
    point_size=1.0,
    cmap="coolwarm",
    export: Optional[str] = None,
):
    """
    cmap: "coolwarm" by default, "rainbow", "plasma", "viridis"
    """
    p = create_plotter()
    # colors = np.random.randint(0, 256, size=(len(pcds), 3))
    # colors = [[255, 0, 0], [0, 0, 255], [0, 255, 0]]
    colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(pcds)))

    for pcd, c in zip(pcds, colors):
        p.add_points(pcd, opacity=0.85, color=c, point_size=point_size)
    if export and export.endswith(".html"):
        p.export_html(export)
    else:
        p.show()


def show_pcds_parallel(
    *pcds: List[np.ndarray],
    titles: Optional[List[str]] = None,
    point_size=1.0,
    color="red",
):
    N = len(pcds)
    titles = titles or ["" for _ in range(N)]
    p = pyvista.Plotter(shape=(1, N))
    for i in range(N):
        p.subplot(0, i)
        p.add_points(pcds[i], color=color, point_size=point_size)
        p.add_title(titles[i], font_size=14)
    p.show()


def pick_point(pcd: np.ndarray, point_size=2.0):
    """
    right click to pick a point from pcd
    return numpy.ndarray if picked, otherwise None
    注意：此函数不可以在 jupyter notebook 中使用
    """
    plotter = create_plotter()
    plotter.add_points(pcd, color="blue", point_size=point_size)
    plotter.enable_point_picking(
        color="red", show_message="Pick a point with right click"
    )
    plotter.show()
    return plotter.picked_point


def pick_points_from_mesh(path: str):
    """
    从 mesh 中选择点，并输出该点的坐标。(Shift + Left Click)
    :param mesh: open3d.geometry.TriangleMesh 对象
    :return: List[numpy.ndarray] 选择的点的坐标
    """
    mesh = o3d.io.read_triangle_mesh(path)
    vis = o3d.visualization.VisualizerWithVertexSelection()
    vis.create_window()
    vis.add_geometry(mesh)
    vis.run()

    # 获取用户选择的点的索引
    picked_points = [p.coord for p in vis.get_picked_points()]
    return picked_points


def show_pcd_with_keypoints(
    pcd: np.ndarray,
    kps: np.ndarray,
    scalars: Optional[np.ndarray] = None,
    pcd_point_size=1.0,
    kps_point_size=5.0,
    export: Optional[str] = None,
):
    """
    pcd: N, 3
    kps: M, 3 keypoints，使用红色点展示
    scalars: 如果设定，点云按照值映射颜色，否则展示为蓝色
    """
    p = create_plotter()
    if scalars is not None:
        p.add_points(pcd, scalars=scalars, point_size=pcd_point_size)
    else:
        p.add_points(pcd, color=[0, 0, 255], point_size=pcd_point_size)
    p.add_points(kps, color=[255, 0, 0], point_size=kps_point_size)

    if export and export.endswith(".html"):
        p.export_html(export)
    else:
        p.show()


def show_transformation(
    source,
    target,
    T,
    *,
    title="Transformation",
    point_size=1.0,
    export: Optional[str] = None,
):
    "检查 transformation 是否正确"
    source_homo = np.concatenate((source, np.ones((source.shape[0], 1))), axis=1)
    source_trans = (source_homo @ T.transpose())[..., :3]  # N, 3
    p = create_plotter()
    p.add_points(source_trans, opacity=0.85, color="red", point_size=point_size)
    p.add_points(target, opacity=0.85, color="blue", point_size=point_size)
    if export and export.endswith(".html"):
        p.export_html(export)
    else:
        p.show()


def show_rgbd_image(depth: np.ndarray, color: np.ndarray, *, colorbar: bool = False):
    "display RGB-D image"
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    img = axs[0].imshow(depth, aspect="equal")  # cmap="coolwarm"
    axs[0].axis("off")
    axs[1].imshow(color, aspect="equal")
    axs[1].axis("off")
    if colorbar:
        fig.colorbar(img)
    plt.tight_layout()
    plt.show()


def get_masked_image(
    color: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
    mask_color: Tuple[int, int, int] = (123, 104, 238),
):
    """
    get RGB-D image overlaped with an instance mask
    color: H, W, 3
    mask: H, W
    alpha: the transparency of the mask
    """
    assert mask.shape == color.shape[:2]
    image = color.copy()
    image[mask == 1] = (1 - alpha) * image[mask == 1] + alpha * np.array(mask_color)
    return image


def show_masked_image(color: np.ndarray, mask: np.ndarray, save: Optional[str] = None):
    "display RGB-D image with mask"
    assert mask.shape == color.shape[:2]
    mask_color = (123, 104, 238, int(0.8 * 255))

    colored_mask = np.zeros((*mask.shape, 4), dtype=np.uint8)
    colored_mask[mask == 1] = mask_color
    plt.figure()
    plt.imshow(color)
    plt.imshow(colored_mask)
    plt.axis("off")
    plt.tight_layout()
    if save:
        plt.savefig(save, bbox_inches="tight", pad_inches=0.0)
    plt.show()


def show_colored_points(
    points: np.ndarray,
    colors: np.ndarray,
    *,
    point_size=1.0,
    export: Optional[str] = None,
):
    """
    display point cloud with color (RGB-D)
    points: N, 3
    colors: N, 3
    """
    plotter = create_plotter()
    plotter.add_points(points, scalars=colors, rgb=True, point_size=point_size)
    if export and export.endswith(".html"):
        plotter.export_html(export)
    else:
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
    edges = np.array(
        [[2, e.source_node_id, e.target_node_id] for e in graph.edges if cond(e)]
    )
    confidence = np.array([e.confidence for e in graph.edges if cond(e)])

    mesh = pyvista.PolyData(nodes, edges)
    mesh.plot(
        scalars=confidence,
        style="wireframe",
        line_width=1,
        cmap="jet",
        show_scalar_bar=True,
    )
