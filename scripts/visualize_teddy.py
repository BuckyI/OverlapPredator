"""
给定视频和预测的物体分割掩膜
导出彩色视频显示追踪结果
掩膜为 HDF5 key -> NP.NDARRAY
"""

# %%
import cv2
import numpy as np

from configs.settings import params
from datasets.tum import TUMDataset
from utils.storage import CacheSE
from utils.visualize import get_masked_image

param = params[5]
dataset = TUMDataset(param[0], param[2])
masks = CacheSE("run/teddy_target_cache.h5", "r")
# video_path = "run/input.mp4"
# data_path = "run/data_compress.hdf5"
output_path = "run/output.mp4"

# %% 加载视频
total_frames = len(dataset.frames)
fps = 30
height, width = dataset.frames[0].depth.shape
print("total frames:", total_frames)
# %% 加载 mask data
total_masks = len(masks.hdf5["mask"])
print("total masks:", total_masks)

# %% 初始化输出视频
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # MP4编码
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
if not out.isOpened():
    print("Error: Could not create output video.")
    raise SystemExit

# %% 验证帧数一致性
# if total_frames != total_masks:
#     print(f"Error: 视频帧数({total_frames})与mask数量({total_masks})不一致")
#     cap.release()
#     cv2.destroyAllWindows()
#     masks.close()
#     raise SystemExit

# 由于还没有处理完毕，所以只处理前一部分
max_id = min(total_frames, total_masks)
print("max_id:", max_id)

# %% 逐帧处理并导出视频
from tqdm import tqdm

tqdm_bar = tqdm(total=max_id, desc="Processing frames")
current_frame_id = 0
while current_frame_id < max_id:
    # 读取新一帧
    frame = dataset.frames[current_frame_id].depth

    depth = frame / 5000
    mask = masks[f"mask/{current_frame_id}"]
    depth_ = depth * mask
    # 将归一化后的深度图映射到 0-255 范围，并转换为 uint8 格式
    depth_ = (depth_ / 5 * 255).astype(np.uint8)
    colored_depth_image = cv2.applyColorMap(depth_, cv2.COLORMAP_TWILIGHT)
    frame = colored_depth_image

    # 转换颜色空间
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # frame = get_masked_image(frame, mask, alpha=0.8)
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # 写入输出视频
    out.write(frame)

    current_frame_id += 1
    tqdm_bar.update(1)

# %% 关闭资源
out.release()
masks.close()
print(f"处理完成！已处理 {current_frame_id} 帧")
print("输出视频已保存至:", output_path)


# %% demo

# current_frame_id = -1
# current_frame_id += 1
# ret, frame = cap.read()  # 读取一帧
# assert isinstance(frame, np.ndarray)
# mask = masks["{:05d}".format(current_frame_id)]
# result = get_masked_image(frame, mask)
