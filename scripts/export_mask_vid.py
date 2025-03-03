"""
给定视频和预测的物体分割掩膜
导出彩色视频显示追踪结果
掩膜为 HDF5 key -> NP.NDARRAY
"""

# %%
import cv2
import numpy as np

from utils.storage import CacheSE
from utils.visualize import get_masked_image

video_path = "run/input.mp4"
data_path = "run/data_compress.hdf5"
output_path = "run/output.mp4"

# %% 加载视频
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    raise SystemExit

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("total frames:", total_frames)
# %% 加载 mask data
masks = CacheSE(data_path, "r")
total_masks = len(masks.hdf5)
print("total masks:", total_masks)

# %% 初始化输出视频
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # MP4编码
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
if not out.isOpened():
    print("Error: Could not create output video.")
    cap.release()
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
    ret, frame = cap.read()
    if not ret:
        print(f"Warning: 提前结束于第 {current_frame_id}/{total_frames} 帧")
        break

    # 处理当前帧
    try:
        mask = masks["{:05d}".format(current_frame_id)]
    except KeyError:
        print(f"Error: 找不到第 {current_frame_id} 帧的mask")
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = get_masked_image(frame, mask, alpha=0.8)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # 写入输出视频
    out.write(frame)

    current_frame_id += 1
    tqdm_bar.update(1)

# %% 关闭资源
cap.release()
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
