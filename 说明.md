# 说明

## 代码来源

该仓库基于 PREDATOR: Registration of 3D Point Clouds with Low Overlap 的论文实现代码
但是在我的项目设计中，该模型只是被用于进行点云配准，即只需要加载模型权重就可以了。

新增的项目代码主要集中在：
- `scripts/` 执行特定任务的脚本文件
- `utils/` 基础调用函数
    - `convert.py` 数据结构转换
    - `evaluate.py` 评估函数
    - `log.py` 日志记录
    - `preprocess.py` 数据预处理，主要是从 RGB-D 图像中基于 mask 提取点云
    - `registration.py` 点云配准，基于 small_gicp
    - `reconstruction.py` 项目定义的数据结构，如 Chunk, Edge 等; 以及对 Open3D TSDF 三维重建方法的封装。
    - `storage.py` 利用 h5py 和 joblib 简化程序运行时数据存储和读取，方便记录日志。有一些数据集也采用了其中定义的类。
    - `visualize.py` 可视化函数的封装，主要基于 pyvista 和 open3d。
- `configs/settings.py` 一些常用的配置（可用的数据集路径），不过在其他电脑环境下需要修改路径
- `datasets/`
    - `dataloader.py` 增加了预处理函数 preporcess，将点云数据的预处理独立出来（原始代码中，数据预处理是数据集加载的一步）
    - `kinect.py` 定义 KinectDataset 类
    - `tum.py` 定义 TUMDataset 类
- `lib/benchmark_utils.py` 增加了 ransac_reigistration 函数，但是好像没有被使用过
- `models/`
    - `architectures.py` 增加了需要的方法，如 forward_with_superpoint，encode，decode，如此编译出的模型可以更灵活地调用。
    - `gcn.py` 修复 bug，用于将模型编译为权重后仍能稳定运行
    - `checker.py` 对配准结果进行检查，这里只涵盖了基于重叠率等基础特征进行检查，以及曾经采用随机森林进行了检查（后来证实有更好的方法，在 evaluate.py/voting_evaluate）
    - `model.py` 封装的 PREDATOR 模型配准类，其读取预先保存好的权重文件，执行点云配准、特征编码功能。
    - `segmenter.py` 对 YOLO 模型的封装，从图片中推理出 mask，并提取点云
- `pipeline/` 该为后期希望重新组织代码时整理的，但是未完成。因为三维重建涉及多个模块协同作用，内容多且有点耦合，不太容易封装。可以参考 scripts 下的三维重建脚本以理解三维重建过程。
    - `data.py`
    - `evaluate.py`
    - `readme.md`
    - `recon.py`
    - `utils.py`

## 环境备份

- `requirements.txt` (`pip install -r requirements.txt`)
- `environment.yml` (`conda env create -f environment.yml`)

不过需要注意，原始 PREDATOR 的仓库中，对点云进行 KNN 查找、降采样的部分，使用了自己编写的 cpp_wrappers，需要编译，进一步也限制了只能使用 python 3.8 版本。
其实点云 KNN 查找哦这部分的代码可以使用更先进的库进行替换，限于时间精力，没有进行修改。
因此本仓库还是作为学习使用比较好（配环境太麻烦了）。



