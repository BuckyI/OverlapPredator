from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.utils import square_distance
from models.blocks import block_decider
from models.gcn import GCN


class KPFCNN(nn.Module):

    def __init__(self, config):
        super(KPFCNN, self).__init__()

        ############
        # Parameters
        ############
        # Current radius of convolution and feature dimension
        layer = 0
        r = config.first_subsampling_dl * config.conv_radius
        in_dim = config.in_feats_dim  # NOTE: 配置文件中该值为 1 （不是特别明白为什么维度为 1）#TODO
        # NOTE: 该变量存储了当前网络最后一层，输入的特征维度数，随着网络层的解析而更新
        out_dim = config.first_feats_dim  # 第一层输出的特征维度
        # NOTE: 该变量存储了当前网络最后一层，输出的特征维度数，随着网络层的解析而更新

        self.K = config.num_kernel_points
        self.epsilon = torch.nn.Parameter(torch.tensor(-5.0))
        self.final_feats_dim = config.final_feats_dim  # 最后一层输出的特征维度

        # NOTE: 两个用于消融实验的参数，默认都为 True，详见 forward 相关代码
        self.condition: bool = config.condition_feature  # True
        self.add_cross_overlap: bool = config.add_cross_score  # True

        #####################
        # List Encoder blocks
        #####################
        # Save all block operations in a list of modules
        self.encoder_blocks = nn.ModuleList()
        self.encoder_skip_dims = []
        self.encoder_skips = []

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architecture):

            # Check equivariance
            # NOTE: 本论文设计的网络架构中，并没有使用到名字包含 equivariant 的 block (configs/models.py)
            if ("equivariant" in block) and (not out_dim % 3 == 0):
                raise ValueError("Equivariant block but features dimension is not a factor of 3")

            # Detect change to next layer for skip connection
            # NOTE: Encoder 每一层的最后一小层输出 skip connection
            if np.any([tmp in block for tmp in ["pool", "strided", "upsample", "global"]]):
                self.encoder_skips.append(block_i)
                self.encoder_skip_dims.append(in_dim)

            # Detect upsampling block to stop
            # NOTE: 这块代码只处理 Encoder 部分的层，所以遇到上采样（Decoder层）就跳出循环
            if "upsample" in block:
                break

            # Apply the good block function defining tf ops
            self.encoder_blocks.append(block_decider(block, r, in_dim, out_dim, layer, config))

            # Update dimension of input from output
            if "simple" in block:
                in_dim = out_dim // 2
            else:
                in_dim = out_dim

            # Detect change to a subsampled layer
            if "pool" in block or "strided" in block:
                # Update radius and feature dimension for next layer
                layer += 1  # strided 表示这是当前层的最后一小层（strided conv 为池化操作）
                r *= 2  # 半径扩大
                out_dim *= 2  # 点特征维度加倍 128 -> 256 -> 512

        #####################
        # bottleneck layer and GNN part
        #####################
        gnn_feats_dim = config.gnn_feats_dim
        self.bottle = nn.Conv1d(in_dim, gnn_feats_dim, kernel_size=1, bias=True)
        k = config.dgcnn_k
        num_head = config.num_head

        # NOTE: self-attention -> cross-attention -> self-attention
        self.gnn = GCN(num_head, gnn_feats_dim, k, config.nets)

        # NOTE: gnn 输出特征又通过了两个 1×1 卷积
        self.proj_gnn = nn.Conv1d(gnn_feats_dim, gnn_feats_dim, kernel_size=1, bias=True)
        self.proj_score = nn.Conv1d(gnn_feats_dim, 1, kernel_size=1, bias=True)

        #####################
        # List Decoder blocks
        #####################
        if self.add_cross_overlap:
            out_dim = gnn_feats_dim + 2
        else:
            out_dim = gnn_feats_dim + 1

        # Save all block operations in a list of modules
        self.decoder_blocks = nn.ModuleList()
        self.decoder_concats = []

        # NOTE: 感觉参数设计上，应该把编码器和解码器分别传入，就不需要这么麻烦地进行判断了。
        # Find first upsampling block
        start_i = 0
        for block_i, block in enumerate(config.architecture):
            if "upsample" in block:
                start_i = block_i
                break

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architecture[start_i:]):

            # Add dimension of skip connection concat
            if block_i > 0 and "upsample" in config.architecture[start_i + block_i - 1]:
                in_dim += self.encoder_skip_dims[layer]
                self.decoder_concats.append(block_i)

            # Apply the good block function defining tf ops
            self.decoder_blocks.append(block_decider(block, r, in_dim, out_dim, layer, config))

            # Update dimension of input from output
            in_dim = out_dim

            # Detect change to a subsampled layer
            if "upsample" in block:
                # Update radius and feature dimension for next layer
                layer -= 1
                r *= 0.5
                out_dim = out_dim // 2
        return

    def regular_score(self, score):
        "NOTE: 把 score 中的 nan 和 inf 都替换为 0"
        score = torch.where(torch.isnan(score), torch.zeros_like(score), score)
        score = torch.where(torch.isinf(score), torch.zeros_like(score), score)
        return score

    def forward(self, batch):
        # Get input features
        x = batch["features"].clone().detach()  # NOTE: 获得一个不跟踪梯度的副本
        len_src_c = batch["stack_lengths"][-1][0]
        len_src_f = batch["stack_lengths"][0][0]
        pcd_c = batch["points"][-1]  # NOTE: 取出的最后一层的点云，这部分用在了 gnn 那部分
        pcd_f = batch["points"][0]
        # NOTE: 这两个 pcd 是 Encoder 最后一层最稀疏的点云，取出来作为 gnn 那部分的输入
        src_pcd_c, tgt_pcd_c = pcd_c[:len_src_c], pcd_c[len_src_c:]

        sigmoid = nn.Sigmoid()
        #################################
        # 1. joint encoder part
        skip_x = []
        for block_i, block_op in enumerate(self.encoder_blocks):
            if block_i in self.encoder_skips:
                skip_x.append(x)  # NOTE: 保存当前层输出用于 skip link
            x = block_op(x, batch)
        # NOTE: (以 indoor.yaml model 结构为例) Encoder，有 4 层，每一层都有 3 小层
        # 这里 self.encoder_blocks 共有 11 个元素，是因为最后一小层实际上是 self.bottle 进行了降维
        # self.encoder_skips = [2, 5, 8, 11]

        # NOTE: Encoder 获取到的点云特征：
        # 🌟 skip_x[0] --> torch.Size([69778, 128])
        # 🌟 skip_x[1] --> torch.Size([6563, 256])
        # 🌟 skip_x[2] --> torch.Size([1995, 512])
        # 🌟 第四层特征 unconditioned_feats --> torch.Size([606, 256])
        # [p.shape for p in batch["points"]] --> 每层点云点数分别为：69778, 6563, 1995, 606

        #################################
        # 2. project the bottleneck features
        feats_c = x.transpose(0, 1).unsqueeze(0)  # [1, C, N] # NOTE: 调整数据结构以适用于 Conv1d
        feats_c = self.bottle(feats_c)  # [1, C, N]
        unconditioned_feats = feats_c.transpose(1, 2).squeeze(0)  # NOTE: 没有经过 GNN 的特征 [N, C]

        #################################
        # 3. apply GNN to communicate the features and get overlap score
        src_feats_c, tgt_feats_c = feats_c[:, :, :len_src_c], feats_c[:, :, len_src_c:]
        # NOTE: 这里传入的 src_feats_c 和 tgt_feats_c 是 Encoder 最后一层点云的坐标
        # 前面不用传入，是因为 KPConv 的层会自己读取对应层的点云。
        src_feats_c, tgt_feats_c = self.gnn(
            src_pcd_c.unsqueeze(0).transpose(1, 2),
            tgt_pcd_c.unsqueeze(0).transpose(1, 2),
            src_feats_c,
            tgt_feats_c,
        )
        feats_c = torch.cat([src_feats_c, tgt_feats_c], dim=-1)

        # NOTE: 下面这俩都是 1x1 卷积
        feats_c = self.proj_gnn(feats_c)  # 没有改变维度，可以当做 gnn 输出的最终特征 256 (1, 256, N)
        scores_c = self.proj_score(feats_c)  # 特征从 256 到 1, 用来预测 score (1, 1, N)

        feats_gnn_norm = (
            F.normalize(feats_c, p=2, dim=1).squeeze(0).transpose(0, 1)
        )  # [N, C], [N, 256] # NOTE: 特征的通道进行归一化处理
        # NOTE: 🌟 GNN 输出特征
        feats_gnn_raw = feats_c.squeeze(0).transpose(0, 1)  # [N, C]
        scores_c_raw = scores_c.squeeze(0).transpose(0, 1)  # [N, 1]

        ####################################
        # 4. decoder part
        # NOTE: 出于某种原因， Decoder 对 gnn 输出的特征进行了归一化
        src_feats_gnn, tgt_feats_gnn = feats_gnn_norm[:len_src_c], feats_gnn_norm[len_src_c:]
        # NOTE: 特征点积获得相似度 (N1, C) @ (N2, C)^T -> (N_1, N_2)
        inner_products = torch.matmul(src_feats_gnn, tgt_feats_gnn.transpose(0, 1))

        src_scores_c, tgt_scores_c = scores_c_raw[:len_src_c], scores_c_raw[len_src_c:]

        # NOTE: 应该是对相似度的分布进行调整，因为 softmax 对较大的输入值无法区分
        temperature = torch.exp(self.epsilon) + 0.03
        # NOTE: F.softmax https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html
        # inner_products (N_1, N_2) 代表了特征的相似度
        # Softmax 将相似度转化为 0-1 的概率分布，沿着 dim=1，也就是说每一行之和为 1
        # 一行 N_2 个数，为 tgt 的点数，代表了当前 src 的点最有可能和 tgt 的哪个点匹配
        # 最后又和 tgt_scores_c 进行矩阵相乘，获得的结果就表示了：**特征相似并且重叠的概率**
        s1 = torch.matmul(F.softmax(inner_products / temperature, dim=1), tgt_scores_c)
        s2 = torch.matmul(F.softmax(inner_products.transpose(0, 1) / temperature, dim=1), src_scores_c)
        # NOTE: 总结来说，scores_saliency 是通过模型预测的 overlap score 加权得到的，权重来自于 src 和 tgt 模型编码特征的相似度
        scores_saliency = torch.cat((s1, s2), dim=0)  # [N, 1]

        # NOTE: RECALL:
        # scores_c_raw          重叠分数
        # scores_saliency       匹配分数
        # feats_gnn_raw         经过 GNN 的点云特征
        # unconditioned_feats   没有经过 GNN 的特征（不包含两个点云之间的关联）
        # NOTE: condition 含义类似“对齐”，控制是否选择用 attention 来让两个点云的特征交互
        # NOTE: add_cross_overlap 的含义应该是，是否把匹配分数（cross overlap score）输出到最终的点云特征中
        if self.condition and self.add_cross_overlap:
            x = torch.cat([scores_c_raw, scores_saliency, feats_gnn_raw], dim=1)
        elif self.condition and not self.add_cross_overlap:
            x = torch.cat([scores_c_raw, feats_gnn_raw], dim=1)
        elif not self.condition and self.add_cross_overlap:
            x = torch.cat([scores_c_raw, scores_saliency, unconditioned_feats], dim=1)
        elif not self.condition and not self.add_cross_overlap:
            x = torch.cat([scores_c_raw, unconditioned_feats], dim=1)

        for block_i, block_op in enumerate(self.decoder_blocks):
            if block_i in self.decoder_concats:
                x = torch.cat([x, skip_x.pop()], dim=1)
            x = block_op(x, batch)
        feats_f = x[:, : self.final_feats_dim]
        scores_overlap = x[:, self.final_feats_dim]
        scores_saliency = x[:, self.final_feats_dim + 1]

        # safe guard our score
        scores_overlap = torch.clamp(sigmoid(scores_overlap.view(-1)), min=0, max=1)
        scores_saliency = torch.clamp(sigmoid(scores_saliency.view(-1)), min=0, max=1)
        scores_overlap = self.regular_score(scores_overlap)
        scores_saliency = self.regular_score(scores_saliency)

        # normalise point-wise features
        feats_f = F.normalize(feats_f, p=2, dim=1)

        return feats_f, scores_overlap, scores_saliency

    @torch.inference_mode()
    def forward_with_superpoint(self, batch):
        "简化 forward 代码，调整返回值，同时额外返回未经过点云信息交互的 superpoint 特征"
        # Get input features
        x = batch["features"].clone().detach()

        #################################
        # 1. joint encoder part
        skip_x = []
        for block_i, block_op in enumerate(self.encoder_blocks):
            if block_i in self.encoder_skips:
                skip_x.append(x)  # NOTE: 保存当前层输出用于 skip link
            x = block_op(x, batch)

        #################################
        # 2. project the bottleneck features
        feats_c = x.transpose(0, 1).unsqueeze(0)  # [1, C, N] # NOTE: 调整数据结构以适用于 Conv1d
        feats_c = self.bottle(feats_c)  # [1, C, N]
        unconditioned_feats = feats_c.transpose(1, 2).squeeze(0)  # NOTE: 没有经过 GNN 的特征 [N, C]

        #################################
        # 3. apply GNN to communicate the features and get overlap score
        len_src_c = batch["stack_lengths"][-1][0]  # 最后一层点云长度
        src_feats_c, tgt_feats_c = feats_c[:, :, :len_src_c], feats_c[:, :, len_src_c:]
        # 最后一层点云坐标
        pcd_c = batch["points"][-1]
        src_pcd_c, tgt_pcd_c = pcd_c[:len_src_c], pcd_c[len_src_c:]
        src_feats_c, tgt_feats_c = self.gnn(
            src_pcd_c.unsqueeze(0).transpose(1, 2),
            tgt_pcd_c.unsqueeze(0).transpose(1, 2),
            src_feats_c,
            tgt_feats_c,
        )
        feats_c = torch.cat([src_feats_c, tgt_feats_c], dim=-1)

        # NOTE: 下面这俩都是 1x1 卷积
        feats_c = self.proj_gnn(feats_c)  # 没有改变维度，可以当做 gnn 输出的最终特征 256 (1, 256, N)
        scores_c = self.proj_score(feats_c)  # 特征从 256 到 1, 用来预测 score (1, 1, N)

        feats_gnn_norm = (
            F.normalize(feats_c, p=2, dim=1).squeeze(0).transpose(0, 1)
        )  # [N, C], [N, 256] # NOTE: 特征的通道进行归一化处理
        # NOTE: 🌟 GNN 输出特征
        feats_gnn_raw = feats_c.squeeze(0).transpose(0, 1)  # [N, C]
        scores_c_raw = scores_c.squeeze(0).transpose(0, 1)  # [N, 1]

        ####################################
        # 4. decoder part
        # NOTE: 出于某种原因， Decoder 对 gnn 输出的特征进行了归一化
        src_feats_gnn, tgt_feats_gnn = feats_gnn_norm[:len_src_c], feats_gnn_norm[len_src_c:]
        # NOTE: 特征点积获得相似度 (N1, C) @ (N2, C)^T -> (N_1, N_2)
        inner_products = torch.matmul(src_feats_gnn, tgt_feats_gnn.transpose(0, 1))

        src_scores_c, tgt_scores_c = scores_c_raw[:len_src_c], scores_c_raw[len_src_c:]

        # NOTE: 应该是对相似度的分布进行调整，因为 softmax 对较大的输入值无法区分
        temperature = torch.exp(self.epsilon) + 0.03
        # NOTE: F.softmax https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html
        # inner_products (N_1, N_2) 代表了特征的相似度
        # Softmax 将相似度转化为 0-1 的概率分布，沿着 dim=1，也就是说每一行之和为 1
        # 一行 N_2 个数，为 tgt 的点数，代表了当前 src 的点最有可能和 tgt 的哪个点匹配
        # 最后又和 tgt_scores_c 进行矩阵相乘，获得的结果就表示了：**特征相似并且重叠的概率**
        s1 = torch.matmul(F.softmax(inner_products / temperature, dim=1), tgt_scores_c)
        s2 = torch.matmul(F.softmax(inner_products.transpose(0, 1) / temperature, dim=1), src_scores_c)
        # NOTE: 总结来说，scores_saliency 是通过模型预测的 overlap score 加权得到的，权重来自于 src 和 tgt 模型编码特征的相似度
        scores_saliency = torch.cat((s1, s2), dim=0)  # [N, 1]

        # NOTE: RECALL:
        # scores_c_raw          重叠分数
        # scores_saliency       匹配分数
        # feats_gnn_raw         经过 GNN 的点云特征
        # unconditioned_feats   没有经过 GNN 的特征（不包含两个点云之间的关联）
        # NOTE: condition 含义类似“对齐”，控制是否选择用 attention 来让两个点云的特征交互
        # NOTE: add_cross_overlap 的含义应该是，是否把匹配分数（cross overlap score）输出到最终的点云特征中
        # self.condition and self.add_cross_overlap:
        x = torch.cat([scores_c_raw, scores_saliency, feats_gnn_raw], dim=1)

        for block_i, block_op in enumerate(self.decoder_blocks):
            if block_i in self.decoder_concats:
                x = torch.cat([x, skip_x.pop()], dim=1)
            x = block_op(x, batch)
        feats_f = x[:, : self.final_feats_dim]
        scores_overlap = x[:, self.final_feats_dim]
        scores_saliency = x[:, self.final_feats_dim + 1]

        # safe guard our score
        sigmoid = nn.Sigmoid()
        scores_overlap = torch.clamp(sigmoid(scores_overlap.view(-1)), min=0, max=1)
        scores_saliency = torch.clamp(sigmoid(scores_saliency.view(-1)), min=0, max=1)
        scores_overlap = self.regular_score(scores_overlap)
        scores_saliency = self.regular_score(scores_saliency)

        # normalise point-wise features
        feats_f = F.normalize(feats_f, p=2, dim=1)

        source_superpoints = unconditioned_feats[:len_src_c]
        target_superpoints = unconditioned_feats[len_src_c:]
        return feats_f, scores_overlap, scores_saliency, source_superpoints, target_superpoints

    @torch.inference_mode()
    def encode(self, batch: Dict[str, List[torch.Tensor]]):
        # Get input features
        pcd = batch["points"][0]
        x = torch.ones((pcd.size(0), 1), dtype=torch.float32, device=pcd.device)

        #################################
        # 1. joint encoder part
        skip_x: List[torch.Tensor] = []
        for block_i, block_op in enumerate(self.encoder_blocks):
            if block_i in self.encoder_skips:
                skip_x.append(x)  # NOTE: 保存当前层输出用于 skip link
            x = block_op(x, batch)

        #################################
        # 2. project the bottleneck features
        feats_c = x.transpose(0, 1).unsqueeze(0)  # [1, C, N] # NOTE: 调整数据结构以适用于 Conv1d
        feats_c = self.bottle(feats_c)  # [1, C, N]
        unconditioned_feats = feats_c.transpose(1, 2).squeeze(0)  # NOTE: 没有经过 GNN 的特征 [N, C]

        skip_x.append(unconditioned_feats)
        # 4 length of features in each layer (N, C)
        assert len(skip_x) == 4
        return tuple(skip_x)  # jit recommend tuple

    @torch.inference_mode()
    def decode(self, batch: Dict[str, List[torch.Tensor]]):
        """
        batch['features'] 存储 encode 的结果，即每层特征
        """
        skip_x = batch["features"][:3]
        unconditioned_feats = batch["features"][3]

        feats_c = unconditioned_feats.transpose(0, 1).unsqueeze(0)

        #################################
        # 3. apply GNN to communicate the features and get overlap score
        len_src_c = batch["stack_lengths"][-1][0]  # 最后一层点云长度
        src_feats_c, tgt_feats_c = feats_c[:, :, :len_src_c], feats_c[:, :, len_src_c:]
        # 最后一层点云坐标
        pcd_c = batch["points"][-1]
        src_pcd_c, tgt_pcd_c = pcd_c[:len_src_c], pcd_c[len_src_c:]
        src_feats_c, tgt_feats_c = self.gnn(
            src_pcd_c.unsqueeze(0).transpose(1, 2),
            tgt_pcd_c.unsqueeze(0).transpose(1, 2),
            src_feats_c,
            tgt_feats_c,
        )
        feats_c = torch.cat([src_feats_c, tgt_feats_c], dim=-1)

        # NOTE: 下面这俩都是 1x1 卷积
        feats_c = self.proj_gnn(feats_c)  # 没有改变维度，可以当做 gnn 输出的最终特征 256 (1, 256, N)
        scores_c = self.proj_score(feats_c)  # 特征从 256 到 1, 用来预测 score (1, 1, N)

        feats_gnn_norm = (
            F.normalize(feats_c, p=2, dim=1).squeeze(0).transpose(0, 1)
        )  # [N, C], [N, 256] # NOTE: 特征的通道进行归一化处理
        # NOTE: 🌟 GNN 输出特征
        feats_gnn_raw = feats_c.squeeze(0).transpose(0, 1)  # [N, C]
        scores_c_raw = scores_c.squeeze(0).transpose(0, 1)  # [N, 1]

        ####################################
        # 4. decoder part
        # NOTE: 出于某种原因， Decoder 对 gnn 输出的特征进行了归一化
        src_feats_gnn, tgt_feats_gnn = feats_gnn_norm[:len_src_c], feats_gnn_norm[len_src_c:]
        # NOTE: 特征点积获得相似度 (N1, C) @ (N2, C)^T -> (N_1, N_2)
        inner_products = torch.matmul(src_feats_gnn, tgt_feats_gnn.transpose(0, 1))

        src_scores_c, tgt_scores_c = scores_c_raw[:len_src_c], scores_c_raw[len_src_c:]

        # NOTE: 应该是对相似度的分布进行调整，因为 softmax 对较大的输入值无法区分
        temperature = torch.exp(self.epsilon) + 0.03
        # NOTE: F.softmax https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html
        # inner_products (N_1, N_2) 代表了特征的相似度
        # Softmax 将相似度转化为 0-1 的概率分布，沿着 dim=1，也就是说每一行之和为 1
        # 一行 N_2 个数，为 tgt 的点数，代表了当前 src 的点最有可能和 tgt 的哪个点匹配
        # 最后又和 tgt_scores_c 进行矩阵相乘，获得的结果就表示了：**特征相似并且重叠的概率**
        s1 = torch.matmul(F.softmax(inner_products / temperature, dim=1), tgt_scores_c)
        s2 = torch.matmul(F.softmax(inner_products.transpose(0, 1) / temperature, dim=1), src_scores_c)
        # NOTE: 总结来说，scores_saliency 是通过模型预测的 overlap score 加权得到的，权重来自于 src 和 tgt 模型编码特征的相似度
        scores_saliency = torch.cat((s1, s2), dim=0)  # [N, 1]

        # NOTE: RECALL:
        # scores_c_raw          重叠分数
        # scores_saliency       匹配分数
        # feats_gnn_raw         经过 GNN 的点云特征
        # unconditioned_feats   没有经过 GNN 的特征（不包含两个点云之间的关联）
        # NOTE: condition 含义类似“对齐”，控制是否选择用 attention 来让两个点云的特征交互
        # NOTE: add_cross_overlap 的含义应该是，是否把匹配分数（cross overlap score）输出到最终的点云特征中
        # self.condition and self.add_cross_overlap:
        x = torch.cat([scores_c_raw, scores_saliency, feats_gnn_raw], dim=1)

        for block_i, block_op in enumerate(self.decoder_blocks):
            if block_i in self.decoder_concats:
                x = torch.cat([x, skip_x.pop()], dim=1)
            x = block_op(x, batch)
        feats_f = x[:, : self.final_feats_dim]
        scores_overlap = x[:, self.final_feats_dim]
        scores_saliency = x[:, self.final_feats_dim + 1]

        # safe guard our score
        sigmoid = nn.Sigmoid()
        scores_overlap = torch.clamp(sigmoid(scores_overlap.view(-1)), min=0, max=1)
        scores_saliency = torch.clamp(sigmoid(scores_saliency.view(-1)), min=0, max=1)
        scores_overlap = self.regular_score(scores_overlap)
        scores_saliency = self.regular_score(scores_saliency)

        # normalise point-wise features
        feats_f = F.normalize(feats_f, p=2, dim=1)

        return feats_f, scores_overlap, scores_saliency
