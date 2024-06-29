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
        pcd_c = batch["points"][-1]
        pcd_f = batch["points"][0]
        src_pcd_c, tgt_pcd_c = pcd_c[:len_src_c], pcd_c[len_src_c:]

        sigmoid = nn.Sigmoid()
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
        src_feats_c, tgt_feats_c = feats_c[:, :, :len_src_c], feats_c[:, :, len_src_c:]
        # NOTE: ??? 奇怪，经过 Encoder 点的数目不是减少了吗，怎么接下来又把源点云传给了 gnn？
        src_feats_c, tgt_feats_c = self.gnn(
            src_pcd_c.unsqueeze(0).transpose(1, 2),
            tgt_pcd_c.unsqueeze(0).transpose(1, 2),
            src_feats_c,
            tgt_feats_c,
        )
        feats_c = torch.cat([src_feats_c, tgt_feats_c], dim=-1)

        feats_c = self.proj_gnn(feats_c)
        scores_c = self.proj_score(feats_c)

        feats_gnn_norm = (
            F.normalize(feats_c, p=2, dim=1).squeeze(0).transpose(0, 1)
        )  # [N, C] # NOTE: 特征的通道进行归一化处理
        feats_gnn_raw = feats_c.squeeze(0).transpose(0, 1)  # [N, C]
        scores_c_raw = scores_c.squeeze(0).transpose(0, 1)  # [N, 1]

        ####################################
        # 4. decoder part
        src_feats_gnn, tgt_feats_gnn = feats_gnn_norm[:len_src_c], feats_gnn_norm[len_src_c:]
        # NOTE: 特征点积获得相似度 (N1, C) @ (N2, C)^T -> (N_1, N_2)
        inner_products = torch.matmul(src_feats_gnn, tgt_feats_gnn.transpose(0, 1))

        src_scores_c, tgt_scores_c = scores_c_raw[:len_src_c], scores_c_raw[len_src_c:]

        # NOTE: 应该是对相似度的分布进行调整，因为 softmax 对较大的输入值无法区分
        temperature = torch.exp(self.epsilon) + 0.03
        s1 = torch.matmul(F.softmax(inner_products / temperature, dim=1), tgt_scores_c)
        s2 = torch.matmul(F.softmax(inner_products.transpose(0, 1) / temperature, dim=1), src_scores_c)
        # NOTE: 总结来说，scores_saliency 是通过模型预测的 overlap score 加权得到的，权重来自于 src 和 tgt 模型编码特征的相似度
        scores_saliency = torch.cat((s1, s2), dim=0)

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
