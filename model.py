# create: Fangyu Liu; Hao Wang
# date: October 2025

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as td
from torchvision import transforms

#Multi-scale Attention Network for Single Image Super-Resolution (CVPR 2024)
#https://arxiv.org/abs/2209.14145

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # Compute mean and variance for channels_first format
            u = x.mean(1, keepdim=True) # Mean across the normalized shape
            s = (x - u).pow(2).mean(1, keepdim=True) # Variance
            x = (x - u) / torch.sqrt(s + self.eps) # Normalize
            x = self.weight[:, None] * x + self.bias[:, None] # Scale and shift
            return x


# Gated Spatial Attention Unit (GSAU)
class GSAU(nn.Module):
    def __init__(self, n_feats, k=3):
        super().__init__()

        i_feats = 2 * n_feats

        self.norm = LayerNorm(n_feats, data_format='channels_first')
        self.scale = nn.Parameter(torch.ones(1, n_feats, 1) * 0.25, requires_grad=True)

        self.X = nn.Conv1d(n_feats, n_feats, k, 1, k // 2, groups=n_feats)

        self.proj_first = nn.Sequential(
            nn.Conv1d(n_feats, i_feats, 1, 1, 0))

        self.proj_last = nn.Sequential(
            nn.Conv1d(n_feats, n_feats, 1, 1, 0))

    def forward(self, x):
        shortcut = x.clone()
        x = self.norm(x)
        x = self.proj_first(x)
        a, x = torch.chunk(x, 2, dim=1)
        att = self.X(a)
        x = self.proj_last(x * att) * self.scale + shortcut
        return x

# Attention Unit
class AU(nn.Module):
    def __init__(self, plane):
        # 初始化函数，plane是输入和输出特征图的通道数
        super(AU, self).__init__()
        # 定义全连接层，conv
        self.conv = nn.Linear(plane, plane)
        
        # 定义softmax操作，用于计算关系矩阵
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        前向传播过程
        Args:
            x: 输入的特征图，形状为 (batch_size, channels, length)
        Returns:
            out: 处理后的特征图，形状为 (batch_size, channels, length)
        """
        batch_size, _, length = x.size()
        
        # 对输入张量进行重塑，获取序列特征
        # 在1D情况下，我们只需要处理序列维度
        feat = x.permute(0, 2, 1).contiguous().view(batch_size * length, -1, 1)  # (B,C,L) → (B,L,C) → (B×L,C,1)
        
        # 对输入张量进行全局平均池化，并通过全连接层进行编码
        encode = self.conv(F.avg_pool1d(x, length).view(batch_size, -1).unsqueeze(1))  # 全局编码
        # 过程: 
        # 1. 平均池化: F.avg_pool1d(x, length) → (B,C,1) 在整个序列维度池化
        # 2. 重塑: .view(batch_size, -1) → (B,C)
        # 3. 增加维度: .unsqueeze(1) → (B,1,C)
        # 4. 全连接: self.conv1 → (B,1,C)
        
        # 计算序列关系矩阵
        energy = torch.matmul(feat, encode.repeat(length, 1, 1))  # 计算序列关系
        # feat: (B×L,C,1), encode.repeat: (B×L,1,C) → 矩阵乘法后: (B×L,C,C)
        
        # 计算经过softmax后的关系矩阵
        full_relation = self.softmax(energy)  # 序列关系, (B×L,C,C)
        
        # 通过矩阵乘法和关系矩阵，对特征进行加权和增强
        full_aug = torch.bmm(full_relation, feat).view(batch_size, length, -1).permute(0, 2, 1)  # 序列增强
        # 过程:
        # 1. 矩阵乘法: torch.bmm(full_relation, feat) → (B×L,C,1)
        # 2. 重塑: .view(batch_size, length, -1) → (B,L,C)  
        # 3. 转置: .permute(0, 2, 1) → (B,C,L)

        out = full_aug
        
        return out  # 返回处理后的特征图

# Multi-View Attention Unit (MVAU)
class MVAU(nn.Module):
    def __init__(self, B, C, L, k=3):
        super().__init__()
        n_feats = C
        i_feats = 2 * n_feats

        self.norm = LayerNorm(n_feats, data_format='channels_first')
        self.scale = nn.Parameter(torch.ones(1, n_feats, 1) * 0.25, requires_grad=True)
        self.scale1 = nn.Parameter(torch.ones(1) * 0.25, requires_grad=True)
        self.scale2 = nn.Parameter(torch.ones(1) * 0.25, requires_grad=True)
        self.scale3 = nn.Parameter(torch.ones(1) * 0.25, requires_grad=True)

        self.sub_att1 = AU(plane=C)
        self.sub_att2 = AU(plane=L)
        self.sub_att3 = AU(plane=B)

        self.proj_first = nn.Sequential(
            nn.Conv1d(n_feats, i_feats, 1, 1, 0))

        self.proj_last = nn.Sequential(
            nn.Conv1d(n_feats, n_feats, k, 1, k // 2, bias=False),
            nn.ReLU()  # ReLU激活函数
        )
        
    def forward(self, x):
        shortcut = x.clone()
        x = self.norm(x)
        x = self.proj_first(x)
        a, x = torch.chunk(x, 2, dim=1)

        # a, x = (B, C, L)
        a1 = a
        att1 = self.sub_att1(a1)
        # a, x = (B, C, L)
        a2 = a
        a2 = a2.permute(0, 2, 1).contiguous() # (B, L, C)
        att2 = self.sub_att2(a2)
        att2 = att2.permute(0, 2, 1).contiguous() # (B, C, L)
        # a, x = (B, C, L)
        a3 = a
        a3 = a3.permute(1, 0, 2).contiguous() # (C, B, L)
        att3 = self.sub_att3(a3)
        att3 = att3.permute(1, 0, 2).contiguous() # (B, C, L)

        att = att1 * self.scale1 + att2 * self.scale2 + att3 * self.scale3
        x = self.proj_last(x * att) * self.scale + shortcut
        return x

# Cross-View Gated Attention (CVGA)
class CVGA(nn.Module):
    def __init__(self, k, num_samples, num_channels, length, periodic=True, channel=True, positional=True, sample=True):
        super(CVGA, self).__init__()
        self.mvau = MVAU(B=num_samples, C=num_channels, L=length, k=k)
        self.gsau = GSAU(n_feats=num_channels, k=k)

    def forward(self, x):
        x = self.mvau(x)
        x = self.gsau(x)
        return x

# 方案1：动态生成Phi向量
class wConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, den_pattern=[0.7, 1.0, 0.7], stride=1, padding=1, groups=1, bias=False, dilation=1):
        super(wConv1d, self).__init__() 
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.groups = groups 
        self.dilation = dilation
        
        # 1D卷积权重
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size))
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu') 
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        # 动态生成Phi向量以匹配卷积核大小
        self.register_buffer('Phi', self._create_phi_vector(kernel_size, den_pattern))

    def _create_phi_vector(self, kernel_size, den_pattern):
        """根据卷积核大小动态生成Phi向量"""
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd for symmetric Phi vector")
        
        center_index = kernel_size // 2
        
        if len(den_pattern) == 3:
            # 使用三元素模式 [left, center, right]
            left_values = torch.linspace(den_pattern[0], den_pattern[1], center_index + 1)[:-1]
            right_values = torch.linspace(den_pattern[1], den_pattern[2], center_index + 1)[1:]
            phi = torch.cat([left_values, torch.tensor([den_pattern[1]]), right_values])
        else:
            # 通用模式：从中心向两边衰减
            center_value = 1.0
            half_size = center_index
            indices = torch.arange(kernel_size).float() - center_index
            phi = torch.exp(-(indices ** 2) / (2 * (half_size / 2) ** 2))  # 高斯衰减
        
        return phi

    def forward(self, x):
        Phi = self.Phi.to(x.device)
        weight_Phi = self.weight * Phi.view(1, 1, -1)
        return F.conv1d(x, weight_Phi, bias=self.bias, stride=self.stride, padding=self.padding, groups=self.groups, dilation=self.dilation)

# wConv with CVGA
class wConv_with_CVGA(nn.Module):
    def __init__(self, batch, in_channels, out_channels, length, kernel_size=3, stride=2, padding=1, groups=1, dilation=1, bias=True, den_pattern=[0.7, 1.0, 0.7]):
        super().__init__()
        self.Conv = nn.Sequential(
            wConv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, den_pattern=den_pattern, stride=stride, padding=padding, groups=groups, bias=bias, dilation=dilation),
            # nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias, dilation=dilation),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1)
        )
        self.multi_att = CVGA(k=kernel_size, num_samples=batch, num_channels=out_channels, length=length)

    def forward(self, x):
        x = self.Conv(x)
        x = self.multi_att(x)
        return x


class GlobalAvgPool(nn.Module):
    def __init__(self, input_channels=512, output_features=18):
        super().__init__()
        # 先用1x1卷积将512通道降维到18通道
        self.channel_reduction = nn.Conv1d(input_channels, output_features, kernel_size=1)
        # 然后全局平均池化（将长度维度池化为1）
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        # x形状: [B, 512, L]
        x = self.channel_reduction(x)  # [B, 18, L]
        x = self.global_avg_pool(x)    # [B, 18, 1]
        x = x.squeeze(-1)              # [B, 18]
        return x


class Our(nn.Module):
    def __init__(self, batch, channels):
        super().__init__()

        self.sub_scale3_1 = wConv_with_CVGA(batch=batch, in_channels=channels, out_channels=2*channels, length=248, kernel_size=3, stride=2, padding=(3 // 2) * 1, groups=1, dilation=1)

        self.sub_scale3_2 = wConv_with_CVGA(batch=batch, in_channels=2*channels, out_channels=4*channels, length=122, kernel_size=3, stride=2, padding=(3 // 2) * 1, groups=2, dilation=1)

        self.sub_scale3_3 = wConv_with_CVGA(batch=batch, in_channels=4*channels, out_channels=8*channels, length=59, kernel_size=3, stride=2, padding=(3 // 2) * 1, groups=4, dilation=1)


        self.sub_scale5_1 = wConv_with_CVGA(batch=batch, in_channels=channels, out_channels=2*channels, length=248, kernel_size=5, stride=2, padding=(5 // 2) * 1, groups=1, dilation=1)

        self.sub_scale5_2 = wConv_with_CVGA(batch=batch, in_channels=2*channels, out_channels=4*channels, length=122, kernel_size=5, stride=2, padding=(5 // 2) * 1, groups=2, dilation=1)

        self.sub_scale5_3 = wConv_with_CVGA(batch=batch, in_channels=4*channels, out_channels=8*channels, length=59, kernel_size=5, stride=2, padding=(5 // 2) * 1, groups=4, dilation=1)
        

        self.sub_scale7_1 = wConv_with_CVGA(batch=batch, in_channels=channels, out_channels=2*channels, length=248, kernel_size=7, stride=2, padding=(7 // 2) * 1, groups=1, dilation=1)

        self.sub_scale7_2 = wConv_with_CVGA(batch=batch, in_channels=2*channels, out_channels=4*channels, length=122, kernel_size=7, stride=2, padding=(7 // 2) * 1, groups=2, dilation=1)

        self.sub_scale7_3 = wConv_with_CVGA(batch=batch, in_channels=4*channels, out_channels=8*channels, length=59, kernel_size=7, stride=2, padding=(7 // 2) * 1, groups=4, dilation=1)

        self.sub_fusion2 = wConv_with_CVGA(batch=batch, in_channels=2*channels, out_channels=4*channels, length=122, kernel_size=7, stride=2, padding=(7 // 2) * 1, groups=2, dilation=1)
        self.sub_fusion3 = wConv_with_CVGA(batch=batch, in_channels=4*channels, out_channels=8*channels, length=59, kernel_size=7, stride=2, padding=(7 // 2) * 1, groups=4, dilation=1)

        self.scale_fusion1 = nn.Conv1d(3*2*channels, 2*channels, 1, 1, 0)
        self.scale_fusion2 = nn.Conv1d(3*4*channels, 4*channels, 1, 1, 0)
        self.scale_fusion3 = nn.Conv1d(3*8*channels, 8*channels, 1, 1, 0)

        self.scale1 = nn.Parameter(torch.ones(1) * 0.25, requires_grad=True)
        self.scale2 = nn.Parameter(torch.ones(1) * 0.25, requires_grad=True)
        self.scale3 = nn.Parameter(torch.ones(1) * 0.25, requires_grad=True)

        self.proj_first = nn.Sequential(
            nn.Conv1d(channels, 3 * channels, 1, 1, 0))

        self.biGRU = nn.GRU(input_size=59, hidden_size=15, num_layers=2, batch_first=True, bidirectional=True)

        self.pool_layer = GlobalAvgPool(input_channels=256, output_features=18)

    def forward(self, sensor_accel, sensor_gyro):
        # (B, 8, 500)
        x = torch.cat((sensor_accel, sensor_gyro), dim=1)
        x = self.proj_first(x)
        a_1, a_2, a_3 = torch.chunk(x, 3, dim=1)

        # step1
        a_1 = self.sub_scale3_1(a_1)
        a_2 = self.sub_scale5_1(a_2)
        a_3 = self.sub_scale7_1(a_3)
        a_cat = torch.cat([a_1, a_2, a_3], dim=1)
        a_fusion = self.scale_fusion1(a_cat) * self.scale1

        # step2
        a_1 = self.sub_scale3_2(a_1)
        a_2 = self.sub_scale5_2(a_2)
        a_3 = self.sub_scale7_2(a_3)
        a_cat = torch.cat([a_1, a_2, a_3], dim=1)
        a_fusion = self.sub_fusion2(a_fusion)
        a_fusion = a_fusion + self.scale_fusion2(a_cat) * self.scale2

        # step3
        a_1 = self.sub_scale3_3(a_1)
        a_2 = self.sub_scale5_3(a_2)
        a_3 = self.sub_scale7_3(a_3)
        a_cat = torch.cat([a_1, a_2, a_3], dim=1)
        a_fusion = self.sub_fusion3(a_fusion)
        a_fusion = a_fusion + self.scale_fusion3(a_cat) * self.scale3

        x = torch.cat([a_1, a_2, a_3, a_fusion], dim=1)
        x, _ = self.biGRU(x)
        x = self.pool_layer(x)
        return x
    

if __name__ == '__main__':
    sensor_accel = torch.randn(4, 4, 500)
    sensor_gyro = torch.randn(4, 4, 500)
    model = Our(4, 8)
    model.eval()

    y_pred = model(sensor_accel, sensor_gyro)

    print(y_pred.shape)

    # import time

    # # 暖身运行（减少首次运行的 overhead）
    # with torch.no_grad():
    #     for _ in range(100):
    #         model(sensor_accel, sensor_gyro)

    # # 测量推理时间
    # num_runs = 1000
    # total_time = 0.0

    # start_time = time.time()
    # # 测量推理时间，100轮
    # with torch.no_grad():
    #     for _ in range(num_runs):
    #         output = model(sensor_accel, sensor_gyro)
    #         end_time = time.time()
    # total_time += (end_time - start_time)

    # inference_time = total_time  / num_runs

    # print(f"多轮平均推理时间: {inference_time} 秒")

    # inference_time_seconds = total_time  / num_runs
    # inference_time_ms = inference_time_seconds * 1000  # 转换为毫秒

    # print(f"多轮平均推理时间: {inference_time_ms} 毫秒")

    # output = model(sensor_accel, sensor_gyro)
    # print(output.shape)

    # print('--------------------------------------------------------------------------------')
    # from thop import profile
    # total = sum([param.nelement() for param in model.parameters()])
    # #1048576 == 1024 * 1024
    # #1073741824 == 1024 * 1024 * 1024
    # #%.2f，保留2位小数点
    # print("Number of parameter: %.2fM" % (total / 1048576))
    # flops, params = profile(model, inputs=(sensor_accel, sensor_gyro,))
    # print("Number of flops: %.2fG" % (flops / 1073741824))
    # print("Number of parameters: %.2fM" % (params / 1048576))
    # print("Number of flops: %.2f" % (flops / 1))
    # print("Number of parameters: %.2f" % (params / 1))


# torch.Size([1, 18])
# 多轮平均推理时间: 0.026763784646987913 秒
# 多轮平均推理时间: 26.763784646987915 毫秒
# torch.Size([1, 18])
# --------------------------------------------------------------------------------
# Number of parameter: 0.54M
# [INFO] Register zero_ops() for <class 'torch.nn.modules.activation.ReLU'>.
# [INFO] Register zero_ops() for <class 'torch.nn.modules.pooling.MaxPool1d'>.
# [INFO] Register zero_ops() for <class 'torch.nn.modules.container.Sequential'>.
# [INFO] Register count_linear() for <class 'torch.nn.modules.linear.Linear'>.
# [INFO] Register count_softmax() for <class 'torch.nn.modules.activation.Softmax'>.
# [INFO] Register count_convNd() for <class 'torch.nn.modules.conv.Conv1d'>.
# [INFO] Register count_gru() for <class 'torch.nn.modules.rnn.GRU'>.
# [INFO] Register count_adap_avgpool() for <class 'torch.nn.modules.pooling.AdaptiveAvgPool1d'>.
# Number of flops: 0.04G
# Number of parameters: 0.52M
# Number of flops: 44492050.00
# Number of parameters: 540958.00