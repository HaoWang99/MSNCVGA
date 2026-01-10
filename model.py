import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as td
from torchvision import transforms

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
            
            u = x.mean(1, keepdim=True) 
            s = (x - u).pow(2).mean(1, keepdim=True) 
            x = (x - u) / torch.sqrt(s + self.eps) 
            x = self.weight[:, None] * x + self.bias[:, None] 
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
        super(AU, self).__init__()
        self.conv = nn.Linear(plane, plane)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, length = x.size()
        feat = x.permute(0, 2, 1).contiguous().view(batch_size * length, -1, 1)  
        encode = self.conv(F.avg_pool1d(x, length).view(batch_size, -1).unsqueeze(1))
        energy = torch.matmul(feat, encode.repeat(length, 1, 1)) 
        full_relation = self.softmax(energy) 
        full_aug = torch.bmm(full_relation, feat).view(batch_size, length, -1).permute(0, 2, 1)
        out = full_au
        return out
        
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
            nn.ReLU() 
        )
        
    def forward(self, x):
        shortcut = x.clone()
        x = self.norm(x)
        x = self.proj_first(x)
        a, x = torch.chunk(x, 2, dim=1)

        
        a1 = a
        att1 = self.sub_att1(a1)
       
        a2 = a
        a2 = a2.permute(0, 2, 1).contiguous() 
        att2 = self.sub_att2(a2)
        att2 = att2.permute(0, 2, 1).contiguous() 
       
        a3 = a
        a3 = a3.permute(1, 0, 2).contiguous() 
        att3 = self.sub_att3(a3)
        att3 = att3.permute(1, 0, 2).contiguous() 
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

class wConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, den_pattern=[0.7, 1.0, 0.7], stride=1, padding=1, groups=1, bias=False, dilation=1):
        super(wConv1d, self).__init__() 
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.groups = groups 
        self.dilation = dilation
        
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size))
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu') 
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        self.register_buffer('Phi', self._create_phi_vector(kernel_size, den_pattern))

    def _create_phi_vector(self, kernel_size, den_pattern):

        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd for symmetric Phi vector")
        
        center_index = kernel_size // 2
        
        if len(den_pattern) == 3:
            left_values = torch.linspace(den_pattern[0], den_pattern[1], center_index + 1)[:-1]
            right_values = torch.linspace(den_pattern[1], den_pattern[2], center_index + 1)[1:]
            phi = torch.cat([left_values, torch.tensor([den_pattern[1]]), right_values])
        else:
            center_value = 1.0
            half_size = center_index
            indices = torch.arange(kernel_size).float() - center_index
            phi = torch.exp(-(indices ** 2) / (2 * (half_size / 2) ** 2)) 
        
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

        self.channel_reduction = nn.Conv1d(input_channels, output_features, kernel_size=1)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):

        x = self.channel_reduction(x) 
        x = self.global_avg_pool(x)   
        x = x.squeeze(-1)            
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


