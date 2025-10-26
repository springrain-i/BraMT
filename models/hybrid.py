import copy
from typing import Optional, Union, Callable
import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from timm.layers import DropPath, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


import os

base_info_files = 'base_info.txt'

class MambaVisionMixer(nn.Module):
    """Mamba mixer from HybridMamba"""
    def __init__(
        self,
        d_model,
        d_state=128,  # 状态空间的维度，决定每个通道的隐状态数量，影响模型的记忆能力和表达能力。 学长另一篇论文用的64
        d_conv=4,  # 卷积核大小。用于局部混合（local mixing），决定卷积操作的感受野。
        expand=2, # 通道扩展倍数。内部隐藏层的维度是 expand * d_model，影响模型容量
        dt_rank="auto",  # 时间步长/离散化参数的低秩维度；控制多时间尺度门控的表达力与开销。
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True, #局部卷积是否有 bias。
        bias=False, # 主线性投影（in_proj / out_proj）是否使用 bias。
        use_fast_path=True,
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        # 记录底层参数信息
        with open(os.path.join(base_info_files), 'a') as f:
            f.write(f"mamba layer--->d_model: {d_model}, d_state: {d_state}, d_conv: {d_conv}, expand: {expand}, conv_bias: {conv_bias}, bias: {bias}\n")
            f.write("-------------------\n")

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        
        # 将输入的特征投影维度扩大，用于分支
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        # SSM分支: 获取B, C, dt
        self.x_proj = nn.Linear(
            self.d_inner//2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        # 
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner//2, bias=True, **factory_kwargs)
        

        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(self.d_inner//2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner//2,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.d_inner//2, device=device))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        # 使用两个卷积分别处理x和z，不共享权重
        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_inner//2,
            out_channels=self.d_inner//2,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner//2,
            **factory_kwargs,
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_inner//2,
            out_channels=self.d_inner//2,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner//2,
            **factory_kwargs,
        )
    # def forward(self, hidden_states):
    #     """处理序列格式的输入"""
    #     B, L, D = hidden_states.shape
    #     x_and_res = self.in_proj(hidden_states)
    #     x, res = x_and_res.split([self.d_inner, self.d_inner], dim=-1)

    #     x = rearrange(x, "b l d -> b d l")
    #     x = self.conv1d(x)[:, :, :L]
    #     x = rearrange(x, "b d l -> b l d")

    #     x = F.silu(x)

    #     y = self.ssm(x)

    #     y = y * F.silu(res)

    #     out = self.out_proj(y)
    #     return out   门控机制得到的输出，原本mamba的方式

    def forward(self, hidden_states):
        B, L, D = hidden_states.shape
        x_z = self.in_proj(hidden_states)
        x_z = rearrange(x_z, "b l d -> b d l")
        x, z = x_z.chunk(2, dim=1)
        x = F.silu(F.conv1d(input=x, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias, padding='same', groups=self.d_inner//2))
        z = F.silu(F.conv1d(input=z, weight=self.conv1d_z.weight, bias=self.conv1d_z.bias, padding='same', groups=self.d_inner//2))

        y = self.ssm(x, L)

        y = torch.cat([y, z], dim=1)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        return out

    def ssm(self, x, L):
        A = -torch.exp(self.A_log.float())
        x_dbl = self.x_proj(rearrange(x,"b d l -> (b l) d"))
        D = self.D.float()
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=L)

        B = rearrange(B, "(b l) dstate -> b dstate l", l=L).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=L).contiguous()
        y = selective_scan_fn(x, 
                              dt, 
                              A, 
                              B, 
                              C, 
                              self.D.float(), 
                              z=None, 
                              delta_bias=self.dt_proj.bias.float(), 
                              delta_softplus=True, 
                              return_last_state=None)
        
        return y


class HybridEncoderLayer(nn.Module):
    """可以选择使用Attention或Mamba"""
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.gelu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = True, norm_first: bool = True,
                 bias: bool = True, use_mamba: bool = False, axis_order: bool = True,
                 # Mamba specific parameters
                 d_state: int = 16, d_conv: int = 4, expand: int = 2, conv_bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        self.use_mamba = use_mamba
        self.axis_order = axis_order
        if use_mamba:
            # 使用Mamba mixer
            print(f"Building MambaVisionMixer with d_model={d_model}, d_state={d_state}, d_conv={d_conv}, expand={expand}")
            if axis_order:
                print("*****use axis_order*****")
                self.in_project = nn.Linear(d_model, d_model*2, bias=bias, **factory_kwargs)
                self.out_project = nn.Linear(d_model*2,d_model, bias=bias, **factory_kwargs)
                self.mixer_time_ch = MambaVisionMixer(
                    d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand,
                    conv_bias=conv_bias, bias=bias,
                    device=device, dtype=dtype
                )
                self.mixer_ch_time = MambaVisionMixer(
                    d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand,
                    conv_bias=conv_bias, bias=bias,
                    device=device, dtype=dtype
                )
            else:
                self.mixer = MambaVisionMixer(
                    d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand,
                    conv_bias=conv_bias, bias=bias,
                    device=device, dtype=dtype
                )

        else:
            # 使用标准的多头自注意力
            print(f"Building MultiheadAttention with d_model={d_model}, nhead={nhead}")
            with open(os.path.join(base_info_files), 'a') as f:
                f.write(f"attn layer--->d_model: {d_model}, nhead: {nhead}, dropout: {dropout} bias: {bias}\n")
                f.write("-------------------\n")
            self.mixer = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                             bias=bias, batch_first=batch_first,
                                             **factory_kwargs)
        
        # Feed Forward Network
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # 激活函数处理
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)
        self.activation = activation

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        """
        处理4D输入 (batch, channels, patches, features)
        将其reshape为适合transformer的格式
        """
        x = src
        if self.norm_first:
            x = x + self._mixer_block(self.norm1(x), src_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._mixer_block(x, src_mask))
            x = self.norm2(x + self._ff_block(x))
        return x

    def _mixer_block(self, x: Tensor, attn_mask: Optional[Tensor]) -> Tensor:
        """Mixer block - 可以是Attention或Mamba"""
        bz, ch_num, patch_num, d_model = x.shape


        if self.use_mamba:
            # Mamba处理
            if self.axis_order: # 考虑不同的顺序影响
                x = self.in_project(x)
                #print(x.shape)
                x_time_ch = x[:,:,:,:d_model]
                x_ch_time = x[:,:,:,d_model:]

                x_time_ch = x_time_ch.contiguous().view(bz, ch_num*patch_num,d_model)
                x_ch_time = x_ch_time.contiguous().view(bz*patch_num, ch_num, d_model)

                output_time_ch = self.mixer_time_ch(x_time_ch)
                output_ch_time = self.mixer_ch_time(x_ch_time)

                output_time_ch = output_time_ch.contiguous().view(bz, ch_num, patch_num, d_model)
                output_ch_time = output_ch_time.contiguous().view(bz, ch_num, patch_num, d_model)
                output = torch.concat((output_time_ch,output_ch_time),dim=3)

                output = self.out_project(output)
            else:
                x_reshaped = x.contiguous().view(bz, ch_num * patch_num, d_model)
                output = self.mixer(x_reshaped)            
                output = output.contiguous().view(bz, ch_num, patch_num, d_model)
        else:
            # 标准自注意力
            x_reshaped = x.view(bz, ch_num * patch_num, d_model)
            output, _ = self.mixer(x_reshaped, x_reshaped, x_reshaped,
                                      attn_mask=attn_mask, need_weights=False)
            output = output.contiguous().view(bz, ch_num, patch_num, d_model)


        return self.dropout1(output)

    def _ff_block(self, x: Tensor) -> Tensor:
        """Feed forward block"""
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    """获取激活函数"""
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise RuntimeError(f"activation should be relu/gelu, not {activation}")


class HybridEncoder(nn.Module):
    def __init__(self, depths, stage_types, norm=None, d_model: int = 200, nhead: int = 8, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.gelu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = True, norm_first: bool = True,
                 bias: bool = True, 
                 # Mamba specific parameters
                 d_state: int = 16, d_conv: int = 4, expand: int = 2, conv_bias: bool = True,
                 device=None, dtype=None) -> None:
        super().__init__()

        open(os.path.join(base_info_files), 'w').close() # 情况base_info

        self.layers = nn.ModuleList()
        # 在这里根据一个list来定义每一层的类型
        for i in range(len(depths)):
            for j in range(depths[i]):
                layer_type = stage_types[i]
                self.layers.append(
                    HybridEncoderLayer(
                        d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,

                        batch_first=True, norm_first=True,
                        activation=F.gelu,
                        # Mamba specific parameters
                        use_mamba= (layer_type == "mamba"), d_state=d_state, d_conv=d_conv, expand=expand, conv_bias=conv_bias
                    )
                )

        # encoder_layers是一个包含不同类型layer的列表
        self.num_layers = sum(depths)
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask)
        if self.norm is not None:
            output = self.norm(output)
        return output




if __name__ == '__main__':
    # 测试代码
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 测试不同的混合模式
    model = HybridEncoder(
        d_model=200,
        d_state=16,
        nhead=8,
        dim_feedforward=800,
        depths=[6,6],
        stage_types=['mamba','attn'],
    )

    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(total_params)
