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


class MambaVisionMixer(nn.Module):
    """Mamba mixer from HybridMamba"""
    def __init__(
        self,
        d_model,
        d_state=128,  # 状态空间的维度，决定每个通道的隐状态数量，影响模型的记忆能力和表达能力。 学长另一篇论文用的64
        d_conv=4,  # 卷积核大小。用于局部混合（local mixing），决定卷积操作的感受野。
        expand=2, # 通道扩展倍数。内部隐藏层的维度是 expand * d_model，影响模型容量
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states):
        # 处理4D输入 (batch, channels, patches, features)
        if hidden_states.dim() == 4:
            bz, ch_num, patch_num, d_model = hidden_states.shape
            # Reshape为序列格式: (batch, seq_len, d_model)
            hidden_states = hidden_states.view(bz, ch_num * patch_num, d_model)
            output = self._forward_sequence(hidden_states)
            # Reshape回原始格式
            return output.view(bz, ch_num, patch_num, d_model)
        else:
            return self._forward_sequence(hidden_states)
    
    def _forward_sequence(self, hidden_states):
        """处理序列格式的输入"""
        B, L, D = hidden_states.shape
        x_and_res = self.in_proj(hidden_states)
        x, res = x_and_res.split([self.d_inner, self.d_inner], dim=-1)

        x = rearrange(x, "b l d -> b d l")
        x = self.conv1d(x)[:, :, :L]
        x = rearrange(x, "b d l -> b l d")

        x = F.silu(x)

        y = self.ssm(x)

        y = y * F.silu(res)

        out = self.out_proj(y)
        return out

    def ssm(self, x):
        D = self.D.float()
        A = -torch.exp(self.A_log.float())
        dt, B, C = torch.split(self.x_proj(x), [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        dt = torch.einsum('bld,nd->bnl', dt, self.dt_proj.weight)
        
        B = rearrange(B, "b l d -> b d l")
        C = rearrange(C, "b l d -> b d l")
        
        y = selective_scan_fn(x.transpose(1,2), 
                              dt, 
                              A, 
                              B, 
                              C, 
                              D, 
                              z=None, 
                              delta_bias=self.dt_proj.bias.float(), 
                              delta_softplus=True, 
                              return_last_state=False)
        
        return y.transpose(1,2)


class HybridEncoderLayer(nn.Module):
    """可以选择使用Attention或Mamba"""
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.gelu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = True, norm_first: bool = True,
                 bias: bool = True, use_mamba: bool = False, 
                 # Mamba specific parameters
                 d_state: int = 16, d_conv: int = 4, expand: int = 2,

                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        self.use_mamba = use_mamba
        
        if use_mamba:
            # 使用Mamba mixer
            self.mixer = MambaVisionMixer(
                d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand,
                device=device, dtype=dtype
            )
        else:
            # 使用标准的多头自注意力
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
        if self.use_mamba:
            # Mamba处理
            mixer_output = self.mixer(x)            #后边再调
            return self.dropout1(mixer_output)
        else:
            # Attention处理4D输入
            bz, ch_num, patch_num, d_model = x.shape
            
            # Reshape为序列格式: (batch, seq_len, d_model)
            x_reshaped = x.view(bz, ch_num * patch_num, d_model)
            
            # 标准自注意力
            attn_output, _ = self.mixer(x_reshaped, x_reshaped, x_reshaped,
                                      attn_mask=attn_mask, need_weights=False)
            
            # Reshape回原始格式
            attn_output = attn_output.view(bz, ch_num, patch_num, d_model)
            
            return self.dropout1(attn_output)

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
                 bias: bool = True, device=None, dtype=None) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        # 在这里根据一个list来定义每一层的类型
        for i in range(len(depths)):
            for j in range(depths[i]):
                layer_type = stage_types[i]
                print(f"Building layer {len(self.layers)} as {layer_type}")
                self.layers.append(
                    HybridEncoderLayer(
                        d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                        batch_first=True, norm_first=True,
                        activation=F.gelu,
                        use_mamba= (layer_type == "mamba")
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
    model = HybridModel(
        in_chans=32,
        patch_size=200,
        d_model=200,
        num_layers=8,
        nhead=8,
        dim_feedforward=800,
        hybrid_mode='alternate',  # 可以改为其他模式测试
        num_classes=2
    )
    model = model.to(device)

    # 测试输入: (batch, channels, patches, features)
    x = torch.randn((4, 32, 30, 200)).to(device)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # 测试自定义层类型
    custom_model = HybridModel(
        in_chans=32,
        patch_size=200,
        d_model=200,
        num_layers=6,
        nhead=8,
        dim_feedforward=800,
        hybrid_mode='custom',
        layer_types=['mamba', 'mamba', 'attention', 'mamba', 'attention', 'attention'],
        num_classes=2
    )
    custom_model = custom_model.to(device)
    y_custom = custom_model(x)
    print(f"Custom model output shape: {y_custom.shape}")
