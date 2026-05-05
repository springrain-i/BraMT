import copy
from typing import Optional, Union, Callable

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


class SimpleTransformerEncoder(nn.Module):
    """简单的Transformer Encoder"""
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        output = src
        for mod in self.layers:
            output = mod(output, src_mask=mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


class SimpleTransformerEncoderLayer(nn.Module):
    """简单的Transformer Encoder Layer - 标准实现"""
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.gelu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = True, norm_first: bool = True,
                 bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        # 标准的多头自注意力
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
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
            x = x + self._sa_block(self.norm1(x), src_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask))
            x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor]) -> Tensor:
        """Self-attention block - 处理4D输入"""
        # 输入: (batch, channels, patches, features)
        bz, ch_num, patch_num, d_model = x.shape
        
        # Reshape为序列格式: (batch, seq_len, d_model)
        x_reshaped = x.view(bz, ch_num * patch_num, d_model)
        
        # 标准自注意力
        attn_output, _ = self.self_attn(x_reshaped, x_reshaped, x_reshaped,
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


def _get_clones(module, N):
    """复制模块N次"""
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


if __name__ == '__main__':
    # 测试代码
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    encoder_layer = SimpleTransformerEncoderLayer(
        d_model=256, nhead=8, dim_feedforward=1024, batch_first=True, norm_first=True,
        activation=F.gelu
    )
    encoder = SimpleTransformerEncoder(encoder_layer, num_layers=6)
    encoder = encoder.to(device)

    # 测试输入: (batch, channels, patches, features)
    a = torch.randn((4, 19, 30, 256)).to(device)
    b = encoder(a)
    print(f"Input shape: {a.shape}")
    print(f"Output shape: {b.shape}")
