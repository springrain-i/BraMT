import torch
import torch.nn as nn
from timm.layers import DropPath, trunc_normal_
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import rearrange, repeat
import math
from Patch_embedding import PatchEmbedding
from einops.layers.torch import Rearrange
# Mamba and Attention blocks from MambaVision
class MambaVisionMixer(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
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
        
        # 修复 dt 的处理
        # dt 形状: (batch, seq_len, dt_rank)
        # dt_proj.weight 形状: (d_inner, dt_rank)
        dt = torch.einsum('bld,nd->bnl', dt, self.dt_proj.weight)  # (batch, d_inner, seq_len)
        
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

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        identity = x  # 保存输入用于残差连接
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        # 在attention模块内部添加残差连接
        x = x + identity
        return x

class HybridBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_attn=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if use_attn:
            self.mixer = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        else:
            self.mixer = MambaVisionMixer(d_model=dim)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        x = x + self.drop_path(self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# Main Model
class HybridMamba(nn.Module):
    """
    HybridMamba模型，支持多种Mamba和Attention的混合模式
    
    混合模式选项:
    - 'alternate': 交错使用Mamba和Attention (默认) - 12层
    - 'mamba_first': 前面更多Mamba，后面Attention - 11层  
    - 'attention_first': 前面更多Attention，后面Mamba - 11层
    - 'all_mamba': 所有层都使用Mamba - 16层
    - 'all_attention': 所有层都使用Attention - 16层
    
    层数配置:
    - mamba_first: [3,3,3,2] = 11layers (前8层Mamba, 后3层Attention)
    - attention_first: [2,3,3,3] = 11layers (前5层Attention, 后6层Mamba)  
    - alternate: [3,3,3,3] = 12layers (交错使用)
    - all_mamba/all_attention: [4,4,4,4] = 16layers
    - 支持自定义层数: custom_depths=[3,4,5,3] = 15layers
    """
    def __init__(self, in_chans , patch_size=200, out_dim=200,
                 d_model=200, depths=[2,2,6,2], num_heads=[4,8,10,20],
                 mlp_ratio=4., qkv_bias=True, drop_rate=0.1, attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm, 
                 hybrid_mode='alternate', custom_depths=None,stage_types=None, num_classes=2):
        super().__init__()
        self.depths = depths
        self.d_model = d_model
        self.hybrid_mode = hybrid_mode
        self.num_classes = num_classes

        # Contain Positional embedding
        self.patch_embed = PatchEmbedding(in_dim=patch_size, out_dim=out_dim, d_model=d_model,seq_len=in_chans)
        
        # 简化层数配置 - 减少模型复杂度以便更好地学习
        if custom_depths is not None:
            depths = custom_depths  # 使用用户自定义的层数
        elif self.hybrid_mode == 'mamba_first':
            depths = [2, 2, 2, 2]  
            stage_types = ['mamba','mamba','atten','atten']
        elif self.hybrid_mode == 'attention_first':
            depths = [2, 2, 2, 2]  
            stage_types = ['atten','atten','mamba','mamba']
        elif self.hybrid_mode == 'alternate':
            depths = [2, 2, 2, 2]  
            stage_types = ['mamba','atten','mamba','atten']
        elif self.hybrid_mode == 'all_mamba':
            depths = [2, 2, 2, 2]  
            stage_types = ['mamba','mamba','mamba','mamba']
        elif self.hybrid_mode == 'all_attention':
            depths = [2, 2, 2, 2]  
            stage_types = ['atten','atten','atten','atten']
        else:
            depths = [2, 2, 2, 2] 
            stage_types = ['mamba','mamba','atten','atten']
        
        total_layers = sum(depths)
        print(f"Using {self.hybrid_mode} mode with {total_layers} total layers")
        
        # 根据实际层数计算drop path rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_layers)]
        
        self.blocks = nn.ModuleList()
        layer_idx = 0
        
        for i in range(len(depths)):
            for j in range(depths[i]):
                # 根据stage_types决定使用哪种mixer
                use_attn = True if stage_types[i] == 'atten' else False
                
                # 确保 num_heads 能被 d_model 整除
                effective_num_heads = min(num_heads[i], d_model)
                while d_model % effective_num_heads != 0:
                    effective_num_heads -= 1
                
                self.blocks.append(
                    HybridBlock(
                        dim=d_model,
                        num_heads=effective_num_heads,
                        mlp_ratio=2.,  # 减少 MLP 比例
                        qkv_bias=qkv_bias,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[layer_idx] if layer_idx < len(dpr) else 0,
                        norm_layer=norm_layer,
                        use_attn=use_attn
                    )
                )
                # 打印每一层使用的mixer类型
                mixer_type = "Attention" if use_attn else "Mamba"
                print(f"Stage {i+1}, Layer {layer_idx}: {mixer_type}")
                
                layer_idx += 1        
        self.proj_out = nn.Sequential(
            # nn.Linear(d_model, d_model*2),
            # nn.GELU(),
            # nn.Linear(d_model*2, d_model),
            # nn.GELU(),
            nn.Linear(d_model, out_dim),
        )
        if num_classes > 2:
            self.classifier = nn.Sequential(   
                Rearrange('b c s d -> b d c s'),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Dropout(0.1),  # 减少 dropout
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(d_model // 2, num_classes),
            )
        else:
            self.classifier = nn.Sequential(   
                Rearrange('b c s d -> b d c s'),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Dropout(0.1),  # 减少 dropout
                nn.Linear(d_model, 1),
                nn.Sigmoid(),
                Rearrange('b 1 -> (b 1)')
            )

        self.norm = norm_layer(d_model)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # 为mamba_first模式使用更小的初始化方差
            if self.hybrid_mode == 'mamba_first':
                trunc_normal_(m.weight, std=.01)  # 更小的初始化
            else:
                trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        bz, ch_num, patch_num, patch_size = x.shape
        x = self.patch_embed(x)  # 输出: (bz, ch_num, patch_num, d_model)
        
        # 重塑为序列格式以适应 HybridBlock
        # 从 (bz, ch_num, patch_num, d_model) -> (bz, ch_num * patch_num, d_model)
        x = x.view(bz, ch_num * patch_num, self.d_model)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        
        # 重塑回原始格式以适应分类器
        # 从 (bz, ch_num * patch_num, d_model) -> (bz, ch_num, patch_num, d_model)
        x = x.view(bz, ch_num, patch_num, self.d_model)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.classifier(x)
        return x  # 统一返回 (batch, num_classes)


