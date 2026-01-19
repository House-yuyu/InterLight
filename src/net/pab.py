"""
PAB (Prompt-Aware Block) - 轻量组合版
结合通道调制 + 门控融合，用于 HV 分支的先验引导
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from net.transformer_utils import LayerNorm


class PAB_CAB(nn.Module):
    """
    Prompt-Aware Cross Attention Block
    门控融合: Q = gate * q1(prompt) + (1-gate) * q2(x_mod)
    """
    def __init__(self, dim, num_heads, bias=False):
        super(PAB_CAB, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # Q 投影 (轻量化: 使用共享的 dwconv)
        self.q1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        
        # 轻量门控: 深度可分离卷积
        self.gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, kernel_size=3, padding=1, groups=dim * 2, bias=bias),
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias),
            nn.Sigmoid()
        )
        
        # KV 投影
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, prompt_feat, x_mod, y):
        """
        Args:
            prompt_feat: prompt 空间特征 [B, C, H, W]
            x_mod: 通道调制后的特征 [B, C, H, W]
            y: I 分支特征 [B, C, H, W]
        """
        b, c, h, w = x_mod.shape

        q1_out = self.q1(prompt_feat)
        q2_out = self.q2(x_mod)
        gate = self.gate(torch.cat([q1_out, q2_out], dim=1))
        q = self.q_dwconv(gate * q1_out + (1 - gate) * q2_out)
        
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = F.softmax(attn, dim=-1)

        out = attn @ v
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        return self.project_out(out)


class PAB_FFN(nn.Module):
    """FFN with Tanh gating"""
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super(PAB_FFN, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.dwconv1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
        self.Tanh = nn.Tanh()

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x1 = self.Tanh(self.dwconv1(x1)) + x1
        x2 = self.Tanh(self.dwconv2(x2)) + x2
        x = x1 * x2
        x = self.project_out(x)
        return x


class PAB(nn.Module):
    """
    Prompt-Aware Block (轻量组合版，支持多 level)
    
    设计:
    1. 通道调制: x' = x * (1+scale) + shift (全局先验)
    2. 门控融合: Q = gate * prompt_spatial + (1-gate) * x' (空间自适应)
    
    多 Level 策略:
    - level=0 (浅层，高分辨率): spatial_size=16，细粒度空间先验
    - level=1 (中层): spatial_size=8，适中空间先验
    - level=2 (深层，低分辨率): spatial_size=4，粗粒度空间先验（更关注语义）
    
    Args:
        dim: 特征通道数
        num_heads: 注意力头数
        prompt_dim: prompt 向量维度
        level: 网络层级 (0=浅层, 1=中层, 2=深层)
        spatial_size: 手动指定空间尺寸（可选，覆盖 level 默认值）
    """
    # 不同 level 对应的默认空间尺寸
    LEVEL_SPATIAL_MAP = {
        0: 16,   
        1: 8,   
        2: 4, 
    }
    
    def __init__(self, dim, num_heads, prompt_dim=512, level=0, spatial_size=None, bias=False):
        super(PAB, self).__init__()
        self.dim = dim
        self.level = level
        
        # 确定空间尺寸
        if spatial_size is not None:
            self.spatial_size = spatial_size
        else:
            self.spatial_size = self.LEVEL_SPATIAL_MAP.get(level, 6)
        
        self.norm = LayerNorm(dim)
        self.ffn = PAB_FFN(dim)
        self.cab = PAB_CAB(dim, num_heads, bias)
        
        # 通道调制
        self.prompt_scale = nn.Linear(prompt_dim, dim)
        self.prompt_shift = nn.Linear(prompt_dim, dim)
        
        # 空间投影: 根据 level 自适应尺寸
        s = self.spatial_size
        self.prompt_spatial = nn.Sequential(
            nn.Linear(prompt_dim, dim * s * s),
            nn.Unflatten(1, (dim, s, s)),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=bias),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=bias),
        )

    def forward(self, x, y, prompt):
        """
        Args:
            x: HV 特征 [B, C, H, W]
            y: I 特征 [B, C, H, W]
            prompt: degraded representations [B, prompt_dim]
        """
        H, W = x.shape[2], x.shape[3]
        
        # 通道调制
        scale = torch.sigmoid(self.prompt_scale(prompt)).unsqueeze(-1).unsqueeze(-1)
        shift = self.prompt_shift(prompt).unsqueeze(-1).unsqueeze(-1)
        x_mod = x * (1 + scale) + shift
        
        # 空间投影 + 上采样
        prompt_feat = self.prompt_spatial(prompt)
        prompt_feat = F.interpolate(prompt_feat, size=(H, W), mode='bilinear', align_corners=False)
        
        # Cross-Attention
        x = x + self.cab(self.norm(prompt_feat), self.norm(x_mod), self.norm(y))
        
        # FFN
        x = x + self.ffn(self.norm(x))
        
        return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("PAB (多 Level 版本) 测试")
    print("=" * 60)
    
    prompt = torch.rand(2, 512).to(device)
    
    # 模拟 U-Net 不同 level 的特征尺寸
    level_configs = [
        (0, 64, 64, "浅层 (高分辨率)"),  # level 0
        (1, 32, 32, "中层"),              # level 1
        (2, 16, 16, "深层 (低分辨率)"),   # level 2
    ]
    
    total_params = 0
    print(f"\nPrompt: {prompt.shape}")
    print("-" * 60)
    
    for level, h, w, desc in level_configs:
        x = torch.rand(2, 72, h, w).to(device)
        y = torch.rand(2, 72, h, w).to(device)
        
        pab = PAB(dim=72, num_heads=4, prompt_dim=512, level=level).to(device)
        out = pab(x, y, prompt)
        params = sum(p.numel() for p in pab.parameters())
        total_params += params
        
        print(f"Level {level} ({desc}):")
        print(f"  输入: {x.shape} → 输出: {out.shape}")
        print(f"  空间尺寸: {pab.spatial_size}x{pab.spatial_size}")
        print(f"  参数量: {params / 1e6:.4f}M")
        print()
    
    print("-" * 60)
    print(f"3个 Level 总参数量: {total_params / 1e6:.4f}M")
    print("=" * 60)
