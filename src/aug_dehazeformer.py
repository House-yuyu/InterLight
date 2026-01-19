import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms as T

class AugNoneOpt(nn.Module):
    """
    内部增强模块，用于自监督学习
    对模型输出进行弱增强和强增强，通过MSR损失约束不同增强下输出的一致性
    """
    def __init__(self):
        super(AugNoneOpt, self).__init__()
        self.weak_aug = nn.Sequential(T.CenterCrop(16))
        self.aggr_aug = nn.Sequential(T.GaussianBlur(kernel_size=(9,21),sigma=(0.1,5)))
        

    def forward(self, source_img):
        augweak_sourge_img = self.weak_aug(source_img)
        augaggr_sourge_img = self.aggr_aug(augweak_sourge_img)

        return augweak_sourge_img, augaggr_sourge_img


class AugExternal(nn.Module):
    """
    外增强模块 (External Augmentation) - 针对低光增强任务深度优化
    
    设计原则：
    1. 低光增强任务对亮度变化极其敏感，增强应该非常温和
    2. 模拟不同相机/传感器的响应曲线差异（轻微的非线性变换）
    3. 模拟轻微的色偏（不同通道的微小差异）
    4. 保护极暗区域，避免对纯噪声背景施加不合理的变换
    5. 在 HVI 变换之前应用，确保模型学习处理"由传感器差异导致的 HVI 分量变化"
    
    参考：
    - ScaleUpDehazing 的 ct.py (Gamma Correction for cross-domain alignment)
    - HVI-CIDNet 的物理模型（基于 I_max 的 HVI 空间）
    - 相机传感器的散粒噪声特性（极低光照下 Gamma 曲线失效）
    """
    def __init__(self, gamma_range=(0.95, 1.05), prob=0.3, mode='symmetric', 
                 dark_threshold=0.05, protect_dark=True):
        super(AugExternal, self).__init__()
        self.gamma_range = gamma_range
        self.prob = prob
        self.mode = mode
        self.dark_threshold = dark_threshold  # 低于此值的像素视为"纯噪声背景"
        self.protect_dark = protect_dark      # 是否保护极暗像素

    def forward(self, x):
        # x: [B, C, H, W], expected range [0, 1]
        if not self.training:
            return x
        
        # 按概率决定是否应用增强
        if random.random() > self.prob:
            return x
            
        B, C, H, W = x.shape
        low, high = self.gamma_range
        
        # 生成随机Gamma值: [B, C, 1, 1]
        # 每个样本的每个通道使用不同的 Gamma，模拟传感器响应差异和色偏
        gamma = torch.rand(B, C, 1, 1, device=x.device) * (high - low) + low  # 空间部分改变
        
        if self.mode == 'symmetric':
            # 对称模式：直接应用幂次变换 out = in^gamma
            # gamma > 1: 压暗 (x^1.05 < x for x∈(0,1))
            # gamma < 1: 提亮 (x^0.95 > x for x∈(0,1))
            out = torch.pow(x.clamp(min=1e-8), gamma)
            
        else:  # 'dehazing_style'
            # 去雾风格：应用倒数 out = in^(1/gamma)
            out = torch.pow(x.clamp(min=1e-8), 1.0 / gamma)
        
        # 保护极暗区域（建议1的实现）
        # 物理依据：传感器在极低光照下主要受散粒噪声主导，而非服从 Gamma 曲线
        # 对这些区域施加 Gamma 变换可能产生非物理的伪影
        if self.protect_dark:
            # 创建一个平滑的衰减 mask：亮度越低，增强效果越弱
            # 使用 I_max (每像素的最大通道值) 作为亮度指标，与 HVI-CIDNet 的定义一致
            intensity = x.max(dim=1, keepdim=True)[0]  # [B, 1, H, W]
            
            # 计算衰减因子：[0, dark_threshold] -> [0, 1] 的平滑过渡
            # 使用 smoothstep 函数确保平滑性
            fade_factor = torch.clamp((intensity - 0) / (self.dark_threshold - 0), 0, 1)
            fade_factor = fade_factor * fade_factor * (3 - 2 * fade_factor)  # smoothstep
            
            # 对极暗区域，线性插值回原始值
            # fade_factor = 0 (极暗) -> 完全保留原值
            # fade_factor = 1 (正常) -> 完全应用 Gamma 变换
            out = fade_factor * out + (1 - fade_factor) * x
        
        return out.clamp(0.0, 1.0)

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       