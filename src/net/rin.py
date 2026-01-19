"""
RIN (Restoration Information Network) - 图像自适应 Prompt 生成器
采用字典学习的思想，根据输入图像的退化特征自适应组合基向量生成恢复引导信息
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RIN(nn.Module):
    """
    Restoration Information Network
    
    Args:
        in_dim: 输入通道数
        atom_num: 字典原子数量 (基向量个数)
        atom_dim: 字典原子维度 (prompt 向量维度)
    
    输出:
        prompt: [B, atom_dim] 的向量，用于指导后续恢复过程
    """
    def __init__(self, in_dim=3, atom_num=32, atom_dim=512):
        super(RIN, self).__init__()
        
        # Condition network - 提取图像退化特征
        hidden_dim = 64
        self.CondNet = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=3, stride=3),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=3),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(hidden_dim, 32, kernel_size=1)
        )
        
        # 字典系数生成
        self.lastOut = nn.Linear(32, atom_num)
        self.act = nn.GELU()
        
        # 可学习字典 [atom_num, atom_dim]
        self.dictionary = nn.Parameter(torch.randn(atom_num, atom_dim), requires_grad=True)
        
    def forward(self, x):
        """
        Args:
            x: 输入图像 [B, C, H, W]
        Returns:
            prompt: [B, atom_dim] prompt 向量
        """
        # 提取全局退化特征
        out = self.CondNet(x)  # [B, 32, H', W']
        out = nn.AdaptiveAvgPool2d(1)(out)  # [B, 32, 1, 1]
        out = out.view(out.size(0), -1)  # [B, 32]
        
        # 生成字典系数
        out = self.lastOut(out)  # [B, atom_num]
        logits = F.softmax(out, dim=-1)  # 归一化权重
        
        # 字典加权组合
        out = logits @ self.dictionary  # [B, atom_dim]
        out = self.act(out)
        
        return out


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.rand(2, 3, 256, 256).to(device)
    
    rin = RIN(in_dim=3, atom_num=32, atom_dim=512).to(device)
    prompt = rin(x)
    print(f"Input shape: {x.shape}")
    print(f"Prompt shape: {prompt.shape}")  # [2, 256]

