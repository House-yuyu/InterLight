"""
LFPV (Learnable Feature Patches and Vectors) Module

可插拔模块，用于增强低光照图像增强性能
基于原始代码仓库实现：https://github.com/xxx/LFPVS

改进版本：针对HVI空间低光任务优化
- 可学习融合系数
- 亮度自适应门控（可选）
- 分支差异化支持（I分支/HV分支）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class LFPVModule(nn.Module):
    """
    LFPV 模块：可学习的特征补丁和向量
    
    根据原始代码实现：
    1. 在编码器特征提取后使用（Query-and-fusion）
    2. 训练时通过 SU 和 MU 更新（直接替换，非残差）
    3. 推理时直接使用已学习的 LFPV
    
    HVI空间优化（可选启用）：
    - fusion_scale: 可学习的融合强度系数
    - intensity_adaptive: 亮度自适应门控
    - branch_type: 分支类型（'I' 或 'HV'），影响默认参数
    """
    
    def __init__(self, channel=144, num_feature=16, patch_size=4,
                 fusion_scale_init=1.0, intensity_adaptive=False, branch_type=None):
        """
        Args:
            channel: 特征通道数（需要与编码器输出匹配）
            num_feature: LFPV 的数量 (论文中的 l)
            patch_size: 补丁大小 (论文中的 k)
            fusion_scale_init: 融合系数初始值（默认1.0，即原始行为）
            intensity_adaptive: 是否启用亮度自适应门控
            branch_type: 分支类型 'I'/'HV'/None，影响默认参数和行为
        """
        super(LFPVModule, self).__init__()
        self.num_feature = num_feature
        self.channel = channel
        self.patch_size = patch_size
        self.branch_type = branch_type
        self.intensity_adaptive = intensity_adaptive
        
        # ============ 可学习的融合系数 ============
        # 初始化为指定值，网络可以自己学习调整
        self.fusion_scale = nn.Parameter(torch.ones(1) * fusion_scale_init)
        
        # ============ 亮度自适应门控（可选）============
        # 思想：暗区域需要更多LFPV增强，亮区域需要保护
        if intensity_adaptive:
            # 轻量级门控网络：从特征统计量预测自适应权重
            self.adaptive_gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),  # 全局平均池化 -> [B, C, 1, 1]
                nn.Conv2d(channel, channel // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 4, 1, 1),
                nn.Sigmoid()  # 输出 [0, 1] 的权重
            )
            # 自适应强度因子：控制门控的影响程度
            # 初始化为0，即默认不启用自适应（等价于原始行为）
            self.adaptive_strength = nn.Parameter(torch.zeros(1))
        
        # ============ 分支特定的偏置（仅I分支可选）============
        # I分支：可学习的亮度提升偏置
        if branch_type == 'I':
            # 初始化为0，不改变原始行为，让网络自己决定是否使用
            self.brightness_bias = nn.Parameter(torch.zeros(1, channel, 1, 1))
        
        # 初始化可学习的通用特征（Cv 和 Cp）
        # 原始代码使用零初始化
        self.common_feature = nn.Parameter(torch.zeros(self.num_feature, self.channel).float())
        self.common_feature.requires_grad = False  # 通过 detach 更新
        
        self.common_feature_patch = nn.Parameter(
            torch.zeros(self.num_feature, self.channel, self.patch_size, self.patch_size).float()
        )
        self.common_feature_patch.requires_grad = False
        
        # Identity Embedding (论文中的 e_j)
        self.embedding = nn.Embedding(self.num_feature, channel)
        
        # Sample-Updater (SU) 网络 - Sv 和 Sp
        # 输入: C_{v,j} ⊕ f_d(x_i, y_i) ⊕ e_j (3*channel)
        # 输出: 新的 C_{v,j}
        self.update_feature = nn.Sequential(
            nn.Conv2d(3*channel, channel*4, 1, 1, 0, bias=True), 
            nn.BatchNorm2d(channel*4), 
            nn.ReLU(), 
            nn.Conv2d(channel*4, channel*4, 1, 1, 0, bias=True), 
            nn.BatchNorm2d(channel*4), 
            nn.ReLU(),
            nn.Conv2d(channel*4, channel*4, 1, 1, 0, bias=True), 
            nn.BatchNorm2d(channel*4), 
            nn.ReLU(),
            nn.Conv2d(channel*4, channel*4, 1, 1, 0, bias=True), 
            nn.BatchNorm2d(channel*4), 
            nn.ReLU(),
            nn.Conv2d(channel*4, channel*4, 1, 1, 0, bias=True), 
            nn.BatchNorm2d(channel*4), 
            nn.ReLU(),
            nn.Conv2d(channel*4, channel, 1, 1, 0, bias=True), 
            nn.BatchNorm2d(channel), 
            nn.ReLU(),
            nn.Conv2d(channel, channel, 1, 1, 0, bias=True)
        )
        
        self.update_feature_patch = nn.Sequential(
            nn.Conv2d(3*channel, channel*4, 3, 1, 1, bias=True), 
            nn.BatchNorm2d(channel*4), 
            nn.ReLU(), 
            nn.Conv2d(channel*4, channel*4, 3, 1, 1, bias=True), 
            nn.BatchNorm2d(channel*4), 
            nn.ReLU(), 
            nn.Conv2d(channel*4, channel*4, 3, 1, 1, bias=True), 
            nn.BatchNorm2d(channel*4), 
            nn.ReLU(), 
            nn.Conv2d(channel*4, channel*4, 3, 1, 1, bias=True), 
            nn.BatchNorm2d(channel*4), 
            nn.ReLU(), 
            nn.Conv2d(channel*4, channel*4, 3, 1, 1, bias=True), 
            nn.BatchNorm2d(channel*4), 
            nn.ReLU(), 
            nn.Conv2d(channel*4, channel, 3, 1, 1, bias=True), 
            nn.BatchNorm2d(channel), 
            nn.ReLU(), 
            nn.Conv2d(channel, channel, 3, 1, 1, bias=True)
        )
        
        # Mutual-Updater (MU) 网络 - Mv 和 Mp
        # 输入: C_{v,j} ⊕ C_{v,h} ⊕ e_j ⊕ e_h (4*channel)
        # 输出: 新的 C_{v,j}
        self.propagate = nn.Sequential(
            nn.Conv2d(channel*4, channel*4, 1, 1, 0, bias=True), 
            nn.BatchNorm2d(channel*4), 
            nn.ReLU(), 
            nn.Conv2d(channel*4, channel*4, 1, 1, 0, bias=True), 
            nn.BatchNorm2d(channel*4), 
            nn.ReLU(), 
            nn.Conv2d(channel*4, channel*4, 1, 1, 0, bias=True), 
            nn.BatchNorm2d(channel*4), 
            nn.ReLU(), 
            nn.Conv2d(channel*4, channel*4, 1, 1, 0, bias=True), 
            nn.BatchNorm2d(channel*4), 
            nn.ReLU(), 
            nn.Conv2d(channel*4, channel*4, 1, 1, 0, bias=True), 
            nn.BatchNorm2d(channel*4), 
            nn.ReLU(), 
            nn.Conv2d(channel*4, channel, 1, 1, 0, bias=True), 
            nn.BatchNorm2d(channel), 
            nn.ReLU(), 
            nn.Conv2d(channel, channel, 1, 1, 0, bias=True)
        )
        
        self.propagate_patch = nn.Sequential(
            nn.Conv2d(channel*4, channel*4, 3, 1, 1, bias=True), 
            nn.BatchNorm2d(channel*4), 
            nn.ReLU(), 
            nn.Conv2d(channel*4, channel*4, 3, 1, 1, bias=True), 
            nn.BatchNorm2d(channel*4), 
            nn.ReLU(), 
            nn.Conv2d(channel*4, channel*4, 3, 1, 1, bias=True), 
            nn.BatchNorm2d(channel*4), 
            nn.ReLU(), 
            nn.Conv2d(channel*4, channel*4, 3, 1, 1, bias=True), 
            nn.BatchNorm2d(channel*4), 
            nn.ReLU(), 
            nn.Conv2d(channel*4, channel*4, 3, 1, 1, bias=True), 
            nn.BatchNorm2d(channel*4), 
            nn.ReLU(), 
            nn.Conv2d(channel*4, channel, 3, 1, 1, bias=True), 
            nn.BatchNorm2d(channel), 
            nn.ReLU(), 
            nn.Conv2d(channel, channel, 3, 1, 1, bias=True)
        )
        
        # Patch 操作
        self.feature_unshuffle = nn.PixelUnshuffle(self.patch_size)
        self.feature_shuffle = nn.PixelShuffle(self.patch_size)
        
        # LayerNorm 层（原始代码对 query 和 key 都做归一化）
        self.norm_query1 = nn.LayerNorm(channel)  # for feature vector query
        self.norm_query2 = nn.LayerNorm(channel)  # for common vector key
        self.norm_query3 = nn.LayerNorm(channel * self.patch_size * self.patch_size)  # for feature patch query
        self.norm_query4 = nn.LayerNorm(channel * self.patch_size * self.patch_size)  # for common patch key
        
        # SU/MU 输出归一化
        self.norm_feature = nn.LayerNorm(channel)
        self.norm_patch = nn.LayerNorm([channel, self.patch_size, self.patch_size])
        self.norm_feature2 = nn.LayerNorm(channel)
        self.norm_patch2 = nn.LayerNorm([channel, self.patch_size, self.patch_size])
        
        # 存储当前更新的特征（用于训练时）
        self._current_common_feature = None
        self._current_common_feature_patch = None
    
    def query_and_fusion(self, feature_map):
        """
        查询和融合机制（推理时使用，论文 Eqs. 6, 7, 8）
        
        Args:
            feature_map: 编码器输出的特征图 [B, C, H, W]
        
        Returns:
            增强后的特征图 [B, C, H, W]
        """
        return self.update_feature_with_common(
            feature_map, 
            self.common_feature, 
            self.common_feature_patch
        )
    
    def forward(self, x, guide_feat=None):
        """
        前向传播：根据训练/推理模式自动选择处理方式
        
        Args:
            x: 特征图 [B, C, H, W]
            guide_feat: 引导特征（可选），用于亮度引导的跨分支交互
        
        Returns:
            增强后的特征图 [B, C, H, W]
        """
        if self.training:
            # 训练时：更新 LFPV 并用其增强特征
            return self.update_and_enhance(x, guide_feat)
        else:
            # 推理时：直接使用已学习的 LFPV 增强特征
            return self.query_and_fusion_with_guide(x, guide_feat)
    
    def query_and_fusion_with_guide(self, feature_map, guide_feat=None):
        """
        带引导的查询和融合（推理时使用）
        """
        return self.update_feature_with_common(
            feature_map, 
            self.common_feature, 
            self.common_feature_patch,
            guide_feat
        )
    
    def update_feature_with_common(self, feature_map_origin, common_feature_this, 
                                   common_feature_patch_this, guide_feat=None):
        """
        使用 LFPV 增强特征图（Query-and-fusion，论文 Eqs. 6, 7, 8）
        
        改进：
        - 可学习融合系数
        - 亮度自适应门控（可选）
        - 分支特定偏置（I分支可选）
        
        Args:
            feature_map_origin: 原始特征图 [B, C, H, W]
            common_feature_this: 通用特征向量 [L, C]
            common_feature_patch_this: 通用特征补丁 [L, C, k, k]
            guide_feat: 引导特征（可选）[B, C, H, W]
        """
        batch_size = feature_map_origin.shape[0]
        channel = feature_map_origin.shape[1]
        assert channel == self.channel
        height = feature_map_origin.shape[2]
        width = feature_map_origin.shape[3]

        feature_map = feature_map_origin
        
        # ============ 1. 向量级查询 (论文 Eq. 6) ============
        # feature_map_vector: [B, C, H, W] -> [B, HW, C]
        feature_map_vector = feature_map.reshape(batch_size, channel, height*width).permute(0, 2, 1)
        feature_map_vector = self.norm_query1(feature_map_vector)  # LayerNorm on query
        
        # common_map_vector: [L, C] -> [B, L, C] -> LayerNorm -> [B, C, L]
        common_map_vector = common_feature_this.reshape(1, self.num_feature, channel).repeat(batch_size, 1, 1)
        common_map_vector = self.norm_query2(common_map_vector).permute(0, 2, 1)  # LayerNorm on key
        
        # attention: [B, HW, C] @ [B, C, L] = [B, HW, L]
        attention_map = torch.bmm(feature_map_vector, common_map_vector)
        attention_map = torch.nn.Softmax(dim=2)(attention_map)
        
        # update: [B, HW, L] @ [B, L, C] = [B, HW, C]
        value_vector = common_feature_this.reshape(1, self.num_feature, channel).repeat(batch_size, 1, 1)
        update_vector = torch.bmm(attention_map, value_vector)
        update_vector = update_vector.reshape(batch_size, height, width, channel).permute(0, 3, 1, 2)

        # ============ 2. 补丁级查询 (论文 Eq. 7) ============
        # 计算需要填充的高度和宽度，使其能被 patch_size 整除
        pad_h = (self.patch_size - height % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - width % self.patch_size) % self.patch_size
        
        if pad_h > 0 or pad_w > 0:
            feature_map_padded = F.pad(feature_map, (0, pad_w, 0, pad_h), mode='reflect')
            padded_height = height + pad_h
            padded_width = width + pad_w
        else:
            feature_map_padded = feature_map
            padded_height = height
            padded_width = width
        
        # PixelUnshuffle: [B, C, H, W] -> [B, C*k*k, H/k, W/k]
        feature_map_patch = self.feature_unshuffle(feature_map_padded)
        patch_h = padded_height // self.patch_size
        patch_w = padded_width // self.patch_size
        
        # Reshape: [B, C*k*k, h', w'] -> [B, h'*w', C*k*k]
        feature_map_patch = feature_map_patch.reshape(
            batch_size, channel * self.patch_size * self.patch_size, 
            patch_h * patch_w
        ).permute(0, 2, 1)
        feature_map_patch = self.norm_query3(feature_map_patch)  # LayerNorm on query
        
        # common_map_patch: [L, C, k, k] -> [B, L, C*k*k] -> LayerNorm -> [B, C*k*k, L]
        common_map_patch = common_feature_patch_this.reshape(
            1, self.num_feature, channel * self.patch_size * self.patch_size
        ).repeat(batch_size, 1, 1)
        common_map_patch = self.norm_query4(common_map_patch).permute(0, 2, 1)  # LayerNorm on key
        
        # attention: [B, h'*w', C*k*k] @ [B, C*k*k, L] = [B, h'*w', L]
        attention_map_patch = torch.bmm(feature_map_patch, common_map_patch)
        attention_map_patch = torch.nn.Softmax(dim=2)(attention_map_patch)
        
        # update: [B, h'*w', L] @ [B, L, C*k*k] = [B, h'*w', C*k*k]
        value_patch = common_feature_patch_this.reshape(
            1, self.num_feature, channel * self.patch_size * self.patch_size
        ).repeat(batch_size, 1, 1)
        update_map = torch.bmm(attention_map_patch, value_patch)
        
        # Reshape back: [B, h'*w', C*k*k] -> [B, C*k*k, h', w'] -> [B, C, H, W]
        update_map = update_map.reshape(
            batch_size, patch_h, patch_w, 
            channel * self.patch_size * self.patch_size
        ).permute(0, 3, 1, 2)
        update_map = self.feature_shuffle(update_map)
        
        # 裁剪回原始尺寸
        if pad_h > 0 or pad_w > 0:
            update_map = update_map[:, :, :height, :width]
        
        # ============ 3. 特征融合 (论文 Eq. 8) - 改进版 ============
        # 基础融合
        fusion_out = update_vector + update_map
        
        # 计算融合权重
        fusion_weight = self.fusion_scale
        
        # 【改进】亮度自适应门控
        if self.intensity_adaptive:
            # 使用引导特征（如果有）或自身特征计算自适应权重
            ref_feat = guide_feat if guide_feat is not None else feature_map_origin
            
            # adaptive_gate 输出 [0, 1]：特征越"暗"，输出越小
            gate_value = self.adaptive_gate(ref_feat)  # [B, 1, 1, 1]
            
            # 自适应调制：暗区域（gate小）获得更多增强
            # adaptive_factor = 1 + strength * (1 - gate)
            # 当 strength=0 时，adaptive_factor=1，即原始行为
            adaptive_factor = 1.0 + torch.sigmoid(self.adaptive_strength) * (1.0 - gate_value)
            fusion_weight = fusion_weight * adaptive_factor
        
        # 应用融合
        output = feature_map_origin + fusion_weight * fusion_out
        
        # 【改进】I分支的亮度偏置
        if self.branch_type == 'I' and hasattr(self, 'brightness_bias'):
            output = output + self.brightness_bias
        
        return output

    def update_common_feature(self, feature_map):
        """
        从训练样本中更新 LFPV (Sample-Updater, 论文 Eqs. 2, 3)
        
        原始代码实现：SU 直接输出新值，不是残差
        
        Args:
            feature_map: 特征图 [B, C, H, W]
        
        Returns:
            common_feature_this: 更新后的特征向量 [L, C]
            common_feature_patch_this: 更新后的特征补丁 [L, C, k, k]
        """
        height = feature_map.shape[2]
        width = feature_map.shape[3]
        batch_size = feature_map.shape[0]
        
        # 随机采样位置
        random_index_height = random.randint(0, height - 1)
        random_index_width = random.randint(0, width - 1)
        random_index_height_patch = random.randint(0, max(0, height - self.patch_size))
        random_index_width_patch = random.randint(0, max(0, width - self.patch_size))
        
        # 提取采样的特征向量: f_d(x_i, y_i) [B, C]
        choosen_feature_vector = feature_map[:, :, random_index_height, random_index_width]
        # 提取采样的特征补丁: f_{p_d}(x_i, y_i) [B, C, k, k]
        choosen_feature_map = feature_map[
            :, :, 
            random_index_height_patch:random_index_height_patch + self.patch_size, 
            random_index_width_patch:random_index_width_patch + self.patch_size
        ]
        
        # ============ SU 更新向量 (论文 Eq. 2) ============
        # C_{v,j}: [L, C] -> [L, B, C] -> [L*B, C, 1, 1]
        common_feature_vector = self.common_feature.reshape(self.num_feature, 1, self.channel).repeat(1, batch_size, 1)
        common_feature_vector = common_feature_vector.reshape(self.num_feature * batch_size, self.channel, 1, 1)
        
        # f_d: [B, C] -> [1, B, C] -> [L, B, C] -> [L*B, C, 1, 1]
        choosen_feature_vector = choosen_feature_vector.reshape(1, batch_size, self.channel).repeat(self.num_feature, 1, 1)
        choosen_feature_vector = choosen_feature_vector.reshape(self.num_feature * batch_size, self.channel, 1, 1)

        # e_j: [L, C] -> [L, B, C] -> [L*B, C, 1, 1]
        index_array = torch.LongTensor([x for x in range(self.num_feature)]).to(feature_map.device)
        position_embedding = self.embedding(index_array).reshape(self.num_feature, 1, self.channel).repeat(1, batch_size, 1)
        position_embedding = position_embedding.reshape(self.num_feature * batch_size, self.channel, 1, 1)

        # SU: concat(C_{v,j}, f_d, e_j) -> new C_{v,j}
        update_vector = self.update_feature(
            torch.cat([common_feature_vector, choosen_feature_vector, position_embedding], dim=1)
        )
        update_vector = update_vector.reshape(self.num_feature, batch_size, self.channel)
        update_vector = torch.mean(update_vector, dim=1)  # 在 batch 维度上平均
        common_feature_this = self.norm_feature(update_vector)

        # ============ SU 更新补丁 (论文 Eq. 3) ============
        # C_{p,j}: [L, C, k, k] -> [L*B, C, k, k]
        common_feature_map = self.common_feature_patch.reshape(
            self.num_feature, 1, self.channel, self.patch_size, self.patch_size
        ).repeat(1, batch_size, 1, 1, 1)
        common_feature_map = common_feature_map.reshape(
            self.num_feature * batch_size, self.channel, self.patch_size, self.patch_size
        )
        
        # f_{p_d}: [B, C, k, k] -> [L*B, C, k, k]
        choosen_feature_map = choosen_feature_map.reshape(
            1, batch_size, self.channel, self.patch_size, self.patch_size
        ).repeat(self.num_feature, 1, 1, 1, 1)
        choosen_feature_map = choosen_feature_map.reshape(
            self.num_feature * batch_size, self.channel, self.patch_size, self.patch_size
        )

        # e_j 扩展到 patch 尺寸
        position_embedding2 = position_embedding.repeat(1, 1, self.patch_size, self.patch_size)

        # SU: concat(C_{p,j}, f_{p_d}, e_j) -> new C_{p,j}
        update_map = self.update_feature_patch(
            torch.cat([common_feature_map, choosen_feature_map, position_embedding2], dim=1)
        )
        update_map = update_map.reshape(
            self.num_feature, batch_size, self.channel, self.patch_size, self.patch_size
        )
        update_map = torch.mean(update_map, dim=1)
        common_feature_patch_this = self.norm_patch(update_map)
        
        return common_feature_this, common_feature_patch_this

    def propagate_feature(self, common_feature_this, common_feature_patch_this):
        """
        在 LFPV 之间传播信息 (Mutual-Updater, 论文 Eq. 4)
        
        原始代码实现：MU 直接输出新值，不是残差
        
        Args:
            common_feature_this: 当前向量特征 [L, C]
            common_feature_patch_this: 当前补丁特征 [L, C, k, k]
        
        Returns:
            common_feature_this: 传播后的向量特征 [L, C]
            common_feature_patch_this: 传播后的补丁特征 [L, C, k, k]
        """
        # 随机选择另一个节点 h
        choosen_list = [random.randint(0, self.num_feature - 1) for _ in range(self.num_feature)]
        choosen_list2 = [random.randint(0, self.num_feature - 1) for _ in range(self.num_feature)]
        
        # C_{v,h}: 从 common_feature_this 中选择
        choosen_feature_vector = common_feature_this[choosen_list, :].reshape(self.num_feature, self.channel, 1, 1)
        # C_{p,h}: 从 common_feature_patch_this 中选择
        choosen_feature_map = common_feature_patch_this[choosen_list2, :, :, :]
        
        # Identity embeddings
        index_array = torch.LongTensor([x for x in range(self.num_feature)]).to(common_feature_this.device)
        # e_j
        position_embedding = self.embedding(index_array).reshape(self.num_feature, self.channel, 1, 1)
        # e_h
        index_array2 = torch.LongTensor(choosen_list).to(common_feature_this.device)
        position_embeddingxx = self.embedding(index_array2).reshape(self.num_feature, self.channel, 1, 1)

        # ============ MU 更新向量 (论文 Eq. 4) ============
        # 输入: C_{v,j} ⊕ C_{v,h} ⊕ e_j ⊕ e_h
        common_feature_vector = common_feature_this.reshape(self.num_feature, self.channel, 1, 1)
        update_feature_vector = self.propagate(
            torch.cat([common_feature_vector, choosen_feature_vector, position_embedding, position_embeddingxx], dim=1)
        )
        update_feature_vector = update_feature_vector.reshape(self.num_feature, self.channel)
        common_feature_this = self.norm_feature2(update_feature_vector)

        # ============ MU 更新补丁 (论文 Eq. 4) ============
        common_feature_map = common_feature_patch_this
        position_embedding2 = position_embedding.repeat(1, 1, self.patch_size, self.patch_size)
        position_embeddingxx2 = position_embeddingxx.repeat(1, 1, self.patch_size, self.patch_size)
        
        update_feature_map = self.propagate_patch(
            torch.cat([common_feature_map, choosen_feature_map, position_embedding2, position_embeddingxx2], dim=1)
        )
        common_feature_patch_this = self.norm_patch2(update_feature_map)
        
        return common_feature_this, common_feature_patch_this
    
    def update_and_enhance(self, feature_map, guide_feat=None):
        """
        训练时：更新 LFPV 并用其增强特征
        
        流程（与原始代码一致）：
        1. 通过 SU 更新 LFPV
        2. 通过 MU 传播更新
        3. 使用更新后的 LFPV 增强特征
        
        Args:
            feature_map: 特征图 [B, C, H, W]
            guide_feat: 引导特征（可选）[B, C, H, W]
        
        Returns:
            增强后的特征图 [B, C, H, W]
        """
        # Step 1: 通过 SU 更新 LFPV
        common_feature_this, common_feature_patch_this = self.update_common_feature(feature_map)
        
        # Step 2: 通过 MU 传播更新
        common_feature_this, common_feature_patch_this = self.propagate_feature(
            common_feature_this, common_feature_patch_this
        )
        
        # 存储更新后的特征（用于后续 update_storage）
        self._current_common_feature = common_feature_this
        self._current_common_feature_patch = common_feature_patch_this
        
        # Step 3: 使用更新后的 LFPV 增强特征
        return self.update_feature_with_common(feature_map, common_feature_this, 
                                               common_feature_patch_this, guide_feat)
    
    def update_storage(self):
        """
        将更新后的 LFPV 存储到参数中（训练时调用）
        
        与原始代码一致：使用 clone().detach() 更新
        """
        if self._current_common_feature is not None:
            self.common_feature.data = self._current_common_feature.clone().detach()
        if self._current_common_feature_patch is not None:
            self.common_feature_patch.data = self._current_common_feature_patch.clone().detach()


# ============ 便捷工厂函数 ============

def create_lfpv_for_i_branch(channel=144, num_feature=16, patch_size=4, 
                              intensity_adaptive=True):
    """
    创建适用于I分支的LFPV
    
    特点：
    - 稍强的融合系数（1.2）：I分支需要更多亮度增强
    - 启用亮度自适应：暗区域获得更多增强
    - 启用亮度偏置：提供直接的亮度提升通道
    """
    return LFPVModule(
        channel=channel,
        num_feature=num_feature,
        patch_size=patch_size,
        fusion_scale_init=1.2,  # 稍强的融合
        intensity_adaptive=intensity_adaptive,
        branch_type='I'
    )


def create_lfpv_for_hv_branch(channel=144, num_feature=16, patch_size=4,
                               intensity_adaptive=True):
    """
    创建适用于HV分支的LFPV
    
    特点：
    - 稍弱的融合系数（0.8）：HV分支需要保护色度信息
    - 启用亮度自适应：但可以选择使用I分支特征作为引导
    - 不使用亮度偏置
    """
    return LFPVModule(
        channel=channel,
        num_feature=num_feature,
        patch_size=patch_size,
        fusion_scale_init=0.8,  # 稍弱的融合，保护色度
        intensity_adaptive=intensity_adaptive,
        branch_type='HV'
    )


if __name__ == '__main__':
    # 测试代码
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("LFPV Module (HVI空间优化版) 测试")
    print("=" * 60)
    
    # 模拟特征
    batch_size = 2
    channel = 144
    height, width = 16, 16
    
    x = torch.randn(batch_size, channel, height, width).to(device)
    guide = torch.randn(batch_size, channel, height, width).to(device)
    
    # 测试原始版本（向后兼容）
    print("\n1. 原始版本（向后兼容）:")
    lfpv_original = LFPVModule(channel=channel).to(device)
    lfpv_original.train()
    out1 = lfpv_original(x)
    print(f"   输入: {x.shape} -> 输出: {out1.shape}")
    print(f"   fusion_scale: {lfpv_original.fusion_scale.item():.4f}")
    
    # 测试I分支版本
    print("\n2. I分支版本:")
    lfpv_i = create_lfpv_for_i_branch(channel=channel).to(device)
    lfpv_i.train()
    out2 = lfpv_i(x, guide)
    print(f"   输入: {x.shape} -> 输出: {out2.shape}")
    print(f"   fusion_scale: {lfpv_i.fusion_scale.item():.4f}")
    print(f"   intensity_adaptive: {lfpv_i.intensity_adaptive}")
    print(f"   brightness_bias 范数: {lfpv_i.brightness_bias.norm().item():.6f}")
    
    # 测试HV分支版本
    print("\n3. HV分支版本:")
    lfpv_hv = create_lfpv_for_hv_branch(channel=channel).to(device)
    lfpv_hv.train()
    out3 = lfpv_hv(x, guide)
    print(f"   输入: {x.shape} -> 输出: {out3.shape}")
    print(f"   fusion_scale: {lfpv_hv.fusion_scale.item():.4f}")
    print(f"   intensity_adaptive: {lfpv_hv.intensity_adaptive}")
    
    # 参数量统计
    print("\n4. 参数量统计:")
    params_original = sum(p.numel() for p in lfpv_original.parameters())
    params_i = sum(p.numel() for p in lfpv_i.parameters())
    params_hv = sum(p.numel() for p in lfpv_hv.parameters())
    print(f"   原始版本: {params_original / 1e6:.4f}M")
    print(f"   I分支版本: {params_i / 1e6:.4f}M (+{(params_i - params_original) / 1e3:.2f}K)")
    print(f"   HV分支版本: {params_hv / 1e6:.4f}M (+{(params_hv - params_original) / 1e3:.2f}K)")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
