import torch
import torch.nn as nn
from net.HVI_transform import RGB_HVI
from net.transformer_utils import *
from net.LCA import *
from net.lfpv_module import LFPVModule, create_lfpv_for_i_branch, create_lfpv_for_hv_branch
from net.rin import RIN
from net.pab import PAB
from huggingface_hub import PyTorchModelHubMixin

class CIDNet(nn.Module, PyTorchModelHubMixin):
    def __init__(self, 
                 channels=[36, 36, 72, 144],
                 heads=[1, 2, 4, 8],
                 norm=False,
                 lfpv_num_feature=16,
                 lfpv_patch_size=4,
                 atom_num=32,
                 atom_dim=512,
                 # HVI空间优化选项
                 lfpv_intensity_adaptive=False,  # 是否启用亮度自适应
                 lfpv_intensity_guide=False,     # 是否启用亮度引导（I引导HV）
                 lfpv_branch_aware=False,         # 是否启用分支感知（不同分支不同参数）
                 use_lfpv=True,                   # 是否启用LFPV模块
                 use_rin_pab=True                 # 是否启用RIN-PAB模块（False时使用原始LCA）
        ):
        super(CIDNet, self).__init__()
        
        [ch1, ch2, ch3, ch4] = channels
        [head1, head2, head3, head4] = heads
        
        # RIN: 生成 prompt 向量（仅在启用RIN-PAB时使用）
        self.use_rin_pab = use_rin_pab
        if use_rin_pab:
            self.rin = RIN(in_dim=3, atom_num=atom_num, atom_dim=atom_dim)
            self.atom_dim = atom_dim
        else:
            self.rin = None
            self.atom_dim = None
        
        # HV_ways
        self.HVE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(3, ch1, 3, stride=1, padding=0, bias=False)
            )
        self.HVE_block1 = NormDownsample(ch1, ch2, use_norm=norm)
        self.HVE_block2 = NormDownsample(ch2, ch3, use_norm=norm)
        self.HVE_block3 = NormDownsample(ch3, ch4, use_norm=norm)
        
        self.HVD_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.HVD_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.HVD_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        self.HVD_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 2, 3, stride=1, padding=0, bias=False)
        )
        
        # I_ways
        self.IE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(1, ch1, 3, stride=1, padding=0, bias=False),
            )
        self.IE_block1 = NormDownsample(ch1, ch2, use_norm=norm)
        self.IE_block2 = NormDownsample(ch2, ch3, use_norm=norm)
        self.IE_block3 = NormDownsample(ch3, ch4, use_norm=norm)
        
        self.ID_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.ID_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.ID_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        self.ID_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 1, 3, stride=1, padding=0, bias=False),
            )
        
        # HV Branch: Encoder 根据use_rin_pab选择使用PAB或LCA
        if use_rin_pab:
            # 使用 PAB (Prompt-Aware, 多 Level)
            # level=0: 浅层(16x16), level=1: 中层(8x8), level=2: 深层(4x4)
            self.HV_PAB1 = PAB(ch2, head2, prompt_dim=atom_dim, level=0)
            self.HV_PAB2 = PAB(ch3, head3, prompt_dim=atom_dim, level=1)
            self.HV_PAB3 = PAB(ch4, head4, prompt_dim=atom_dim, level=2)
            # 不使用LCA作为替代
            self.HV_LCA1 = None
            self.HV_LCA2 = None
            self.HV_LCA3 = None
        else:
            # 使用原始的 LCA 替代 PAB
            self.HV_LCA1 = HV_LCA(ch2, head2)
            self.HV_LCA2 = HV_LCA(ch3, head3)
            self.HV_LCA3 = HV_LCA(ch4, head4)
            # 不使用PAB
            self.HV_PAB1 = None
            self.HV_PAB2 = None
            self.HV_PAB3 = None
        
        # HV Branch: Decoder 保持原有的 HV_LCA
        self.HV_LCA4 = HV_LCA(ch4, head4)
        self.HV_LCA5 = HV_LCA(ch3, head3)
        self.HV_LCA6 = HV_LCA(ch2, head2)
        
        # I Branch: 保持原有的 I_LCA
        self.I_LCA1 = I_LCA(ch2, head2)
        self.I_LCA2 = I_LCA(ch3, head3)
        self.I_LCA3 = I_LCA(ch4, head4)
        self.I_LCA4 = I_LCA(ch4, head4)
        self.I_LCA5 = I_LCA(ch3, head3)
        self.I_LCA6 = I_LCA(ch2, head2)
        
        self.trans = RGB_HVI()
        
        # LFPV 模块配置
        self.lfpv_num_feature = lfpv_num_feature
        self.lfpv_patch_size = lfpv_patch_size
        self.lfpv_intensity_guide = lfpv_intensity_guide
        self.use_lfpv = use_lfpv
        
        # 创建 LFPV 模块
        if use_lfpv:
            if lfpv_branch_aware:
                # 分支感知模式：I分支和HV分支使用不同的参数
                self.lfpv_i = create_lfpv_for_i_branch(
                    channel=ch4, 
                    num_feature=lfpv_num_feature, 
                    patch_size=lfpv_patch_size,
                    intensity_adaptive=lfpv_intensity_adaptive
                )
                self.lfpv_hv = create_lfpv_for_hv_branch(
                    channel=ch4, 
                    num_feature=lfpv_num_feature, 
                    patch_size=lfpv_patch_size,
                    intensity_adaptive=lfpv_intensity_adaptive
                )
            else:
                # 原始模式：两个分支使用相同的默认参数（向后兼容）
                self.lfpv_i = LFPVModule(
                    channel=ch4, 
                    num_feature=lfpv_num_feature, 
                    patch_size=lfpv_patch_size,
                    fusion_scale_init=1.0,
                    intensity_adaptive=lfpv_intensity_adaptive,
                    branch_type='I' if lfpv_intensity_adaptive else None
                )
                self.lfpv_hv = LFPVModule(
                    channel=ch4, 
                    num_feature=lfpv_num_feature, 
                    patch_size=lfpv_patch_size,
                    fusion_scale_init=1.0,
                    intensity_adaptive=lfpv_intensity_adaptive,
                    branch_type='HV' if lfpv_intensity_adaptive else None
                )
        else:
            # 禁用LFPV时，创建恒等映射模块（直接返回输入）
            class IdentityModule(nn.Module):
                def __init__(self):
                    super(IdentityModule, self).__init__()
                def forward(self, x, guide_feat=None):
                    return x
                def update_storage(self):
                    pass
            
            self.lfpv_i = IdentityModule()
            self.lfpv_hv = IdentityModule()
    
    def decode_features(self, i_enc4, hv_4, i_enc3, hv_3, i_enc2, hv_2, i_enc1, hv_1, 
                       i_enc0, hv_0, i_jump0, hv_jump0, v_jump1, hv_jump1, v_jump2, hv_jump2, hvi):
        """
        解码特征到最终输出
        """
        i_dec4 = self.I_LCA4(i_enc4, hv_4)
        hv_4 = self.HV_LCA4(hv_4, i_enc4)
        
        hv_3 = self.HVD_block3(hv_4, hv_jump2)
        i_dec3 = self.ID_block3(i_dec4, v_jump2)
        i_dec2 = self.I_LCA5(i_dec3, hv_3)
        hv_2 = self.HV_LCA5(hv_3, i_dec3)
        
        hv_2 = self.HVD_block2(hv_2, hv_jump1)
        i_dec2 = self.ID_block2(i_dec3, v_jump1)
        
        i_dec1 = self.I_LCA6(i_dec2, hv_2)
        hv_1 = self.HV_LCA6(hv_2, i_dec2)
        
        i_dec1 = self.ID_block1(i_dec1, i_jump0)
        i_dec0 = self.ID_block0(i_dec1)
        hv_1 = self.HVD_block1(hv_1, hv_jump0)
        hv_0 = self.HVD_block0(hv_1)
        
        output_hvi = torch.cat([hv_0, i_dec0], dim=1) + hvi
        output_rgb = self.trans.PHVIT(output_hvi)
        
        return output_rgb
        
    def forward(self, x):
        dtypes = x.dtype
        
        # 1. 生成 prompt 向量（仅在启用RIN-PAB时）
        if self.use_rin_pab:
            prompt = self.rin(x)  # [B, atom_dim]
        else:
            prompt = None
        
        # 2. RGB -> HVI 变换
        hvi = self.trans.HVIT(x)
        i = hvi[:, 2, :, :].unsqueeze(1).to(dtypes)
        
        # ============ Encoder ============
        # Level 0
        i_enc0 = self.IE_block0(i)
        i_enc1 = self.IE_block1(i_enc0)
        hv_0 = self.HVE_block0(hvi)
        hv_1 = self.HVE_block1(hv_0)
        i_jump0 = i_enc0
        hv_jump0 = hv_0
        
        # Level 1: I 用 LCA, HV 用 PAB 或 LCA
        i_enc2 = self.I_LCA1(i_enc1, hv_1)
        if self.use_rin_pab:
            hv_2 = self.HV_PAB1(hv_1, i_enc1, prompt)
        else:
            hv_2 = self.HV_LCA1(hv_1, i_enc1)
        v_jump1 = i_enc2
        hv_jump1 = hv_2
        i_enc2 = self.IE_block2(i_enc2)
        hv_2 = self.HVE_block2(hv_2)
        
        # Level 2: I 用 LCA, HV 用 PAB 或 LCA
        i_enc3 = self.I_LCA2(i_enc2, hv_2)
        if self.use_rin_pab:
            hv_3 = self.HV_PAB2(hv_2, i_enc2, prompt)
        else:
            hv_3 = self.HV_LCA2(hv_2, i_enc2)
        v_jump2 = i_enc3
        hv_jump2 = hv_3
        i_enc3 = self.IE_block3(i_enc2)
        hv_3 = self.HVE_block3(hv_2)
        
        # Level 3: I 用 LCA, HV 用 PAB 或 LCA (Bottleneck 前)
        i_enc4 = self.I_LCA3(i_enc3, hv_3)
        if self.use_rin_pab:
            hv_4 = self.HV_PAB3(hv_3, i_enc3, prompt)
        else:
            hv_4 = self.HV_LCA3(hv_3, i_enc3)
        
        # LFPV 增强
        if not self.use_lfpv:
            # LFPV被禁用时：直接解码，返回单路径输出
            output_rgb = self.decode_features(
                i_enc4, hv_4, i_enc3, hv_3, i_enc2, hv_2, i_enc1, hv_1,
                i_enc0, hv_0, i_jump0, hv_jump0, v_jump1, hv_jump1, v_jump2, hv_jump2, hvi
            )
            return output_rgb
        elif self.training:
            # 训练时：双路径策略
            output_without_lfpv = self.decode_features(
                i_enc4, hv_4, i_enc3, hv_3, i_enc2, hv_2, i_enc1, hv_1,
                i_enc0, hv_0, i_jump0, hv_jump0, v_jump1, hv_jump1, v_jump2, hv_jump2, hvi
            )
            
            # LFPV 增强（支持亮度引导）
            # I分支：自身特征作为引导
            i_enc4_enhanced = self.lfpv_i(i_enc4, guide_feat=None)
            
            # HV分支：使用I分支特征作为引导（如果启用）
            if self.lfpv_intensity_guide:
                hv_4_enhanced = self.lfpv_hv(hv_4, guide_feat=i_enc4)
            else:
                hv_4_enhanced = self.lfpv_hv(hv_4, guide_feat=None)
            
            self.lfpv_i.update_storage()
            self.lfpv_hv.update_storage()
            
            output_with_lfpv = self.decode_features(
                i_enc4_enhanced, hv_4_enhanced, i_enc3, hv_3, i_enc2, hv_2, i_enc1, hv_1,
                i_enc0, hv_0, i_jump0, hv_jump0, v_jump1, hv_jump1, v_jump2, hv_jump2, hvi
            )
            
            return output_with_lfpv, output_without_lfpv
        else:
            # 推理时：LFPV 增强（支持亮度引导）
            i_enc4 = self.lfpv_i(i_enc4, guide_feat=None)
            
            if self.lfpv_intensity_guide:
                hv_4 = self.lfpv_hv(hv_4, guide_feat=i_enc4)
            else:
                hv_4 = self.lfpv_hv(hv_4, guide_feat=None)
            
            output_rgb = self.decode_features(
                i_enc4, hv_4, i_enc3, hv_3, i_enc2, hv_2, i_enc1, hv_1,
                i_enc0, hv_0, i_jump0, hv_jump0, v_jump1, hv_jump1, v_jump2, hv_jump2, hvi
            )
            
            return output_rgb
    
    def HVIT(self, x):
        hvi = self.trans.HVIT(x)
        return hvi
