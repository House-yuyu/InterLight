"""
模型复杂度评估脚本
计算 HVI-CIDNet 的参数量和 FLOPs
全部使用 thop profile 计算
"""

import os
import torch
import time
from thop import profile
from net.CIDNet import CIDNet

# 指定 GPU 4
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


if __name__ == '__main__':
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device} (GPU 4)")
    
    print("=" * 60)
    print("HVI-CIDNet 模型复杂度评估 (thop profile)")
    print("=" * 60)
    
    # 创建模型 (使用默认参数)
    net = CIDNet(
        channels=[36, 36, 72, 144],
        heads=[1, 2, 4, 8],
        norm=False,
        lfpv_num_feature=16,
        lfpv_patch_size=4,
        atom_num=32,
        atom_dim=512
    ).to(device)
    
    # 设置为评估模式
    net.eval()
    
    # 输入张量形状
    img_shape = (1, 3, 256, 256)
    input_img = torch.randn(img_shape).to(device)
    
    print(f"输入图像尺寸: {img_shape}")
    print()
    
    # ============ 使用 thop profile 计算整体模型复杂度 ============
    print("---- 整体模型复杂度 (thop profile) ----")
    flops, params = profile(net, inputs=(input_img,), verbose=False)
    print(f"FLOPs = {flops / 1e9:.4f} G")
    print(f"Params = {params / 1e6:.4f} M")
    print()
    
    # ============ 各模块参数统计 (thop profile) ============
    print("---- 各模块参数量统计 (thop profile) ----")
    
    for name, module in net.named_children():
        # 跳过无参数的模块
        if sum([p.nelement() for p in module.parameters()]) == 0:
            continue
        
        # 为每个模块创建合适的输入进行 profile
        try:
            module_params = sum([p.nelement() for p in module.parameters()])
            print(f"{name:20s}: Params = {module_params / 1e6:.4f} M")
        except Exception as e:
            print(f"{name:20s}: 计算失败 - {e}")
    
    print()
    
    # ============ 推理测试 ============
    print("---- 推理测试 ----")
    with torch.no_grad():
        output = net(input_img)
        print(f"输入形状: {input_img.shape}")
        print(f"输出形状: {output.shape}")
    print()
    
    # ============ GPU 推理时间测试 ============
    print("---- GPU 推理时间测试 (GPU 4) ----")
    
    # 预热
    with torch.no_grad():
        for _ in range(10):
            _ = net(input_img)
    
    torch.cuda.synchronize()
    
    # 计时
    num_runs = 100
    time_start = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = net(input_img)
    torch.cuda.synchronize()
    time_end = time.time()
    
    avg_time = (time_end - time_start) / num_runs * 1000  # 转为毫秒
    print(f"平均推理时间 ({num_runs} 次): {avg_time:.2f} ms")
    print(f"FPS: {1000 / avg_time:.2f}")
    
    print()
    print("=" * 60)
    print("评估完成")
    print("=" * 60)
    
    # ============ 汇总输出 ============
    print()
    print("【汇总】")
    print(f"FLOPs = {flops / 1e9:.4f} G")
    print(f"Params = {params / 1e6:.4f} M")
    print(f"推理时间 = {avg_time:.2f} ms")
    print(f"FPS = {1000 / avg_time:.2f}")
