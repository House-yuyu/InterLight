import os
# 关键：在导入torch之前先解析参数并设置CUDA_VISIBLE_DEVICES
from data.options import option
opt = option().parse_args()

# 在导入torch之前设置CUDA_VISIBLE_DEVICES
if opt.gpu_ids is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
    print(f'===> Setting CUDA_VISIBLE_DEVICES={opt.gpu_ids} (before importing torch)')

# 现在可以安全地导入torch了
import torch
import random
import math
import json
import shutil
from torchvision import transforms
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from net.CIDNet import CIDNet
from measure import metrics
from eval import eval
from data.data import *
from loss.losses import *
from data.scheduler import *
from tqdm import tqdm
from datetime import datetime
from aug_dehazeformer import AugNoneOpt, AugExternal
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt

# 验证GPU设置
if torch.cuda.is_available():
    print(f'===> CUDA_VISIBLE_DEVICES was set to {os.environ.get("CUDA_VISIBLE_DEVICES", "not set")} before importing torch')
    print(f'===> Physical GPU(s) {opt.gpu_ids if opt.gpu_ids else "all"} mapped to logical GPU(s) starting from 0')
    print(f'===> PyTorch detects {torch.cuda.device_count()} GPU(s)')
    for i in range(torch.cuda.device_count()):
        print(f'  Logical GPU {i}: {torch.cuda.get_device_name(i)}')
    if opt.gpu_ids:
        gpu_list = [int(x.strip()) for x in opt.gpu_ids.split(',')]
        print(f'===> Physical GPU(s) {",".join(map(str, gpu_list))} is/are being used as logical GPU(s) 0-{len(gpu_list)-1}')

def calculate_beta(beta_base, current_step, total_steps):
    """
    使用余弦衰减计算beta值
    beta = beta_base * (1/2 * (cos(pi * current_step / total_steps) + 1))
    """
    beta = beta_base * (1 / 2 * (math.cos(math.pi * current_step / total_steps) + 1))
    return beta

def seed_torch():
    seed = random.randint(1, 1000000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
def get_model_module(model):
    """获取实际的模型对象（兼容DataParallel）"""
    if isinstance(model, nn.DataParallel):
        return model.module
    return model

def train_init():
    seed_torch()
    cudnn.benchmark = True
    # GPU设置已经在导入torch之前完成，这里只需要验证
    # 如果gpu_ids为None，则使用环境变量中已设置的CUDA_VISIBLE_DEVICES
    cuda = opt.gpu_mode
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    
def train(epoch):
    model.train()
    loss_print = 0
    pic_cnt = 0
    loss_last_10 = 0
    pic_last_10 = 0
    train_len = len(training_data_loader)
    iter = 0
    torch.autograd.set_detect_anomaly(opt.grad_detect)
    
    # 初始化内部增强模块（如果启用自监督学习）
    aug_opt = None
    if opt.ssl_aug:
        aug_opt = AugNoneOpt().cuda()
        # 如果使用多GPU，也需要将aug_opt包装为DataParallel
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            aug_opt = nn.DataParallel(aug_opt)
            
    # 初始化外部增强模块（如果启用）
    ext_aug = None
    if opt.ext_aug:
        # 解析 gamma 范围参数
        try:
            gamma_range = tuple(map(float, opt.ext_aug_range.split(',')))
            if len(gamma_range) != 2:
                raise ValueError
        except:
            print("===> Warning: Invalid ext_aug_range, using default (0.95, 1.05)")
            gamma_range = (0.95, 1.05)
        
        # 获取增强模式和保护参数
        mode = getattr(opt, 'ext_aug_mode', 'symmetric')
        dark_threshold = getattr(opt, 'ext_aug_dark_threshold', 0.05)
        protect_dark = getattr(opt, 'ext_aug_protect_dark', True)
        
        ext_aug = AugExternal(
            gamma_range=gamma_range, 
            prob=opt.ext_aug_prob, 
            mode=mode,
            dark_threshold=dark_threshold,
            protect_dark=protect_dark
        ).cuda()
        
        print(f"===> External Augmentation Enabled:")
        print(f"     - Gamma range: {gamma_range}")
        print(f"     - Probability: {opt.ext_aug_prob}")
        print(f"     - Mode: {mode}")
        print(f"     - Dark protection: {'ON' if protect_dark else 'OFF'} (threshold={dark_threshold})")
        print(f"     - Rationale:")
        print(f"       * Apply BEFORE HVI transform to simulate sensor-induced HVI variations")
        print(f"       * Use mild gamma (±5%) to preserve physical noise characteristics")
        print(f"       * Protect dark pixels where shot noise dominates over gamma response")
        # External Augmentation 不需要 DataParallel，因为它没有可学习的参数，且直接作用于batch
    
    for batch in tqdm(training_data_loader):
        im1, im2, path1, path2 = batch[0], batch[1], batch[2], batch[3]
        im1 = im1.cuda()
        im2 = im2.cuda()
        
        # 应用外部增强 (External Augmentation)
        # 模拟外部数据分布，提升泛化能力
        if opt.ext_aug and ext_aug is not None:
            im1 = ext_aug(im1)
        
        # use random gamma function (enhancement curve) to improve generalization
        if opt.gamma:
            gamma = random.randint(opt.start_gamma,opt.end_gamma) / 100.0
            model_output = model(im1 ** gamma)  
        else:
            model_output = model(im1)  
            
        gt_rgb = im2
        model_module = get_model_module(model)
        
        # 处理 LFPV 双路径输出
        if model.training and isinstance(model_output, tuple):
            # 训练时：双路径输出 (output_with_lfpv, output_without_lfpv)
            output_rgb, output_rgb_without_lfpv = model_output
            
            # 路径1损失：不使用 LFPV 的输出（让baseline学习尽可能多的知识）
            output_hvi_1 = model_module.HVIT(output_rgb_without_lfpv)
            gt_hvi = model_module.HVIT(gt_rgb)
            loss_hvi_1 = L1_loss(output_hvi_1, gt_hvi) + D_loss(output_hvi_1, gt_hvi) + E_loss(output_hvi_1, gt_hvi) + opt.P_weight * P_loss(output_hvi_1, gt_hvi)[0]
            loss_rgb_1 = L1_loss(output_rgb_without_lfpv, gt_rgb) + D_loss(output_rgb_without_lfpv, gt_rgb) + E_loss(output_rgb_without_lfpv, gt_rgb) + opt.P_weight * P_loss(output_rgb_without_lfpv, gt_rgb)[0]
            loss1 = loss_rgb_1 + opt.HVI_weight * loss_hvi_1
            
            # 路径2损失：使用 LFPV 的输出（学习剩余知识）
            output_hvi_2 = model_module.HVIT(output_rgb)
            loss_hvi_2 = L1_loss(output_hvi_2, gt_hvi) + D_loss(output_hvi_2, gt_hvi) + E_loss(output_hvi_2, gt_hvi) + opt.P_weight * P_loss(output_hvi_2, gt_hvi)[0]
            loss_rgb_2 = L1_loss(output_rgb, gt_rgb) + D_loss(output_rgb, gt_rgb) + E_loss(output_rgb, gt_rgb) + opt.P_weight * P_loss(output_rgb, gt_rgb)[0]
            loss2 = loss_rgb_2 + opt.HVI_weight * loss_hvi_2
            
            # 组合损失（Eq. 9）：loss = loss1 + lambda * loss2
            # lambda 通过 opt.lfpv_weight 控制
            loss = loss1 + opt.lfpv_weight * loss2
            
            # 释放中间变量以节省内存（双路径分支）
            del output_hvi_1, output_hvi_2, gt_hvi
        else:
            # 推理时或单路径输出：正常处理
            output_rgb = model_output if not isinstance(model_output, tuple) else model_output[0]
            output_hvi = model_module.HVIT(output_rgb)
            gt_hvi = model_module.HVIT(gt_rgb)
            loss_hvi = L1_loss(output_hvi, gt_hvi) + D_loss(output_hvi, gt_hvi) + E_loss(output_hvi, gt_hvi) + opt.P_weight * P_loss(output_hvi, gt_hvi)[0]
            loss_rgb = L1_loss(output_rgb, gt_rgb) + D_loss(output_rgb, gt_rgb) + E_loss(output_rgb, gt_rgb) + opt.P_weight * P_loss(output_rgb, gt_rgb)[0]
            loss = loss_rgb + opt.HVI_weight * loss_hvi
            
            # 释放中间变量以节省内存（单路径分支）
            del output_hvi, gt_hvi
        
        # 自监督学习的内部增强和MSR损失
        # 只对最终的增强图像（output_rgb）进行增强，而不是中间特征
        if opt.ssl_aug and aug_opt is not None:
            # 对最终的增强图像进行内部增强（弱增强和强增强）
            weak_output, aggr_output = aug_opt(output_rgb)
            
            # 计算MSR损失（弱增强和强增强输出的MSE）
            # 这个损失约束模型在不同增强下输出的一致性
            msr_loss = torch.nn.MSELoss()(weak_output, aggr_output)
            
            # 计算beta值（余弦衰减），beta_base作为MSR损失的权重基础值
            current_step = (epoch - 1) * train_len + iter
            beta = calculate_beta(opt.beta_base, current_step, opt.beta_total_steps)
            
            # 将MSR损失添加到总损失中（与ScaleUpDehazing保持一致：loss = main_loss + beta * msr_loss）
            loss = loss + beta * msr_loss
            
            # 释放中间变量以节省内存
            del weak_output, aggr_output, msr_loss
        
        iter += 1
        
        if opt.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01, norm_type=2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 释放loss变量并清理GPU缓存（多GPU训练时有助于减少内存占用）
        loss_value = loss.item()
        del loss
        if iter % 10 == 0:  # 每10个iteration清理一次缓存
            torch.cuda.empty_cache()
        
        loss_print = loss_print + loss_value
        loss_last_10 = loss_last_10 + loss_value
        pic_cnt += 1
        pic_last_10 += 1
        if iter == train_len:
            print("===> Epoch[{}]: Loss: {:.4f} || Learning rate: lr={}.".format(epoch,
                loss_last_10/pic_last_10, optimizer.param_groups[0]['lr']))
            loss_last_10 = 0
            pic_last_10 = 0
            output_img = transforms.ToPILImage()((output_rgb)[0].squeeze(0))
            gt_img = transforms.ToPILImage()((gt_rgb)[0].squeeze(0))
            if not os.path.exists(opt.val_folder+'training'):          
                os.mkdir(opt.val_folder+'training') 
            output_img.save(opt.val_folder+'training/test.png')
            gt_img.save(opt.val_folder+'training/gt.png')
    return loss_print, pic_cnt
                

def checkpoint(epoch):
    # 根据数据集名称和实验名称创建不同的目录
    if opt.exp_name:
        dataset_dir = os.path.join("./weights", f"{opt.dataset}_{opt.exp_name}")
    else:
        dataset_dir = os.path.join("./weights", opt.dataset)
    if not os.path.exists("./weights"):          
        os.mkdir("./weights") 
    if not os.path.exists(dataset_dir):          
        os.mkdir(dataset_dir)  
    model_out_path = os.path.join(dataset_dir, "epoch_{}.pth".format(epoch))
    # 保存模型时，如果是DataParallel，保存时去掉'module.'前缀以便单GPU加载
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), model_out_path)
    else:
        torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))
    return model_out_path

def save_best_checkpoint(epoch, metric_name, metric_value, dataset_name):
    """保存最佳指标的权重"""
    # 根据实验名称构建目录路径
    if opt.exp_name:
        dataset_dir = os.path.join("./weights", f"{dataset_name}_{opt.exp_name}")
    else:
        dataset_dir = os.path.join("./weights", dataset_name)
    best_model_path = os.path.join(dataset_dir, f"best_{metric_name}.pth")
    current_model_path = os.path.join(dataset_dir, f"epoch_{epoch}.pth")
    
    # 复制当前权重到最佳权重
    if os.path.exists(current_model_path):
        shutil.copy2(current_model_path, best_model_path)
        print(f"===> Saved best {metric_name} checkpoint: {best_model_path} (value: {metric_value:.4f})")
    return best_model_path

def reset_best_results(dataset_name):
    """重置最佳结果记录，并备份原有记录（包括权重文件）"""
    results_dir = "./results/best_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # 根据实验名称构建文件名
    if opt.exp_name:
        result_name = f"{dataset_name}_{opt.exp_name}"
    else:
        result_name = dataset_name
    
    json_file = os.path.join(results_dir, f"{result_name}.json")
    md_file = os.path.join(results_dir, f"{result_name}.md")
    
    # 备份原有文件（如果存在）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建备份目录
    backup_dir = os.path.join(results_dir, f"{result_name}_backup_{timestamp}")
    weights_backed_up = False
    
    # 如果存在旧的JSON文件，读取并备份权重文件
    if os.path.exists(json_file):
        try:
            with open(json_file, 'r') as f:
                content = f.read()
                content = content.replace('Infinity', '999999.0').replace('infinity', '999999.0')
                old_results = json.loads(content)
            
            # 收集所有需要备份的权重文件路径
            weight_keys = ["best_psnr", "best_ssim", "best_lpips", 
                          "best_psnr_gtmean", "best_ssim_gtmean", "best_lpips_gtmean"]
            weight_paths = set()  # 使用set避免重复
            
            for key in weight_keys:
                if key in old_results and isinstance(old_results[key], dict):
                    best_weight_path = old_results[key].get("best_weight_path", "")
                    if best_weight_path and os.path.exists(best_weight_path):
                        weight_paths.add(best_weight_path)
            
            # 备份权重文件
            if weight_paths:
                if not os.path.exists(backup_dir):
                    os.makedirs(backup_dir)
                print(f"===> Backing up {len(weight_paths)} best weight file(s) to {backup_dir}")
                for weight_path in weight_paths:
                    weight_name = os.path.basename(weight_path)
                    backup_weight_path = os.path.join(backup_dir, weight_name)
                    shutil.copy2(weight_path, backup_weight_path)
                    print(f"     - {weight_path} -> {backup_weight_path}")
                weights_backed_up = True
                
        except Exception as e:
            print(f"===> Warning: Failed to backup weights: {e}")
    
    # 备份JSON文件
    if os.path.exists(json_file):
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)
        backup_json = os.path.join(backup_dir, f"{result_name}.json")
        shutil.copy2(json_file, backup_json)
        print(f"===> Backed up existing JSON to {backup_json}")
    
    # 备份Markdown文件
    if os.path.exists(md_file):
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)
        backup_md = os.path.join(backup_dir, f"{result_name}.md")
        shutil.copy2(md_file, backup_md)
        print(f"===> Backed up existing Markdown to {backup_md}")
    
    if weights_backed_up or os.path.exists(json_file) or os.path.exists(md_file):
        print(f"===> All backups saved to: {backup_dir}")
    
    # 重置为初始值 - 每个最佳指标同时保存所有三个指标值
    best_results = {
        "dataset": dataset_name,
        # 普通模式 (use_GT_mean=False)
        "best_psnr": {"psnr": 0.0, "ssim": 0.0, "lpips": 999999.0, "epoch": 0, "weight_path": "", "best_weight_path": ""},
        "best_ssim": {"psnr": 0.0, "ssim": 0.0, "lpips": 999999.0, "epoch": 0, "weight_path": "", "best_weight_path": ""},
        "best_lpips": {"psnr": 0.0, "ssim": 0.0, "lpips": 999999.0, "epoch": 0, "weight_path": "", "best_weight_path": ""},
        # GT Mean 模式 (use_GT_mean=True)
        "best_psnr_gtmean": {"psnr": 0.0, "ssim": 0.0, "lpips": 999999.0, "epoch": 0, "weight_path": "", "best_weight_path": ""},
        "best_ssim_gtmean": {"psnr": 0.0, "ssim": 0.0, "lpips": 999999.0, "epoch": 0, "weight_path": "", "best_weight_path": ""},
        "best_lpips_gtmean": {"psnr": 0.0, "ssim": 0.0, "lpips": 999999.0, "epoch": 0, "weight_path": "", "best_weight_path": ""},
        "last_updated": ""
    }
    
    # 保存JSON文件
    with open(json_file, 'w') as f:
        json.dump(best_results, f, indent=4)
    
    # 保存Markdown文件
    with open(md_file, 'w') as f:
        f.write(f"# Best Results for {dataset_name}\n\n")
        f.write(f"**Last Updated:** \n\n")
        
        # 普通模式
        f.write("# Normal Mode (use_GT_mean=False)\n\n")
        f.write("## Best PSNR\n\n")
        f.write(f"- **PSNR:** 0.0 dB | **SSIM:** 0.0 | **LPIPS:** 999999.0\n")
        f.write(f"- **Epoch:** 0\n")
        f.write(f"- **Best Weight Path:** ``\n\n")
        f.write("## Best SSIM\n\n")
        f.write(f"- **PSNR:** 0.0 dB | **SSIM:** 0.0 | **LPIPS:** 999999.0\n")
        f.write(f"- **Epoch:** 0\n")
        f.write(f"- **Best Weight Path:** ``\n\n")
        f.write("## Best LPIPS\n\n")
        f.write(f"- **PSNR:** 0.0 dB | **SSIM:** 0.0 | **LPIPS:** 999999.0\n")
        f.write(f"- **Epoch:** 0\n")
        f.write(f"- **Best Weight Path:** ``\n\n")
        
        # GT Mean 模式
        f.write("---\n\n")
        f.write("# GT Mean Mode (use_GT_mean=True)\n\n")
        f.write("## Best PSNR (GT Mean)\n\n")
        f.write(f"- **PSNR:** 0.0 dB | **SSIM:** 0.0 | **LPIPS:** 999999.0\n")
        f.write(f"- **Epoch:** 0\n")
        f.write(f"- **Best Weight Path:** ``\n\n")
        f.write("## Best SSIM (GT Mean)\n\n")
        f.write(f"- **PSNR:** 0.0 dB | **SSIM:** 0.0 | **LPIPS:** 999999.0\n")
        f.write(f"- **Epoch:** 0\n")
        f.write(f"- **Best Weight Path:** ``\n\n")
        f.write("## Best LPIPS (GT Mean)\n\n")
        f.write(f"- **PSNR:** 0.0 dB | **SSIM:** 0.0 | **LPIPS:** 999999.0\n")
        f.write(f"- **Epoch:** 0\n")
        f.write(f"- **Best Weight Path:** ``\n\n")
        
        f.write("---\n\n")
        f.write("## Summary Table (Normal Mode)\n\n")
        f.write("| Best By | PSNR | SSIM | LPIPS | Epoch | Weight Path |\n")
        f.write("|---------|------|------|-------|-------|-------------|\n")
        f.write(f"| PSNR | 0.0 dB | 0.0 | 999999.0 | 0 | `` |\n")
        f.write(f"| SSIM | 0.0 dB | 0.0 | 999999.0 | 0 | `` |\n")
        f.write(f"| LPIPS | 0.0 dB | 0.0 | 999999.0 | 0 | `` |\n\n")
        
        f.write("## Summary Table (GT Mean Mode)\n\n")
        f.write("| Best By | PSNR | SSIM | LPIPS | Epoch | Weight Path |\n")
        f.write("|---------|------|------|-------|-------|-------------|\n")
        f.write(f"| PSNR | 0.0 dB | 0.0 | 999999.0 | 0 | `` |\n")
        f.write(f"| SSIM | 0.0 dB | 0.0 | 999999.0 | 0 | `` |\n")
        f.write(f"| LPIPS | 0.0 dB | 0.0 | 999999.0 | 0 | `` |\n")
    
    print(f"===> Reset best results for {dataset_name}")

def update_best_results(dataset_name, epoch, psnr, ssim, lpips, model_path, 
                        psnr_gtmean=None, ssim_gtmean=None, lpips_gtmean=None):
    """更新并保存每个数据集的最佳结果
    
    Args:
        dataset_name: 数据集名称
        epoch: 当前epoch
        psnr, ssim, lpips: 普通模式的指标值
        model_path: 模型权重路径
        psnr_gtmean, ssim_gtmean, lpips_gtmean: GT Mean模式的指标值（可选）
    """
    results_dir = "./results/best_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # 根据实验名称构建文件名
    if opt.exp_name:
        result_name = f"{dataset_name}_{opt.exp_name}"
    else:
        result_name = dataset_name
    
    # JSON文件用于程序读取
    json_file = os.path.join(results_dir, f"{result_name}.json")
    # Markdown文件用于人类阅读
    md_file = os.path.join(results_dir, f"{result_name}.md")
    
    # 默认初始值 - 新格式：每个最佳指标保存所有三个指标值
    best_results = {
        "dataset": dataset_name,
        # 普通模式 (use_GT_mean=False)
        "best_psnr": {"psnr": 0.0, "ssim": 0.0, "lpips": 999999.0, "epoch": 0, "weight_path": "", "best_weight_path": ""},
        "best_ssim": {"psnr": 0.0, "ssim": 0.0, "lpips": 999999.0, "epoch": 0, "weight_path": "", "best_weight_path": ""},
        "best_lpips": {"psnr": 0.0, "ssim": 0.0, "lpips": 999999.0, "epoch": 0, "weight_path": "", "best_weight_path": ""},
        # GT Mean 模式 (use_GT_mean=True)
        "best_psnr_gtmean": {"psnr": 0.0, "ssim": 0.0, "lpips": 999999.0, "epoch": 0, "weight_path": "", "best_weight_path": ""},
        "best_ssim_gtmean": {"psnr": 0.0, "ssim": 0.0, "lpips": 999999.0, "epoch": 0, "weight_path": "", "best_weight_path": ""},
        "best_lpips_gtmean": {"psnr": 0.0, "ssim": 0.0, "lpips": 999999.0, "epoch": 0, "weight_path": "", "best_weight_path": ""},
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            content = f.read()
            # 处理无效的Infinity值（兼容旧格式）
            content = content.replace('Infinity', '999999.0').replace('infinity', '999999.0')
            loaded_results = json.loads(content)
            
            # 兼容旧格式：如果是旧格式（只有value字段），转换为新格式
            for key in ["best_psnr", "best_ssim", "best_lpips"]:
                if key in loaded_results:
                    if "value" in loaded_results[key]:
                        # 旧格式转换为新格式
                        old_val = loaded_results[key]["value"]
                        loaded_results[key]["psnr"] = old_val if key == "best_psnr" else 0.0
                        loaded_results[key]["ssim"] = old_val if key == "best_ssim" else 0.0
                        loaded_results[key]["lpips"] = old_val if key == "best_lpips" else 999999.0
                        del loaded_results[key]["value"]
                    if "best_weight_path" not in loaded_results[key]:
                        loaded_results[key]["best_weight_path"] = ""
            
            # 添加缺失的gtmean字段
            for key in ["best_psnr_gtmean", "best_ssim_gtmean", "best_lpips_gtmean"]:
                if key not in loaded_results:
                    loaded_results[key] = {"psnr": 0.0, "ssim": 0.0, "lpips": 999999.0, "epoch": 0, "weight_path": "", "best_weight_path": ""}
            
            best_results = loaded_results
    
    updated = False
    
    # ========== 普通模式 ==========
    # 更新PSNR（越大越好）
    if psnr > best_results["best_psnr"]["psnr"]:
        old_epoch = best_results["best_psnr"]["epoch"]
        old_value = best_results["best_psnr"]["psnr"]
        best_results["best_psnr"]["psnr"] = float(psnr)
        best_results["best_psnr"]["ssim"] = float(ssim)
        best_results["best_psnr"]["lpips"] = float(lpips)
        best_results["best_psnr"]["epoch"] = epoch
        best_results["best_psnr"]["weight_path"] = model_path
        best_results["best_psnr"]["best_weight_path"] = save_best_checkpoint(epoch, "psnr", psnr, dataset_name)
        updated = True
        print(f"===> New best PSNR: {psnr:.4f} dB at epoch {epoch} (previous best: {old_value:.4f} dB at epoch {old_epoch})")
        print(f"     [All metrics at this point: PSNR={psnr:.4f}, SSIM={ssim:.4f}, LPIPS={lpips:.4f}]")
    
    # 更新SSIM（越大越好）
    if ssim > best_results["best_ssim"]["ssim"]:
        old_epoch = best_results["best_ssim"]["epoch"]
        old_value = best_results["best_ssim"]["ssim"]
        best_results["best_ssim"]["psnr"] = float(psnr)
        best_results["best_ssim"]["ssim"] = float(ssim)
        best_results["best_ssim"]["lpips"] = float(lpips)
        best_results["best_ssim"]["epoch"] = epoch
        best_results["best_ssim"]["weight_path"] = model_path
        best_results["best_ssim"]["best_weight_path"] = save_best_checkpoint(epoch, "ssim", ssim, dataset_name)
        updated = True
        print(f"===> New best SSIM: {ssim:.4f} at epoch {epoch} (previous best: {old_value:.4f} at epoch {old_epoch})")
        print(f"     [All metrics at this point: PSNR={psnr:.4f}, SSIM={ssim:.4f}, LPIPS={lpips:.4f}]")
    
    # 更新LPIPS（越小越好）
    if lpips < best_results["best_lpips"]["lpips"]:
        old_epoch = best_results["best_lpips"]["epoch"]
        old_value = best_results["best_lpips"]["lpips"]
        best_results["best_lpips"]["psnr"] = float(psnr)
        best_results["best_lpips"]["ssim"] = float(ssim)
        best_results["best_lpips"]["lpips"] = float(lpips)
        best_results["best_lpips"]["epoch"] = epoch
        best_results["best_lpips"]["weight_path"] = model_path
        best_results["best_lpips"]["best_weight_path"] = save_best_checkpoint(epoch, "lpips", lpips, dataset_name)
        updated = True
        print(f"===> New best LPIPS: {lpips:.4f} at epoch {epoch} (previous best: {old_value:.4f} at epoch {old_epoch})")
        print(f"     [All metrics at this point: PSNR={psnr:.4f}, SSIM={ssim:.4f}, LPIPS={lpips:.4f}]")
    
    # ========== GT Mean 模式 ==========
    if psnr_gtmean is not None and ssim_gtmean is not None and lpips_gtmean is not None:
        # 更新PSNR (GT Mean)
        if psnr_gtmean > best_results["best_psnr_gtmean"]["psnr"]:
            old_epoch = best_results["best_psnr_gtmean"]["epoch"]
            old_value = best_results["best_psnr_gtmean"]["psnr"]
            best_results["best_psnr_gtmean"]["psnr"] = float(psnr_gtmean)
            best_results["best_psnr_gtmean"]["ssim"] = float(ssim_gtmean)
            best_results["best_psnr_gtmean"]["lpips"] = float(lpips_gtmean)
            best_results["best_psnr_gtmean"]["epoch"] = epoch
            best_results["best_psnr_gtmean"]["weight_path"] = model_path
            best_results["best_psnr_gtmean"]["best_weight_path"] = save_best_checkpoint(epoch, "psnr_gtmean", psnr_gtmean, dataset_name)
            updated = True
            print(f"===> New best PSNR (GT Mean): {psnr_gtmean:.4f} dB at epoch {epoch} (previous best: {old_value:.4f} dB at epoch {old_epoch})")
            print(f"     [All metrics at this point: PSNR={psnr_gtmean:.4f}, SSIM={ssim_gtmean:.4f}, LPIPS={lpips_gtmean:.4f}]")
        
        # 更新SSIM (GT Mean)
        if ssim_gtmean > best_results["best_ssim_gtmean"]["ssim"]:
            old_epoch = best_results["best_ssim_gtmean"]["epoch"]
            old_value = best_results["best_ssim_gtmean"]["ssim"]
            best_results["best_ssim_gtmean"]["psnr"] = float(psnr_gtmean)
            best_results["best_ssim_gtmean"]["ssim"] = float(ssim_gtmean)
            best_results["best_ssim_gtmean"]["lpips"] = float(lpips_gtmean)
            best_results["best_ssim_gtmean"]["epoch"] = epoch
            best_results["best_ssim_gtmean"]["weight_path"] = model_path
            best_results["best_ssim_gtmean"]["best_weight_path"] = save_best_checkpoint(epoch, "ssim_gtmean", ssim_gtmean, dataset_name)
            updated = True
            print(f"===> New best SSIM (GT Mean): {ssim_gtmean:.4f} at epoch {epoch} (previous best: {old_value:.4f} at epoch {old_epoch})")
            print(f"     [All metrics at this point: PSNR={psnr_gtmean:.4f}, SSIM={ssim_gtmean:.4f}, LPIPS={lpips_gtmean:.4f}]")
        
        # 更新LPIPS (GT Mean)
        if lpips_gtmean < best_results["best_lpips_gtmean"]["lpips"]:
            old_epoch = best_results["best_lpips_gtmean"]["epoch"]
            old_value = best_results["best_lpips_gtmean"]["lpips"]
            best_results["best_lpips_gtmean"]["psnr"] = float(psnr_gtmean)
            best_results["best_lpips_gtmean"]["ssim"] = float(ssim_gtmean)
            best_results["best_lpips_gtmean"]["lpips"] = float(lpips_gtmean)
            best_results["best_lpips_gtmean"]["epoch"] = epoch
            best_results["best_lpips_gtmean"]["weight_path"] = model_path
            best_results["best_lpips_gtmean"]["best_weight_path"] = save_best_checkpoint(epoch, "lpips_gtmean", lpips_gtmean, dataset_name)
            updated = True
            print(f"===> New best LPIPS (GT Mean): {lpips_gtmean:.4f} at epoch {epoch} (previous best: {old_value:.4f} at epoch {old_epoch})")
            print(f"     [All metrics at this point: PSNR={psnr_gtmean:.4f}, SSIM={ssim_gtmean:.4f}, LPIPS={lpips_gtmean:.4f}]")
    
    if updated:
        best_results["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # 保存JSON文件（用于程序读取）
        with open(json_file, 'w') as f:
            json.dump(best_results, f, indent=4)
        
        # 保存Markdown文件（用于人类阅读）
        with open(md_file, 'w') as f:
            f.write(f"# Best Results for {dataset_name}\n\n")
            f.write(f"**Last Updated:** {best_results['last_updated']}\n\n")
            
            # 普通模式
            f.write("# Normal Mode (use_GT_mean=False)\n\n")
            f.write("## Best PSNR\n\n")
            f.write(f"- **PSNR:** {best_results['best_psnr']['psnr']:.4f} dB | **SSIM:** {best_results['best_psnr']['ssim']:.4f} | **LPIPS:** {best_results['best_psnr']['lpips']:.4f}\n")
            f.write(f"- **Epoch:** {best_results['best_psnr']['epoch']}\n")
            f.write(f"- **Best Weight Path:** `{best_results['best_psnr']['best_weight_path']}`\n\n")
            f.write("## Best SSIM\n\n")
            f.write(f"- **PSNR:** {best_results['best_ssim']['psnr']:.4f} dB | **SSIM:** {best_results['best_ssim']['ssim']:.4f} | **LPIPS:** {best_results['best_ssim']['lpips']:.4f}\n")
            f.write(f"- **Epoch:** {best_results['best_ssim']['epoch']}\n")
            f.write(f"- **Best Weight Path:** `{best_results['best_ssim']['best_weight_path']}`\n\n")
            f.write("## Best LPIPS\n\n")
            f.write(f"- **PSNR:** {best_results['best_lpips']['psnr']:.4f} dB | **SSIM:** {best_results['best_lpips']['ssim']:.4f} | **LPIPS:** {best_results['best_lpips']['lpips']:.4f}\n")
            f.write(f"- **Epoch:** {best_results['best_lpips']['epoch']}\n")
            f.write(f"- **Best Weight Path:** `{best_results['best_lpips']['best_weight_path']}`\n\n")
            
            # GT Mean 模式
            f.write("---\n\n")
            f.write("# GT Mean Mode (use_GT_mean=True)\n\n")
            f.write("## Best PSNR (GT Mean)\n\n")
            f.write(f"- **PSNR:** {best_results['best_psnr_gtmean']['psnr']:.4f} dB | **SSIM:** {best_results['best_psnr_gtmean']['ssim']:.4f} | **LPIPS:** {best_results['best_psnr_gtmean']['lpips']:.4f}\n")
            f.write(f"- **Epoch:** {best_results['best_psnr_gtmean']['epoch']}\n")
            f.write(f"- **Best Weight Path:** `{best_results['best_psnr_gtmean']['best_weight_path']}`\n\n")
            f.write("## Best SSIM (GT Mean)\n\n")
            f.write(f"- **PSNR:** {best_results['best_ssim_gtmean']['psnr']:.4f} dB | **SSIM:** {best_results['best_ssim_gtmean']['ssim']:.4f} | **LPIPS:** {best_results['best_ssim_gtmean']['lpips']:.4f}\n")
            f.write(f"- **Epoch:** {best_results['best_ssim_gtmean']['epoch']}\n")
            f.write(f"- **Best Weight Path:** `{best_results['best_ssim_gtmean']['best_weight_path']}`\n\n")
            f.write("## Best LPIPS (GT Mean)\n\n")
            f.write(f"- **PSNR:** {best_results['best_lpips_gtmean']['psnr']:.4f} dB | **SSIM:** {best_results['best_lpips_gtmean']['ssim']:.4f} | **LPIPS:** {best_results['best_lpips_gtmean']['lpips']:.4f}\n")
            f.write(f"- **Epoch:** {best_results['best_lpips_gtmean']['epoch']}\n")
            f.write(f"- **Best Weight Path:** `{best_results['best_lpips_gtmean']['best_weight_path']}`\n\n")
            
            f.write("---\n\n")
            f.write("## Summary Table (Normal Mode)\n\n")
            f.write("| Best By | PSNR | SSIM | LPIPS | Epoch | Weight Path |\n")
            f.write("|---------|------|------|-------|-------|-------------|\n")
            f.write(f"| PSNR | {best_results['best_psnr']['psnr']:.4f} dB | {best_results['best_psnr']['ssim']:.4f} | {best_results['best_psnr']['lpips']:.4f} | {best_results['best_psnr']['epoch']} | `{best_results['best_psnr']['best_weight_path']}` |\n")
            f.write(f"| SSIM | {best_results['best_ssim']['psnr']:.4f} dB | {best_results['best_ssim']['ssim']:.4f} | {best_results['best_ssim']['lpips']:.4f} | {best_results['best_ssim']['epoch']} | `{best_results['best_ssim']['best_weight_path']}` |\n")
            f.write(f"| LPIPS | {best_results['best_lpips']['psnr']:.4f} dB | {best_results['best_lpips']['ssim']:.4f} | {best_results['best_lpips']['lpips']:.4f} | {best_results['best_lpips']['epoch']} | `{best_results['best_lpips']['best_weight_path']}` |\n\n")
            
            f.write("## Summary Table (GT Mean Mode)\n\n")
            f.write("| Best By | PSNR | SSIM | LPIPS | Epoch | Weight Path |\n")
            f.write("|---------|------|------|-------|-------|-------------|\n")
            f.write(f"| PSNR | {best_results['best_psnr_gtmean']['psnr']:.4f} dB | {best_results['best_psnr_gtmean']['ssim']:.4f} | {best_results['best_psnr_gtmean']['lpips']:.4f} | {best_results['best_psnr_gtmean']['epoch']} | `{best_results['best_psnr_gtmean']['best_weight_path']}` |\n")
            f.write(f"| SSIM | {best_results['best_ssim_gtmean']['psnr']:.4f} dB | {best_results['best_ssim_gtmean']['ssim']:.4f} | {best_results['best_ssim_gtmean']['lpips']:.4f} | {best_results['best_ssim_gtmean']['epoch']} | `{best_results['best_ssim_gtmean']['best_weight_path']}` |\n")
            f.write(f"| LPIPS | {best_results['best_lpips_gtmean']['psnr']:.4f} dB | {best_results['best_lpips_gtmean']['ssim']:.4f} | {best_results['best_lpips_gtmean']['lpips']:.4f} | {best_results['best_lpips_gtmean']['epoch']} | `{best_results['best_lpips_gtmean']['best_weight_path']}` |\n")
        
        print(f"===> Best results updated and saved to {json_file} and {md_file}")
    
    return best_results

def init_metrics_log(dataset_name, start_epoch, opt):
    """初始化评估指标日志文件（实时保存）"""
    log_dir = "./results/training"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    # 根据实验名称构建文件名
    if opt.exp_name:
        metrics_file = os.path.join(log_dir, f"metrics_{dataset_name}_{opt.exp_name}_{timestamp}.md")
    else:
        metrics_file = os.path.join(log_dir, f"metrics_{dataset_name}_{timestamp}.md")
    
    # 写入训练配置信息和表头
    with open(metrics_file, 'w') as f:
        f.write("dataset: " + dataset_name + "\n")
        f.write(f"lr: {opt.lr}\n")
        f.write(f"batch size: {opt.batchSize}\n")
        f.write(f"crop size: {opt.cropSize}\n")
        f.write(f"HVI_weight: {opt.HVI_weight}\n")
        f.write(f"L1_weight: {opt.L1_weight}\n")
        f.write(f"D_weight: {opt.D_weight}\n")
        f.write(f"E_weight: {opt.E_weight}\n")
        f.write(f"P_weight: {opt.P_weight}\n")
        f.write(f"SSL_aug: {opt.ssl_aug}\n")
        if opt.ssl_aug:
            f.write(f"beta_base: {opt.beta_base}\n")
            f.write(f"beta_total_steps: {opt.beta_total_steps}\n")
        f.write("\n## Normal Mode (use_GT_mean=False)\n\n")
        f.write("| Epochs | PSNR | SSIM | LPIPS |\n")
        f.write("|--------|------|------|-------|\n")
    
    print(f"===> Metrics log initialized: {metrics_file}")
    return metrics_file

def write_evaluation_log(metrics_file, epoch, psnr, ssim, lpips, 
                         psnr_gtmean=None, ssim_gtmean=None, lpips_gtmean=None):
    """实时写入评估结果到日志文件
    
    Args:
        metrics_file: 日志文件路径
        epoch: 当前epoch
        psnr, ssim, lpips: 普通模式的指标值
        psnr_gtmean, ssim_gtmean, lpips_gtmean: GT Mean模式的指标值（可选）
    """
    with open(metrics_file, 'a') as f:
        f.write(f"| {epoch} | {psnr:.4f} | {ssim:.4f} | {lpips:.4f} |\n")
        
        # 如果有GT Mean结果，在同一行后面添加（通过注释形式）
        if psnr_gtmean is not None and ssim_gtmean is not None and lpips_gtmean is not None:
            f.write(f"| {epoch} (GT Mean) | {psnr_gtmean:.4f} | {ssim_gtmean:.4f} | {lpips_gtmean:.4f} |\n")

def plot_evaluation_metrics(dataset_name, epochs, psnr_list, ssim_list, lpips_list):
    """绘制评估指标的折线图"""
    results_dir = "./results/best_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # 创建图表
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # PSNR图
    axes[0].plot(epochs, psnr_list, 'b-o', linewidth=2, markersize=6, label='PSNR')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('PSNR (dB)', fontsize=12)
    axes[0].set_title('PSNR over Epochs', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].tick_params(labelsize=10)
    
    # SSIM图
    axes[1].plot(epochs, ssim_list, 'g-o', linewidth=2, markersize=6, label='SSIM')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('SSIM', fontsize=12)
    axes[1].set_title('SSIM over Epochs', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].tick_params(labelsize=10)
    
    # LPIPS图（注意LPIPS越小越好）
    axes[2].plot(epochs, lpips_list, 'r-o', linewidth=2, markersize=6, label='LPIPS')
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('LPIPS', fontsize=12)
    axes[2].set_title('LPIPS over Epochs (lower is better)', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    axes[2].tick_params(labelsize=10)
    
    plt.tight_layout()
    
    # 保存图片
    plot_path = os.path.join(results_dir, f"{dataset_name}_metrics.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"===> Evaluation metrics plot saved to {plot_path}")
    
    return plot_path
    
def load_datasets():
    print(f'===> Loading datasets: {opt.dataset}')
    if opt.dataset == 'lol_v1':
        train_set = get_lol_training_set(opt.data_train_lol_v1,size=opt.cropSize)
        test_set = get_eval_set(opt.data_val_lol_v1)
        
    elif opt.dataset == 'lol_blur':
        train_set = get_training_set_blur(opt.data_train_lol_blur,size=opt.cropSize)
        test_set = get_eval_set(opt.data_val_lol_blur)

    elif opt.dataset == 'lolv2_real':
        train_set = get_lol_v2_training_set(opt.data_train_lolv2_real,size=opt.cropSize)
        test_set = get_eval_set(opt.data_val_lolv2_real)
        
    elif opt.dataset == 'lolv2_syn':
        train_set = get_lol_v2_syn_training_set(opt.data_train_lolv2_syn,size=opt.cropSize)
        test_set = get_eval_set(opt.data_val_lolv2_syn)
    
    elif opt.dataset == 'SID':
        train_set = get_SID_training_set(opt.data_train_SID,size=opt.cropSize)
        test_set = get_eval_set(opt.data_val_SID)
        
    elif opt.dataset == 'SICE_mix':
        train_set = get_SICE_training_set(opt.data_train_SICE,size=opt.cropSize)
        test_set = get_SICE_eval_set(opt.data_val_SICE_mix)
        
    elif opt.dataset == 'SICE_grad':
        train_set = get_SICE_training_set(opt.data_train_SICE,size=opt.cropSize)
        test_set = get_SICE_eval_set(opt.data_val_SICE_grad)
        
    elif opt.dataset == 'fivek':
        train_set = get_fivek_training_set(opt.data_train_fivek,size=opt.cropSize)
        test_set = get_fivek_eval_set(opt.data_test_fivek)
        
    elif opt.dataset == 'sdsd_indoor':
        train_set = get_sdsd_indoor_training_set(opt.data_train_sdsd_indoor, size=opt.cropSize)
        test_set = get_sdsd_indoor_eval_set(opt.data_val_sdsd_indoor)
        
    elif opt.dataset == 'sdsd_outdoor':
        train_set = get_sdsd_outdoor_training_set(opt.data_train_sdsd_outdoor, size=opt.cropSize)
        test_set = get_sdsd_outdoor_eval_set(opt.data_val_sdsd_outdoor)
        
    elif opt.dataset == 'smid':
        train_set = get_smid_training_set(opt.data_train_smid, size=opt.cropSize)
        test_set = get_smid_eval_set(opt.data_val_smid)
        
    elif opt.dataset == 'lsrw':
        train_set = get_lsrw_training_set(opt.data_train_lsrw, size=opt.cropSize)
        test_set = get_lsrw_eval_set(opt.data_val_lsrw)
        
    elif opt.dataset == 'lsrw_huawei':
        train_set = get_lsrw_huawei_training_set(opt.data_train_lsrw, size=opt.cropSize)
        test_set = get_lsrw_huawei_eval_set(opt.data_val_lsrw)
        
    elif opt.dataset == 'lsrw_nikon':
        train_set = get_lsrw_nikon_training_set(opt.data_train_lsrw, size=opt.cropSize)
        test_set = get_lsrw_nikon_eval_set(opt.data_val_lsrw)
    else:
        raise Exception("should choose a dataset")
    
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
    testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
    return training_data_loader, testing_data_loader

def build_model():
    print('===> Building model ')
    # 创建模型实例，传入配置参数
    lfpv_num_feature = getattr(opt, 'lfpv_num_feature', 16)
    lfpv_patch_size = getattr(opt, 'lfpv_patch_size', 4)
    atom_num = getattr(opt, 'atom_num', 32)
    atom_dim = getattr(opt, 'atom_dim', 256)
    
    # LFPV HVI空间优化选项
    lfpv_intensity_adaptive = getattr(opt, 'lfpv_intensity_adaptive', False)
    lfpv_intensity_guide = getattr(opt, 'lfpv_intensity_guide', False)
    lfpv_branch_aware = getattr(opt, 'lfpv_branch_aware', False)
    use_lfpv = getattr(opt, 'use_lfpv', True)
    
    # RIN-PAB选项
    use_rin_pab = getattr(opt, 'use_rin_pab', True)
    
    model = CIDNet(
        lfpv_num_feature=lfpv_num_feature, 
        lfpv_patch_size=lfpv_patch_size,
        atom_num=atom_num,
        atom_dim=atom_dim,
        lfpv_intensity_adaptive=lfpv_intensity_adaptive,
        lfpv_intensity_guide=lfpv_intensity_guide,
        lfpv_branch_aware=lfpv_branch_aware,
        use_lfpv=use_lfpv,
        use_rin_pab=use_rin_pab
    )
    if use_rin_pab:
        print(f'===> RIN-PAB enabled: atom_num={atom_num}, atom_dim={atom_dim}')
    else:
        print(f'===> RIN-PAB disabled: using original LCA in HV encoder')
    if use_lfpv:
        print(f'===> LFPV enabled: intensity_adaptive={lfpv_intensity_adaptive}, '
              f'intensity_guide={lfpv_intensity_guide}, branch_aware={lfpv_branch_aware}')
    else:
        print(f'===> LFPV disabled')
    
    # 支持多GPU训练
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f'===> Using {torch.cuda.device_count()} GPUs for training')
        model = nn.DataParallel(model)
    
    model = model.cuda()
    
    if opt.start_epoch > 0:
        # 从对应数据集和实验名称的目录加载模型
        if opt.exp_name:
            dataset_dir = os.path.join("./weights", f"{opt.dataset}_{opt.exp_name}")
        else:
            dataset_dir = os.path.join("./weights", opt.dataset)
        pth = os.path.join(dataset_dir, f"epoch_{opt.start_epoch}.pth")
        if not os.path.exists(pth):
            # 如果新路径不存在，尝试旧路径（向后兼容）
            old_dataset_dir = os.path.join("./weights", opt.dataset)
            old_pth = os.path.join(old_dataset_dir, f"epoch_{opt.start_epoch}.pth")
            if os.path.exists(old_pth):
                print(f"===> Warning: Using old path {old_pth}, consider migrating to {pth}")
                pth = old_pth
            else:
                # 尝试更旧的路径
                very_old_pth = f"./weights/train/epoch_{opt.start_epoch}.pth"
                if os.path.exists(very_old_pth):
                    print(f"===> Warning: Using very old path {very_old_pth}, consider migrating to {pth}")
                    pth = very_old_pth
                else:
                    raise FileNotFoundError(f"Checkpoint not found: {pth}")
        state_dict = torch.load(pth, map_location=lambda storage, loc: storage)
        # 处理DataParallel的state_dict键名（如果模型使用了DataParallel）
        if isinstance(model, nn.DataParallel):
            # 如果保存的模型是DataParallel，需要去掉'module.'前缀
            if any(k.startswith('module.') for k in state_dict.keys()):
                # 保存的模型已经是DataParallel格式，直接加载
                model.load_state_dict(state_dict, strict=False)
            else:
                # 保存的模型不是DataParallel格式，需要添加'module.'前缀
                new_state_dict = {}
                for k, v in state_dict.items():
                    new_state_dict['module.' + k] = v
                model.load_state_dict(new_state_dict, strict=False)
        else:
            # 如果当前模型不是DataParallel，但保存的模型是，需要去掉'module.'前缀
            if any(k.startswith('module.') for k in state_dict.keys()):
                new_state_dict = {}
                for k, v in state_dict.items():
                    new_state_dict[k.replace('module.', '')] = v
                model.load_state_dict(new_state_dict, strict=False)
            else:
                model.load_state_dict(state_dict, strict=False)
        print(f'===> Loaded checkpoint from epoch {opt.start_epoch} (strict=False, allowing missing/unexpected keys)')
    return model

def make_scheduler():
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)      
    if opt.cos_restart_cyclic:
        if opt.start_warmup:
            scheduler_step = CosineAnnealingRestartCyclicLR(optimizer=optimizer, periods=[(opt.nEpochs//4)-opt.warmup_epochs, (opt.nEpochs*3)//4], restart_weights=[1,1],eta_mins=[0.0002,0.0000001])
            scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=opt.warmup_epochs, after_scheduler=scheduler_step)
        else:
            scheduler = CosineAnnealingRestartCyclicLR(optimizer=optimizer, periods=[opt.nEpochs//4, (opt.nEpochs*3)//4], restart_weights=[1,1],eta_mins=[0.0002,0.0000001])
    elif opt.cos_restart:
        if opt.start_warmup:
            scheduler_step = CosineAnnealingRestartLR(optimizer=optimizer, periods=[opt.nEpochs - opt.warmup_epochs - opt.start_epoch], restart_weights=[1],eta_min=1e-7)
            scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=opt.warmup_epochs, after_scheduler=scheduler_step)
        else:
            scheduler = CosineAnnealingRestartLR(optimizer=optimizer, periods=[opt.nEpochs - opt.start_epoch], restart_weights=[1],eta_min=1e-7)
    else:
        raise Exception("should choose a scheduler")
    return optimizer,scheduler

def init_loss():
    L1_weight   = opt.L1_weight
    D_weight    = opt.D_weight 
    E_weight    = opt.E_weight 
    P_weight    = 1.0
    
    L1_loss= L1Loss(loss_weight=L1_weight, reduction='mean').cuda()
    D_loss = SSIM(weight=D_weight).cuda()
    E_loss = EdgeLoss(loss_weight=E_weight).cuda()
    P_loss = PerceptualLoss({'conv1_2': 1, 'conv2_2': 1,'conv3_4': 1,'conv4_4': 1}, perceptual_weight = P_weight ,criterion='mse').cuda()
    return L1_loss,P_loss,E_loss,D_loss

if __name__ == '__main__':  
    
    '''
    preparision
    '''
    train_init()
    training_data_loader, testing_data_loader = load_datasets()
    model = build_model()
    optimizer,scheduler = make_scheduler()
    L1_loss,P_loss,E_loss,D_loss = init_loss()
    
    '''
    train
    '''
    psnr = []
    ssim = []
    lpips = []
    eval_epochs = []  # 记录每次评估的epoch
    start_epoch=0
    if opt.start_epoch > 0:
        start_epoch = opt.start_epoch
    
    # 只有从第0轮开始训练时才重置最佳结果记录
    if start_epoch == 0:
        reset_best_results(opt.dataset)
    else:
        print(f"===> Starting from epoch {start_epoch}, keeping existing best results")
    
    # 初始化评估指标日志文件（实时保存）
    metrics_file = init_metrics_log(opt.dataset, start_epoch, opt)
    
    if not os.path.exists(opt.val_folder):          
        os.mkdir(opt.val_folder)
        
    for epoch in range(start_epoch+1, opt.nEpochs + start_epoch + 1):
        epoch_loss, pic_num = train(epoch)
        scheduler.step()
        
        if epoch % opt.snapshots == 0:
            model_out_path = checkpoint(epoch) 
            norm_size = True
            img_ext = '*.png'  # 默认使用png格式 (LOL系列数据集)

            # LOL three subsets
            if opt.dataset == 'lol_v1':
                output_folder = 'LOLv1/'
                label_dir = opt.data_valgt_lol_v1
            if opt.dataset == 'lolv2_real':
                output_folder = 'LOLv2_real/'
                label_dir = opt.data_valgt_lolv2_real
            if opt.dataset == 'lolv2_syn':
                output_folder = 'LOLv2_syn/'
                label_dir = opt.data_valgt_lolv2_syn
            
            # LOL-blur dataset with low_blur and high_sharp_scaled
            if opt.dataset == 'lol_blur':
                output_folder = 'LOL_blur/'
                label_dir = opt.data_valgt_lol_blur
                
            if opt.dataset == 'SID':
                output_folder = 'SID/'
                label_dir = opt.data_valgt_SID
                npy = True
            if opt.dataset == 'SICE_mix':
                output_folder = 'SICE_mix/'
                label_dir = opt.data_valgt_SICE_mix
                norm_size = False
                img_ext = '*.png'  # SICE标准测试集(SICE_Mix)使用png格式
            if opt.dataset == 'SICE_grad':
                output_folder = 'SICE_grad/'
                label_dir = opt.data_valgt_SICE_grad
                norm_size = False
                img_ext = '*.png'  # SICE标准测试集(SICE_Grad)使用png格式
                
            if opt.dataset == 'fivek':
                output_folder = 'fivek/'
                label_dir = opt.data_valgt_fivek
                norm_size = False
                img_ext = '*.jpg'  # fivek数据集使用jpg格式
                
            if opt.dataset == 'sdsd_indoor':
                output_folder = 'SDSD_indoor/'
                label_dir = opt.data_valgt_sdsd_indoor
                norm_size = False
                
            if opt.dataset == 'sdsd_outdoor':
                output_folder = 'SDSD_outdoor/'
                label_dir = opt.data_valgt_sdsd_outdoor
                norm_size = False
                
            if opt.dataset == 'smid':
                output_folder = 'SMID/'
                label_dir = opt.data_valgt_smid
                norm_size = False
                
            if opt.dataset == 'lsrw':
                output_folder = 'LSRW/'
                label_dir = opt.data_valgt_lsrw
                norm_size = False
                img_ext = '*.jpg'  # LSRW数据集使用jpg格式
                
            if opt.dataset == 'lsrw_huawei':
                output_folder = 'LSRW_Huawei/'
                label_dir = './datasets/LSRW/Eval/Huawei/high/'  # Huawei的GT目录
                norm_size = False
                img_ext = '*.jpg'  # LSRW数据集使用jpg格式
                
            if opt.dataset == 'lsrw_nikon':
                output_folder = 'LSRW_Nikon/'
                label_dir = './datasets/LSRW/Eval/Nikon/high/'  # Nikon的GT目录
                norm_size = False
                img_ext = '*.jpg'  # LSRW数据集使用jpg格式

            # 如果指定了实验名称，将其加入到输出文件夹路径中
            if opt.exp_name:
                # 在output_folder的末尾（在斜杠之前）添加exp_name
                if output_folder.endswith('/'):
                    output_folder = output_folder[:-1] + f'_{opt.exp_name}/'
                else:
                    output_folder = output_folder + f'_{opt.exp_name}/'

            im_dir = opt.val_folder + output_folder + img_ext
            is_lol_v1 = (opt.dataset == 'lol_v1')
            is_lolv2_real = (opt.dataset == 'lolv2_real')
            eval(model, testing_data_loader, model_out_path, opt.val_folder+output_folder, 
                 norm_size=norm_size, LOL=is_lol_v1, v2=is_lolv2_real, alpha=0.8)
            
            # 计算普通模式的指标 (use_GT_mean=False)
            avg_psnr, avg_ssim, avg_lpips = metrics(im_dir, label_dir, use_GT_mean=False)
            print("===> [Normal Mode] Avg.PSNR: {:.4f} dB ".format(avg_psnr))
            print("===> [Normal Mode] Avg.SSIM: {:.4f} ".format(avg_ssim))
            print("===> [Normal Mode] Avg.LPIPS: {:.4f} ".format(avg_lpips))
            
            # 根据选项决定是否计算GT Mean模式的指标
            avg_psnr_gtmean = None
            avg_ssim_gtmean = None
            avg_lpips_gtmean = None
            if opt.eval_gtmean:
                avg_psnr_gtmean, avg_ssim_gtmean, avg_lpips_gtmean = metrics(im_dir, label_dir, use_GT_mean=True)
                print("===> [GT Mean Mode] Avg.PSNR: {:.4f} dB ".format(avg_psnr_gtmean))
                print("===> [GT Mean Mode] Avg.SSIM: {:.4f} ".format(avg_ssim_gtmean))
                print("===> [GT Mean Mode] Avg.LPIPS: {:.4f} ".format(avg_lpips_gtmean))
            else:
                print("===> [GT Mean Mode] Skipped (--eval_gtmean=False)")
            
            psnr.append(avg_psnr)
            ssim.append(avg_ssim)
            lpips.append(avg_lpips)
            eval_epochs.append(epoch)
            
            # 实时保存评估结果到日志文件（根据选项决定是否包括GT Mean模式）
            write_evaluation_log(metrics_file, epoch, avg_psnr, avg_ssim, avg_lpips,
                               avg_psnr_gtmean, avg_ssim_gtmean, avg_lpips_gtmean)
            
            # 更新并保存最佳结果（根据选项决定是否包括GT Mean模式）
            best_results = update_best_results(opt.dataset, epoch, avg_psnr, avg_ssim, avg_lpips, model_out_path,
                                              avg_psnr_gtmean, avg_ssim_gtmean, avg_lpips_gtmean)
            
            # 绘制并保存评估指标折线图
            plot_evaluation_metrics(opt.dataset, eval_epochs, psnr, ssim, lpips)
            
            print(psnr)
            print(ssim)
            print(lpips)
        torch.cuda.empty_cache()
    
    print(f"===> Metrics log saved: {metrics_file}")  
        