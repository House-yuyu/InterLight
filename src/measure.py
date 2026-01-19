import os
# GPU设置由环境变量或命令行参数控制，不在这里硬编码
import torch
import glob
import cv2
import lpips
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
import platform



def ssim(prediction, target):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    img1 = prediction.astype(np.float64)
    img2 = target.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5] 
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(target, ref):
    '''
    calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    img1 = np.array(target, dtype=np.float64)
    img2 = np.array(ref, dtype=np.float64)
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def calculate_psnr(target, ref):
    img1 = np.array(target, dtype=np.float32)
    img2 = np.array(ref, dtype=np.float32)
    diff = img1 - img2
    psnr = 10.0 * np.log10(255.0 * 255.0 / (np.mean(np.square(diff)) + 1e-8))
    return psnr

def metrics(im_dir, label_dir, use_GT_mean):
    avg_psnr = 0
    avg_ssim = 0
    avg_lpips = 0
    n = 0
    skipped_count = 0
    skipped_files = []
    # loss_fn = lpips.LPIPS(net='alex')
    # loss_fn.cuda()
    
    # 检查输入目录是否存在
    im_files = sorted(glob.glob(im_dir))
    if len(im_files) == 0:
        raise ValueError(f"未找到任何图片文件！请检查路径: {im_dir}\n"
                        f"提示：请确保已经运行评估脚本生成了输出图片，或者检查输出目录是否存在。")
    
    # 检查标签目录是否存在
    if not os.path.exists(label_dir):
        raise ValueError(f"标签目录不存在！请检查路径: {label_dir}")
    
    print(f"===> 找到 {len(im_files)} 张输出图片")
    print(f"===> GT目录: {label_dir}")
    
    for item in tqdm(im_files, desc="计算指标"):
        im1 = Image.open(item).convert('RGB') 
        
        os_name = platform.system()
        if os_name.lower() == 'windows':
            name = item.split('\\')[-1]
        elif os_name.lower() == 'linux':
            name = item.split('/')[-1]
        else:
            name = item.split('/')[-1]
            
        label_path = label_dir + name
        if not os.path.exists(label_path):
            # 尝试不同的扩展名大小写组合 (处理 .jpg vs .JPG 等情况)
            name_base, name_ext = os.path.splitext(name)
            found = False
            for ext_variant in [name_ext.lower(), name_ext.upper(), name_ext.capitalize()]:
                alt_label_path = label_dir + name_base + ext_variant
                if os.path.exists(alt_label_path):
                    label_path = alt_label_path
                    found = True
                    break
            if not found:
                # GT文件不存在，跳过该图片
                skipped_count += 1
                skipped_files.append(name)
                continue
            
        im2 = Image.open(label_path).convert('RGB')
        (h, w) = im2.size
        im1 = im1.resize((h, w))  
        im1 = np.array(im1) 
        im2 = np.array(im2)
        
        if use_GT_mean:
            mean_restored = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY).mean()
            mean_target = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY).mean()
            im1 = np.clip(im1 * (mean_target/mean_restored), 0, 255)
        
        score_psnr = calculate_psnr(im1, im2)
        score_ssim = calculate_ssim(im1, im2)
        # ex_p0 = lpips.im2tensor(im1).cuda()
        # ex_ref = lpips.im2tensor(im2).cuda()
        

        # score_lpips = loss_fn.forward(ex_ref, ex_p0)
    
        avg_psnr += score_psnr
        avg_ssim += score_ssim
        # avg_lpips += score_lpips.item()
        n += 1
        torch.cuda.empty_cache()
    
    if n == 0:
        raise ValueError(f"没有有效的图片对可以计算指标！\n"
                        f"  输入图片数: {len(im_files)}\n"
                        f"  跳过的图片数: {skipped_count}\n"
                        f"  请检查输入图片和标签文件是否匹配。")
    
    # 打印统计信息
    if skipped_count > 0:
        print(f"\n===> 统计信息:")
        print(f"  总输出图片数: {len(im_files)}")
        print(f"  有效图片对数: {n}")
        print(f"  跳过图片数: {skipped_count} (无对应GT)")
        if len(skipped_files) <= 10:
            print(f"  跳过的文件: {', '.join(skipped_files)}")
        else:
            print(f"  跳过的文件 (前10个): {', '.join(skipped_files[:10])} ...")
        print(f"  计算指标时只使用有GT的 {n} 张图片\n")

    avg_psnr = avg_psnr / n
    avg_ssim = avg_ssim / n
    avg_lpips = avg_lpips / n
    return avg_psnr, avg_ssim, avg_lpips


if __name__ == '__main__':
    
    mea_parser = argparse.ArgumentParser(description='Measure')
    mea_parser.add_argument('--use_GT_mean', action='store_true', help='Use the mean of GT to rectify the output of the model')
    mea_parser.add_argument('--lol', action='store_true', help='measure lolv1 dataset')
    mea_parser.add_argument('--lol_v2_real', action='store_true', help='measure lol_v2_real dataset')
    mea_parser.add_argument('--lol_v2_syn', action='store_true', help='measure lol_v2_syn dataset')
    mea_parser.add_argument('--SICE_grad', action='store_true', help='measure SICE_grad dataset')
    mea_parser.add_argument('--SICE_mix', action='store_true', help='measure SICE_mix dataset')
    mea_parser.add_argument('--fivek', action='store_true', help='measure fivek dataset')
    mea_parser.add_argument('--lsrw', action='store_true', help='measure LSRW dataset (Huawei+Nikon)')
    mea_parser.add_argument('--lsrw_huawei', action='store_true', help='measure LSRW Huawei dataset')
    mea_parser.add_argument('--lsrw_nikon', action='store_true', help='measure LSRW Nikon dataset')
    mea_parser.add_argument('--sdsd_indoor', action='store_true', help='measure SDSD indoor dataset')
    mea_parser.add_argument('--sdsd_outdoor', action='store_true', help='measure SDSD outdoor dataset')
    mea_parser.add_argument('--smid', action='store_true', help='measure SMID dataset')
    mea = mea_parser.parse_args()

    if mea.lol:
        # 优先尝试 output 目录，如果不存在则尝试 results 目录
        if os.path.exists('./output/LOLv1/'):
            im_dir = './output/LOLv1/*.png'
        elif os.path.exists('./results/LOLv1/'):
            im_dir = './results/LOLv1/*.png'
            print("使用 results/LOLv1/ 目录（训练过程中的评估结果）")
        else:
            im_dir = './output/LOLv1/*.png'  # 默认路径，会在metrics函数中报错
        label_dir = './datasets/LOLdataset/eval15/high/'
    if mea.lol_v2_real:
        if os.path.exists('./output/LOLv2_real/'):
            im_dir = './output/LOLv2_real/*.png'
        elif os.path.exists('./results/LOLv2_real/'):
            im_dir = './results/LOLv2_real/*.png'
            print("使用 results/LOLv2_real/ 目录（训练过程中的评估结果）")
        else:
            im_dir = './output/LOLv2_real/*.png'
        label_dir = './datasets/LOLv2/Real_captured/Test/Normal/'
    if mea.lol_v2_syn:
        if os.path.exists('./output/LOLv2_syn/'):
            im_dir = './output/LOLv2_syn/*.png'
        elif os.path.exists('./results/LOLv2_syn/'):
            im_dir = './results/LOLv2_syn/*.png'
            print("使用 results/LOLv2_syn/ 目录（训练过程中的评估结果）")
        else:
            im_dir = './output/LOLv2_syn/*.png'
        label_dir = './datasets/LOLv2/Synthetic/Test/Normal/'
    if mea.SICE_grad:
        if os.path.exists('./output/SICE_grad/'):
            im_dir = './output/SICE_grad/*.png'
        elif os.path.exists('./results/SICE_grad/'):
            im_dir = './results/SICE_grad/*.png'
            print("使用 results/SICE_grad/ 目录（训练过程中的评估结果）")
        else:
            im_dir = './output/SICE_grad/*.png'
        label_dir = './datasets/SICE/SICE_Reshape/'
    if mea.SICE_mix:
        if os.path.exists('./output/SICE_mix/'):
            im_dir = './output/SICE_mix/*.png'
        elif os.path.exists('./results/SICE_mix/'):
            im_dir = './results/SICE_mix/*.png'
            print("使用 results/SICE_mix/ 目录（训练过程中的评估结果）")
        else:
            im_dir = './output/SICE_mix/*.png'
        label_dir = './datasets/SICE/SICE_Reshape/'
    if mea.fivek:
        if os.path.exists('./output/fivek/'):
            im_dir = './output/fivek/*.jpg'
        elif os.path.exists('./results/fivek/'):
            im_dir = './results/fivek/*.jpg'
            print("使用 results/fivek/ 目录（训练过程中的评估结果）")
        else:
            im_dir = './output/fivek/*.jpg'
        label_dir = './datasets/FiveK/test/target/'
        
    if mea.lsrw:
        if os.path.exists('./output/LSRW/'):
            im_dir = './output/LSRW/*.jpg'
        elif os.path.exists('./results/LSRW/'):
            im_dir = './results/LSRW/*.jpg'
            print("使用 results/LSRW/ 目录（训练过程中的评估结果）")
        else:
            im_dir = './output/LSRW/*.jpg'
        label_dir = './datasets/LSRW/Eval/'
        
    if mea.lsrw_huawei:
        if os.path.exists('./output/LSRW_Huawei/'):
            im_dir = './output/LSRW_Huawei/*.jpg'
        elif os.path.exists('./results/LSRW_Huawei/'):
            im_dir = './results/LSRW_Huawei/*.jpg'
            print("使用 results/LSRW_Huawei/ 目录（训练过程中的评估结果）")
        else:
            im_dir = './output/LSRW_Huawei/*.jpg'
        label_dir = './datasets/LSRW/Eval/Huawei/high/'
        
    if mea.lsrw_nikon:
        if os.path.exists('./output/LSRW_Nikon/'):
            im_dir = './output/LSRW_Nikon/*.jpg'
        elif os.path.exists('./results/LSRW_Nikon/'):
            im_dir = './results/LSRW_Nikon/*.jpg'
            print("使用 results/LSRW_Nikon/ 目录（训练过程中的评估结果）")
        else:
            im_dir = './output/LSRW_Nikon/*.jpg'
        label_dir = './datasets/LSRW/Eval/Nikon/high/'
        
    if mea.sdsd_indoor:
        if os.path.exists('./output/SDSD_indoor/'):
            im_dir = './output/SDSD_indoor/*.png'
        elif os.path.exists('./results/SDSD_indoor/'):
            im_dir = './results/SDSD_indoor/*.png'
            print("使用 results/SDSD_indoor/ 目录（训练过程中的评估结果）")
        else:
            im_dir = './output/SDSD_indoor/*.png'
        label_dir = './datasets/SDSD_indoor_png/eval/high/'
        
    if mea.sdsd_outdoor:
        if os.path.exists('./output/SDSD_outdoor/'):
            im_dir = './output/SDSD_outdoor/*.png'
        elif os.path.exists('./results/SDSD_outdoor/'):
            im_dir = './results/SDSD_outdoor/*.png'
            print("使用 results/SDSD_outdoor/ 目录（训练过程中的评估结果）")
        else:
            im_dir = './output/SDSD_outdoor/*.png'
        label_dir = './datasets/SDSD_outdoor_png/eval/high/'
        
    if mea.smid:
        if os.path.exists('./output/SMID/'):
            im_dir = './output/SMID/*.png'
        elif os.path.exists('./results/SMID/'):
            im_dir = './results/SMID/*.png'
            print("使用 results/SMID/ 目录（训练过程中的评估结果）")
        else:
            im_dir = './output/SMID/*.png'
        label_dir = './datasets/SMID_png/eval/high/'

    avg_psnr, avg_ssim, avg_lpips = metrics(im_dir, label_dir, mea.use_GT_mean)
    print("===> Avg.PSNR: {:.4f} dB ".format(avg_psnr))
    print("===> Avg.SSIM: {:.4f} ".format(avg_ssim))
    print("===> Avg.LPIPS: {:.4f} ".format(avg_lpips))
