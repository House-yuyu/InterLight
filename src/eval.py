import os
import subprocess

# 自动选择空闲GPU
def auto_select_gpu():
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
            encoding='utf-8'
        )
        gpu_info = []
        for line in result.strip().split('\n'):
            parts = line.split(',')
            gpu_id = int(parts[0].strip())
            mem_free = int(parts[1].strip())
            gpu_info.append((gpu_id, mem_free))
        
        # 选择空闲显存最多的GPU
        best_gpu = max(gpu_info, key=lambda x: x[1])
        print(f"===> Auto-selected GPU {best_gpu[0]} with {best_gpu[1]} MB free memory")
        return str(best_gpu[0])
    except:
        return '0'

# 在导入torch之前设置GPU（仅当直接运行eval.py时）
# 注意：当从train.py导入时，不应覆盖已设置的CUDA_VISIBLE_DEVICES
if __name__ == '__main__' or 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = auto_select_gpu()

# GPU设置由环境变量或命令行参数控制，不在这里硬编码
import argparse
import torch.nn as nn
from tqdm import tqdm
from data.data import *
from torchvision import transforms
from torch.utils.data import DataLoader
from loss.losses import *
from net.CIDNet import CIDNet

def get_model_module(model):
    """获取实际的模型对象（兼容DataParallel）"""
    if isinstance(model, nn.DataParallel):
        return model.module
    return model


def eval(model, testing_data_loader, model_path, output_folder,norm_size=True,LOL=False,v2=False,unpaired=False,alpha=1.0,gamma=1.0):
    torch.set_grad_enabled(False)
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    
    # 处理模型加载：兼容DataParallel和普通模型格式
    # train.py保存时已经去掉了module.前缀，但需要兼容旧格式
    if isinstance(model, nn.DataParallel):
        # 如果eval时使用DataParallel，需要处理键名
        if any(k.startswith('module.') for k in state_dict.keys()):
            # 保存的模型有module.前缀，直接加载
            model.load_state_dict(state_dict, strict=False)
        else:
            # 保存的模型没有module.前缀，需要添加
            new_state_dict = {}
            for k, v in state_dict.items():
                new_state_dict['module.' + k] = v
            model.load_state_dict(new_state_dict, strict=False)
    else:
        # eval时使用普通模型
        if any(k.startswith('module.') for k in state_dict.keys()):
            # 保存的模型有module.前缀，需要去掉
            new_state_dict = {}
            for k, v in state_dict.items():
                new_state_dict[k.replace('module.', '')] = v
            model.load_state_dict(new_state_dict, strict=False)
        else:
            # 保存的模型没有module.前缀，直接加载
            model.load_state_dict(state_dict, strict=False)
    
    print('Pre-trained model is loaded.')
    model.eval()
    print('Evaluation:')
    # 兼容DataParallel：访问子模块需要使用module属性
    model_module = get_model_module(model)
    if LOL:
        model_module.trans.gated = True
    elif v2:
        model_module.trans.gated2 = True
        model_module.trans.alpha = alpha
    elif unpaired:
        model_module.trans.gated2 = True
        model_module.trans.alpha = alpha
    for batch in tqdm(testing_data_loader):
        with torch.no_grad():
            if norm_size:
                input, name = batch[0], batch[1]
            else:
                input, name, h, w = batch[0], batch[1], batch[2], batch[3]
            
            input = input.cuda()
            model_output = model(input**gamma)
            # 处理可能的双路径输出（训练时）或单输出（推理时）
            output = model_output if not isinstance(model_output, tuple) else model_output[0] 
            
        if not os.path.exists(output_folder):          
            os.mkdir(output_folder)  
            
        output = torch.clamp(output.cuda(),0,1).cuda()
        if not norm_size:
            output = output[:, :, :h, :w]
        
        output_img = transforms.ToPILImage()(output.squeeze(0))
        output_img.save(output_folder + name[0])
        torch.cuda.empty_cache()
    print('===> End evaluation')
    if LOL:
        model_module.trans.gated = False
    elif v2:
        model_module.trans.gated2 = False
    torch.set_grad_enabled(True)
    
if __name__ == '__main__':
    
    eval_parser = argparse.ArgumentParser(description='Eval')
    eval_parser.add_argument('--perc', action='store_true', help='trained with perceptual loss')
    eval_parser.add_argument('--lol', action='store_true', help='output lolv1 dataset')
    eval_parser.add_argument('--lol_v2_real', action='store_true', help='output lol_v2_real dataset')
    eval_parser.add_argument('--lol_v2_syn', action='store_true', help='output lol_v2_syn dataset')
    eval_parser.add_argument('--SICE_grad', action='store_true', help='output SICE_grad dataset')
    eval_parser.add_argument('--SICE_mix', action='store_true', help='output SICE_mix dataset')
    eval_parser.add_argument('--fivek', action='store_true', help='output FiveK dataset')
    eval_parser.add_argument('--lsrw', action='store_true', help='output LSRW dataset (Huawei+Nikon)')
    eval_parser.add_argument('--lsrw_huawei', action='store_true', help='output LSRW Huawei dataset')
    eval_parser.add_argument('--lsrw_nikon', action='store_true', help='output LSRW Nikon dataset')
    eval_parser.add_argument('--sdsd_indoor', action='store_true', help='output SDSD indoor dataset')
    eval_parser.add_argument('--sdsd_outdoor', action='store_true', help='output SDSD outdoor dataset')
    eval_parser.add_argument('--smid', action='store_true', help='output SMID dataset')

    eval_parser.add_argument('--best_GT_mean', action='store_true', help='output lol_v2_real dataset best_GT_mean')
    eval_parser.add_argument('--best_PSNR', action='store_true', help='output lol_v2_real dataset best_PSNR')
    eval_parser.add_argument('--best_SSIM', action='store_true', help='output lol_v2_real dataset best_SSIM')

    eval_parser.add_argument('--custome', action='store_true', help='output custome dataset')
    eval_parser.add_argument('--custome_path', type=str, default='./YOLO')
    eval_parser.add_argument('--unpaired', action='store_true', help='output unpaired dataset')
    eval_parser.add_argument('--DICM', action='store_true', help='output DICM dataset')
    eval_parser.add_argument('--LIME', action='store_true', help='output LIME dataset')
    eval_parser.add_argument('--MEF', action='store_true', help='output MEF dataset')
    eval_parser.add_argument('--NPE', action='store_true', help='output NPE dataset')
    eval_parser.add_argument('--VV', action='store_true', help='output VV dataset')
    eval_parser.add_argument('--alpha', type=float, default=1.0)
    eval_parser.add_argument('--gamma', type=float, default=1.0)
    eval_parser.add_argument('--unpaired_weights', type=str, default='./weights/LOLv2_syn/w_perc.pth')

    ep = eval_parser.parse_args()


    cuda = True
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, or need to change CUDA_VISIBLE_DEVICES number")
    
    if not os.path.exists('./output'):          
            os.mkdir('./output')  
    
    norm_size = True
    num_workers = 1
    alpha = None
    if ep.lol:
        eval_data = DataLoader(dataset=get_eval_set("./datasets/LOLdataset/eval15/low"), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = './output/LOLv1/'
        if ep.perc:
            weight_path = './weights/LOLv1/w_perc.pth'
        else:
            weight_path = './weights/LOLv1/wo_perc.pth'
        
            
    elif ep.lol_v2_real:
        eval_data = DataLoader(dataset=get_eval_set("./datasets/LOLv2/Real_captured/Test/Low"), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = './output/LOLv2_real/'
        if ep.best_GT_mean:
            weight_path = './weights/LOLv2_real/w_perc.pth'
            alpha = 0.84
        elif ep.best_PSNR:
            weight_path = './weights/LOLv2_real/best_PSNR.pth'
            alpha = 0.8
        elif ep.best_SSIM:
            weight_path = './weights/LOLv2_real/best_SSIM.pth'
            alpha = 0.82
            
    elif ep.lol_v2_syn:
        eval_data = DataLoader(dataset=get_eval_set("./datasets/LOLv2/Synthetic/Test/Low"), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = './output/LOLv2_syn/'
        if ep.perc:
            weight_path = './weights/LOLv2_syn/w_perc.pth'
        else:
            weight_path = './weights/LOLv2_syn/wo_perc.pth'
            
    elif ep.SICE_grad:
        eval_data = DataLoader(dataset=get_SICE_eval_set("./datasets/SICE/SICE_Grad"), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = './output/SICE_grad/'
        weight_path = './weights/SICE_grad/best_psnr.pth'
        norm_size = False
        
    elif ep.SICE_mix:
        eval_data = DataLoader(dataset=get_SICE_eval_set("./datasets/SICE/SICE_Mix"), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = './output/SICE_mix/'
        weight_path = './weights/SICE_mix/best_psnr.pth'
        norm_size = False
        
    elif ep.fivek:
        eval_data = DataLoader(dataset=get_SICE_eval_set("./datasets/FiveK/test/input"), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = './output/fivek/'
        weight_path = './weights/fivek.pth'
        norm_size = False
    
    elif ep.lsrw:
        eval_data = DataLoader(dataset=get_lsrw_eval_set("./datasets/LSRW/Eval"), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = './output/LSRW/'
        weight_path = './weights/lsrw/best_psnr.pth'
        norm_size = False
        
    elif ep.lsrw_huawei:
        eval_data = DataLoader(dataset=get_lsrw_huawei_eval_set("./datasets/LSRW/Eval"), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = './output/LSRW_Huawei/'
        weight_path = './weights/lsrw_huawei/best_psnr.pth'
        norm_size = False
        
    elif ep.lsrw_nikon:
        eval_data = DataLoader(dataset=get_lsrw_nikon_eval_set("./datasets/LSRW/Eval"), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = './output/LSRW_Nikon/'
        weight_path = './weights/lsrw_nikon/best_psnr.pth'
        norm_size = False
        
    elif ep.sdsd_indoor:
        eval_data = DataLoader(dataset=get_SICE_eval_set("./datasets/SDSD_indoor_png/eval/low"), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = './output/SDSD_indoor/'
        weight_path = './weights/sdsd_indoor/best_psnr.pth'
        norm_size = False
        
    elif ep.sdsd_outdoor:
        eval_data = DataLoader(dataset=get_SICE_eval_set("./datasets/SDSD_outdoor_png/eval/low"), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = './output/SDSD_outdoor/'
        weight_path = './weights/sdsd_outdoor/best_psnr.pth'
        norm_size = False
        
    elif ep.smid:
        eval_data = DataLoader(dataset=get_SICE_eval_set("./datasets/SMID_png/eval/low"), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = './output/SMID/'
        weight_path = './weights/smid/best_psnr.pth'
        norm_size = False
    
    elif ep.unpaired: 
        if ep.DICM:
            eval_data = DataLoader(dataset=get_SICE_eval_set("./datasets/DICM"), num_workers=num_workers, batch_size=1, shuffle=False)
            output_folder = './output/DICM/'
        elif ep.LIME:
            eval_data = DataLoader(dataset=get_SICE_eval_set("./datasets/LIME"), num_workers=num_workers, batch_size=1, shuffle=False)
            output_folder = './output/LIME/'
        elif ep.MEF:
            eval_data = DataLoader(dataset=get_SICE_eval_set("./datasets/MEF"), num_workers=num_workers, batch_size=1, shuffle=False)
            output_folder = './output/MEF/'
        elif ep.NPE:
            eval_data = DataLoader(dataset=get_SICE_eval_set("./datasets/NPE"), num_workers=num_workers, batch_size=1, shuffle=False)
            output_folder = './output/NPE/'
        elif ep.VV:
            eval_data = DataLoader(dataset=get_SICE_eval_set("./datasets/VV"), num_workers=num_workers, batch_size=1, shuffle=False)
            output_folder = './output/VV/'
        elif ep.custome:
            eval_data = DataLoader(dataset=get_SICE_eval_set(ep.custome_path), num_workers=num_workers, batch_size=1, shuffle=False)
            output_folder = './output/custome/'
        alpha = ep.alpha
        norm_size = False
        weight_path = ep.unpaired_weights
        
    eval_net = CIDNet().cuda()
    eval(eval_net, eval_data, weight_path, output_folder,norm_size=norm_size,LOL=ep.lol,v2=ep.lol_v2_real,unpaired=ep.unpaired,alpha=alpha,gamma=ep.gamma)

