
import os
import random
import torch
import torch.utils.data as data
import numpy as np
from os import listdir
from os.path import join
from data.util import *
from torchvision import transforms as t

    
class LOLDatasetFromFolder(data.Dataset):
    def __init__(self, data_dir, transform=None):
        super(LOLDatasetFromFolder, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.norm = t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __getitem__(self, index):

        folder = self.data_dir+'/low'
        folder2= self.data_dir+'/high'
        data_filenames = [join(folder, x) for x in listdir(folder) if is_image_file(x)]
        data_filenames2 = [join(folder2, x) for x in listdir(folder2) if is_image_file(x)]
        num = len(data_filenames)

        im1 = load_img(data_filenames[index])
        im2 = load_img(data_filenames2[index])
        _, file1 = os.path.split(data_filenames[index])
        _, file2 = os.path.split(data_filenames2[index])
        seed = random.randint(1, 1000000)
        seed = np.random.randint(seed) # make a seed with numpy generator 
        if self.transform:
            random.seed(seed) # apply this seed to img tranfsorms
            torch.manual_seed(seed) # needed for torchvision 0.7
            im1 = self.transform(im1)
            random.seed(seed)
            torch.manual_seed(seed)         
            im2 = self.transform(im2) 
        return im1, im2, file1, file2

    def __len__(self):
        return 485

    
class LOLv2DatasetFromFolder(data.Dataset):
    def __init__(self, data_dir, transform=None):
        super(LOLv2DatasetFromFolder, self).__init__()
        self.data_dir = data_dir
        self.transform = transform

    def __getitem__(self, index):

        folder = self.data_dir+'/Low'
        folder2= self.data_dir+'/Normal'
        data_filenames = [join(folder, x) for x in listdir(folder) if is_image_file(x)]
        data_filenames2 = [join(folder2, x) for x in listdir(folder2) if is_image_file(x)]
        
        im1 = load_img(data_filenames[index])
        im2 = load_img(data_filenames2[index])
        _, file1 = os.path.split(data_filenames[index])
        _, file2 = os.path.split(data_filenames2[index])
        seed = random.randint(1, 1000000)
        seed = np.random.randint(seed) # make a seed with numpy generator 
        if self.transform:
            random.seed(seed) # apply this seed to img tranforms
            torch.manual_seed(seed) # needed for torchvision 0.7
            im1 = self.transform(im1)      
            random.seed(seed) # apply this seed to img tranforms
            torch.manual_seed(seed) # needed for torchvision 0.7 
            im2 = self.transform(im2)
        return im1, im2, file1, file2

    def __len__(self):
        return 685



class LOLv2SynDatasetFromFolder(data.Dataset):
    def __init__(self, data_dir, transform=None):
        super(LOLv2SynDatasetFromFolder, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.norm = t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __getitem__(self, index):

        folder = self.data_dir+'/Low'
        folder2= self.data_dir+'/Normal'
        data_filenames = [join(folder, x) for x in listdir(folder) if is_image_file(x)]
        data_filenames2 = [join(folder2, x) for x in listdir(folder2) if is_image_file(x)]


        im1 = load_img(data_filenames[index])
        im2 = load_img(data_filenames2[index])
        _, file1 = os.path.split(data_filenames[index])
        _, file2 = os.path.split(data_filenames2[index])
        seed = random.randint(1, 1000000)
        seed = np.random.randint(seed) # make a seed with numpy generator 
        if self.transform:
            random.seed(seed) # apply this seed to img tranfsorms
            torch.manual_seed(seed) # needed for torchvision 0.7
            im1 = self.transform(im1)
            random.seed(seed)
            torch.manual_seed(seed)         
            im2 = self.transform(im2)
        return im1, im2, file1, file2

    def __len__(self):
        return 900


class SDSDIndoorDatasetFromFolder(data.Dataset):
    """SDSD Indoor/Outdoor/SMID 数据集加载器
    
    数据集结构:
    - train/low: 低光图片
    - train/high: 正常光图片
    - eval/low: 低光图片
    - eval/high: 正常光图片
    
    支持的数据集:
    - SDSD Indoor: train 1655张, eval 308张
    - SDSD Outdoor: train 2650张, eval 500张
    - SMID: train 15763张, eval 5046张
    """
    def __init__(self, data_dir, transform=None):
        super(SDSDIndoorDatasetFromFolder, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        
        # 预先加载文件列表并排序，确保 low 和 high 对应
        folder_low = join(data_dir, 'low')
        folder_high = join(data_dir, 'high')
        self.data_filenames_low = sorted([join(folder_low, x) for x in listdir(folder_low) if is_image_file(x)])
        self.data_filenames_high = sorted([join(folder_high, x) for x in listdir(folder_high) if is_image_file(x)])
        self.num_samples = len(self.data_filenames_low)

    def __getitem__(self, index):
        im1 = load_img(self.data_filenames_low[index])
        im2 = load_img(self.data_filenames_high[index])
        _, file1 = os.path.split(self.data_filenames_low[index])
        _, file2 = os.path.split(self.data_filenames_high[index])
        
        seed = random.randint(1, 1000000)
        seed = np.random.randint(seed)
        if self.transform:
            random.seed(seed)
            torch.manual_seed(seed)
            im1 = self.transform(im1)
            random.seed(seed)
            torch.manual_seed(seed)         
            im2 = self.transform(im2)
        return im1, im2, file1, file2

    def __len__(self):
        return self.num_samples


class LSRWDatasetFromFolder(data.Dataset):
    """LSRW 数据集加载器 (参考DarkIR实现)
    
    数据集结构 (DarkIR格式):
    - Train/
      - Huawei/
        - low/
        - high/
      - Nikon/
        - low/
        - high/
    - Eval/
      - Huawei/
        - low/
        - high/
      - Nikon/
        - low/
        - high/
    
    或者简化格式:
    - train/
      - low/
      - high/
    - eval/
      - low/
      - high/
    
    参数:
        data_dir: 数据集根目录
        transform: 数据增强变换
        camera: 指定相机类型，可选 'Huawei', 'Nikon', 'all'（默认使用全部）
    """
    def __init__(self, data_dir, transform=None, camera='all'):
        super(LSRWDatasetFromFolder, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        
        self.data_filenames_low = []
        self.data_filenames_high = []
        
        # 确定要加载的相机
        if camera == 'all':
            cameras = ['Huawei', 'Nikon']
        else:
            cameras = [camera]
        
        # 检测数据集结构
        # 首先检查是否是DarkIR的多相机格式
        huawei_low = join(data_dir, 'Huawei', 'low')
        nikon_low = join(data_dir, 'Nikon', 'low')
        simple_low = join(data_dir, 'low')
        
        if os.path.exists(huawei_low) or os.path.exists(nikon_low):
            # DarkIR 多相机格式
            for cam in cameras:
                folder_low = join(data_dir, cam, 'low')
                folder_high = join(data_dir, cam, 'high')
                if os.path.exists(folder_low) and os.path.exists(folder_high):
                    low_files = sorted([join(folder_low, x) for x in listdir(folder_low) if is_image_file(x)])
                    high_files = sorted([join(folder_high, x) for x in listdir(folder_high) if is_image_file(x)])
                    self.data_filenames_low.extend(low_files)
                    self.data_filenames_high.extend(high_files)
        elif os.path.exists(simple_low):
            # 简化格式: train/low, train/high
            folder_low = simple_low
            folder_high = join(data_dir, 'high')
            self.data_filenames_low = sorted([join(folder_low, x) for x in listdir(folder_low) if is_image_file(x)])
            self.data_filenames_high = sorted([join(folder_high, x) for x in listdir(folder_high) if is_image_file(x)])
        else:
            raise ValueError(f"Cannot find LSRW dataset structure in {data_dir}")
        
        self.num_samples = len(self.data_filenames_low)
        camera_info = camera if camera != 'all' else 'Huawei+Nikon'
        print(f"LSRW Dataset ({camera_info}): loaded {self.num_samples} image pairs from {data_dir}")

    def __getitem__(self, index):
        im1 = load_img(self.data_filenames_low[index])
        im2 = load_img(self.data_filenames_high[index])
        _, file1 = os.path.split(self.data_filenames_low[index])
        _, file2 = os.path.split(self.data_filenames_high[index])
        
        seed = random.randint(1, 1000000)
        seed = np.random.randint(seed)
        if self.transform:
            random.seed(seed)
            torch.manual_seed(seed)
            im1 = self.transform(im1)
            random.seed(seed)
            torch.manual_seed(seed)         
            im2 = self.transform(im2)
        return im1, im2, file1, file2

    def __len__(self):
        return self.num_samples


class LSRWDatasetFromFolderEval(data.Dataset):
    """LSRW 评估数据集加载器 (参考DarkIR实现)
    
    支持单独评估 Huawei 或 Nikon，或两者一起
    
    参数:
        data_dir: 数据集根目录
        transform: 数据增强变换
        camera: 指定相机类型，可选 'Huawei', 'Nikon', 'all'（默认使用全部）
    """
    def __init__(self, data_dir, transform=None, camera='all'):
        super(LSRWDatasetFromFolderEval, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        
        self.data_filenames_low = []
        self.data_filenames_high = []
        
        # 确定要加载的相机
        if camera == 'all':
            cameras = ['Huawei', 'Nikon']
        else:
            cameras = [camera]
        
        # 检测数据集结构
        huawei_low = join(data_dir, 'Huawei', 'low')
        nikon_low = join(data_dir, 'Nikon', 'low')
        simple_low = join(data_dir, 'low')
        
        if os.path.exists(huawei_low) or os.path.exists(nikon_low):
            # DarkIR 多相机格式
            for cam in cameras:
                folder_low = join(data_dir, cam, 'low')
                folder_high = join(data_dir, cam, 'high')
                if os.path.exists(folder_low) and os.path.exists(folder_high):
                    low_files = sorted([join(folder_low, x) for x in listdir(folder_low) if is_image_file(x)])
                    high_files = sorted([join(folder_high, x) for x in listdir(folder_high) if is_image_file(x)])
                    self.data_filenames_low.extend(low_files)
                    self.data_filenames_high.extend(high_files)
        elif os.path.exists(simple_low):
            # 简化格式
            folder_low = simple_low
            folder_high = join(data_dir, 'high')
            self.data_filenames_low = sorted([join(folder_low, x) for x in listdir(folder_low) if is_image_file(x)])
            self.data_filenames_high = sorted([join(folder_high, x) for x in listdir(folder_high) if is_image_file(x)])
        else:
            raise ValueError(f"Cannot find LSRW dataset structure in {data_dir}")
        
        self.num_samples = len(self.data_filenames_low)
        camera_info = camera if camera != 'all' else 'Huawei+Nikon'
        print(f"LSRW Eval Dataset ({camera_info}): loaded {self.num_samples} image pairs from {data_dir}")

    def __getitem__(self, index):
        import torch.nn.functional as F
        input = load_img(self.data_filenames_low[index])
        _, file = os.path.split(self.data_filenames_low[index])

        if self.transform:
            input = self.transform(input)
            factor = 8
            h, w = input.shape[1], input.shape[2]
            H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            input = F.pad(input.unsqueeze(0), (0, padw, 0, padh), 'reflect').squeeze(0)
        return input, file, h, w

    def __len__(self):
        return self.num_samples



    

