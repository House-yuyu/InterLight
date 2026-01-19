from torchvision.transforms import Compose, ToTensor, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip
from data.LOLdataset import *
from data.eval_sets import *
from data.SICE_blur_SID import *
from data.fivek import *
# LSRW数据集类已在LOLdataset.py中定义

def transform1(size=256):
    return Compose([
        RandomCrop((size, size)),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        ToTensor(),
    ])

def transform2():
    return Compose([ToTensor()])



def get_lol_training_set(data_dir,size):
    return LOLDatasetFromFolder(data_dir, transform=transform1(size))


def get_lol_v2_training_set(data_dir,size):
    return LOLv2DatasetFromFolder(data_dir, transform=transform1(size))


def get_training_set_blur(data_dir,size):
    return LOLBlurDatasetFromFolder(data_dir, transform=transform1(size))


def get_lol_v2_syn_training_set(data_dir,size):
    return LOLv2SynDatasetFromFolder(data_dir, transform=transform1(size))


def get_SID_training_set(data_dir,size):
    return SIDDatasetFromFolder(data_dir, transform=transform1(size))


def get_SICE_training_set(data_dir,size):
    return SICEDatasetFromFolder(data_dir, transform=transform1(size))

def get_SICE_eval_set(data_dir):
    return SICEDatasetFromFolderEval(data_dir, transform=transform2())

def get_eval_set(data_dir):
    return DatasetFromFolderEval(data_dir, transform=transform2())

def get_fivek_training_set(data_dir,size):
    return FiveKDatasetFromFolder(data_dir, transform=transform1(size))

def get_fivek_eval_set(data_dir):
    return SICEDatasetFromFolderEval(data_dir, transform=transform2())

def get_sdsd_indoor_training_set(data_dir, size):
    return SDSDIndoorDatasetFromFolder(data_dir, transform=transform1(size))

def get_sdsd_indoor_eval_set(data_dir):
    return SICEDatasetFromFolderEval(data_dir, transform=transform2())

def get_sdsd_outdoor_training_set(data_dir, size):
    return SDSDIndoorDatasetFromFolder(data_dir, transform=transform1(size))  # 结构相同，复用类

def get_sdsd_outdoor_eval_set(data_dir):
    return SICEDatasetFromFolderEval(data_dir, transform=transform2())

def get_smid_training_set(data_dir, size):
    return SDSDIndoorDatasetFromFolder(data_dir, transform=transform1(size))  # 结构相同，复用类

def get_smid_eval_set(data_dir):
    return SICEDatasetFromFolderEval(data_dir, transform=transform2())

# ===== LSRW Dataset (参考DarkIR实现) =====
def get_lsrw_training_set(data_dir, size, camera='all'):
    """获取LSRW训练集
    
    支持两种数据集结构:
    1. DarkIR多相机格式: data_dir/Huawei/low, data_dir/Huawei/high, data_dir/Nikon/low, data_dir/Nikon/high
    2. 简化格式: data_dir/low, data_dir/high
    
    参数:
        camera: 'all' (默认), 'Huawei', 或 'Nikon'
    """
    return LSRWDatasetFromFolder(data_dir, transform=transform1(size), camera=camera)

def get_lsrw_eval_set(data_dir, camera='all'):
    """获取LSRW评估集
    
    参数:
        camera: 'all' (默认), 'Huawei', 或 'Nikon'
    """
    return LSRWDatasetFromFolderEval(data_dir, transform=transform2(), camera=camera)

# LSRW Huawei 单独
def get_lsrw_huawei_training_set(data_dir, size):
    return LSRWDatasetFromFolder(data_dir, transform=transform1(size), camera='Huawei')

def get_lsrw_huawei_eval_set(data_dir):
    return LSRWDatasetFromFolderEval(data_dir, transform=transform2(), camera='Huawei')

# LSRW Nikon 单独
def get_lsrw_nikon_training_set(data_dir, size):
    return LSRWDatasetFromFolder(data_dir, transform=transform1(size), camera='Nikon')

def get_lsrw_nikon_eval_set(data_dir):
    return LSRWDatasetFromFolderEval(data_dir, transform=transform2(), camera='Nikon')