import argparse

def _str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def option():
    # Training settings
    parser = argparse.ArgumentParser(description='CIDNet')
    parser.add_argument('--batchSize', type=int, default=16, help='training batch size')
    parser.add_argument('--cropSize', type=int, default=256, help='image crop size (patch size)')
    parser.add_argument('--nEpochs', type=int, default=1000, help='number of epochs to train for end')
    parser.add_argument('--start_epoch', type=int, default=0, help='number of epochs to start, >0 is retrained a pre-trained pth')
    parser.add_argument('--snapshots', type=int, default=10, help='Snapshots for save checkpoints pth')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate')
    parser.add_argument('--gpu_mode', type=_str2bool, default=True)
    parser.add_argument('--gpu_ids', type=str, default=None, help='GPU IDs to use (e.g., "0,1,2,3" or "0" for single GPU). If None, use CUDA_VISIBLE_DEVICES from environment')
    parser.add_argument('--shuffle', type=_str2bool, default=True)
    parser.add_argument('--threads', type=int, default=16, help='number of threads for dataloader to use')

    # choose a scheduler
    parser.add_argument('--cos_restart_cyclic', type=_str2bool, default=False)
    parser.add_argument('--cos_restart', type=_str2bool, default=True)

    # warmup training
    parser.add_argument('--warmup_epochs', type=int, default=3, help='warmup_epochs')
    parser.add_argument('--start_warmup', type=_str2bool, default=True, help='turn False to train without warmup') 

    # train datasets
    parser.add_argument('--data_train_lol_blur'     , type=str, default='./datasets/LOL_blur/train')
    parser.add_argument('--data_train_lol_v1'       , type=str, default='./datasets/LOLdataset/our485')
    parser.add_argument('--data_train_lolv2_real'   , type=str, default='./datasets/LOLv2/Real_captured/Train')
    parser.add_argument('--data_train_lolv2_syn'    , type=str, default='./datasets/LOLv2/Synthetic/Train')
    parser.add_argument('--data_train_SID'          , type=str, default='./datasets/Sony_total_dark/train')
    parser.add_argument('--data_train_SICE'         , type=str, default='./datasets/SICE/Dataset/train')
    parser.add_argument('--data_train_fivek'        , type=str, default='./datasets/FiveK/train')
    parser.add_argument('--data_train_sdsd_indoor'  , type=str, default='./datasets/SDSD_indoor_png/train')
    parser.add_argument('--data_train_sdsd_outdoor' , type=str, default='./datasets/SDSD_outdoor_png/train')
    parser.add_argument('--data_train_smid'         , type=str, default='./datasets/SMID_png/train')
    parser.add_argument('--data_train_lsrw'         , type=str, default='./datasets/LSRW/Training data')

    # validation input
    parser.add_argument('--data_val_lol_blur'       , type=str, default='./datasets/LOL_blur/eval/low_blur')
    parser.add_argument('--data_val_lol_v1'         , type=str, default='./datasets/LOLdataset/eval15/low')
    parser.add_argument('--data_val_lolv2_real'     , type=str, default='./datasets/LOLv2/Real_captured/Test/Low')
    parser.add_argument('--data_val_lolv2_syn'      , type=str, default='./datasets/LOLv2/Synthetic/Test/Low')
    parser.add_argument('--data_val_SID'            , type=str, default='./datasets/Sony_total_dark/eval/short')
    parser.add_argument('--data_val_SICE_mix'       , type=str, default='./datasets/SICE/SICE_Mix')
    parser.add_argument('--data_val_SICE_grad'      , type=str, default='./datasets/SICE/SICE_Grad')
    parser.add_argument('--data_test_fivek'         , type=str, default='./datasets/FiveK/test/input')
    parser.add_argument('--data_val_sdsd_indoor'    , type=str, default='./datasets/SDSD_indoor_png/eval/low')
    parser.add_argument('--data_val_sdsd_outdoor'   , type=str, default='./datasets/SDSD_outdoor_png/eval/low')
    parser.add_argument('--data_val_smid'           , type=str, default='./datasets/SMID_png/eval/low')
    parser.add_argument('--data_val_lsrw'           , type=str, default='./datasets/LSRW/Eval')

    # validation groundtruth
    parser.add_argument('--data_valgt_lol_blur'     , type=str, default='./datasets/LOL_blur/eval/high_sharp_scaled/')
    parser.add_argument('--data_valgt_lol_v1'       , type=str, default='./datasets/LOLdataset/eval15/high/')
    parser.add_argument('--data_valgt_lolv2_real'   , type=str, default='./datasets/LOLv2/Real_captured/Test/Normal/')
    parser.add_argument('--data_valgt_lolv2_syn'    , type=str, default='./datasets/LOLv2/Synthetic/Test/Normal/')
    parser.add_argument('--data_valgt_SID'          , type=str, default='./datasets/Sony_total_dark/eval/long/')
    parser.add_argument('--data_valgt_SICE_mix'     , type=str, default='./datasets/SICE/SICE_Reshape/')
    parser.add_argument('--data_valgt_SICE_grad'    , type=str, default='./datasets/SICE/SICE_Reshape/')
    parser.add_argument('--data_valgt_fivek'        , type=str, default='./datasets/FiveK/test/target/')
    parser.add_argument('--data_valgt_sdsd_indoor'  , type=str, default='./datasets/SDSD_indoor_png/eval/high/')
    parser.add_argument('--data_valgt_sdsd_outdoor' , type=str, default='./datasets/SDSD_outdoor_png/eval/high/')
    parser.add_argument('--data_valgt_smid'         , type=str, default='./datasets/SMID_png/eval/high/')
    parser.add_argument('--data_valgt_lsrw'         , type=str, default='./datasets/LSRW/Eval/')

    parser.add_argument('--val_folder', default='./results/', help='Location to save validation datasets')

    # loss weights
    parser.add_argument('--HVI_weight', type=float, default=1.0)
    parser.add_argument('--L1_weight', type=float, default=1.0)
    parser.add_argument('--D_weight',  type=float, default=0.5)
    parser.add_argument('--E_weight',  type=float, default=50.0)
    parser.add_argument('--P_weight',  type=float, default=1e-2)
    
    # LFPV weight (for dual-path training loss)
    parser.add_argument('--lfpv_weight', type=float, default=1.0, help='weight for LFPV path loss (loss = loss1 + lfpv_weight * loss2)')
    
    # Enable/disable LFPV module
    parser.add_argument('--use_lfpv', type=_str2bool, default=True, help='enable LFPV module (default: True). Set to False to disable LFPV completely')
    
    # LFPV module configuration
    parser.add_argument('--lfpv_num_feature', type=int, default=16, help='number of LFPV features (default: 16)')
    parser.add_argument('--lfpv_patch_size', type=int, default=4, help='LFPV patch size (default: 4)')
    
    # LFPV HVI空间优化选项
    parser.add_argument('--lfpv_intensity_adaptive', type=_str2bool, default=False, 
                        help='enable intensity-adaptive gating (dark regions get more enhancement)')
    parser.add_argument('--lfpv_intensity_guide', type=_str2bool, default=False, 
                        help='enable intensity guidance (I-branch guides HV-branch)')
    parser.add_argument('--lfpv_branch_aware', type=_str2bool, default=False, 
                        help='enable branch-aware LFPV (different params for I/HV branches)')
    
    # RIN module configuration
    parser.add_argument('--atom_num', type=int, default=32, help='RIN dictionary atom number (default: 32)')
    parser.add_argument('--atom_dim', type=int, default=512, help='RIN prompt vector dimension (default: 512, same as DACLIP)')
    
    # Enable/disable RIN-PAB module
    parser.add_argument('--use_rin_pab', type=_str2bool, default=True, help='enable RIN-PAB module (default: True). Set to False to use original LCA instead of PAB in HV encoder')
    
    # use random gamma function (enhancement curve) to improve generalization
    parser.add_argument('--gamma', type=_str2bool, default=False)
    parser.add_argument('--start_gamma', type=int, default=60)
    parser.add_argument('--end_gamma', type=int, default=120)

    # self-supervised learning with internal augmentation
    parser.add_argument('--ssl_aug', type=_str2bool, default=False, help='enable self-supervised learning with internal augmentation')
    parser.add_argument('--beta_base', type=float, default=0.1, help='base value for beta in MSR loss (weight for MSR loss, with cosine decay)')
    parser.add_argument('--beta_total_steps', type=int, default=1000, help='total steps for beta cosine decay')

    # external augmentation
    parser.add_argument('--ext_aug', type=_str2bool, default=False, help='enable external augmentation (channel-wise random gamma) for domain alignment simulation')
    parser.add_argument('--ext_aug_prob', type=float, default=0.3, help='probability of applying external augmentation (default: 0.3, conservative for low-light)')
    parser.add_argument('--ext_aug_range', type=str, default='0.95,1.05', help='gamma range for external augmentation (default: 0.95,1.05, very mild for low-light task)')
    parser.add_argument('--ext_aug_mode', type=str, default='symmetric', choices=['symmetric', 'dehazing_style'], help='augmentation mode: symmetric (x^gamma) or dehazing_style (x^(1/gamma))')
    parser.add_argument('--ext_aug_dark_threshold', type=float, default=0.05, help='threshold for protecting extremely dark pixels (physical noise floor)')
    parser.add_argument('--ext_aug_protect_dark', type=_str2bool, default=True, help='enable dark pixel protection to avoid amplifying shot noise')

    # auto grad, turn off to speed up training
    parser.add_argument('--grad_detect', type=_str2bool, default=False, help='if gradient explosion occurs, turn-on it')
    parser.add_argument('--grad_clip', type=_str2bool, default=True, help='if gradient fluctuates too much, turn-on it')
    
    # evaluation options
    parser.add_argument('--eval_gtmean', type=_str2bool, default=True, help='whether to compute GT Mean mode metrics during training evaluation (default: True)')
    
    # experiment identifier for ablation studies
    parser.add_argument('--exp_name', type=str, default='', help='experiment name/identifier for distinguishing different ablation experiments (default: empty, uses dataset name only)')
    
    # choose which dataset you want to train
    parser.add_argument('--dataset', type=str, default='lol_v1',
    choices=['lol_v1',
             'lolv2_real',
             'lolv2_syn',
             'lol_blur', 
             'SID',
             'SICE_mix',
             'SICE_grad',
             'fivek',
             'sdsd_indoor',
             'sdsd_outdoor',
             'smid',
             'lsrw',
             'lsrw_huawei',
             'lsrw_nikon'],
    help='Select the dataset to train on (default: %(default)s)')

    return parser
