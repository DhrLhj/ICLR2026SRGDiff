import argparse
from omegaconf import OmegaConf
import numpy as np
import torch
import random
from dataset_SEED_DEAP_new import load_SEED_data,load_DEAP_data_mean_std,DEAP,SEED_condition
from torch.utils.data import DataLoader,TensorDataset
from generative.networks.nets import AutoencoderKL, PatchDiscriminator
import os
import time

def set_seed(seed: int):
    """Set the seed for all random number generators"""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_all_data(cfg):
    """获取全部数据（train+test）并分别返回"""
    train_eeg, train_labels, train_mask, test_eeg, test_labels, test_mask, subject_train, subject_test = load_SEED_data(
        cfg.patient_id,
        cfg.signal_length,
        cfg.shuffle,
        cfg.dataset_dir,
    )
    print(f"Train data shape: {train_eeg.shape}")
    print(f"Test data shape: {test_eeg.shape}")
    print(f"Train labels shape: {train_labels.shape}")
    print(f"Test labels shape: {test_labels.shape}")
    return (
        torch.from_numpy(train_eeg).float(), torch.from_numpy(train_labels),
        torch.from_numpy(test_eeg).float(), torch.from_numpy(test_labels)
    )

def mask_eeg_channels(eeg, mask_ratio=0.5, mask_channel_names=None, ch_names=None):
    """
    对EEG数据进行通道掩码
    支持通过通道名指定掩码通道，其余随机掩码
    """
    N, C, L = eeg.shape
    masked_eeg = eeg.clone()
    if mask_channel_names is not None and ch_names is not None:
        # 通过通道名指定掩码
        mask_channel_indices = []
        for name in mask_channel_names:
            if name not in ch_names:
                raise ValueError(f"通道名 {name} 不在ch_names列表中！")
            mask_channel_indices.append(ch_names.index(name))
        print(f"Masking fixed channels: {mask_channel_indices} ({mask_channel_names})")
        masked_eeg[:, mask_channel_indices, :] = 0
        # 如果掩码通道数不足mask_ratio，还可以随机补齐
        num_mask = int(C * mask_ratio)
        n_fixed = len(mask_channel_indices)
        if n_fixed < num_mask:
            for i in range(N):
                remain_indices = list(set(range(C)) - set(mask_channel_indices))
                n_to_fill = num_mask - n_fixed
                if n_to_fill > 0:
                    fill_indices = np.random.choice(remain_indices, n_to_fill, replace=False)
                    masked_eeg[i, fill_indices, :] = 0
    else:
        # 随机掩码
        num_mask = int(C * mask_ratio)
        print(f"Masking {num_mask} out of {C} channels ({mask_ratio*100}%)")
        for i in range(N):
            idx = torch.randperm(C)[:num_mask]
            masked_eeg[i, idx, :] = 0
    return masked_eeg

def extract_z(model, data, device, batch_size=64):
    """提取潜在变量z"""
    model.eval()
    all_z = []
    with torch.no_grad():
        for i in range(0, data.shape[0], batch_size):
            batch = data[i:i+batch_size].to(device)
            _, z_mu, z_sigma = model(batch)
            all_z.append(z_mu.cpu())
            if (i // batch_size + 1) % 10 == 0:
                print(f"Processed batch {i // batch_size + 1}/{(data.shape[0] + batch_size - 1) // batch_size}")
    all_z = torch.cat(all_z, dim=0)
    print(f"Extracted z shape: {all_z.shape}")
    return all_z

def parse_args():
    parser = argparse.ArgumentParser(description="Autoencoder Z Extraction Script")
    parser.add_argument('--dataset_name', type=str, default='SEED', help='Name of the dataset')
    parser.add_argument('--patient_id', type=str, default='all', help='Patient ID for the dataset')
    parser.add_argument('--shuffle', type=bool, default=False, help='Whether to shuffle the dataset')
    parser.add_argument('--num_channel', type=int, default=62, help='Number of channels in the signal')
    parser.add_argument('--signal_length', type=int, default=800, help='Length of the signal')
    parser.add_argument('--dataset_dir', type=str, default='/data/zhouleyu/Preprocessed_EEG', help='Directory of the dataset')
    parser.add_argument('--save_dir', type=str, default='/data/zhouleyu/autoencoder_for_SEED_new/checkpoint/SEED_all_klbig_noshuffle/checkpoint/best_model.pth', help='Directory to save results')
    parser.add_argument('--down_stream_dir', type=str, default="/data/zhouleyu/ddpm_for_SEED/data/SEED_all_klbig_50%_nomask_noshuffle")
    parser.add_argument('--device', type=str, default='cuda:1', help='Device to use (e.g., cuda:1, cpu)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--train_batch_size', type=int, default=64, help='Batch size for processing')
    parser.add_argument('--config_file', type=str, default="/data/zhouleyu/autoencoder_for_SEED_new/config_big_klbig.yaml")
    # 新增AutoencoderKL网络架构参数
    parser.add_argument('--ae_num_channels', type=int, nargs='+', default=[64, 32], 
                       help='Number of channels for each layer in AutoencoderKL (e.g., --ae_num_channels 64 32)')
    parser.add_argument('--ae_latent_channels', type=int, default=32, 
                       help='Number of latent channels in AutoencoderKL')
    # 新增：指定掩码通道名
    parser.add_argument('--mask_channel_names', type=str, nargs='*', default=["AF3", "AF4", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "TP7", "CP5", "CP3", "CP1", 
"CPZ", "CP2", "CP4", "CP6", "TP8", "POO7", "POO8", "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8"],
        help='需要掩码的通道名称列表（如 --mask_channel_names FZ CZ PZ）')
    return parser.parse_args()

"""
[ "AF3", "AF4", "FC5", "FC6", "CP5", "CP6", "PO5", "PO6"]
['FP1','F5','T7','C3','P5','O1','FZ','CZ','PZ','FP2','F8','T8','C4','P8','O2']
["AF3", "AF4", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "TP7", "CP5", "CP3", "CP1", 
"CPZ", "CP2", "CP4", "CP6", "TP8", "POO7", "POO8", "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8"]
"""

def load_best_model(cfg, config, device):
    """加载最佳模型"""
    best_model_path = cfg.save_dir
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Best model not found: {best_model_path}")
    print(f"Loading best model from {best_model_path}")
    checkpoint = torch.load(best_model_path, map_location=device)
    autoencoder_args = config.autoencoderkl.params
    autoencoder_args['num_channels'] = cfg.ae_num_channels
    autoencoder_args['latent_channels'] = cfg.ae_latent_channels
    model = AutoencoderKL(**autoencoder_args).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    best_val_loss = checkpoint.get('best_loss', 'Unknown')
    print(f"Best model loaded successfully! (Best Val Loss: {best_val_loss})")
    print(f"Model architecture - num_channels: {cfg.ae_num_channels}, latent_channels: {cfg.ae_latent_channels}")
    return model

def main():
    # 通道名列表（顺序需与数据一致）
    ch_names = [
        'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
        'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ',
        'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7',
        'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6',
        'PO8', 'POO7',  'O1', 'OZ', 'O2','POO8'
    ]
    # 解析参数
    args = parse_args()
    config = OmegaConf.load(args.config_file)
    # 创建配置对象 - 包含所有parser参数
    cfg = OmegaConf.create()
    cfg.dataset_name = args.dataset_name
    cfg.patient_id = args.patient_id
    cfg.shuffle = args.shuffle
    cfg.num_channel = args.num_channel
    cfg.signal_length = args.signal_length
    cfg.dataset_dir = args.dataset_dir
    cfg.save_dir = args.save_dir
    cfg.down_stream_dir = args.down_stream_dir
    cfg.device = args.device
    cfg.seed = args.seed
    cfg.train_batch_size = args.train_batch_size
    cfg.config_file = args.config_file
    cfg.ae_num_channels = args.ae_num_channels
    cfg.ae_latent_channels = args.ae_latent_channels
    cfg.mask_channel_names = args.mask_channel_names
    # 设置随机种子
    if cfg.seed is not None:
        set_seed(cfg.seed)
    # 设置设备
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # 确保输出目录存在
    os.makedirs(cfg.down_stream_dir, exist_ok=True)
    # 保存完整的参数信息到txt
    cfg_args_path = os.path.join(cfg.down_stream_dir, "extraction_config_full.txt")
    with open(cfg_args_path, "w") as f:
        f.write("="*80 + "\n")
        f.write("AutoencoderKL Z Extraction Configuration\n")
        f.write("="*80 + "\n\n")
        f.write("📋 Parser Arguments (Command Line):\n")
        f.write("-"*50 + "\n")
        parser_args = vars(args)
        for key, value in parser_args.items():
            f.write(f"  {key:<25}: {value}\n")
        f.write(f"\n📊 Final Configuration (Used in Extraction):\n")
        f.write("-"*50 + "\n")
        for key, value in cfg.items():
            f.write(f"  {key:<25}: {value}\n")
        f.write(f"\n🏗️  Model Architecture:\n")
        f.write("-"*50 + "\n")
        f.write(f"  AutoencoderKL Channels : {cfg.ae_num_channels}\n")
        f.write(f"  Latent Channels        : {cfg.ae_latent_channels}\n")
        f.write(f"\n📈 Extraction Settings:\n")
        f.write("-"*50 + "\n")
        f.write(f"  Dataset Name           : {cfg.dataset_name}\n")
        f.write(f"  Patient ID             : {cfg.patient_id}\n")
        f.write(f"  Data Shuffle           : {cfg.shuffle}\n")
        f.write(f"  Signal Length          : {cfg.signal_length}\n")
        f.write(f"  Number of Channels     : {cfg.num_channel}\n")
        f.write(f"  Batch Size             : {cfg.train_batch_size}\n")
        f.write(f"  Device                 : {cfg.device}\n")
        f.write(f"  Random Seed            : {cfg.seed}\n")
        f.write(f"\n📁 Paths:\n")
        f.write("-"*50 + "\n")
        f.write(f"  Dataset Directory      : {cfg.dataset_dir}\n")
        f.write(f"  Model Save Directory   : {cfg.save_dir}\n")
        f.write(f"  Output Directory       : {cfg.down_stream_dir}\n")
        f.write(f"  Config File            : {cfg.config_file}\n")
        import datetime
        f.write(f"\n⏰ Configuration saved at: {datetime.datetime.now()}\n")
        f.write("="*80 + "\n")
    print(f"已保存完整配置参数到: {cfg_args_path}")
    # 加载最佳模型
    model = load_best_model(cfg, config, device)
    # 获取分离的train和test数据
    print("\nLoading train and test data separately...")
    train_eeg, train_labels, test_eeg, test_labels = get_all_data(cfg)
    # ================ 处理训练集数据 ================
    print("\n" + "="*60)
    print("Processing TRAIN data...")
    # 训练集: 提取z100（无掩码）
    print("Task 1: Extracting train z100 (no masking)...")
    train_z100 = extract_z(model, train_eeg, device, cfg.train_batch_size)
    train_z100_path = os.path.join(cfg.down_stream_dir, "train_z100.pth")
    torch.save(train_z100, train_z100_path)
    print(f"Train z100 saved to: {train_z100_path}")
    # 训练集: 提取z50（50%通道掩码或指定通道掩码）
    print("Task 2: Extracting train z50 (masking)...")
    train_masked_eeg = mask_eeg_channels(
        train_eeg, mask_ratio=0.5, mask_channel_names=cfg.mask_channel_names, ch_names=ch_names
    )
    train_z50 = extract_z(model, train_masked_eeg, device, cfg.train_batch_size)
    train_z50_path = os.path.join(cfg.down_stream_dir, "train_z50.pth")
    torch.save(train_z50, train_z50_path)
    print(f"Train z50 saved to: {train_z50_path}")
    # 保存训练集标签
    train_labels_path = os.path.join(cfg.down_stream_dir, "train_labels.pth")
    torch.save(train_labels, train_labels_path)
    print(f"Train labels saved to: {train_labels_path}")
    # ================ 处理测试集数据 ================
    print("\n" + "="*60)
    print("Processing TEST data...")
    # 测试集: 提取z100（无掩码）
    print("Task 3: Extracting test z100 (no masking)...")
    test_z100 = extract_z(model, test_eeg, device, cfg.train_batch_size)
    test_z100_path = os.path.join(cfg.down_stream_dir, "test_z100.pth")
    torch.save(test_z100, test_z100_path)
    print(f"Test z100 saved to: {test_z100_path}")
    # 测试集: 提取z50（50%通道掩码或指定通道掩码）
    print("Task 4: Extracting test z50 (masking)...")
    test_masked_eeg = mask_eeg_channels(
        test_eeg, mask_ratio=0.5, mask_channel_names=cfg.mask_channel_names, ch_names=ch_names
    )
    test_z50 = extract_z(model, test_masked_eeg, device, cfg.train_batch_size)
    test_z50_path = os.path.join(cfg.down_stream_dir, "test_z50.pth")
    torch.save(test_z50, test_z50_path)
    print(f"Test z50 saved to: {test_z50_path}")
    # 保存测试集标签
    test_labels_path = os.path.join(cfg.down_stream_dir, "test_labels.pth")
    torch.save(test_labels, test_labels_path)
    print(f"Test labels saved to: {test_labels_path}")
    # ================ 保存原始数据（用于对比） ================
    print("\n" + "="*60)
    print("Saving original data for comparison...")
    train_orig_path = os.path.join(cfg.down_stream_dir, "train_original.pth")
    torch.save(train_eeg, train_orig_path)
    print(f"Train original data saved to: {train_orig_path}")
    test_orig_path = os.path.join(cfg.down_stream_dir, "test_original.pth")
    torch.save(test_eeg, test_orig_path)
    print(f"Test original data saved to: {test_orig_path}")
    train_masked_path = os.path.join(cfg.down_stream_dir, "train_masked_50percent.pth")
    torch.save(train_masked_eeg, train_masked_path)
    print(f"Train masked data saved to: {train_masked_path}")
    test_masked_path = os.path.join(cfg.down_stream_dir, "test_masked_50percent.pth")
    torch.save(test_masked_eeg, test_masked_path)
    print(f"Test masked data saved to: {test_masked_path}")
    # ================ 保存配置文件 ================
    cfg_path = os.path.join(cfg.down_stream_dir, "extraction_config.txt")
    with open(cfg_path, "w") as f:
        f.write("Extraction Configuration:\n")
        f.write("="*40 + "\n")
        for k, v in cfg.items():
            f.write(f"{k}: {v}\n")
        f.write(f"\nData Shapes:\n")
        f.write(f"Train data: {train_eeg.shape}\n")
        f.write(f"Test data: {test_eeg.shape}\n")
        f.write(f"Train z100: {train_z100.shape}\n")
        f.write(f"Train z50: {train_z50.shape}\n")
        f.write(f"Test z100: {test_z100.shape}\n")
        f.write(f"Test z50: {test_z50.shape}\n")
        f.write(f"Train labels: {train_labels.shape}\n")
        f.write(f"Test labels: {test_labels.shape}\n")
        f.write(f"\nModel Architecture:\n")
        f.write(f"ae_num_channels: {cfg.ae_num_channels}\n")
        f.write(f"ae_latent_channels: {cfg.ae_latent_channels}\n")
        f.write(f"\nSaved Files:\n")
        f.write(f"Train files:\n")
        f.write(f"  - train_z100.pth: Latent variables (no masking)\n")
        f.write(f"  - train_z50.pth: Latent variables (masking)\n")
        f.write(f"  - train_labels.pth: Training labels\n")
        f.write(f"  - train_original.pth: Original training data\n")
        f.write(f"  - train_masked_50percent.pth: Masked training data\n")
        f.write(f"Test files:\n")
        f.write(f"  - test_z100.pth: Latent variables (no masking)\n")
        f.write(f"  - test_z50.pth: Latent variables (masking)\n")
        f.write(f"  - test_labels.pth: Test labels\n")
        f.write(f"  - test_original.pth: Original test data\n")
        f.write(f"  - test_masked_50percent.pth: Masked test data\n")
    print(f"Basic configuration saved to: {cfg_path}")
    print("\n" + "="*60)
    print("🎉 All tasks completed successfully!")
    print(f"🏗️  Model Architecture:")
    print(f"  AutoencoderKL Channels: {cfg.ae_num_channels}")
    print(f"  Latent Channels: {cfg.ae_latent_channels}")
    print(f"📊 Data Shapes:")
    print(f"  Train data: {train_eeg.shape} → Train z100: {train_z100.shape}, Train z50: {train_z50.shape}")
    print(f"  Test data: {test_eeg.shape} → Test z100: {test_z100.shape}, Test_z50: {test_z50.shape}")
    print(f"  Train labels: {train_labels.shape}, Test labels: {test_labels.shape}")
    print(f"📁 Results saved in: {cfg.down_stream_dir}")
    print(f"📋 Files saved:")
    print(f"  Training set: train_z100.pth, train_z50.pth, train_labels.pth, train_original.pth, train_masked_50percent.pth")
    print(f"  Test set: test_z100.pth, test_z50.pth, test_labels.pth, test_original.pth, test_masked_50percent.pth")
    print(f"  Config files: extraction_config_full.txt, extraction_config.txt")

if __name__ == "__main__":
    main()