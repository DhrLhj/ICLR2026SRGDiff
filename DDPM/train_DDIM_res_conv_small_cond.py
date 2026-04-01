import argparse
from omegaconf import OmegaConf
import numpy as np
import torch
import random
import os
import time
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import MultiStepLR
from generative.networks.schedulers import DDIMScheduler
from unet import UNetModel
from torch.nn import MSELoss
from network import CatConv, AdaConv, AdaConv_Res,AdaConv_Res_Small
from res_time_net import TimeAwareResidualPredictor

# === 导入ConvEncoder ===
from train_conv_new import ConvEncoder

def parse_args():
    parser = argparse.ArgumentParser(description="Latent DDIM Training")
    parser.add_argument('--latent_data_dir', type=str, default='/data/zhouleyu/ddpm_for_SEED/data/SEED_all_klbig_90%_nomask')
    parser.add_argument('--save_dir', type=str, default='/data/zhouleyu/ddpm_for_SEED/checkpoint_resconv/adanew_90%_cos_cond_small')
    parser.add_argument('--down_stream_dir', type=str, default="/data/zhouleyu/ddpm_for_SEED/downstream_nomask_resconv/adanew_90%_s_0.02_with_ConvEncoder_cond_small")
    parser.add_argument('--epoch', type=int, default=80)
    parser.add_argument('--device', type=str, default='cuda:3')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train_batch_size', type=int, default=64*4)
    parser.add_argument('--num_train_timesteps', type=int, default=1000)
    parser.add_argument('--num_inference_steps', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--config_file', type=str, default="/data/zhouleyu/autoencoder_for_SEED_new/config.yaml")
    parser.add_argument('--conv_encoder_ckpt', type=str, default="/data/zhouleyu/ddpm_for_SEED/best_conv_result/90%_time0.2/best_conv_encoder.pth")
    parser.add_argument('--conv_encoder_time_emb_dim', type=int, default=64)
    parser.add_argument('--conv_encoder_T', type=int, default=1000)
    parser.add_argument('--scheduler_s', type=float, default=0.02)
    parser.add_argument('--scheduler_schedule', type=str, default='cosine')
    return parser.parse_args()

def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
def train_residual_model(res_model, train_loader, optimizer, device, num_epochs, scheduler_steps):
    res_model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for step, batch in enumerate(train_loader):
            z100_target = batch[0].to(device)
            z50 = batch[1].to(device)
            batch_size = z100_target.shape[0]
            timesteps = torch.randint(0, scheduler_steps, (batch_size,), device=device).long()
            res_target = z100_target - z50
            optimizer.zero_grad()
            res_pred = res_model(z50, timesteps)
            res_loss = torch.nn.functional.l1_loss(res_pred, res_target)
            res_loss.backward()
            optimizer.step()
            epoch_loss += res_loss.item()
        avg_loss = epoch_loss / len(train_loader)
        print(f"[Residual] Epoch {epoch+1}/{num_epochs} - Avg Res Loss: {avg_loss:.6f}")
    return res_model

def load_conv_encoder(conv_encoder_ckpt, in_channels, latent_channels, T, time_emb_dim, device):
    conv_encoder = ConvEncoder(
        in_channels=in_channels,
        latent_channels=latent_channels,
        T=T,
        time_emb_dim=time_emb_dim
    ).to(device)
    conv_encoder.load_state_dict(torch.load(conv_encoder_ckpt, map_location=device))
    conv_encoder.eval()
    for p in conv_encoder.parameters():
        p.requires_grad = False
    return conv_encoder

def train_latent_ddim(model, scheduler, train_loader, optimizer, device, epoch, cfg, conv_encoder):
    model.train()
    epoch_loss = 0.0
    num_batches = len(train_loader)
    for step, batch in enumerate(train_loader):
        z100_target = batch[0].to(device)
        z50 = batch[1].to(device)
        batch_size = z100_target.shape[0]
        timesteps = torch.randint(0, scheduler.num_train_timesteps, (batch_size,), device=device).long()
        # === 用ConvEncoder生成zcond ===
        zcond = conv_encoder(z50, timesteps)
        noise = torch.randn_like(z100_target)
        noisy_z100 = scheduler.add_noise(z100_target, noise, timesteps)
        optimizer.zero_grad()
        noise_pred = model(noisy_z100, timesteps, zcond)
        loss = torch.nn.functional.smooth_l1_loss(noise_pred, noise)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item()
        if (step + 1) % 10 == 0 or step == 0:
            avg_loss = epoch_loss / (step + 1)
            print(f"Batch [{step+1}/{num_batches}] - Loss: {loss.item():.6f}, Avg Loss: {avg_loss:.6f}")
    return epoch_loss / num_batches

def validate_latent_ddim(model, scheduler, test_loader, device, conv_encoder):
    model.eval()
    val_loss = 0.0
    num_batches = len(test_loader)
    with torch.no_grad():
        for batch in test_loader:
            z100_target = batch[0].to(device)
            z50 = batch[1].to(device)
            batch_size = z100_target.shape[0]
            timesteps = torch.randint(0, scheduler.num_train_timesteps, (batch_size,), device=device).long()
            zcond = conv_encoder(z50, timesteps)
            noise = torch.randn_like(z100_target)
            noisy_z100 = scheduler.add_noise(z100_target, noise, timesteps)
            noise_pred = model(noisy_z100, timesteps, zcond)
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            val_loss += loss.item()
    return val_loss / num_batches

def sample_latent_ddim(model, scheduler, z50, conv_encoder, device, num_inference_steps=50):
    model.eval()
    with torch.no_grad():
        scheduler.set_timesteps(num_inference_steps)
        batch_size = z50.shape[0]
        z100_sample = torch.randn_like(z50)  # shape与zcond一致即可
        for t in scheduler.timesteps:
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            # 每一步都用当前t重新生成zcond
            zcond = conv_encoder(z50, t_batch)
            noise_pred = model(z100_sample, t_batch, zcond)
            scheduler_output = scheduler.step(noise_pred, t, z100_sample)
            if hasattr(scheduler_output, 'prev_sample'):
                z100_sample = scheduler_output.prev_sample
            elif isinstance(scheduler_output, tuple):
                z100_sample = scheduler_output[0]
            else:
                z100_sample = scheduler_output
    return z100_sample


def setup_unet_model(latent_shape, config_file, device):
    """设置UNet模型"""
    config = OmegaConf.load(config_file)
    
    # 获取潜在空间维度
    if len(latent_shape) == 4:
        _, C, H, W = latent_shape
        print(f"潜在空间维度: C={C}, H={H}, W={W}")
    elif len(latent_shape) == 3:
        _, C, L = latent_shape
        print(f"潜在空间维度: C={C}, L={L}")
    else:
        raise ValueError(f"未知的潜在空间shape: {latent_shape}")
    
    # UNet配置
    unet_params = config['ddpm']['params']['unet_config']['params']
    unet_params['in_channels'] = C * 2  # z50 + z100 拼接
    unet_params['out_channels'] = C     # 预测z100的噪声
    
    print(f"UNet配置:")
    print(f"  输入通道: {unet_params['in_channels']} (z50 + z100)")
    print(f"  输出通道: {unet_params['out_channels']} (噪声预测)")
    print(f"  其他参数: {unet_params}")
    
    # 创建模型
    model = UNetModel(**unet_params)
    model.to(device)
    
    # 统计参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"UNet模型参数: 总计 {total_params:,}, 可训练 {trainable_params:,}")
    
    return model

def load_latent_data(latent_data_dir):
    """加载潜在空间数据并进行标准化"""
    print(f"从 {latent_data_dir} 加载潜在空间数据...")
    
    # 🔥 修改：分别查找训练集和测试集的数据文件
    train_z100_files = [f for f in os.listdir(latent_data_dir) if f.startswith('train_z100') and f.endswith('.pth')]
    train_z50_files = [f for f in os.listdir(latent_data_dir) if f.startswith('train_z50') and f.endswith('.pth')]
    test_z100_files = [f for f in os.listdir(latent_data_dir) if f.startswith('test_z100') and f.endswith('.pth')]
    test_z50_files = [f for f in os.listdir(latent_data_dir) if f.startswith('test_z50') and f.endswith('.pth')]
    
    if not train_z100_files or not train_z50_files:
        raise FileNotFoundError(f"在 {latent_data_dir} 中未找到 train_z100.pth 或 train_z50.pth 文件")
    
    if not test_z100_files or not test_z50_files:
        raise FileNotFoundError(f"在 {latent_data_dir} 中未找到 test_z100.pth 或 test_z50.pth 文件")
    
    # 取最新的文件（如果有多个）
    train_z100_file = sorted(train_z100_files)[-1]
    train_z50_file = sorted(train_z50_files)[-1]
    test_z100_file = sorted(test_z100_files)[-1]
    test_z50_file = sorted(test_z50_files)[-1]
    
    train_z100_path = os.path.join(latent_data_dir, train_z100_file)
    train_z50_path = os.path.join(latent_data_dir, train_z50_file)
    test_z100_path = os.path.join(latent_data_dir, test_z100_file)
    test_z50_path = os.path.join(latent_data_dir, test_z50_file)
    
    print(f"加载训练集 z100: {train_z100_path}")
    print(f"加载训练集 z50: {train_z50_path}")
    print(f"加载测试集 z100: {test_z100_path}")
    print(f"加载测试集 z50: {test_z50_path}")
    
    # 🔥 分别加载训练集和测试集数据
    train_z100_data = torch.load(train_z100_path, map_location='cpu')  # [N_train, C, H, W] - 训练集高质量目标
    train_z50_data = torch.load(train_z50_path, map_location='cpu')    # [N_train, C, H, W] - 训练集低质量条件
    test_z100_data = torch.load(test_z100_path, map_location='cpu')    # [N_test, C, H, W] - 测试集高质量目标
    test_z50_data = torch.load(test_z50_path, map_location='cpu')      # [N_test, C, H, W] - 测试集低质量条件
    
    print(f"训练集 z100 shape: {train_z100_data.shape}")
    print(f"训练集 z50 shape: {train_z50_data.shape}")
    print(f"测试集 z100 shape: {test_z100_data.shape}")
    print(f"测试集 z50 shape: {test_z50_data.shape}")
    
    # 检查形状一致性
    if train_z100_data.shape != train_z50_data.shape:
        raise ValueError(f"训练集z100和z50的形状不匹配: {train_z100_data.shape} vs {train_z50_data.shape}")
    
    if test_z100_data.shape != test_z50_data.shape:
        raise ValueError(f"测试集z100和z50的形状不匹配: {test_z100_data.shape} vs {test_z50_data.shape}")
    
    # 🔥 合并训练集和测试集数据用于统计
    all_z100_data = torch.cat([train_z100_data, test_z100_data], dim=0)
    all_z50_data = torch.cat([train_z50_data, test_z50_data], dim=0)
    
    # 原始数据统计
    print(f"原始 z100 数据范围: [{all_z100_data.min():.4f}, {all_z100_data.max():.4f}]")
    print(f"原始 z50 数据范围: [{all_z50_data.min():.4f}, {all_z50_data.max():.4f}]")
    print(f"原始 z100 均值: {all_z100_data.mean():.4f}, 标准差: {all_z100_data.std():.4f}")
    print(f"原始 z50 均值: {all_z50_data.mean():.4f}, 标准差: {all_z50_data.std():.4f}")
    
    # 🔥 使用所有数据计算z50的逐通道均值和标准差用于标准化
    if len(all_z50_data.shape) == 4:  # [N, C, H, W]
        # 沿着 N, H, W 维度计算均值和标准差，保留 C 维度
        z50_mean = all_z50_data.mean(dim=(0, 2, 3), keepdim=True)  # [1, C, 1, 1]
        z50_std = all_z50_data.std(dim=(0, 2, 3), keepdim=True)    # [1, C, 1, 1]
    elif len(all_z50_data.shape) == 3:  # [N, C, L]
        # 沿着 N, L 维度计算均值和标准差，保留 C 维度
        z50_mean = all_z50_data.mean(dim=(0, 2), keepdim=True)     # [1, C, 1]
        z50_std = all_z50_data.std(dim=(0, 2), keepdim=True)       # [1, C, 1]
    else:
        raise ValueError(f"未知的潜在空间shape: {all_z50_data.shape}")
    
    # 避免除以零
    z50_std = torch.clamp(z50_std, min=1e-8)
    
    print(f"z50 逐通道均值: {z50_mean.squeeze()}")
    print(f"z50 逐通道标准差: {z50_std.squeeze()}")
    
    # 🔥 分别对训练集和测试集进行标准化
    train_z50_normalized = (train_z50_data - z50_mean) / z50_std
    train_z100_normalized = (train_z100_data - z50_mean) / z50_std  # 使用z50的统计量
    
    test_z50_normalized = (test_z50_data - z50_mean) / z50_std
    test_z100_normalized = (test_z100_data - z50_mean) / z50_std
    
    # 重新合并标准化后的数据用于统计
    all_z100_normalized = torch.cat([train_z100_normalized, test_z100_normalized], dim=0)
    all_z50_normalized = torch.cat([train_z50_normalized, test_z50_normalized], dim=0)
    
    # 标准化后的数据统计
    print(f"标准化后 z100 数据范围: [{all_z100_normalized.min():.4f}, {all_z100_normalized.max():.4f}]")
    print(f"标准化后 z50 数据范围: [{all_z50_normalized.min():.4f}, {all_z50_normalized.max():.4f}]")
    print(f"标准化后 z100 均值: {all_z100_normalized.mean():.4f}, 标准差: {all_z100_normalized.std():.4f}")
    print(f"标准化后 z50 均值: {all_z50_normalized.mean():.4f}, 标准差: {all_z50_normalized.std():.4f}")
    
    # 🔥 返回分离的训练集和测试集数据以及标准化参数
    return (
        train_z100_normalized.float(),  # 训练集高质量目标
        train_z50_normalized.float(),   # 训练集低质量条件
        test_z100_normalized.float(),   # 测试集高质量目标
        test_z50_normalized.float(),    # 测试集低质量条件
        z50_mean.float(),               # 标准化均值
        z50_std.float()                 # 标准化标准差
    )

def create_data_loaders(train_z100_data, train_z50_data, test_z100_data, test_z50_data, batch_size):
    """创建训练和测试数据加载器"""
    # 🔥 修改：直接使用预分割的训练集和测试集数据
    print(f"创建数据加载器...")
    
    # 训练数据
    train_dataset = TensorDataset(train_z100_data, train_z50_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # 测试数据
    test_dataset = TensorDataset(test_z100_data, test_z50_data)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"训练样本数: {len(train_dataset)}")
    print(f"测试样本数: {len(test_dataset)}")
    
    return train_loader, test_loader


def denormalize_data(normalized_data, mean, std):
    """反标准化数据"""
    return normalized_data * std + mean

def test_sampling(model, scheduler, test_loader, device, cfg, conv_encoder, num_inference_steps=50, z50_mean=None, z50_std=None):
    """测试采样效果，并保存zcond"""
    print(f"\n🔍 测试DDIM采样 (推理步数: {num_inference_steps})")
    
    model.eval()
    test_results = {
        'original_z100': [],
        'condition_z50': [],
        'generated_z100': [],
        'original_z100_denorm': [],
        'condition_z50_denorm': [],
        'generated_z100_denorm': [],
        'mse_scores': [],
        'mse_scores_denorm': [],
        'zcond': []  # 新增：保存每个batch的zcond
    }
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            z100_target = batch[0].to(device)
            z50_condition = batch[1].to(device)
            batch_size = z100_target.shape[0]
            # 用t=0生成zcond（可根据实际采样逻辑调整）
            t0 = torch.zeros(batch_size, device=device, dtype=torch.long)
            zcond = conv_encoder(z50_condition, t0)
            test_results['zcond'].append(zcond.cpu())
            
            # DDIM采样生成
            z100_generated = sample_latent_ddim(
                model, scheduler, z50_condition, conv_encoder, device, num_inference_steps
            )
            
            mse = torch.mean((z100_generated - z100_target) ** 2).item()
            
            if z50_mean is not None and z50_std is not None:
                z50_mean_device = z50_mean.to(device)
                z50_std_device = z50_std.to(device)
                z100_target_denorm = denormalize_data(z100_target, z50_mean_device, z50_std_device)
                z50_condition_denorm = denormalize_data(z50_condition, z50_mean_device, z50_std_device)
                z100_generated_denorm = denormalize_data(z100_generated, z50_mean_device, z50_std_device)
                mse_denorm = torch.mean((z100_generated_denorm - z100_target_denorm) ** 2).item()
                test_results['original_z100_denorm'].append(z100_target_denorm.cpu())
                test_results['condition_z50_denorm'].append(z50_condition_denorm.cpu())
                test_results['generated_z100_denorm'].append(z100_generated_denorm.cpu())
                test_results['mse_scores_denorm'].append(mse_denorm)
            else:
                mse_denorm = None
            
            test_results['original_z100'].append(z100_target.cpu())
            test_results['condition_z50'].append(z50_condition.cpu())
            test_results['generated_z100'].append(z100_generated.cpu())
            test_results['mse_scores'].append(mse)
            
            print(f"  Batch {batch_idx+1}: MSE = {mse:.6f}" + 
                  (f", MSE_denorm = {mse_denorm:.6f}" if mse_denorm is not None else ""))
    
    # 合并结果
    for key in ['original_z100', 'condition_z50', 'generated_z100', 'zcond']:
        test_results[key] = torch.cat(test_results[key], dim=0)
    
    if z50_mean is not None and z50_std is not None:
        for key in ['original_z100_denorm', 'condition_z50_denorm', 'generated_z100_denorm']:
            test_results[key] = torch.cat(test_results[key], dim=0)
        avg_mse_denorm = np.mean(test_results['mse_scores_denorm'])
        print(f"平均采样MSE (反标准化): {avg_mse_denorm:.6f}")
    
    avg_mse = np.mean(test_results['mse_scores'])
    print(f"平均采样MSE (标准化): {avg_mse:.6f}")
    
    # 保存标准化参数
    test_results['z50_mean'] = z50_mean
    test_results['z50_std'] = z50_std
    
    # 保存zcond到文件
    zcond_save_path = os.path.join(cfg.down_stream_dir, f"zcond_test_{int(time.time())}.pth")
    torch.save(test_results['zcond'], zcond_save_path)
    print(f"zcond已保存到: {zcond_save_path}")
    
    return test_results, avg_mse
    
    return test_results, avg_mse
def main(cfg):
    args = parse_args()
    # 更新配置
    cfg.latent_data_dir = args.latent_data_dir
    cfg.save_dir = args.save_dir
    cfg.down_stream_dir = args.down_stream_dir
    cfg.num_epochs = args.epoch
    cfg.device = args.device
    cfg.seed = args.seed
    cfg.train_batch_size = args.train_batch_size
    cfg.num_train_timesteps = args.num_train_timesteps
    cfg.num_inference_steps = args.num_inference_steps
    cfg.learning_rate = args.learning_rate
    cfg.config_file = args.config_file
    cfg.scheduler_s = args.scheduler_s
    cfg.scheduler_schedule = args.scheduler_schedule
    cfg.conv_encoder_ckpt = args.conv_encoder_ckpt
    cfg.conv_encoder_time_emb_dim = args.conv_encoder_time_emb_dim
    cfg.conv_encoder_T = args.conv_encoder_T

    # 设置随机种子
    if cfg.seed is not None:
        set_seed(cfg.seed)

    # 创建保存目录
    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs(cfg.down_stream_dir, exist_ok=True)

    # 加载数据
    train_z100_data, train_z50_data, test_z100_data, test_z50_data, z50_mean, z50_std = load_latent_data(cfg.latent_data_dir)
    train_loader, test_loader = create_data_loaders(
        train_z100_data, train_z50_data, test_z100_data, test_z50_data, cfg.train_batch_size
    )

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # 加载ConvEncoder
    conv_encoder = load_conv_encoder(
        conv_encoder_ckpt=cfg.conv_encoder_ckpt,
        in_channels=train_z50_data.shape[1],
        latent_channels=train_z100_data.shape[1],
        T=cfg.conv_encoder_T,
        time_emb_dim=cfg.conv_encoder_time_emb_dim,
        device=device
    )

    # 主模型
    model = AdaConv_Res_Small(
        signal_length=train_z100_data.shape[-1],
        signal_channel=train_z100_data.shape[-2],
        hidden_channel=12,
        in_kernel_size=7,
        out_kernel_size=7,
        slconv_kernel_size=17,
        num_scales=4,
        num_blocks=4,
        num_off_diag=8,
        use_pos_emb=True,
        padding_mode="circular",
        use_fft_conv=True,
    ).to(device)

    # 残差预测模型
    res_model = TimeAwareResidualPredictor(
        channel_dim=train_z100_data.shape[-2],
        signal_length=train_z100_data.shape[-1],
        time_embed_dim=128,
        max_timesteps=cfg.num_train_timesteps,
        num_layers=2
    ).to(device)

    # DDIM调度器
    scheduler = DDIMScheduler(
        num_train_timesteps=cfg.num_train_timesteps,
        schedule=cfg.scheduler_schedule,
        s=cfg.scheduler_s,
        clip_sample=False,
        steps_offset=1
    )

    # 优化器和学习率调度器（仅优化残差模型）
    res_optimizer = torch.optim.Adam(res_model.parameters(), lr=cfg.learning_rate)
        # 优化器和学习率调度器（仅优化残差模型）
    res_checkpoint_path = os.path.join(cfg.save_dir, "res_model_checkpoint.pth")
    if os.path.exists(res_checkpoint_path):
        print(f"检测到残差模型checkpoint，恢复自: {res_checkpoint_path}")
        res_checkpoint = torch.load(res_checkpoint_path, map_location=device)
        res_model.load_state_dict(res_checkpoint['model_state_dict'])
        res_optimizer.load_state_dict(res_checkpoint['optimizer_state_dict'])
        print("残差模型参数和优化器已恢复。")
    else:
        print("\n=== 先训练残差预测模型 ===")
        train_residual_model(
            res_model, train_loader, res_optimizer, device, num_epochs=500, scheduler_steps=cfg.num_train_timesteps
        )
        torch.save(res_model.state_dict(), os.path.join(cfg.save_dir, "res_model_pretrained.pth"))
        print("残差模型预训练完成并保存。")
        res_checkpoint = {
            'model_state_dict': res_model.state_dict(),
            'optimizer_state_dict': res_optimizer.state_dict(),
            'epoch': 500,
        }
        torch.save(res_checkpoint, res_checkpoint_path)
        print("残差模型checkpoint已保存。")

    # 主模型优化器和学习率调度器（不再优化残差模型）
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    lr_scheduler = MultiStepLR(optimizer, milestones=[100, 150], gamma=0.5)

    # checkpoint恢复
    checkpoint_dir = cfg.save_dir
    best_model_path = os.path.join(cfg.save_dir, "best_latent_ddim_model.pth")
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_epoch_") and f.endswith(".pth")]
    start_epoch = 0
    best_train_loss = float("inf")
    train_losses = []

    if checkpoint_files:
        latest_checkpoint_file = max(checkpoint_files, key=lambda f: int(f.split("_")[-1].split(".")[0]))
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint_file)
        print(f"检测到checkpoint，恢复自: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_train_loss = checkpoint.get('best_train_loss', float("inf"))
        train_losses = checkpoint.get('train_losses', [])
        print(f"恢复训练: 从epoch {start_epoch}，best_train_loss={best_train_loss:.6f}")
    else:
        print("未检测到checkpoint，将从头开始训练。")

    # 训练循环（只优化主模型）
    for epoch in range(start_epoch, cfg.num_epochs):
        model.train()
        res_model.eval()  # 固定残差模型
        epoch_loss = 0.0
        num_batches = len(train_loader)
        for step, batch in enumerate(train_loader):
            z100_target = batch[0].to(device)
            z50 = batch[1].to(device)
            batch_size = z100_target.shape[0]
            timesteps = torch.randint(0, scheduler.num_train_timesteps, (batch_size,), device=device).long()
            zcond = conv_encoder(z50, timesteps)
            noise = torch.randn_like(z100_target)
            noisy_z100 = scheduler.add_noise(z100_target, noise, timesteps)

            optimizer.zero_grad()
            # DDPM噪声预测
            noise_pred = model(noisy_z100, timesteps, zcond)
            ddpm_loss = torch.nn.functional.smooth_l1_loss(noise_pred, noise)
            ddpm_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += ddpm_loss.item()

            if (step + 1) % 10 == 0 or step == 0:
                avg_loss = epoch_loss / (step + 1)
                print(f"Batch [{step+1}/{num_batches}] - Loss: {ddpm_loss.item():.6f}, Avg Loss: {avg_loss:.6f}")

        avg_epoch_loss = epoch_loss / num_batches
        lr_scheduler.step()
        train_losses.append(avg_epoch_loss)
        print(f"\nEpoch {epoch+1}/{cfg.num_epochs} 总结:")
        print(f"  训练损失: {avg_epoch_loss:.6f}")
        print(f"  学习率: {optimizer.param_groups[0]['lr']:.2e}")

        # 保存最佳模型
        if avg_epoch_loss < best_train_loss:
            best_train_loss = avg_epoch_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'train_loss': avg_epoch_loss,
                'best_train_loss': best_train_loss,
                'train_losses': train_losses,
                'config': dict(cfg),
                'model_type': 'latent_ddim_with_res'
            }, best_model_path)
            print(f"  🎉 新的最佳模型已保存! 训练损失: {best_train_loss:.6f}")

        # 定期保存检查点
        if (epoch + 1) % 20 == 0:
            checkpoint_path = os.path.join(cfg.save_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'train_loss': avg_epoch_loss,
                'train_losses': train_losses,
                'best_train_loss': best_train_loss,
                'config': dict(cfg)
            }, checkpoint_path)
            print(f"  💾 检查点已保存: {checkpoint_path}")

    # 保存最终模型
    final_model_path = os.path.join(cfg.save_dir, "final_latent_ddim_model.pth")
    torch.save({
        'epoch': cfg.num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': lr_scheduler.state_dict(),
        'train_losses': train_losses,
        'best_train_loss': best_train_loss,
        'config': dict(cfg),
        'model_type': 'latent_ddim_with_res'
    }, final_model_path)
    print(f"\n🎉 潜在空间DDIM训练完成!")
    print(f"最佳训练损失: {best_train_loss:.6f}")
    print(f"最终模型已保存: {final_model_path}")

    # ========== 测试集采样 ==========
    print(f"\n🔍 开始采样测试...")
    if os.path.exists(best_model_path):
        best_checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(best_checkpoint['model_state_dict'])
        res_model.load_state_dict(torch.load(os.path.join(cfg.save_dir, "res_model_pretrained.pth"), map_location=device))
        test_results, avg_mse = test_sampling(
            model, scheduler, test_loader, device, cfg, conv_encoder, cfg.num_inference_steps, z50_mean, z50_std
        )
        test_sample_path = os.path.join(cfg.down_stream_dir, f"ddim_sampling_test_{int(time.time())}.pth")
        torch.save(test_results, test_sample_path)
        print(f"测试集采样结果已保存: {test_sample_path}")
        print("\n🔍 对训练集进行采样验证...")
        train_results, train_avg_mse = test_sampling(
            model, scheduler, train_loader, device, cfg, conv_encoder, cfg.num_inference_steps, z50_mean, z50_std
        )
        train_sample_path = os.path.join(cfg.down_stream_dir, f"ddim_sampling_train_{int(time.time())}.pth")
        torch.save(train_results, train_sample_path)
        print(f"训练集采样结果已保存: {train_sample_path}")
    else:
        print("未找到最佳模型，跳过采样测试。")
        test_results, avg_mse = None, None

    # 保存训练历史
    history_path = os.path.join(cfg.down_stream_dir, "training_history.pth")
    torch.save({
        'train_losses': train_losses,
        'best_train_loss': best_train_loss,
        'final_test_mse': avg_mse,
        'config': dict(cfg),
        'scheduler_schedule': cfg.scheduler_schedule,
        'scheduler_s': cfg.scheduler_s
    }, history_path)
    print(f"\n✅ 全部完成!")
    print(f"🏗️  配置总结:")
    print(f"  调度器类型: {cfg.scheduler_schedule}")
    print(f"  S参数: {cfg.scheduler_s}")
    print(f"  训练步数: {cfg.num_train_timesteps}")
    print(f"  推理步数: {cfg.num_inference_steps}")
    print(f"模型保存目录: {cfg.save_dir}")
    print(f"结果保存目录: {cfg.down_stream_dir}")
    if avg_mse is not None:
        print(f"最终测试MSE: {avg_mse:.6f}")

if __name__ == "__main__":
    args = parse_args()
    cfg = OmegaConf.create()
    main(cfg)