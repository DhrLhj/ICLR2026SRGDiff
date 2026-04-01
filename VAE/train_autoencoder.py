#仿照德国人写的代码
import argparse
from omegaconf import OmegaConf
import numpy as np
import torch
import random
from nimingcode.VAE.dataset_SEED_DEAP import load_SEED_data,load_DEAP_data_mean_std,DEAP,SEED_condition
from torch.utils.data import DataLoader,TensorDataset
from torch.optim.lr_scheduler import MultiStepLR
from generative.losses import JukeboxLoss, PatchAdversarialLoss
from generative.networks.nets import AutoencoderKL, PatchDiscriminator
import os
from torch.nn import L1Loss
import time
from torch import optim

def parse_args():
    
    parser = argparse.ArgumentParser(description="Diffusion Training Script")
    parser.add_argument('--dataset_name', type=str, default='SEED', help='Name of the dataset')
    parser.add_argument('--patient_id', type=str, default='4', help='Patient ID for the dataset')
    parser.add_argument('--shuffle', type=bool, default=True, help='Whether to shuffle the dataset')
    parser.add_argument('--num_channel', type=int, default=62, help='Number of channels in the signal')
    parser.add_argument('--signal_length', type=int, default=800, help='Length of the signal')
    parser.add_argument('--dataset_dir', type=str, default='/data/zhouleyu/Preprocessed_EEG', help='Directory of the dataset')
    parser.add_argument('--save_dir', type=str, default='/data/zhouleyu/autoencoder_for_SEED/checkpoint/SEED_4', help='Directory to save results')
    parser.add_argument('--down_stream_dir', type=str, default="/data/zhouleyu/autoencoder_for_SEED/downstream/SEED_4")
    parser.add_argument('--epoch', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--device', type=str, default='cuda:1', help='Device to use (e.g., cuda:1, cpu)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--mask_training', type=bool, default="true", help='是否mask_training')
    parser.add_argument('--train_batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--network', type=str, default="CatConv_new", help='Network architecture to use (e.g., CatConv_new)')
    parser.add_argument('--num_timesteps', type=int, default=50)
    parser.add_argument('--noise', type=str, default="white", help='noise type (white)')
    parser.add_argument('--config_file', type=str, default="/data/zhouleyu/autoencoder_for_SEED/config.yaml")
    return parser.parse_args()

def set_seed(seed: int):
    """
    Set the seed for all random number generators and switch to deterministic algorithms.
    This can hurt performance!

    Args:
        seed: The random seed.
    """

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def init_dataset(cfg):
    train_eeg, train_labels, train_mask, test_eeg, test_labels, test_mask = load_SEED_data(
        cfg.patient_id,
        cfg.signal_length,
        cfg.shuffle,
        cfg.dataset_dir,
    )
    
    # 确保数据是tensor格式
    if isinstance(train_eeg, np.ndarray):
        train_eeg = torch.from_numpy(train_eeg).float()
    if isinstance(test_eeg, np.ndarray):
        test_eeg = torch.from_numpy(test_eeg).float()
    if isinstance(train_labels, np.ndarray):
        train_labels = torch.from_numpy(train_labels)
    if isinstance(test_labels, np.ndarray):
        test_labels = torch.from_numpy(test_labels)
    
    # 创建TensorDataset
    train_dataset = TensorDataset(train_eeg, train_labels)
    test_dataset = TensorDataset(test_eeg, test_labels)
    
    # 设置DataLoader参数
    # if cfg.device.startswith("cuda") and torch.cuda.is_available():
    #     dataloader_kwargs = {"num_workers": 4, "pin_memory": True}
    # else:
    
    dataloader_kwargs = {"num_workers": 0}

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        **dataloader_kwargs
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.train_batch_size,
        shuffle=False,
        **dataloader_kwargs
    )
    
    return train_loader, test_loader, train_eeg, test_eeg, train_labels, test_labels

    
    
    
def test_and_save_reconstruction(model, test_loader, cfg, device):
    """
    测试数据重建并保存原始和重建数据
    """
    print("\n开始测试数据重建...")
    
    model.eval()
    original_data_list = []
    reconstructed_data_list = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            eeg_data = batch[0].to(device)  # [batch, 62, 800]
            
            # 通过autoencoder重建
            reconstruction, miu, sigma = model(eeg_data)
            print("miu.shape",miu.shape)
            print("sigma.shape",sigma.shape)
            # 保存到CPU
            original_data_list.append(eeg_data.cpu())
            reconstructed_data_list.append(reconstruction.cpu())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Processing batch {batch_idx + 1}/{len(test_loader)}")
    
    # 合并所有batch的数据
    original_data = torch.cat(original_data_list, dim=0)  # [total_samples, 62, 800]
    reconstructed_data = torch.cat(reconstructed_data_list, dim=0)  # [total_samples, 62, 800]
    
    print(f"原始数据形状: {original_data.shape}")
    print(f"重建数据形状: {reconstructed_data.shape}")
    
    # 确保保存目录存在，如果存在则继续（用于覆盖文件）
    if not os.path.exists(cfg.down_stream_dir):
        os.makedirs(cfg.down_stream_dir, exist_ok=True)
        print(f"创建目录: {cfg.down_stream_dir}")
    else:
        print(f"目录已存在，将覆盖现有文件: {cfg.down_stream_dir}")
    
    # 保存数据（覆盖模式）
    original_path = os.path.join(cfg.down_stream_dir, "original_test_data.pth")
    reconstructed_path = os.path.join(cfg.down_stream_dir, "reconstructed_test_data.pth")
    
    # 检查文件是否存在并提示覆盖
    if os.path.exists(original_path):
        print(f"覆盖现有文件: {original_path}")
    if os.path.exists(reconstructed_path):
        print(f"覆盖现有文件: {reconstructed_path}")
    
    torch.save(original_data, original_path)
    torch.save(reconstructed_data, reconstructed_path)
    
    print(f"原始测试数据已保存至: {original_path}")
    print(f"重建测试数据已保存至: {reconstructed_path}")
    
    # 计算重建质量指标
    mse_loss = torch.mean((original_data - reconstructed_data) ** 2)
    mae_loss = torch.abs(original_data - reconstructed_data).mean()
    
    print(f"重建质量指标:")
    print(f"  MSE Loss: {mse_loss:.6f}")
    print(f"  MAE Loss: {mae_loss:.6f}")
    
    # 可选：计算每个通道的重建质量
    channel_mse = torch.mean((original_data - reconstructed_data) ** 2, dim=(0, 2))  # [62]
    print(f"  各通道平均MSE: {channel_mse.mean():.6f} ± {channel_mse.std():.6f}")
    
    return original_data, reconstructed_data

# ...existing code...

def single_inference_speed_test(model, test_loader, device):
    """
    单条数据推理速度测试
    从测试数据中随机选择一条数据，测试模型推理速度
    """
    import time
    
    print("\n开始单条数据推理速度测试...")
    
    model.eval()
    
    # 从测试数据中随机选择一条数据
    test_dataset = test_loader.dataset
    random_idx = torch.randint(0, len(test_dataset), (1,)).item()
    single_sample = test_dataset[random_idx]
    single_eeg = single_sample[0].unsqueeze(0).to(device).float()  # [1, 62, 800]
    
    print(f"随机选择的数据索引: {random_idx}")
    print(f"输入数据形状: {single_eeg.shape}")
    
    
    # 记录开始时间
    start_time = time.time()
    
    with torch.no_grad():
        # 进行单次推理
        reconstruction, z_mu, z_sigma = model(single_eeg)
    
    # 记录结束时间
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"推理完成!")
    print(f"输出数据形状: {reconstruction.shape}")
    print(f"潜在空间均值形状: {z_mu.shape}")
    print(f"潜在空间方差形状: {z_sigma.shape}")
    print(f"单次推理耗时: {elapsed_time:.6f} 秒")
    print(f"推理速度: {1/elapsed_time:.2f} samples/second")
    
    return elapsed_time, reconstruction.cpu()



def main(cfg):
    """
    Main function.

    Args:
        cfg: The config object.
    """
    # 解析命令行参数
    args = parse_args()
    config = OmegaConf.load(args.config_file)

    # 更新 cfg 的参数
    cfg.name = args.dataset_name
    cfg.num_channel = args.num_channel
    cfg.signal_length = args.signal_length
    cfg.dataset_dir = args.dataset_dir
    cfg.save_dir = args.save_dir
    cfg.num_epochs = args.epoch
    cfg.device = args.device
    cfg.seed = args.seed
    cfg.patient_id = args.patient_id
    cfg.mask_training = args.mask_training
    cfg.train_batch_size = args.train_batch_size
    cfg.down_stream_dir = args.down_stream_dir
    cfg.num_timesteps = args.num_timesteps
    cfg.noise = args.noise
    cfg.network = args.network
    cfg.shuffle = args.shuffle
    print("cfg.shuffle", cfg.shuffle)
    
    if cfg.seed is not None:
        set_seed(cfg.seed)

    # 使用命令行传入的 device
    if cfg.device.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(cfg.device)
        print(device)
        environ_kwargs = {"num_workers": 0, "pin_memory": True}
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        environ_kwargs = {}
        print("Using CPU")

    train_loader, test_loader, _, _, _, _ = init_dataset(cfg)

    print(f"Train dataset size: {len(train_loader.dataset)}, Test dataset size: {len(test_loader.dataset)}")

    autoencoder_args = config.autoencoderkl.params
    autoencoder_args['num_channels'] = [32,24,16]
    autoencoder_args['latent_channels'] = 8
    
    model = AutoencoderKL(**autoencoder_args)
    discriminator_dict = config.patchdiscriminator.params
    discriminator = PatchDiscriminator(**discriminator_dict)

    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    device = cfg.device

    model = model.to(device)
    discriminator = discriminator.to(device)
    
    # 优化器
    optimizer_g = torch.optim.Adam(params=model.parameters(),
                                   lr=config.models.optimizer_g_lr)
    optimizer_d = torch.optim.Adam(params=discriminator.parameters(),
                                   lr=config.models.optimizer_d_lr)
    
    # 损失函数
    l1_loss = L1Loss()
    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    adv_weight = config.models.adv_weight
    #jukebox_loss = JukeboxLoss(spatial_dims=1, reduction="sum")

    # 训练配置
    kl_weight = config.models.kl_weight
    n_epochs = config.train.n_epochs
    val_interval = config.train.val_interval
    spectral_weight = config.models.spectral_weight
    
    # 记录损失
    epoch_recon_loss_list = []
    epoch_gen_loss_list = []
    epoch_disc_loss_list = []
    epoch_spectral_loss_list = []
    val_recon_epoch_loss_list = []
    best_loss = float("inf")
    start_epoch = 0
    
    # 检查点目录
    checkpoint_dir = os.path.join(cfg.save_dir, "checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 检查点加载
    if os.path.exists(checkpoint_dir):
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("epoch_") and f.endswith(".pth")]
        if checkpoint_files:
            latest_checkpoint_file = max(checkpoint_files, key=lambda f: int(f.split("_")[1].split(".")[0]))
            checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint_file)
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
            optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
            start_epoch = checkpoint['epoch']
            best_loss = checkpoint.get('best_loss', float("inf"))
            print(f"Resuming training from epoch {start_epoch}, best loss: {best_loss:.6f}")
    else:
        print("No checkpoint found. Starting training from scratch.")

    print(f"Starting training for {n_epochs} epochs...")
    
    for epoch in range(start_epoch, n_epochs):
        model.train()
        discriminator.train()
        
        # 训练指标
        epoch_loss = 0
        gen_epoch_loss = 0
        disc_epoch_loss = 0
        spectral_epoch_loss = 0
        num_batches = len(train_loader)
        
        print(f"\nEpoch {epoch+1}/{n_epochs}")
        print("-" * 50)
        
        for step, batch in enumerate(train_loader):
            eeg_data = batch[0].to(device)  # TensorDataset返回的是元组

            # 生成器训练
            optimizer_g.zero_grad(set_to_none=True)
            reconstruction, z_mu, z_sigma = model(eeg_data)
            
            # 重建损失
            recons_loss = l1_loss(reconstruction.float(), eeg_data.float())
            
            # 频谱损失
            #recons_spectral = jukebox_loss(reconstruction.float(), eeg_data.float())
            
            # KL散度损失
            kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1])
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
            
            # 对抗损失
            logits_fake = discriminator(reconstruction.contiguous().float())[-1]
            generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
            
            # 总生成器损失
            if hasattr(args, 'spe') and args.spe == "spectral":
                loss_g = recons_loss + kl_weight * kl_loss + adv_weight * generator_loss #+ recons_spectral * spectral_weight
            else:
                loss_g = recons_loss + kl_weight * kl_loss + adv_weight * generator_loss
            
            loss_g.backward()
            optimizer_g.step()

            # 判别器训练
            optimizer_d.zero_grad(set_to_none=True)
            
            logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
            loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
            logits_real = discriminator(eeg_data.contiguous().detach())[-1]
            loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
            discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
            
            loss_d = adv_weight * discriminator_loss
            loss_d.backward()
            optimizer_d.step()
            
            # 累积损失
            epoch_loss += recons_loss.item()
            gen_epoch_loss += generator_loss.item()
            disc_epoch_loss += discriminator_loss.item()
            #spectral_epoch_loss += recons_spectral.item()
            
            # 每10个batch打印一次进度
            if (step + 1) % 10 == 0 or step == 0:
                print(f"Batch [{step+1}/{num_batches}] - "
                      f"Recon: {recons_loss.item():.6f}, "
                      f"Gen: {generator_loss.item():.6f}, "
                      f"Disc: {discriminator_loss.item():.6f}")
        
        # 计算平均损失
        avg_recon_loss = epoch_loss / num_batches
        avg_gen_loss = gen_epoch_loss / num_batches
        avg_disc_loss = disc_epoch_loss / num_batches
        avg_spectral_loss = spectral_epoch_loss / num_batches
        
        # 记录损失
        epoch_recon_loss_list.append(avg_recon_loss)
        epoch_gen_loss_list.append(avg_gen_loss)
        epoch_disc_loss_list.append(avg_disc_loss)
        epoch_spectral_loss_list.append(avg_spectral_loss)
        
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Reconstruction Loss: {avg_recon_loss:.6f}")
        print(f"  Generator Loss: {avg_gen_loss:.6f}")
        print(f"  Discriminator Loss: {avg_disc_loss:.6f}")
        print(f"  Spectral Loss: {avg_spectral_loss:.6f}")
        
        # 定期保存检查点
        if (epoch + 1) % 50 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch + 1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_g_state_dict': optimizer_g.state_dict(),
                'optimizer_d_state_dict': optimizer_d.state_dict(),
                'best_loss': best_loss,
                'recon_loss_history': epoch_recon_loss_list,
                'gen_loss_history': epoch_gen_loss_list,
                'disc_loss_history': epoch_disc_loss_list,
            }, checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")

    print("\nTraining completed!")
    print(f"Best validation loss: {best_loss:.6f}")
    
    # 保存最终模型和损失历史
    final_model_path = os.path.join(cfg.save_dir, "final_model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'recon_loss_history': epoch_recon_loss_list,
        'gen_loss_history': epoch_gen_loss_list,
        'disc_loss_history': epoch_disc_loss_list,
        'val_loss_history': val_recon_epoch_loss_list,
    }, final_model_path)
    print(f"Final model saved: {final_model_path}")

    print("\n" + "="*60)
    print("开始推理速度测试...")
    
    inference_time, sample_reconstruction = single_inference_speed_test(
        model, test_loader, device
    )
    
    print(f"\n推理速度测试完成，单次推理时间: {inference_time:.6f} 秒")
    
     # 新增：测试数据重建和保存
    print("\n" + "="*60)
    print("开始测试阶段...")
    
    
    # 进行测试数据重建
    original_test_data, reconstructed_test_data = test_and_save_reconstruction(
        model, test_loader, cfg, device
    )

    
    
    
    
if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()
    # 创建一个空的配置对象
    cfg = OmegaConf.create()
    # 运行主函数
    main(cfg)