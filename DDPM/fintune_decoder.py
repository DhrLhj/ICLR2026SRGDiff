import torch
import os
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from omegaconf import OmegaConf
import argparse
import random
import numpy as np

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class GatedConvFusion(nn.Module):
    def __init__(self, in_channels=32, kernel_size=3, delta_weight=0.2):
        super().__init__()
        padding = kernel_size // 2
        self.delta_conv = nn.Conv1d(in_channels, in_channels, kernel_size, padding=padding)
        self.gate_conv = nn.Sequential(
            nn.Conv1d(in_channels * 2, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.delta_weight = delta_weight  # 残差比重

    def forward(self, low_feat, high_feat):
        delta = self.delta_conv((high_feat - low_feat))  # [B, C, L]
        gate_input = torch.cat([low_feat, high_feat], dim=1)  # [B, 2*C, L]
        gate = self.gate_conv(gate_input)  # [B, C, L]
        fused = low_feat + gate * delta * self.delta_weight
        return fused

def load_data(sample_path, zcond_path, z100_path):
    sample_data = torch.load(sample_path)
    zcond = torch.load(zcond_path)
    sample = sample_data.get("generated_z100_denorm")
    z100 = torch.load(z100_path)
    assert sample is not None and zcond is not None and z100 is not None
    return sample, zcond, z100

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune GatedConvFusion Model")
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)

    # 路径配置
    train_sample_path = "/data/zhouleyu/ddpm_for_SEED/downstream_nomask_resconv/adanew_75%_s_0.02_with_ConvEncoder/ddim_sampling_train_1753256105.pth"
    train_zcond_path = "/data/zhouleyu/ddpm_for_SEED/best_conv_result/75%_time0.2/train_reconstructed_latent.pth"
    test_sample_path = "/data/zhouleyu/ddpm_for_SEED/downstream_nomask_resconv/adanew_75%_s_0.02_with_ConvEncoder/ddim_sampling_test_1753255746.pth"
    test_zcond_path = "/data/zhouleyu/ddpm_for_SEED/best_conv_result/75%_time0.2/test_reconstructed_latent.pth"
    train_z100_path = "/data/zhouleyu/ddpm_for_SEED/data/SEED_all_klbig_75%_nomask/train_z100.pth"
    test_z100_path = "/data/zhouleyu/ddpm_for_SEED/data/SEED_all_klbig_75%_nomask/test_z100.pth"
    model_path = "/data/zhouleyu/autoencoder_for_SEED_new/checkpoint/SEED_all_klbig/checkpoint/best_model.pth"
    config_file = "/data/zhouleyu/autoencoder_for_SEED_new/config_big_klbig.yaml"
    save_dir = "/data/zhouleyu/ddpm_for_SEED/downstream_fuse_crossattention/75%_0.01_cond_new"
    os.makedirs(save_dir, exist_ok=True)

    # 读取train和test数据
    train_sample, train_zcond, train_z100 = load_data(train_sample_path, train_zcond_path, train_z100_path)
    test_sample, test_zcond, test_z100 = load_data(test_sample_path, test_zcond_path, test_z100_path)

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    # 加载VAE decoder（用于对比和test推理）
    config = OmegaConf.load(config_file)
    autoencoder_args = config.autoencoderkl.params
    in_channels = train_zcond.shape[1]
    autoencoder_args['num_channels'] = [64,32]
    autoencoder_args['latent_channels'] = in_channels
    model = torch.load(model_path, map_location=device)
    if isinstance(model, dict) and 'model_state_dict' in model:
        from generative.networks.nets import AutoencoderKL
        vae = AutoencoderKL(**autoencoder_args).to(device)
        vae.load_state_dict(model['model_state_dict'])
    else:
        vae = model.to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    # ====== 直接将zcond输出decoder并保存（用于对比）======
    with torch.no_grad():
        recon_zcond = vae.decode(test_zcond.to(device)).cpu()
    torch.save(recon_zcond, os.path.join(save_dir, "test_zcond_decoder_recon.pth"))
    print(f"test_zcond经过decoder重建数据已保存到 {os.path.join(save_dir, 'test_zcond_decoder_recon.pth')}")

    # 构造融合模块
    fusion = GatedConvFusion(in_channels=in_channels, kernel_size=3, delta_weight=0.01).to(device)

    # checkpoint路径
    fusion_ckpt_path = os.path.join(save_dir, "fusion_model_checkpoint.pth")
    start_epoch = 0

    # 构造train和test数据集
    train_dataset = TensorDataset(train_sample, train_zcond, train_z100)
    test_dataset = TensorDataset(test_sample, test_zcond, test_z100)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.Adam(fusion.parameters(), lr=1e-4)
    mse_loss = nn.MSELoss()

    # 检查是否有checkpoint
    if os.path.exists(fusion_ckpt_path):
        print(f"检测到模型checkpoint，恢复自: {fusion_ckpt_path}")
        checkpoint = torch.load(fusion_ckpt_path, map_location=device)
        fusion.load_state_dict(checkpoint['fusion_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        print(f"恢复训练: 从epoch {start_epoch}")

    num_epochs = args.num_epochs
    for epoch in range(start_epoch, num_epochs):
        fusion.train()
        epoch_loss = 0.0
        for sample, zcond, z100 in train_loader:
            sample = sample.to(device)
            zcond = zcond.to(device)
            z100 = z100.to(device)
            fused = fusion(zcond, sample)
            loss = mse_loss(fused, z100)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * sample.size(0)
        avg_loss = epoch_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs} - Train Latent MSE: {avg_loss:.6f}")

        # 保存checkpoint
        if (epoch + 1) % 10 == 0 or (epoch + 1) == num_epochs:
            checkpoint = {
                'fusion_state_dict': fusion.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'train_loss': avg_loss
            }
            torch.save(checkpoint, fusion_ckpt_path)
            print(f"Checkpoint saved at epoch {epoch+1}")

        # 每轮在test集评估
        fusion.eval()
        test_loss = 0.0
        with torch.no_grad():
            for sample, zcond, z100 in test_loader:
                sample = sample.to(device)
                zcond = zcond.to(device)
                z100 = z100.to(device)
                fused = fusion(zcond, sample)
                loss = mse_loss(fused, z100)
                test_loss += loss.item() * sample.size(0)
        avg_test_loss = test_loss / len(test_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs} - Test Latent MSE: {avg_test_loss:.6f}")

    # 保存微调后的融合模块
    torch.save(fusion.state_dict(), os.path.join(save_dir, "gated_conv_fusion.pth"))
    print(f"已保存融合模块到 {save_dir}")

    # ========== test集推理并保存重建数据 ==========
    fusion.eval()
    vae.eval()
    recon_list = []
    with torch.no_grad():
        for sample, zcond, z100 in test_loader:
            sample = sample.to(device)
            zcond = zcond.to(device)
            fused = fusion(zcond, sample)
            recon = vae.decode(fused)
            recon_list.append(recon.cpu())
    recon_all = torch.cat(recon_list, dim=0)
    torch.save(recon_all, os.path.join(save_dir, "test_reconstructed_data.pth"))
    print(f"test集重建数据已保存到 {os.path.join(save_dir, 'test_reconstructed_data.pth')}")

if __name__ == "__main__":
    main()