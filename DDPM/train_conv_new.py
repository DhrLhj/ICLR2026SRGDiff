import torch
import torch.nn as nn
import torch.optim as optim
import os
from omegaconf import OmegaConf
import math
from generative.networks.nets import AutoencoderKL

class ConvEncoder(nn.Module):
    def __init__(self, in_channels, latent_channels, T=1000, mode='linear', min_gamma=0.1, time_emb_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, latent_channels, kernel_size=3, padding=1)
        )
        self.T = T
        self.mode = mode
        self.min_gamma = min_gamma
        self.time_emb_dim = time_emb_dim
        self.time_proj = nn.Linear(time_emb_dim, latent_channels)

    def get_time_embedding(self, t):
        device = t.device
        half_dim = self.time_emb_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t * emb
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb

    def forward(self, x, t):
        h = self.conv(x)  # [B, latent_channels, L]
        t = t.float().view(-1, 1, 1)
        t_scalar = t[:, 0, 0].view(-1, 1)
        if self.mode == 'linear':
            gamma = 1.0 - 0.2 * (t / self.T)
        else:
            raise ValueError("Unknown mode for gamma")
        gamma = torch.clamp(gamma, min=self.min_gamma, max=1.0)
        time_emb = self.get_time_embedding(t_scalar)
        time_emb = self.time_proj(time_emb)
        time_emb = time_emb.unsqueeze(-1)
        time_emb = time_emb.expand(-1, -1, h.shape[2])
        out = gamma * h + time_emb * (1-gamma)  
        return out
    
import torch

def save_train_infer(conv_encoder, vae, args, device):
    train_z50 = torch.load(os.path.join(args.data_dir, "train_z50.pth")).float()
    train_z100 = torch.load(os.path.join(args.data_dir, "train_z100.pth")).float()
    train_origin = torch.load(os.path.join(args.data_dir, "train_original.pth")).float()
    batch_size = 64

    conv_encoder.eval()
    vae.eval()

    z_train_list = []
    recon_train_list = []
    train_z100_list = []
    train_origin_list = []

    with torch.no_grad():
        for i in range(0, train_z50.shape[0], batch_size):
            batch_z50 = train_z50[i:i+batch_size].to(device)
            batch_z100 = train_z100[i:i+batch_size]
            batch_origin = train_origin[i:i+batch_size]
            t_train = torch.zeros(batch_z50.shape[0], device=device)
            z_train = conv_encoder(batch_z50, t_train)
            recon_train = vae.decode(z_train).cpu()
            # 裁剪长度一致
            min_len = min(recon_train.shape[2], batch_origin.shape[2])
            recon_train = recon_train[..., :min_len]
            batch_origin = batch_origin[..., :min_len]
            z_train = z_train[..., :min_len]
            batch_z100 = batch_z100[..., :min_len]
            z_train_list.append(z_train.cpu())
            recon_train_list.append(recon_train.cpu())
            train_z100_list.append(batch_z100)
            train_origin_list.append(batch_origin)

    # 拼接所有batch
    z_train_all = torch.cat(z_train_list, dim=0)
    recon_train_all = torch.cat(recon_train_list, dim=0)
    train_z100_all = torch.cat(train_z100_list, dim=0)
    train_origin_all = torch.cat(train_origin_list, dim=0)

    torch.save(z_train_all, os.path.join(args.save_dir, "train_reconstructed_latent.pth"))
    torch.save(train_z100_all, os.path.join(args.save_dir, "train_target_latent.pth"))
    torch.save(recon_train_all, os.path.join(args.save_dir, "train_reconstructed_data.pth"))
    torch.save(train_origin_all, os.path.join(args.save_dir, "train_origin_data.pth"))
    print(f"训练集推理结果已分批保存！")

# 在 main() 里调用
# save_train_infer(conv_encoder, vae, args, device)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="训练ConvEncoder + 固定VAE重建")
    parser.add_argument('--data_dir', type=str, default="/data/zhouleyu/ddpm_for_SEED/data/SEED_all_klbig_50%_nomask", help='数据目录')
    parser.add_argument('--vae_model_path', type=str, default="/data/zhouleyu/autoencoder_for_SEED_new/checkpoint/SEED_all_klbig/checkpoint/best_model.pth", help='VAE最优模型路径')
    parser.add_argument('--save_dir', type=str, default="/data/zhouleyu/ddpm_for_SEED/best_conv_result/50%_time0.2", help='重建结果保存目录')
    parser.add_argument('--config_file', type=str, default="/data/zhouleyu/autoencoder_for_SEED_new/config_big_klbig.yaml", help='VAE结构配置yaml')
    parser.add_argument('--device', type=str, default='cuda:2', help='设备')
    parser.add_argument('--epochs', type=int, default=300, help='训练轮数')
    parser.add_argument('--lr', type=float, default=5*1e-4, help='学习率')
    parser.add_argument('--checkpoint_interval', type=int, default=20, help='每多少轮保存一次checkpoint')
    parser.add_argument('--resume', type=str, default=None, help='checkpoint路径，若指定则恢复训练')
    parser.add_argument('--latent_loss_weight', type=float, default=100.0, help='隐空间loss权重')
    parser.add_argument('--recon_loss_weight', type=float, default=1.0, help='重建loss权重')
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    # 1. 读取训练数据
    train_z50 = torch.load(os.path.join(args.data_dir, "train_z50.pth"))  # [N, C, L]
    train_z100 = torch.load(os.path.join(args.data_dir, "train_z100.pth"))  # [N, C, L]
    train_origin = torch.load(os.path.join(args.data_dir, "train_original.pth"))  # [N, C, L]
    print(f"train_z50: {train_z50.shape}, train_z100: {train_z100.shape}, train_origin: {train_origin.shape}")

    # 2. 加载VAE结构参数
    config = OmegaConf.load(args.config_file)
    autoencoder_args = config.autoencoderkl.params
    in_channels = train_z50.shape[1]
    latent_channels = int(autoencoder_args.get('latent_channels', 32))
    autoencoder_args['num_channels'] = [int(x) for x in autoencoder_args.get('num_channels', [64,32])]
    autoencoder_args['latent_channels'] = latent_channels

    # 3. 加载VAE模型
    vae = AutoencoderKL(**autoencoder_args).to(device)
    vae_ckpt = torch.load(args.vae_model_path, map_location=device)
    vae.load_state_dict(vae_ckpt['model_state_dict'] if 'model_state_dict' in vae_ckpt else vae_ckpt)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    # 4. 构造ConvEncoder
    conv_encoder = ConvEncoder(in_channels, latent_channels).to(device)

    # 5. 优化器与损失
    optimizer = optim.Adam(conv_encoder.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # 6. 数据准备
    train_z50 = train_z50.float().to(device)
    train_z100 = train_z100.float().to(device)
    train_origin = train_origin.float().to(device)
    batch_size = 64
    dataset = torch.utils.data.TensorDataset(train_z50, train_z100, train_origin)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 7. 恢复checkpoint（如有）
    start_epoch = 0
    best_loss = float('inf')
    if args.resume is not None and os.path.isfile(args.resume):
        print(f"恢复训练: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        conv_encoder.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint.get('epoch', 0)
        best_loss = checkpoint.get('best_loss', float('inf'))

    # 8. 训练ConvEncoder并保存checkpoint
    conv_encoder.train()
    for epoch in range(start_epoch, args.epochs):
        total_loss = 0
        total_latent_loss = 0
        total_recon_loss = 0
        for batch_z, batch_z100, batch_origin in dataloader:
            batch_z = batch_z.to(device)
            batch_z100 = batch_z100.to(device)
            batch_origin = batch_origin.to(device)
            t = torch.randint(0, args.epochs + 1, (batch_z.size(0),), device=device)  # 或 args.T + 1
            optimizer.zero_grad()
            z = conv_encoder(batch_z,t)  # [B, latent_channels, L]
            recon = vae.decode(z)
            latent_loss = criterion(z, batch_z100)
            recon_loss = criterion(recon, batch_origin)
            loss = args.latent_loss_weight * latent_loss + args.recon_loss_weight * recon_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_z.size(0)
            total_latent_loss += latent_loss.item() * batch_z.size(0)
            total_recon_loss += recon_loss.item() * batch_z.size(0)
        avg_loss = total_loss / len(dataset)
        avg_latent_loss = total_latent_loss / len(dataset)
        avg_recon_loss = total_recon_loss / len(dataset)
        print(f"Epoch {epoch+1}/{args.epochs}, Total Loss: {avg_loss:.6f}, Latent MSE: {avg_latent_loss:.6f}, Recon MSE: {avg_recon_loss:.6f}")

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(conv_encoder.state_dict(), os.path.join(args.save_dir, "best_conv_encoder.pth"))
        # 定期保存checkpoint
        if (epoch + 1) % args.checkpoint_interval == 0 or (epoch + 1) == args.epochs:
            checkpoint = {
                'model': conv_encoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
                'best_loss': best_loss
            }
            torch.save(checkpoint, os.path.join(args.save_dir, f"conv_encoder_epoch{epoch+1}.pth"))
            
            
    #save_train_infer(conv_encoder, vae, args, device)

    # 9. 测试集推理与保存
    # ...前面代码保持不变...

    # 9. 测试集推理与保存
    test_z50 = torch.load(os.path.join(args.data_dir, "test_z50.pth")).float().to(device)
    test_z100 = torch.load(os.path.join(args.data_dir, "test_z100.pth")).float()
    test_origin = torch.load(os.path.join(args.data_dir, "test_original.pth")).float()
    # 加载最佳模型参数（支持断点续训和直接推理）
    best_ckpt_path = os.path.join(args.save_dir, "best_conv_encoder.pth")
    if os.path.exists(best_ckpt_path):
        conv_encoder.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    else:
        print(f"未找到 {best_ckpt_path}，使用当前模型参数进行推理。")
    conv_encoder.eval()
    with torch.no_grad():
        t_test = torch.zeros(test_z50.shape[0], device=device)  # 推理时t=0
        z = conv_encoder(test_z50, t_test)
        recon = vae.decode(z).cpu()
        if recon.shape[2] != test_origin.shape[2]:
            min_len = min(recon.shape[2], test_origin.shape[2])
            recon = recon[..., :min_len]
            test_origin = test_origin[..., :min_len]
            z = z[..., :min_len]
            test_z100 = test_z100[..., :min_len]
    torch.save(z.cpu(), os.path.join(args.save_dir, "test_reconstructed_latent.pth"))
    torch.save(test_z100, os.path.join(args.save_dir, "test_target_latent.pth"))
    torch.save(recon, os.path.join(args.save_dir, "test_reconstructed_data.pth"))
    torch.save(test_origin, os.path.join(args.save_dir, "test_origin_data.pth"))
    print(f"测试集原始数据已保存至: {os.path.join(args.save_dir, 'test_origin_data.pth')}")
    print(f"测试集重建数据已保存至: {os.path.join(args.save_dir, 'test_reconstructed_data.pth')}")
    print(f"测试集隐空间重建已保存至: {os.path.join(args.save_dir, 'test_reconstructed_latent.pth')}")
    
    save_train_infer(conv_encoder, vae, args, device)

    
     # ========== 保存训练集推理结果 ==========
    # train_z50 = torch.load(os.path.join(args.data_dir, "train_z50.pth")).float().to(device)
    # train_z100 = torch.load(os.path.join(args.data_dir, "train_z100.pth")).float()
    # train_origin = torch.load(os.path.join(args.data_dir, "train_original.pth")).float()
    # t_train = torch.zeros(train_z50.shape[0], device=device)
    # z_train = conv_encoder(train_z50, t_train)
    # recon_train = vae.decode(z_train).cpu()
    # if recon_train.shape[2] != train_origin.shape[2]:
    #     min_len = min(recon_train.shape[2], train_origin.shape[2])
    #     recon_train = recon_train[..., :min_len]
    #     train_origin = train_origin[..., :min_len]
    #     z_train = z_train[..., :min_len]
    #     train_z100 = train_z100[..., :min_len]
    # torch.save(z_train.cpu(), os.path.join(args.save_dir, "train_reconstructed_latent.pth"))
    # torch.save(train_z100, os.path.join(args.save_dir, "train_target_latent.pth"))
    # torch.save(recon_train, os.path.join(args.save_dir, "train_reconstructed_data.pth"))
    # torch.save(train_origin, os.path.join(args.save_dir, "train_origin_data.pth"))
    # print(f"训练集原始数据已保存至: {os.path.join(args.save_dir, 'train_origin_data.pth')}")
    # print(f"训练集重建数据已保存至: {os.path.join(args.save_dir, 'train_reconstructed_data.pth')}")
    # print(f"训练集隐空间重建已保存至: {os.path.join(args.save_dir, 'train_reconstructed_latent.pth')}")

if __name__ == "__main__":
    main()