import torch
import os
import numpy as np
from omegaconf import OmegaConf
from generative.networks.nets import AutoencoderKL

def load_data(sample_path, zcond_path, original_path):
    sample_data = torch.load(sample_path)
    zcond = torch.load(zcond_path)
    original = torch.load(original_path)
    assert original is not None and zcond is not None
    return zcond, original

def mse(x, y):
    return np.mean((x - y) ** 2)

def nmse(x, y):
    return np.sum((x - y) ** 2) / np.sum(y ** 2)

def pcc(x, y):
    x_flat = x.flatten()
    y_flat = y.flatten()
    return np.corrcoef(x_flat, y_flat)[0, 1]

def snr(x, y):
    signal_power = np.mean(x ** 2)
    noise_power = np.mean((x - y) ** 2)
    return 10 * np.log10(signal_power / (noise_power + 1e-8))

def evaluate(zcond, original, decoder, device):
    zcond = zcond.to(device)
    original = original.to(device)
    with torch.no_grad():
        recon = decoder(zcond)
    # 断言保证 recon 和 original 形状一致
    assert recon.shape == original.shape, f"recon shape {recon.shape} != original shape {original.shape}"
    recon_np = recon.cpu().numpy()
    original_np = original.cpu().numpy()
    
    print("recon_np",recon_np[0][0][:10])
    print("original_np",original_np[0][0][:10])
    mse_score = mse(recon_np, original_np)
    nmse_score = nmse(recon_np, original_np)
    pcc_score = pcc(recon_np, original_np)
    snr_score = snr(original_np, recon_np)
    return mse_score, nmse_score, pcc_score, snr_score

def main():
    # 路径配置
    train_sample_path = "/data/zhouleyu/ddpm_for_SEED/downstream_nomask_resconv/adanew_90%_s_0.02_with_ConvEncoder/ddim_sampling_train_1753174959.pth"
    train_zcond_path = "/data/zhouleyu/ddpm_for_SEED/downstream_nomask_resconv/adanew_90%_s_0.02_with_ConvEncoder/zcond_train_1753174957.pth"
    test_sample_path = "/data/zhouleyu/ddpm_for_SEED/downstream_nomask_resconv/adanew_90%_s_0.02_with_ConvEncoder/ddim_sampling_test_1753174460.pth"
    test_zcond_path = "/data/zhouleyu/ddpm_for_SEED/best_conv/test_reconstructed_latent.pth"
    train_original_path = "/data/zhouleyu/ddpm_for_SEED/data/SEED_all_klbig_90%_nomask/train_original.pth"
    test_original_path = "/data/zhouleyu/ddpm_for_SEED/best_conv/test_origin_data.pth"
    model_path = "/data/zhouleyu/autoencoder_for_SEED_new/checkpoint/SEED_all_klbig/checkpoint/best_model.pth"
    config_file = "/data/zhouleyu/autoencoder_for_SEED_new/config_big_klbig.yaml"

    # 读取数据
    train_zcond, train_original = load_data(train_sample_path, train_zcond_path, train_original_path)
    test_zcond, test_original = load_data(test_sample_path, test_zcond_path, test_original_path)

    # 加载VAE decoder（只推理，不训练）
    config = OmegaConf.load(config_file)
    autoencoder_args = config.autoencoderkl.params
    autoencoder_args['num_channels'] = [64,32]
    autoencoder_args['latent_channels'] = 32
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = AutoencoderKL(**autoencoder_args).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    decoder = model.decode
    model.eval()
    for param in model.parameters():
        param.requires_grad = False  # 冻结decoder参数

    # 评估train
    # print("Train Set:")
    # train_mse, train_nmse, train_pcc, train_snr = evaluate(train_zcond, train_original, decoder, device)
    # print(f"MSE : {train_mse:.6f}")
    # print(f"NMSE: {train_nmse:.6f}")
    # print(f"PCC : {train_pcc:.6f}")
    # print(f"SNR : {train_snr:.6f}")

    # 评估test
    print("\nTest Set:")
    test_mse, test_nmse, test_pcc, test_snr = evaluate(test_zcond, test_original, decoder, device)
    print(f"MSE : {test_mse:.6f}")
    print(f"NMSE: {test_nmse:.6f}")
    print(f"PCC : {test_pcc:.6f}")
    print(f"SNR : {test_snr:.6f}")

if __name__ == "__main__":
    main()