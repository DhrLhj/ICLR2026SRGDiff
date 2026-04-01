import os
import re
import scipy.io as sio
import numpy as np
import torch
import xarray as xr
from scipy import io
from torch.utils.data.dataset import Dataset
from sklearn.model_selection import train_test_split


def load_SEED_data(pat_id,signal_length=800, shuffle = False,filepath=None):
    """
    加载并处理SEED数据集，返回与参考代码相同格式的数据
    
    :param pat_id: 患者ID，"all"表示所有患者，"1_"、"2_"等表示特定患者
    :param filepath: 包含.mat文件的目录路径
    :param time_before_event: 事件前时间(保留参数，实际不使用)
    :param time_after_event: 事件后时间(保留参数，实际不使用)
    :param lower_q: 下四分位数(保留参数，实际不使用)
    :param upper_q: 上四分位数(保留参数，实际不使用)
    :param iqr_factor: IQR因子(保留参数，实际不使用)
    :return: (X_train, y_train, train_mask, X_test, y_test, test_mask, fin_mean, fin_std)
    """
    # 定义默认数据目录（如果filepath为None）
    if filepath is None:
        filepath = "./data"  # 默认路径
    
    # 定义试验的标签（1=positive, 0=neutral, -1=negative）
    trial_labels = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]
    false_data = ["1_20131027","5_20140411","6_20130712","15_20130709"]
    # 获取目录中所有.mat文件并根据pat_id筛选
    all_files = [f for f in os.listdir(filepath) if f.endswith('.mat')]
    
    # 根据pat_id筛选文件
    if pat_id == "all":
        mat_files = all_files
    else:
        # 确保pat_id格式正确（如"1_"）
        if not pat_id.endswith('_'):
            pat_id += '_'
        mat_files = [f for f in all_files if f.startswith(pat_id)]
    
    mat_files.sort()  # 确保文件顺序一致

    # 初始化列表存储所有样本和对应的标签
    all_samples = []
    all_labels = []

    # 遍历每个.mat文件
    for file_name in mat_files:
        file_base = os.path.splitext(file_name)[0]
        if file_base in false_data:
            print(f"跳过异常文件: {file_name}")
            continue
        
        file_path = os.path.join(filepath, file_name)
        mat_data = sio.loadmat(file_path)

        # 使用正则表达式匹配以_eeg1到_eeg15结尾的键
        eeg_keys = [key for key in mat_data.keys() if re.match(r'.*_eeg[1-9][0-9]?$', key)]

        # 遍历每个EEG数据键
        for key in eeg_keys:
            eeg_data = mat_data[key]  # 当前EEG数据 (62, T)
            T = eeg_data.shape[1]  # 当前段的长度
            num_samples = T // 800  # 可以分割的样本数量

            for i in range(num_samples):
                sample = eeg_data[:, i * 800:(i + 1) * 800]  # 提取长度为800的样本
                all_samples.append(sample)

                # 根据试验编号（key中的数字部分）获取对应的标签
                trial_index = int(re.search(r'_eeg(\d+)', key).group(1)) - 1
                all_labels.append(trial_labels[trial_index])

        print(f"已处理文件: {file_name}, 累计样本数: {len(all_samples)}")

    # 检查是否找到数据
    if len(all_samples) == 0:
        raise ValueError(f"未找到匹配患者ID '{pat_id}'的数据")

    # 转换为NumPy数组
    all_samples = np.array(all_samples)  # (N, 62, 800)
    all_labels = np.array(all_labels)   # (N,)

    # 数据分割 (80%训练，20%测试) - 先打乱数据
    # indices = np.random.permutation(len(all_samples))  # 生成随机排列的索引
    # shuffled_samples = all_samples[indices]  # 按索引打乱样本
    # shuffled_labels = all_labels[indices]    # 按索引打乱标签
    
    shuffled_samples = all_samples 
    shuffled_labels = all_labels

    # 分割数据
    if pat_id == "all":
            X_train, X_test, y_train, y_test = train_test_split(
                shuffled_samples, shuffled_labels, train_size=0.8, random_state=42,shuffle=shuffle
            )

    else:
        X_train, X_test, y_train, y_test = train_test_split(
            shuffled_samples, shuffled_labels, train_size=0.8, random_state=42, shuffle=shuffle
            )

    # 创建全1的mask（表示所有数据点都有效）
    train_mask = np.ones_like(X_train)  # (N_train, 62, 800)
    test_mask = np.ones_like(X_test)    # (N_test, 62, 800)

    return (
        X_train,          # 训练数据 (N_train, 62, 800)
        y_train,          # 训练标签 (N_train,)
        train_mask,       # 训练mask (N_train, 62, 800)，全1表示无mask
        X_test,           # 测试数据 (N_test, 62, 800)
        y_test,           # 测试标签 (N_test,)
        test_mask,        # 测试mask (N_test, 62, 800)，全1表示无mask
    )
    
    
    
    
    
class SEED_condition(Dataset):
    """
    SEED dataset.

    Provides class information and supports conditional and masked training.
    """

    def __init__(
        self,
        signal_length=1001,
        data_array=None,
        label_array=None,
        mask_array=None,
        train_mean=None,
        train_std=None,
        conditional_training=True,
        lr_channels=None,
        hr_channels=None,
        masked_training=True,
    ):
        super().__init__()
        self.conditional_training = conditional_training
        self.masked_training = masked_training

        _num_time_windows, self.num_channels, self.signal_length = data_array.shape
        assert self.signal_length == signal_length
        self.data_array = data_array
        self.label_array = label_array
        self.mask_array = mask_array#那些需要忽视的异常值
        self.train_mean = train_mean
        self.train_std = train_std
        self.hr_channels = hr_channels
        self.lr_channels = lr_channels

    def __getitem__(self, index):
        return_dict = {}
        return_dict["signal"] = torch.from_numpy(np.float32(self.data_array[index]))#signal没有加mask
        return_dict["label"] = torch.tensor([np.float32(self.label_array[index])])
        cond = self.get_cond(index=index)#只需要更改这里就行
        if cond is not None and self.masked_training:
            return_dict["mask"] = torch.from_numpy(np.float32(self.mask_array[index]))
            cond[self.num_channels :] *= return_dict["mask"]
            cond[: self.num_channels] *= return_dict["mask"]#condition加了mask
            #相当于是消除异常值用的
            return_dict["cond"] = cond
        if cond is not None:
            return_dict["cond"] = cond
        if self.masked_training:
            return_dict["mask"] = torch.from_numpy(np.float32(self.mask_array[index]))
        return return_dict

    def get_cond(self, index=0):
        cond = None
        if self.conditional_training:
            condition_mask = torch.zeros(self.num_channels, self.signal_length)
            num_cond_channels = np.random.randint(int(self.num_channels * 0.3),int(self.num_channels*0.7))
            cond_channel = list(
                np.random.choice(
                    self.num_channels,
                    size=num_cond_channels,
                    replace=False,
                )
            )
            # Cond channels are indicated by 1.0
            condition_mask[cond_channel, :] = 1.0
            condition_signal = (
                torch.from_numpy(np.float32(self.data_array[index])) * condition_mask
            )
            cond = torch.cat((condition_mask, condition_signal), dim=0)
        return cond #cond前半部确保哪些通道可用，后半部分就是原始数据*conditional_mask
    
    def get_condition_guding(self, index=0):
        cond = None
        if self.conditional_training:
            condition_mask = torch.zeros(self.num_channels, self.signal_length)
            # 固定 cond_channel 为 lr_channels 对应的索引
            cond_channel = [self.hr_channels.index(ch) for ch in self.lr_channels if ch in self.hr_channels]

            # 将选中的通道标记为 1.0
            condition_mask[cond_channel, :] = 1.0

            # 生成 conditional_signal
            condition_signal = (
                torch.from_numpy(np.float32(self.data_array[index])) * condition_mask
            )

            # 拼接 condition_mask 和 condition_signal
            cond = torch.cat((condition_mask, condition_signal), dim=0)

        return cond #cond前半部确保哪些通道可用，后半部分就是原始数据*conditional_mask

    def get_train_mean_and_std(self):
        return torch.from_numpy(np.float32(self.train_mean)), torch.from_numpy(
            np.float32(self.train_std)
        )#返回均值和方差

    def __len__(self):
        return len(self.data_array)
    
    
    
class DEAP(Dataset):
    """
    AJILE dataset.

    Provides class information and supports conditional and masked training.
    """

    def __init__(
        self,
        signal_length=512,
        masked_training=True,
        data_array=None,
        label_array=None,
        mask_array=None,
        train_mean=None,
        train_std=None,
    ):
        super().__init__()
        self.masked_training = masked_training

        _num_time_windows, self.num_channels, self.signal_length = data_array.shape
        assert self.signal_length == signal_length
        self.data_array = data_array
        self.label_array = label_array
        self.mask_array = mask_array#那些需要忽视的异常值
        self.train_mean = train_mean
        self.train_std = train_std

    def __getitem__(self, index):
        return_dict = {}
        return_dict["signal"] = torch.from_numpy(np.float32(self.data_array[index]))
        return_dict["label"] = torch.tensor([np.float32(self.label_array[index])])
        cond = self.get_cond(index=index)
        if cond is not None and self.masked_training:
            return_dict["mask"] = torch.from_numpy(np.float32(self.mask_array[index]))
            cond[self.num_channels :] *= return_dict["mask"]
            cond[: self.num_channels] *= return_dict["mask"]
            #相当于是消除异常值用的
            return_dict["cond"] = cond
        if cond is not None:
            return_dict["cond"] = cond
        return_dict["mask"] = torch.from_numpy(np.float32(self.mask_array[index]))
        return return_dict

    def get_cond(self, index=0):
        cond = None
        condition_mask = torch.zeros(self.num_channels, self.signal_length)
        num_cond_channels = np.random.randint(self.num_channels + 1)
        cond_channel = list(
            np.random.choice(
                self.num_channels,
                size=num_cond_channels,
                replace=False,
            )
        )
        # Cond channels are indicated by 1.0
        condition_mask[cond_channel, :] = 1.0
        condition_signal = (
            torch.from_numpy(np.float32(self.data_array[index])) * condition_mask
        )
        cond = torch.cat((condition_mask, condition_signal), dim=0)
        return cond #cond前半部确保哪些通道可用，后半部分就是原始数据*conditional_mask

    def get_train_mean_and_std(self):
        return torch.from_numpy(np.float32(self.train_mean)), torch.from_numpy(
            np.float32(self.train_std)
        )#返回均值和方差

    def __len__(self):
        return len(self.data_array)
    
    
    
def load_DEAP_data_mean_std(pat_id, window_size=512,root_dir = ""):
    
    baseline_samples=384
    train_ratio=0.8
    
    eeg_data = []
    labels = []
    subject_ids = []

    # 确定要处理的被试列表
    if pat_id == "all":
        subj_list = [f"s{i:02d}" for i in range(1, 33)]
    else:
        subj_list = [pat_id]

    for subj_id in subj_list:
        file_path = os.path.join(root_dir, f"{subj_id}_all_trials.npz")
        
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found. Skipping.")
            continue
        
        # 加载数据
        data = np.load(file_path)
        subj_eeg = data['eeg']  # (40, 32, 8064)
        subj_labels = data['labels']  # (40, 4)
        
        # 遍历每个实验
        for trial_idx in range(subj_eeg.shape[0]):
            # 移除基线（前3秒）
            trial_eeg = subj_eeg[trial_idx, :, baseline_samples:]  # (32, 8064-384)
            trial_label = subj_labels[trial_idx]  # (4,)
            
            # 计算可用的窗口数
            num_windows = trial_eeg.shape[1] // window_size
            
            # 分窗
            for win_idx in range(num_windows):
                start = win_idx * window_size
                end = start + window_size
                window_eeg = trial_eeg[:, start:end]  # (32, 128)
                
                # 添加到结果
                eeg_data.append(window_eeg)
                # 重复标签以匹配窗口数
                labels.append(np.tile(trial_label, (32, 1)))  # (32, 4)
                subject_ids.append(subj_id)
    
    # 转换为NumPy数组
    eeg_data = np.array(eeg_data)  # (N, 32, 128)
    labels = np.array(labels)      # (N, 32, 4)
    
    # 按8:2分割训练集和测试集
    train_eeg, test_eeg, train_labels, test_labels = train_test_split(
        eeg_data, labels, train_size=train_ratio, random_state=42
    )
    
    # 计算训练集每个通道的均值和标准差
    # train_eeg: (N_train, 32, 128)
    channel_means = np.mean(train_eeg, axis=(0, 2))  # 沿样本和时间轴平均，形状 (32,)
    channel_stds = np.std(train_eeg, axis=(0, 2))   # 沿样本和时间轴计算标准差，形状 (32,)
    
    # 创建掩码，全设为1
    train_mask = np.ones_like(train_eeg)  # (N_train, 32, 128)
    test_mask = np.ones_like(test_eeg)    # (N_test, 32, 128)

    train_eeg = (train_eeg - channel_means[None, :, None]) / channel_stds[None, :, None]
    test_eeg = (test_eeg - channel_means[None, :, None]) / channel_stds[None, :, None]

    print("train_eeg shape:", train_eeg.shape)
    print("train_labels shape:", train_labels.shape)
    print("train_mask shape:", train_mask.shape)
    print("test_eeg shape:", test_eeg.shape)
    print("test_labels shape:", test_labels.shape)
    print("test_mask shape:", test_mask.shape)
    
    
    return (
        train_eeg,          # 训练数据 (N_train, 62, 800)
        train_labels,          # 训练标签 (N_train,)
        train_mask,       # 训练mask (N_train, 62, 800)，全1表示无mask
        test_eeg,           # 测试数据 (N_test, 62, 800)
        test_labels,           # 测试标签 (N_test,)
        test_mask,        # 测试mask (N_test, 62, 800)，全1表示无mask
        channel_means,         # 均值 (1, 62, 1)
        channel_stds,  # 标准差 (1, 62, 1)
    )

