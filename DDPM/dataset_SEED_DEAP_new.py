import os
import re
import scipy.io as sio
import numpy as np
import torch
import xarray as xr
from scipy import io
from torch.utils.data.dataset import Dataset
from sklearn.model_selection import train_test_split


def load_SEED_data(pat_id, signal_length=800, shuffle=False, filepath=None):
    """
    加载并处理SEED数据集，shuffle=True时从预处理文件夹读取，否则从.mat文件读取
    
    :param pat_id: 患者ID，"all"表示所有患者，"4"等表示特定患者
    :param signal_length: 信号长度，默认800
    :param shuffle: 是否使用shuffle数据（True时从预处理文件夹读取）
    :param filepath: 包含.mat文件的目录路径（仅当shuffle=False时使用）
    :return: (X_train, y_train, train_mask, X_test, y_test, test_mask, subject_train, subject_test)
    """
    
    if shuffle:
        # 🔥 从预处理的文件夹读取数据
        print(f"🔄 从预处理文件夹读取shuffle数据...")
        
        # 根据pat_id确定数据文件夹
        if pat_id == "all":
            data_folder = "/data/zhouleyu/autoencoder_for_SEED_new/SEED_data_shuffle_all"
        else:
            data_folder = f"/data/zhouleyu/autoencoder_for_SEED_new/SEED_data_shuffle_{pat_id}"
        
        print(f"📁 数据文件夹: {data_folder}")
        
        # 检查文件夹是否存在
        if not os.path.exists(data_folder):
            raise FileNotFoundError(f"预处理数据文件夹不存在: {data_folder}")
        
        try:
            # 加载NumPy格式的数据
            X_train = np.load(os.path.join(data_folder, "X_train.npy"))
            y_train = np.load(os.path.join(data_folder, "y_train.npy"))
            train_mask = np.load(os.path.join(data_folder, "train_mask.npy"))
            subject_train = np.load(os.path.join(data_folder, "subject_train.npy"))
            
            X_test = np.load(os.path.join(data_folder, "X_test.npy"))
            y_test = np.load(os.path.join(data_folder, "y_test.npy"))
            test_mask = np.load(os.path.join(data_folder, "test_mask.npy"))
            subject_test = np.load(os.path.join(data_folder, "subject_test.npy"))
            
            print(f"✅ 成功从预处理文件夹加载数据!")
            print(f"  训练集: {X_train.shape}, 测试集: {X_test.shape}")
            print(f"  训练标签: {y_train.shape}, 测试标签: {y_test.shape}")
            print(f"  训练个体: {np.unique(subject_train)}, 测试个体: {np.unique(subject_test)}")
            
            return (
                X_train,          # 训练数据 (N_train, 62, 800)
                y_train,          # 训练标签 (N_train,)
                train_mask,       # 训练mask (N_train, 62, 800)，全1表示无mask
                X_test,           # 测试数据 (N_test, 62, 800)
                y_test,           # 测试标签 (N_test,)
                test_mask,        # 测试mask (N_test, 62, 800)，全1表示无mask
                subject_train,    # 训练集个体ID (N_train,)
                subject_test,     # 测试集个体ID (N_test,)
            )
            
        except Exception as e:
            print(f"❌ 从预处理文件夹加载数据失败: {e}")
            print(f"📝 提示: 请先运行main()函数生成预处理数据")
            raise
    
    else:
        # 🔥 原始的从.mat文件读取数据的逻辑
        print(f"🔄 从.mat文件读取数据...")
        
        # 定义默认数据目录（如果filepath为None）
        if filepath is None:
            filepath = "./data"  # 默认路径
        
        # 定义试验的标签（1=positive, 0=neutral, -1=negative）
        trial_labels = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]
        false_data = []
        
        # 读取filtered_trial_abs_le_2000.txt文件
        filtered_file = '/data/zhouleyu/autoencoder_for_SEED_new/filtered_trial_2000_and_sensor_final.txt'
        valid_trials = {}  # {filename: [trial_names]}
        
        with open(filtered_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    fname, trial = parts
                    if fname not in valid_trials:
                        valid_trials[fname] = []
                    valid_trials[fname].append(trial)
        
        # 初始化列表存储所有样本和对应的标签、个体信息
        all_samples = []
        all_labels = []
        all_subjects = []  # 存储每个样本对应的个体ID

        # 遍历有效的文件和试验
        for file_name, trial_list in valid_trials.items():
            # 检查是否在false_data中
            file_base = os.path.splitext(file_name)[0]
            if file_base in false_data:
                print(f"跳过异常文件: {file_name}")
                continue
            
            # 根据pat_id筛选
            if pat_id != "all":
                if not pat_id.endswith('_'):
                    pat_id += '_'
                if not file_name.startswith(pat_id):
                    continue
            
            # 提取个体ID（从文件名中提取数字部分）
            subject_id = re.match(r'(\d+)_.*', file_name).group(1)
            
            file_path = os.path.join(filepath, file_name)
            if not os.path.exists(file_path):
                print(f"文件不存在: {file_path}")
                continue
                
            mat_data = sio.loadmat(file_path)
            
            # 遍历该文件的有效试验
            for trial_name in trial_list:
                if trial_name not in mat_data:
                    print(f"试验 {trial_name} 在文件 {file_name} 中不存在")
                    continue
                    
                eeg_data = mat_data[trial_name]  # 当前EEG数据 (62, T)
                T = eeg_data.shape[1]  # 当前段的长度
                
                # 如果数据长度不足800，从中间截取
                if T < signal_length:
                    print(f"警告: {file_name}的{trial_name}长度不足{signal_length}, 实际长度{T}, 跳过")
                    continue
                
                # 计算可以分割的窗口数量
                num_windows = T // signal_length
                
                # 如果有剩余部分，计算中心对齐的起始位置
                if T % signal_length != 0:
                    start_offset = (T % signal_length) // 2
                else:
                    start_offset = 0
                
                # 分窗处理
                for i in range(num_windows):
                    start_idx = start_offset + i * signal_length
                    end_idx = start_idx + signal_length
                    sample = eeg_data[:, start_idx:end_idx]  # 提取长度为signal_length的样本
                    
                    all_samples.append(sample)
                    
                    # 根据试验编号获取对应的标签
                    trial_index = int(re.search(r'_eeg(\d+)', trial_name).group(1)) - 1
                    all_labels.append(trial_labels[trial_index])
                    
                    # 记录个体ID
                    all_subjects.append(subject_id)
            
            print(f"已处理文件: {file_name}, 处理试验数: {len(trial_list)}, 累计样本数: {len(all_samples)}")

        # 检查是否找到数据
        if len(all_samples) == 0:
            raise ValueError(f"未找到匹配患者ID '{pat_id}'的有效数据")

        # 转换为NumPy数组
        all_samples = np.array(all_samples)  # (N, 62, 800)
        all_labels = np.array(all_labels)   # (N,)
        all_subjects = np.array(all_subjects)  # (N,) 个体ID数组

        print(f"总共加载样本数: {len(all_samples)}")
        print(f"样本形状: {all_samples.shape}")
        print(f"包含的个体ID: {np.unique(all_subjects)}")

        if pat_id == "all":    # 数据分割
            # 按个体ID划分
            test_ids = ['13', '14', '15']
            train_mask = np.isin(all_subjects, test_ids, invert=True)
            test_mask = np.isin(all_subjects, test_ids)
            X_train = all_samples[train_mask]
            y_train = all_labels[train_mask]
            subject_train = all_subjects[train_mask]
            X_test = all_samples[test_mask]
            y_test = all_labels[test_mask]
            subject_test = all_subjects[test_mask]
        else:
            # 如果是单个患者ID，直接分割
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test, subject_train, subject_test = train_test_split(
                all_samples, all_labels, all_subjects, train_size=0.8, random_state=42, shuffle=False
            )

        # 创建全1的mask（表示所有数据点都有效）
        train_mask = np.ones_like(X_train)  # (N_train, 62, 800)
        test_mask = np.ones_like(X_test)    # (N_test, 62, 800)

        print(f"训练集样本数: {len(X_train)}, 测试集样本数: {len(X_test)}")
        print(f"训练集个体分布: {np.unique(subject_train, return_counts=True)}")
        print(f"测试集个体分布: {np.unique(subject_test, return_counts=True)}")

        return (
            X_train,          # 训练数据 (N_train, 62, 800)
            y_train,          # 训练标签 (N_train,)
            train_mask,       # 训练mask (N_train, 62, 800)，全1表示无mask
            X_test,           # 测试数据 (N_test, 62, 800)
            y_test,           # 测试标签 (N_test,)
            test_mask,        # 测试mask (N_test, 62, 800)，全1表示无mask
            subject_train,    # 训练集个体ID (N_train,)
            subject_test,     # 测试集个体ID (N_test,)
        )
        
        
        
def load_SEED_data_6815(pat_id, signal_length=800, shuffle=False, filepath=None):
    """
    加载并处理SEED数据集，shuffle=True时从预处理文件夹读取，否则从.mat文件读取
    
    :param pat_id: 患者ID，"all"表示所有患者，"4"等表示特定患者
    :param signal_length: 信号长度，默认800
    :param shuffle: 是否使用shuffle数据（True时从预处理文件夹读取）
    :param filepath: 包含.mat文件的目录路径（仅当shuffle=False时使用）
    :return: (X_train, y_train, train_mask, X_test, y_test, test_mask, subject_train, subject_test)
    """
    
    if shuffle:
        # 🔥 从预处理的文件夹读取数据
        print(f"🔄 从预处理文件夹读取shuffle数据...")
        
        # 根据pat_id确定数据文件夹
        if pat_id == "all":
            data_folder = "/data/zhouleyu/autoencoder_for_SEED_new/SEED_data_shuffle_all"
        else:
            data_folder = f"/data/zhouleyu/autoencoder_for_SEED_new/SEED_data_shuffle_{pat_id}"
        
        print(f"📁 数据文件夹: {data_folder}")
        
        # 检查文件夹是否存在
        if not os.path.exists(data_folder):
            raise FileNotFoundError(f"预处理数据文件夹不存在: {data_folder}")
        
        try:
            # 加载NumPy格式的数据
            X_train = np.load(os.path.join(data_folder, "X_train.npy"))
            y_train = np.load(os.path.join(data_folder, "y_train.npy"))
            train_mask = np.load(os.path.join(data_folder, "train_mask.npy"))
            subject_train = np.load(os.path.join(data_folder, "subject_train.npy"))
            
            X_test = np.load(os.path.join(data_folder, "X_test.npy"))
            y_test = np.load(os.path.join(data_folder, "y_test.npy"))
            test_mask = np.load(os.path.join(data_folder, "test_mask.npy"))
            subject_test = np.load(os.path.join(data_folder, "subject_test.npy"))
            
            print(f"✅ 成功从预处理文件夹加载数据!")
            print(f"  训练集: {X_train.shape}, 测试集: {X_test.shape}")
            print(f"  训练标签: {y_train.shape}, 测试标签: {y_test.shape}")
            print(f"  训练个体: {np.unique(subject_train)}, 测试个体: {np.unique(subject_test)}")
            
            return (
                X_train,          # 训练数据 (N_train, 62, 800)
                y_train,          # 训练标签 (N_train,)
                train_mask,       # 训练mask (N_train, 62, 800)，全1表示无mask
                X_test,           # 测试数据 (N_test, 62, 800)
                y_test,           # 测试标签 (N_test,)
                test_mask,        # 测试mask (N_test, 62, 800)，全1表示无mask
                subject_train,    # 训练集个体ID (N_train,)
                subject_test,     # 测试集个体ID (N_test,)
            )
            
        except Exception as e:
            print(f"❌ 从预处理文件夹加载数据失败: {e}")
            print(f"📝 提示: 请先运行main()函数生成预处理数据")
            raise
    
    else:
        # 🔥 原始的从.mat文件读取数据的逻辑
        print(f"🔄 从.mat文件读取数据...")
        
        # 定义默认数据目录（如果filepath为None）
        if filepath is None:
            filepath = "./data"  # 默认路径
        
        # 定义试验的标签（1=positive, 0=neutral, -1=negative）
        trial_labels = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]
        false_data = []
        
        # 读取filtered_trial_abs_le_2000.txt文件
        filtered_file = '/data/zhouleyu/autoencoder_for_SEED_new/filtered_trial_2000_and_sensor_final.txt'
        valid_trials = {}  # {filename: [trial_names]}
        
        with open(filtered_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    fname, trial = parts
                    if fname not in valid_trials:
                        valid_trials[fname] = []
                    valid_trials[fname].append(trial)
        
        # 初始化列表存储所有样本和对应的标签、个体信息
        all_samples = []
        all_labels = []
        all_subjects = []  # 存储每个样本对应的个体ID

        # 遍历有效的文件和试验
        for file_name, trial_list in valid_trials.items():
            # 检查是否在false_data中
            file_base = os.path.splitext(file_name)[0]
            if file_base in false_data:
                print(f"跳过异常文件: {file_name}")
                continue
            
            # 根据pat_id筛选
            if pat_id != "all":
                if not pat_id.endswith('_'):
                    pat_id += '_'
                if not file_name.startswith(pat_id):
                    continue
            
            # 提取个体ID（从文件名中提取数字部分）
            subject_id = re.match(r'(\d+)_.*', file_name).group(1)
            
            file_path = os.path.join(filepath, file_name)
            if not os.path.exists(file_path):
                print(f"文件不存在: {file_path}")
                continue
                
            mat_data = sio.loadmat(file_path)
            
            # 遍历该文件的有效试验
            for trial_name in trial_list:
                if trial_name not in mat_data:
                    print(f"试验 {trial_name} 在文件 {file_name} 中不存在")
                    continue
                    
                eeg_data = mat_data[trial_name]  # 当前EEG数据 (62, T)
                T = eeg_data.shape[1]  # 当前段的长度
                
                # 如果数据长度不足800，从中间截取
                if T < signal_length:
                    print(f"警告: {file_name}的{trial_name}长度不足{signal_length}, 实际长度{T}, 跳过")
                    continue
                
                # 计算可以分割的窗口数量
                num_windows = T // signal_length
                
                # 如果有剩余部分，计算中心对齐的起始位置
                if T % signal_length != 0:
                    start_offset = (T % signal_length) // 2
                else:
                    start_offset = 0
                
                # 分窗处理
                for i in range(num_windows):
                    start_idx = start_offset + i * signal_length
                    end_idx = start_idx + signal_length
                    sample = eeg_data[:, start_idx:end_idx]  # 提取长度为signal_length的样本
                    
                    all_samples.append(sample)
                    
                    # 根据试验编号获取对应的标签
                    trial_index = int(re.search(r'_eeg(\d+)', trial_name).group(1)) - 1
                    all_labels.append(trial_labels[trial_index])
                    
                    # 记录个体ID
                    all_subjects.append(subject_id)
            
            print(f"已处理文件: {file_name}, 处理试验数: {len(trial_list)}, 累计样本数: {len(all_samples)}")

        # 检查是否找到数据
        if len(all_samples) == 0:
            raise ValueError(f"未找到匹配患者ID '{pat_id}'的有效数据")

        # 转换为NumPy数组
        all_samples = np.array(all_samples)  # (N, 62, 800)
        all_labels = np.array(all_labels)   # (N,)
        all_subjects = np.array(all_subjects)  # (N,) 个体ID数组

        print(f"总共加载样本数: {len(all_samples)}")
        print(f"样本形状: {all_samples.shape}")
        print(f"包含的个体ID: {np.unique(all_subjects)}")

        if pat_id == "all":    # 数据分割
            # 按个体ID划分
            test_ids = ['6', '8', '15']
            train_mask = np.isin(all_subjects, test_ids, invert=True)
            test_mask = np.isin(all_subjects, test_ids)
            X_train = all_samples[train_mask]
            y_train = all_labels[train_mask]
            subject_train = all_subjects[train_mask]
            X_test = all_samples[test_mask]
            y_test = all_labels[test_mask]
            subject_test = all_subjects[test_mask]
        else:
            # 如果是单个患者ID，直接分割
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test, subject_train, subject_test = train_test_split(
                all_samples, all_labels, all_subjects, train_size=0.8, random_state=42, shuffle=False
            )

        # 创建全1的mask（表示所有数据点都有效）
        train_mask = np.ones_like(X_train)  # (N_train, 62, 800)
        test_mask = np.ones_like(X_test)    # (N_test, 62, 800)

        print(f"训练集样本数: {len(X_train)}, 测试集样本数: {len(X_test)}")
        print(f"训练集个体分布: {np.unique(subject_train, return_counts=True)}")
        print(f"测试集个体分布: {np.unique(subject_test, return_counts=True)}")

        return (
            X_train,          # 训练数据 (N_train, 62, 800)
            y_train,          # 训练标签 (N_train,)
            train_mask,       # 训练mask (N_train, 62, 800)，全1表示无mask
            X_test,           # 测试数据 (N_test, 62, 800)
            y_test,           # 测试标签 (N_test,)
            test_mask,        # 测试mask (N_test, 62, 800)，全1表示无mask
            subject_train,    # 训练集个体ID (N_train,)
            subject_test,     # 测试集个体ID (N_test,)
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
    
    
    
def load_DEAP_data_mean_std(pat_id, window_size=512,root_dir = "/data/zhouleyu/DEAP_data"):
    
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






def main():
    """
    主函数：加载SEED数据并保存所有结果
    """
    import os
    import torch
    
    # 设置参数
    RANDOM_SEED = 42
    SHUFFLE = True
    PATIENT_ID = '4'
    SIGNAL_LENGTH = 800
    FILEPATH = '/data/zhouleyu/Preprocessed_EEG'
    SAVE_DIR = '/data/zhouleyu/autoencoder_for_SEED_new/SEED_data_shuffle_4'
    
    print("="*80)
    print("SEED数据加载和保存程序")
    print("="*80)
    print(f"参数设置:")
    print(f"  Random Seed: {RANDOM_SEED}")
    print(f"  Shuffle: {SHUFFLE}")
    print(f"  Patient ID: {PATIENT_ID}")
    print(f"  Signal Length: {SIGNAL_LENGTH}")
    print(f"  Data Path: {FILEPATH}")
    print(f"  Save Directory: {SAVE_DIR}")
    print("-"*80)
    
    # 设置随机种子
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    
    # 创建保存目录
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"📁 创建保存目录: {SAVE_DIR}")
    
    # 加载SEED数据
    print("\n🔄 开始加载SEED数据...")
    try:
        (
            X_train,          # 训练数据 (N_train, 62, 800)
            y_train,          # 训练标签 (N_train,)
            train_mask,       # 训练mask (N_train, 62, 800)，全1表示无mask
            X_test,           # 测试数据 (N_test, 62, 800)
            y_test,           # 测试标签 (N_test,)
            test_mask,        # 测试mask (N_test, 62, 800)，全1表示无mask
            subject_train,    # 训练集个体ID (N_train,)
            subject_test,     # 测试集个体ID (N_test,)
        ) = load_SEED_data(
            pat_id=PATIENT_ID,
            signal_length=SIGNAL_LENGTH,
            shuffle=SHUFFLE,
            filepath=FILEPATH
        )
        
        print("✅ SEED数据加载成功!")
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return
    
    # 打印数据统计信息
    print(f"\n📊 数据统计信息:")
    print(f"  训练集:")
    print(f"    数据形状: {X_train.shape}")
    print(f"    标签形状: {y_train.shape}")
    print(f"    掩码形状: {train_mask.shape}")
    print(f"    个体ID形状: {subject_train.shape}")
    print(f"    标签分布: {np.unique(y_train, return_counts=True)}")
    print(f"    个体分布: {np.unique(subject_train, return_counts=True)}")
    
    print(f"  测试集:")
    print(f"    数据形状: {X_test.shape}")
    print(f"    标签形状: {y_test.shape}")
    print(f"    掩码形状: {test_mask.shape}")
    print(f"    个体ID形状: {subject_test.shape}")
    print(f"    标签分布: {np.unique(y_test, return_counts=True)}")
    print(f"    个体分布: {np.unique(subject_test, return_counts=True)}")
    
    # 保存所有数据
    print(f"\n💾 开始保存数据到: {SAVE_DIR}")
    
    # 保存训练集数据
    train_data_path = os.path.join(SAVE_DIR, "X_train.npy")
    np.save(train_data_path, X_train)
    print(f"  ✅ 训练数据已保存: {train_data_path}")
    
    train_labels_path = os.path.join(SAVE_DIR, "y_train.npy")
    np.save(train_labels_path, y_train)
    print(f"  ✅ 训练标签已保存: {train_labels_path}")
    
    train_mask_path = os.path.join(SAVE_DIR, "train_mask.npy")
    np.save(train_mask_path, train_mask)
    print(f"  ✅ 训练掩码已保存: {train_mask_path}")
    
    subject_train_path = os.path.join(SAVE_DIR, "subject_train.npy")
    np.save(subject_train_path, subject_train)
    print(f"  ✅ 训练个体ID已保存: {subject_train_path}")
    
    # 保存测试集数据
    test_data_path = os.path.join(SAVE_DIR, "X_test.npy")
    np.save(test_data_path, X_test)
    print(f"  ✅ 测试数据已保存: {test_data_path}")
    
    test_labels_path = os.path.join(SAVE_DIR, "y_test.npy")
    np.save(test_labels_path, y_test)
    print(f"  ✅ 测试标签已保存: {test_labels_path}")
    
    test_mask_path = os.path.join(SAVE_DIR, "test_mask.npy")
    np.save(test_mask_path, test_mask)
    print(f"  ✅ 测试掩码已保存: {test_mask_path}")
    
    subject_test_path = os.path.join(SAVE_DIR, "subject_test.npy")
    np.save(subject_test_path, subject_test)
    print(f"  ✅ 测试个体ID已保存: {subject_test_path}")
    
    # 保存PyTorch格式的数据（方便后续使用）
    print(f"\n💾 保存PyTorch格式数据...")
    
    torch_train_data = torch.from_numpy(X_train).float()
    torch_train_labels = torch.from_numpy(y_train).long()
    torch_test_data = torch.from_numpy(X_test).float()
    torch_test_labels = torch.from_numpy(y_test).long()
    
    torch.save(torch_train_data, os.path.join(SAVE_DIR, "X_train.pth"))
    torch.save(torch_train_labels, os.path.join(SAVE_DIR, "y_train.pth"))
    torch.save(torch_test_data, os.path.join(SAVE_DIR, "X_test.pth"))
    torch.save(torch_test_labels, os.path.join(SAVE_DIR, "y_test.pth"))
    print(f"  ✅ PyTorch格式数据已保存")
    
    # 保存数据配置信息
    config_info = {
        'random_seed': RANDOM_SEED,
        'shuffle': SHUFFLE,
        'patient_id': PATIENT_ID,
        'signal_length': SIGNAL_LENGTH,
        'data_filepath': FILEPATH,
        'save_directory': SAVE_DIR,
        'train_data_shape': X_train.shape,
        'test_data_shape': X_test.shape,
        'train_labels_shape': y_train.shape,
        'test_labels_shape': y_test.shape,
        'train_subjects': np.unique(subject_train).tolist(),
        'test_subjects': np.unique(subject_test).tolist(),
        'train_label_distribution': dict(zip(*np.unique(y_train, return_counts=True))),
        'test_label_distribution': dict(zip(*np.unique(y_test, return_counts=True))),
        'num_channels': X_train.shape[1],
        'signal_length_actual': X_train.shape[2],
    }
    
    # 保存配置信息到txt文件
    config_path = os.path.join(SAVE_DIR, "data_config.txt")
    with open(config_path, 'w') as f:
        f.write("SEED数据集配置信息\n")
        f.write("="*50 + "\n\n")
        
        f.write("加载参数:\n")
        f.write(f"  Random Seed: {config_info['random_seed']}\n")
        f.write(f"  Shuffle: {config_info['shuffle']}\n")
        f.write(f"  Patient ID: {config_info['patient_id']}\n")
        f.write(f"  Signal Length: {config_info['signal_length']}\n")
        f.write(f"  Data Path: {config_info['data_filepath']}\n")
        
        f.write(f"\n数据形状:\n")
        f.write(f"  训练数据: {config_info['train_data_shape']}\n")
        f.write(f"  测试数据: {config_info['test_data_shape']}\n")
        f.write(f"  训练标签: {config_info['train_labels_shape']}\n")
        f.write(f"  测试标签: {config_info['test_labels_shape']}\n")
        f.write(f"  通道数: {config_info['num_channels']}\n")
        f.write(f"  信号长度: {config_info['signal_length_actual']}\n")
        
        f.write(f"\n个体分布:\n")
        f.write(f"  训练集个体: {config_info['train_subjects']}\n")
        f.write(f"  测试集个体: {config_info['test_subjects']}\n")
        
        f.write(f"\n标签分布:\n")
        f.write(f"  训练集: {config_info['train_label_distribution']}\n")
        f.write(f"  测试集: {config_info['test_label_distribution']}\n")
        
        import datetime
        f.write(f"\n保存时间: {datetime.datetime.now()}\n")
    
    print(f"  ✅ 配置信息已保存: {config_path}")
    
    # 保存文件列表
    file_list_path = os.path.join(SAVE_DIR, "file_list.txt")
    with open(file_list_path, 'w') as f:
        f.write("保存的文件列表:\n")
        f.write("="*30 + "\n")
        f.write("NumPy格式:\n")
        f.write("  X_train.npy - 训练数据\n")
        f.write("  y_train.npy - 训练标签\n")
        f.write("  train_mask.npy - 训练掩码\n")
        f.write("  subject_train.npy - 训练个体ID\n")
        f.write("  X_test.npy - 测试数据\n")
        f.write("  y_test.npy - 测试标签\n")
        f.write("  test_mask.npy - 测试掩码\n")
        f.write("  subject_test.npy - 测试个体ID\n")
        f.write("\nPyTorch格式:\n")
        f.write("  X_train.pth - 训练数据\n")
        f.write("  y_train.pth - 训练标签\n")
        f.write("  X_test.pth - 测试数据\n")
        f.write("  y_test.pth - 测试标签\n")
        f.write("\n配置文件:\n")
        f.write("  data_config.txt - 数据配置信息\n")
        f.write("  file_list.txt - 文件列表说明\n")
    
    print(f"  ✅ 文件列表已保存: {file_list_path}")
    
    print(f"\n🎉 所有数据保存完成!")
    print(f"📁 保存位置: {SAVE_DIR}")
    print(f"📋 保存文件:")
    print(f"  - 8个NumPy文件 (.npy)")
    print(f"  - 4个PyTorch文件 (.pth)")
    print(f"  - 2个配置文件 (.txt)")
    print("="*80)


if __name__ == "__main__":
    main()