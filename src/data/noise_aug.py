# 显微成像物理噪声模拟器
# 模拟: Shot Noise (Poisson) + Read Noise (Gaussian) + Background Offset

import numpy as np
import torch

class MicroPhysicsNoise:
    """
    模拟荧光显微镜成像链路的噪声注入器。
    用于在纯净或高信噪比图像上合成低质量输入，训练模型的去噪与超分能力。
    """
    def __init__(self, signal_range=(50, 200), read_noise_sigma=(0.01, 0.05), baseline_offset=0.0):
        """
        Args:
            signal_range (tuple): 模拟光子数的范围 (min_photons, max_photons)。
                                  值越小，图像越暗，泊松噪声(相对)越强。
            read_noise_sigma (tuple): 读出噪声的标准差范围 (高斯噪声)。
                                      相对于归一化后的 [0,1] 图像强度。
            baseline_offset (float): 模拟相机的暗电流基线漂移。
        """
        self.signal_range = signal_range
        self.read_noise_sigma = read_noise_sigma
        self.baseline_offset = baseline_offset

    def __call__(self, img_tensor):
        """
        Args:
            img_tensor (Tensor): [C, H, W], 值域 [0, 1]
        Returns:
            noisy_img (Tensor): [C, H, W], 值域 [0, 1]
        """
        # 确保输入不为负
        img = torch.clamp(img_tensor, min=0.0)
        
        # 1. 随机采样物理参数
        # alpha: 模拟的峰值光子数 (Peak Photons)
        # 例如 alpha=100 意味着图像中最亮处对应 100 个光子，此时 Shot Noise 显著
        alpha = np.random.uniform(*self.signal_range)
        
        # sigma: 读出噪声强度
        sigma = np.random.uniform(*self.read_noise_sigma)

        # 2. 模拟泊松噪声 (Shot Noise)
        # 过程: 归一化强度 -> 光子数 -> 泊松采样 -> 归一化强度
        # Poisson(N) 的方差为 N，信噪比为 sqrt(N)
        img_photons = img * alpha
        noisy_poisson = torch.poisson(img_photons) / alpha

        # 3. 模拟高斯读出噪声 (Read Noise)
        gaussian_noise = torch.randn_like(img) * sigma
        
        # 4. 模拟基线漂移 (Baseline)
        offset = np.random.uniform(0, self.baseline_offset)
        
        # 5. 合成
        noisy_image = noisy_poisson + gaussian_noise + offset
        
        # 6. 截断回合法范围
        noisy_image = torch.clamp(noisy_image, 0.0, 1.0)
        
        return noisy_image

# 单元测试
if __name__ == "__main__":
    injector = MicroPhysicsNoise(signal_range=(30, 100), read_noise_sigma=(0.01, 0.02))
    dummy_img = torch.rand(1, 128, 128)
    noisy = injector(dummy_img)
    print(f"Noise injected. Mean: {noisy.mean():.4f}, Std: {noisy.std():.4f}")