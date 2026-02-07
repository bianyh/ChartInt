import numpy as np
import pywt
import cv2

def calculate_subband_entropy(image_path, wavelet='bior1.3', level=3):
    """
    计算子带熵 (Subband Entropy)
    原理：对图像进行多级小波分解，计算每个子带（水平、垂直、对角）系数的香农熵。
    """
    # 1. 读取图像并转为灰度 (通常 SE 是基于亮度的，除非特定说明是对 RGB 分别计算)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image not found: {image_path}")
    
    # 2. 小波分解
    # 论文中使用了小波分解 
    coeffs = pywt.wavedec2(img, wavelet, level=level)
    
    # coeffs[0] 是近似分量 (低频)，coeffs[1:] 是细节分量 (高频)
    # 通常 Subband Entropy 关注的是细节分量（Detail Coefficients）
    
    entropies = []
    
    # 3. 计算每个细节子带的熵
    for i in range(1, len(coeffs)):
        (cH, cV, cD) = coeffs[i] # 水平、垂直、对角分量
        
        for subband in [cH, cV, cD]:
            # 归一化子带系数以便计算概率分布
            # 使用绝对值，因为系数可能有负数
            data = np.abs(subband)
            
            # 计算直方图
            hist, _ = np.histogram(data, bins=256, density=True)
            
            # 移除 0 值以避免 log2(0)
            hist = hist[hist > 0]
            
            # 计算香农熵
            entropy = -np.sum(hist * np.log2(hist))
            entropies.append(entropy)
            
    # 4. 汇总 (通常取平均值或总和，作为图像整体的 SE 分数)
    return np.mean(entropies)

# 使用示例
try:
    se_score = calculate_subband_entropy("1.png")
    print(f"O.SE (Subband Entropy): {se_score}")
except Exception as e:
    print(e)