import pandas as pd
import numpy as np
from O_MeC import calculate_local_image_omec
from O_TextInkRatio import calculate_local_image_otir
import ast
from MetricsComputation import ImageFeatureExtractor
from SE import calculate_subband_entropy
from paddleocr import PaddleOCR

class VisualComplexityCalculator:
    def __init__(self, coeff_file_path):
        """
        初始化计算器
        :param coeff_file_path: 包含模型系数的 CSV 文件路径 (PLS_feature_pvalue_n5.csv)
        """
        # 1. 从 CSV 文件加载模型系数
        self.coefficients = self._load_coefficients(coeff_file_path)

        # 2. 训练数据的统计值 (Min, Max) - 用于归一化
        # 这些是模型预处理的固有参数，必须固定使用原训练集的统计值
        self.train_stats = {
            'O.IE':   {'min': 0.476,   'max': 7.907},
            'O.KC':   {'min': 12852,   'max': 1347415},
            'O.SE':   {'min': 0.553,   'max': 5.018},
            'O.IG':   {'min': 0.125,   'max': 5.733},
            'O.FC':   {'min': 1.344,   'max': 15.034},
            'O.H':    {'min': -0.966,  'max': -0.029},
            'O.CF':   {'min': 14.81,   'max': 198.294},
            'O.ERGB': {'min': 0.675,   'max': 17.626},
            'O.ED':   {'min': 0.002,   'max': 0.361},
            'O.FP':   {'min': 0,       'max': 593},
            'O.TiR':  {'min': 0.0,     'max': 0.778},
            'O.MeC':  {'min': 1,       'max': 43}
        }
        
        # 默认截距 (假设数据中心化处理)
        self.intercept = 0.0

    def _load_coefficients(self, file_path):
        """内部方法：读取 CSV 并转换为字典"""
        try:
            df = pd.read_csv(file_path)
            # 确保列名存在（兼容大小写）
            if 'Feature' not in df.columns or 'coefficient' not in df.columns:
                raise ValueError("CSV 文件必须包含 'Feature' 和 'coefficient' 列")
            
            # 转换为字典 {'O.IE': -0.018, ...}
            coef_dict = dict(zip(df['Feature'], df['coefficient']))
            print(f"成功加载模型系数，共 {len(coef_dict)} 个特征。")
            return coef_dict
        except Exception as e:
            raise RuntimeError(f"读取系数文件失败: {e}")

    def normalize(self, feature_name, value):
        """
        对单个特征值进行归一化
        公式: x_norm = ((x - min) / (max - min) + 0.0001) / 1.0001
        """
        if feature_name not in self.train_stats:
            # 如果 CSV 里有这个特征但统计表里没有，无法归一化，报错
            raise ValueError(f"缺少特征 '{feature_name}' 的归一化统计值(Min/Max)")
        
        stats = self.train_stats[feature_name]
        min_val = stats['min']
        max_val = stats['max']
        
        # Min-Max 缩放
        scaled = (value - min_val) / (max_val - min_val)
        
        # 微调 (Epsilon Adjustment)
        normalized = (scaled + 0.0001) / 1.0001
        
        return normalized

    def calculate_score(self, input_data):
        """
        计算视觉复杂性得分
        :param input_data: 包含 12 个特征值的字典 {'O.KC': 12345, ...}
        :return: 预测的复杂性得分 (0.0 - 1.0 之间)
        """
        score = self.intercept
        
        # 检查输入数据中是否缺少系数文件中定义的特征
        missing_features = [f for f in self.coefficients.keys() if f not in input_data]
        if missing_features:
            print(f"⚠️ 警告: 输入数据缺少特征 {missing_features}，计算时将默认这些项为 0")
        
        for feature, coef in self.coefficients.items():
            # 获取输入值，如果未提供则默认为该特征的最小值 (即归一化后的 0)
            raw_value = input_data.get(feature, self.train_stats[feature]['min'])
            
            # 1. 归一化
            norm_value = self.normalize(feature, raw_value)
            
            # 2. 乘以系数累加
            term = norm_value * coef
            score += term
            
        return score
    

class ComputeComplex:
    def __init__(self, image_path, 
                 pls_model_path='PLS_feature_pvalue_n5.csv',
                 color_naming_path='./HeerStone_colorNaming.xlsx',
                 color_sim_path='./HeerStone_colorSimilarity.xlsx',
                 ocr_model=None):
        """
        初始化计算器，加载必要的模型和数据文件。
        """
        self.image_path = image_path
        if ocr_model is None:
            self.ocr_model = PaddleOCR(use_doc_orientation_classify=False,use_doc_unwarping=False,use_textline_orientation=False,lang="ch")
        else:
            self.ocr_model = ocr_model
        
        # 1. 初始化计算器和提取器
        # 确保 'PLS_feature_pvalue_n5.csv' 等文件路径正确
        try:
            self.calculator = VisualComplexityCalculator(pls_model_path)
            self.extractor = ImageFeatureExtractor()
            
            # 2. 预加载颜色数据 (HeerStone 数据集)
            self.color_df = pd.read_excel(color_naming_path)
            color_data = pd.read_excel(color_sim_path)
            
            # 处理相似颜色列 (字符串转列表)
            color_data['Similar_name'] = color_data['Similar_name'].apply(ast.literal_eval)
            # 创建颜色映射字典
            self.similar_colors_dict = dict(zip(color_data['Color_name'], color_data['Similar_name']))
            
            self.is_initialized = True
        except Exception as e:
            print(f"初始化 ComputeComplex 失败: {e}")
            self.is_initialized = False

    def compute_all_features(self, image_path=None):
        """
        执行特征提取并计算最终的复杂性得分。
        返回: complexity_score (float)
        """
        if image_path is not None:
            self.image_path = image_path
        if not self.is_initialized:
            raise RuntimeError("类未成功初始化，无法执行计算。")

        try:
            # 1. 提取基础特征
            sample_data = self.extractor.extract_features(self.image_path)

            # 2. 计算特定指标 (O.MeC, O.TiR)
            # 注意：这里调用了外部定义的全局函数
            sample_data['O.MeC'] = calculate_local_image_omec(
                self.image_path, 
                self.color_df, 
                self.similar_colors_dict, 
                threshold=14
            )
            
            sample_data['O.TiR'], _ = calculate_local_image_otir(self.image_path, ocr_model=self.ocr_model)
            
            # 如果有其他特征，如 O.SE 或 O.FC，可以在此处取消注释或添加
            sample_data['O.SE'] = calculate_subband_entropy(self.image_path)
            sample_data['O.FC'] = 5.0 

            print(f"提取的特征: {sample_data}")

            # 3. 计算最终得分
            complexity_score = self.calculator.calculate_score(sample_data)
            
            return complexity_score, sample_data

        except Exception as e:
            print(f"计算特征时发生错误: {e}")
            return None

# ================= 使用示例 =================
if __name__ == "__main__":
    # 定义文件路径
    # img_path = "C:\\Users\\11494\\Desktop\\data\\output\\output\\97\\chart.png"
    img_path = "78.jpg"
    
    # 实例化类
    # 如果你的 CSV/Excel 文件不在当前目录，请在初始化时传入具体路径参数
    complex_computer = ComputeComplex(img_path)
    
    # 执行计算
    score = complex_computer.compute_all_features()
    
    if score is not None:
        print("-" * 40)
        print(f"预测结果: {score:.4f}")
        print(f"转换百分制: {score * 100:.2f}")
        print("-" * 40)