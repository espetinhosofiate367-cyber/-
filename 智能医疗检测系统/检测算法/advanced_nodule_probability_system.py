"""
高级结节概率预测系统
Advanced Nodule Probability Prediction System

结合多模态特征融合、时序分析、机器学习和医学先验知识的智能预测系统
"""

import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score
from collections import deque
import pickle
import time
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class AdvancedNoduleProbabilitySystem:
    """高级结节概率预测系统"""
    
    def __init__(self, history_length=10, ensemble_size=3):
        """
        初始化高级概率预测系统
        
        Args:
            history_length: 历史数据长度
            ensemble_size: 集成模型数量
        """
        self.history_length = history_length
        self.ensemble_size = ensemble_size
        
        # 历史数据缓存
        self.feature_history = deque(maxlen=history_length)
        self.probability_history = deque(maxlen=history_length)
        self.temporal_features = deque(maxlen=history_length)
        
        # 多模态特征提取器
        self.feature_extractors = {
            'morphological': MorphologicalFeatureExtractor(),
            'texture': TextureFeatureExtractor(),
            'intensity': IntensityFeatureExtractor(),
            'spatial': SpatialFeatureExtractor(),
            'temporal': TemporalFeatureExtractor()
        }
        
        # 集成学习模型
        self.ensemble_models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'neural_network': MLPClassifier(hidden_layer_sizes=(128, 64, 32), random_state=42)
        }
        
        # 特征缩放器
        self.feature_scaler = StandardScaler()
        self.probability_scaler = MinMaxScaler()
        
        # 医学先验知识权重
        self.medical_weights = {
            'size_factor': 0.25,      # 尺寸因子
            'shape_factor': 0.20,     # 形状因子
            'texture_factor': 0.20,   # 纹理因子
            'intensity_factor': 0.15, # 强度因子
            'temporal_factor': 0.20   # 时序因子
        }
        
        # 系统状态
        self.is_trained = False
        self.model_performance = {}
        
    def extract_comprehensive_features(self, matrix: np.ndarray, 
                                     nodule_mask: np.ndarray,
                                     elastography_data: Optional[Dict] = None) -> Dict:
        """
        提取综合特征
        
        Args:
            matrix: 传感器矩阵数据
            nodule_mask: 结节掩码
            elastography_data: 弹性成像数据
            
        Returns:
            综合特征字典
        """
        features = {}
        
        # 1. 形态学特征
        features.update(self.feature_extractors['morphological'].extract(nodule_mask))
        
        # 2. 纹理特征
        features.update(self.feature_extractors['texture'].extract(matrix, nodule_mask))
        
        # 3. 强度特征
        features.update(self.feature_extractors['intensity'].extract(matrix, nodule_mask))
        
        # 4. 空间特征
        features.update(self.feature_extractors['spatial'].extract(matrix, nodule_mask))
        
        # 5. 弹性成像特征（如果可用）
        if elastography_data:
            features.update(self._extract_elastography_features(elastography_data))
        
        # 6. 时序特征（如果有历史数据）
        if len(self.feature_history) > 0:
            features.update(self.feature_extractors['temporal'].extract(
                list(self.feature_history), features))
        
        return features
    
    def _extract_elastography_features(self, elastography_data: Dict) -> Dict:
        """提取弹性成像特征"""
        features = {}
        
        if 'youngs_modulus' in elastography_data:
            features['youngs_modulus'] = elastography_data['youngs_modulus']
        
        if 'elasticity_contrast' in elastography_data:
            contrast = elastography_data['elasticity_contrast']
            features['elasticity_mean'] = np.mean(contrast)
            features['elasticity_std'] = np.std(contrast)
            features['elasticity_max'] = np.max(contrast)
        
        if 'stress_data' in elastography_data:
            stress = elastography_data['stress_data']
            if isinstance(stress, dict) and 'stress_matrix' in stress:
                stress_matrix = stress['stress_matrix']
                features['stress_mean'] = np.mean(stress_matrix)
                features['stress_std'] = np.std(stress_matrix)
        
        return features
    
    def predict_probability(self, matrix: np.ndarray, 
                          nodule_mask: np.ndarray,
                          elastography_data: Optional[Dict] = None) -> Dict:
        """
        预测结节概率
        
        Args:
            matrix: 传感器矩阵数据
            nodule_mask: 结节掩码
            elastography_data: 弹性成像数据
            
        Returns:
            预测结果字典
        """
        try:
            # 提取综合特征
            features = self.extract_comprehensive_features(matrix, nodule_mask, elastography_data)
            
            # 如果模型已训练，使用集成预测
            if self.is_trained:
                probability = self._ensemble_predict(features)
            else:
                # 使用基于规则的预测
                probability = self._rule_based_predict(features)
            
            # 应用医学先验知识调整
            adjusted_probability = self._apply_medical_priors(probability, features)
            
            # 时序平滑
            smoothed_probability = self._temporal_smoothing(adjusted_probability)
            
            # 更新历史数据
            self.feature_history.append(features)
            self.probability_history.append(smoothed_probability)
            
            # 计算置信度
            confidence = self._calculate_confidence(features, smoothed_probability)
            
            return {
                'probability': smoothed_probability,
                'confidence': confidence,
                'features': features,
                'raw_probability': probability,
                'adjusted_probability': adjusted_probability,
                'feature_importance': self._get_feature_importance(features)
            }
            
        except Exception as e:
            print(f"概率预测错误: {e}")
            return {
                'probability': 0.5,
                'confidence': 0.0,
                'features': {},
                'raw_probability': 0.5,
                'adjusted_probability': 0.5,
                'feature_importance': {}
            }
    
    def _ensemble_predict(self, features: Dict) -> float:
        """集成模型预测"""
        feature_vector = self._features_to_vector(features)
        feature_vector = self.feature_scaler.transform([feature_vector])[0]
        
        predictions = []
        weights = []
        
        for model_name, model in self.ensemble_models.items():
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba([feature_vector])[0][1]  # 假设二分类
                predictions.append(prob)
                # 使用模型性能作为权重
                weight = self.model_performance.get(model_name, 1.0)
                weights.append(weight)
        
        if predictions:
            # 加权平均
            weights = np.array(weights)
            weights = weights / np.sum(weights)
            return np.average(predictions, weights=weights)
        else:
            return 0.5
    
    def _rule_based_predict(self, features: Dict) -> float:
        """基于规则的预测（未训练时使用）"""
        probability = 0.5
        
        # 尺寸因子
        if 'area' in features:
            area_score = min(features['area'] / 50.0, 1.0)  # 归一化面积
            probability += 0.2 * area_score
        
        # 形状因子
        if 'circularity' in features:
            # 结节通常较圆，但过于规则可能是良性
            shape_score = 1.0 - abs(features['circularity'] - 0.7)
            probability += 0.15 * shape_score
        
        # 强度因子
        if 'intensity_mean' in features:
            intensity_score = features['intensity_mean']
            probability += 0.15 * intensity_score
        
        # 纹理因子
        if 'texture_contrast' in features:
            texture_score = min(features['texture_contrast'] / 100.0, 1.0)
            probability += 0.1 * texture_score
        
        # 弹性因子
        if 'elasticity_mean' in features:
            elasticity_score = min(features['elasticity_mean'] / 10.0, 1.0)
            probability += 0.2 * elasticity_score
        
        return np.clip(probability, 0.0, 1.0)
    
    def _apply_medical_priors(self, probability: float, features: Dict) -> float:
        """应用医学先验知识"""
        adjusted_prob = probability
        
        # 尺寸先验：过小或过大的区域概率降低
        if 'area' in features:
            area = features['area']
            if area < 5 or area > 200:  # 经验阈值
                adjusted_prob *= 0.7
        
        # 形状先验：过于规则的形状可能是良性
        if 'circularity' in features:
            if features['circularity'] > 0.95:  # 过于圆形
                adjusted_prob *= 0.8
        
        # 强度先验：强度过低可能是噪声
        if 'intensity_mean' in features:
            if features['intensity_mean'] < 0.1:
                adjusted_prob *= 0.6
        
        return np.clip(adjusted_prob, 0.0, 1.0)
    
    def _temporal_smoothing(self, probability: float) -> float:
        """时序平滑"""
        if len(self.probability_history) == 0:
            return probability
        
        # 指数移动平均
        alpha = 0.3  # 平滑因子
        recent_probs = list(self.probability_history)[-5:]  # 最近5个值
        
        if recent_probs:
            smoothed = alpha * probability + (1 - alpha) * np.mean(recent_probs)
            return smoothed
        
        return probability
    
    def _calculate_confidence(self, features: Dict, probability: float) -> float:
        """计算预测置信度"""
        confidence = 0.5
        
        # 基于特征完整性
        feature_completeness = len(features) / 20.0  # 假设最多20个特征
        confidence += 0.3 * min(feature_completeness, 1.0)
        
        # 基于历史一致性
        if len(self.probability_history) > 3:
            recent_probs = list(self.probability_history)[-3:]
            consistency = 1.0 - np.std(recent_probs)
            confidence += 0.2 * max(consistency, 0.0)
        
        return np.clip(confidence, 0.0, 1.0)
    
    def _get_feature_importance(self, features: Dict) -> Dict:
        """获取特征重要性"""
        importance = {}
        
        # 基于医学权重
        for feature_name, value in features.items():
            if 'area' in feature_name or 'size' in feature_name:
                importance[feature_name] = self.medical_weights['size_factor']
            elif 'circularity' in feature_name or 'shape' in feature_name:
                importance[feature_name] = self.medical_weights['shape_factor']
            elif 'texture' in feature_name:
                importance[feature_name] = self.medical_weights['texture_factor']
            elif 'intensity' in feature_name:
                importance[feature_name] = self.medical_weights['intensity_factor']
            else:
                importance[feature_name] = 0.1
        
        return importance
    
    def _features_to_vector(self, features: Dict) -> np.ndarray:
        """将特征字典转换为向量"""
        # 定义特征顺序
        feature_names = [
            'area', 'perimeter', 'circularity', 'solidity', 'extent',
            'intensity_mean', 'intensity_std', 'intensity_max', 'intensity_min',
            'texture_contrast', 'texture_dissimilarity', 'texture_homogeneity',
            'spatial_centroid_x', 'spatial_centroid_y', 'spatial_orientation',
            'youngs_modulus', 'elasticity_mean', 'elasticity_std',
            'temporal_stability', 'temporal_trend'
        ]
        
        vector = []
        for name in feature_names:
            vector.append(features.get(name, 0.0))
        
        return np.array(vector)


class MorphologicalFeatureExtractor:
    """形态学特征提取器"""
    
    def extract(self, mask: np.ndarray) -> Dict:
        """提取形态学特征"""
        features = {}
        
        try:
            # 确保mask是二值图像
            if mask.dtype != np.uint8:
                mask = (mask > 0).astype(np.uint8)
            
            # 查找轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # 选择最大轮廓
                largest_contour = max(contours, key=cv2.contourArea)
                
                # 基本几何特征
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                
                features['area'] = area
                features['perimeter'] = perimeter
                
                # 圆形度
                if perimeter > 0:
                    features['circularity'] = 4 * np.pi * area / (perimeter ** 2)
                else:
                    features['circularity'] = 0.0
                
                # 凸包特征
                hull = cv2.convexHull(largest_contour)
                hull_area = cv2.contourArea(hull)
                if hull_area > 0:
                    features['solidity'] = area / hull_area
                else:
                    features['solidity'] = 0.0
                
                # 边界矩形
                x, y, w, h = cv2.boundingRect(largest_contour)
                features['extent'] = area / (w * h) if w * h > 0 else 0.0
                features['aspect_ratio'] = w / h if h > 0 else 1.0
                
                # 椭圆拟合
                if len(largest_contour) >= 5:
                    ellipse = cv2.fitEllipse(largest_contour)
                    features['ellipse_major'] = max(ellipse[1])
                    features['ellipse_minor'] = min(ellipse[1])
                    features['ellipse_ratio'] = features['ellipse_minor'] / features['ellipse_major']
            
        except Exception as e:
            print(f"形态学特征提取错误: {e}")
        
        return features


class TextureFeatureExtractor:
    """纹理特征提取器"""
    
    def extract(self, matrix: np.ndarray, mask: np.ndarray) -> Dict:
        """提取纹理特征"""
        features = {}
        
        try:
            # 获取掩码区域的数据
            masked_data = matrix[mask > 0]
            
            if len(masked_data) > 0:
                # 基本统计特征
                features['texture_mean'] = np.mean(masked_data)
                features['texture_std'] = np.std(masked_data)
                features['texture_skewness'] = self._calculate_skewness(masked_data)
                features['texture_kurtosis'] = self._calculate_kurtosis(masked_data)
                
                # 灰度共生矩阵特征（简化版）
                glcm_features = self._calculate_glcm_features(matrix, mask)
                features.update(glcm_features)
                
                # 局部二值模式（简化版）
                lbp_features = self._calculate_lbp_features(matrix, mask)
                features.update(lbp_features)
        
        except Exception as e:
            print(f"纹理特征提取错误: {e}")
        
        return features
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """计算偏度"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """计算峰度"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _calculate_glcm_features(self, matrix: np.ndarray, mask: np.ndarray) -> Dict:
        """计算灰度共生矩阵特征（简化版）"""
        features = {}
        
        try:
            # 简化的对比度计算
            masked_matrix = matrix * mask
            if np.sum(mask) > 1:
                # 计算相邻像素差异
                diff_h = np.abs(np.diff(masked_matrix, axis=1))
                diff_v = np.abs(np.diff(masked_matrix, axis=0))
                
                features['texture_contrast'] = np.mean(diff_h) + np.mean(diff_v)
                features['texture_dissimilarity'] = np.std(diff_h) + np.std(diff_v)
                features['texture_homogeneity'] = 1.0 / (1.0 + features['texture_contrast'])
        
        except Exception as e:
            print(f"GLCM特征计算错误: {e}")
        
        return features
    
    def _calculate_lbp_features(self, matrix: np.ndarray, mask: np.ndarray) -> Dict:
        """计算局部二值模式特征（简化版）"""
        features = {}
        
        try:
            # 简化的LBP计算
            masked_matrix = matrix * mask
            h, w = masked_matrix.shape
            
            if h > 2 and w > 2:
                # 计算局部方差
                local_var = []
                for i in range(1, h-1):
                    for j in range(1, w-1):
                        if mask[i, j] > 0:
                            neighborhood = masked_matrix[i-1:i+2, j-1:j+2]
                            local_var.append(np.var(neighborhood))
                
                if local_var:
                    features['lbp_variance'] = np.mean(local_var)
                    features['lbp_uniformity'] = np.std(local_var)
        
        except Exception as e:
            print(f"LBP特征计算错误: {e}")
        
        return features


class IntensityFeatureExtractor:
    """强度特征提取器"""
    
    def extract(self, matrix: np.ndarray, mask: np.ndarray) -> Dict:
        """提取强度特征"""
        features = {}
        
        try:
            masked_data = matrix[mask > 0]
            
            if len(masked_data) > 0:
                features['intensity_mean'] = np.mean(masked_data)
                features['intensity_std'] = np.std(masked_data)
                features['intensity_max'] = np.max(masked_data)
                features['intensity_min'] = np.min(masked_data)
                features['intensity_range'] = features['intensity_max'] - features['intensity_min']
                
                # 强度分布特征
                features['intensity_q25'] = np.percentile(masked_data, 25)
                features['intensity_q75'] = np.percentile(masked_data, 75)
                features['intensity_iqr'] = features['intensity_q75'] - features['intensity_q25']
                
                # 相对强度（与背景比较）
                background_data = matrix[mask == 0]
                if len(background_data) > 0:
                    bg_mean = np.mean(background_data)
                    features['intensity_contrast_ratio'] = features['intensity_mean'] / (bg_mean + 1e-6)
        
        except Exception as e:
            print(f"强度特征提取错误: {e}")
        
        return features


class SpatialFeatureExtractor:
    """空间特征提取器"""
    
    def extract(self, matrix: np.ndarray, mask: np.ndarray) -> Dict:
        """提取空间特征"""
        features = {}
        
        try:
            # 质心
            y_coords, x_coords = np.where(mask > 0)
            if len(x_coords) > 0:
                features['spatial_centroid_x'] = np.mean(x_coords)
                features['spatial_centroid_y'] = np.mean(y_coords)
                
                # 相对位置（归一化到[0,1]）
                h, w = matrix.shape
                features['spatial_centroid_x_norm'] = features['spatial_centroid_x'] / w
                features['spatial_centroid_y_norm'] = features['spatial_centroid_y'] / h
                
                # 空间分布
                features['spatial_spread_x'] = np.std(x_coords)
                features['spatial_spread_y'] = np.std(y_coords)
                
                # 主轴方向
                if len(x_coords) > 1:
                    cov_matrix = np.cov(x_coords, y_coords)
                    eigenvals, eigenvecs = np.linalg.eig(cov_matrix)
                    features['spatial_orientation'] = np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0])
        
        except Exception as e:
            print(f"空间特征提取错误: {e}")
        
        return features


class TemporalFeatureExtractor:
    """时序特征提取器"""
    
    def extract(self, history: List[Dict], current_features: Dict) -> Dict:
        """提取时序特征"""
        features = {}
        
        try:
            if len(history) < 2:
                features['temporal_stability'] = 1.0
                features['temporal_trend'] = 0.0
                return features
            
            # 提取关键特征的历史值
            key_features = ['area', 'intensity_mean', 'circularity']
            
            for key in key_features:
                if key in current_features:
                    # 收集历史值
                    historical_values = []
                    for hist_features in history:
                        if key in hist_features:
                            historical_values.append(hist_features[key])
                    
                    if len(historical_values) > 1:
                        # 计算稳定性（变异系数的倒数）
                        mean_val = np.mean(historical_values)
                        std_val = np.std(historical_values)
                        if mean_val > 0:
                            cv = std_val / mean_val
                            features[f'{key}_stability'] = 1.0 / (1.0 + cv)
                        
                        # 计算趋势（线性回归斜率）
                        x = np.arange(len(historical_values))
                        if len(x) > 1:
                            slope = np.polyfit(x, historical_values, 1)[0]
                            features[f'{key}_trend'] = slope
            
            # 整体稳定性和趋势
            stability_values = [v for k, v in features.items() if 'stability' in k]
            if stability_values:
                features['temporal_stability'] = np.mean(stability_values)
            
            trend_values = [v for k, v in features.items() if 'trend' in k]
            if trend_values:
                features['temporal_trend'] = np.mean(np.abs(trend_values))
        
        except Exception as e:
            print(f"时序特征提取错误: {e}")
        
        return features


# 使用示例和测试函数
def test_advanced_system():
    """测试高级概率预测系统"""
    print("测试高级结节概率预测系统...")
    
    # 创建系统实例
    system = AdvancedNoduleProbabilitySystem()
    
    # 模拟数据
    matrix = np.random.rand(20, 20)
    mask = np.zeros((20, 20))
    mask[8:12, 8:12] = 1  # 模拟结节区域
    
    # 模拟弹性成像数据
    elastography_data = {
        'youngs_modulus': 15.0,
        'elasticity_contrast': np.random.rand(20, 20) * 5,
        'stress_data': {'stress_matrix': np.random.rand(20, 20)}
    }
    
    # 预测概率
    result = system.predict_probability(matrix, mask, elastography_data)
    
    print(f"预测概率: {result['probability']:.3f}")
    print(f"置信度: {result['confidence']:.3f}")
    print(f"特征数量: {len(result['features'])}")
    print("测试完成！")

if __name__ == "__main__":
    test_advanced_system()