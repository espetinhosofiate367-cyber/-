"""
SureTouch风格的应力弹性成像系统
基于触觉成像技术进行结节检测和边界标记
参考文献：Medical Tactile Inc. SureTouch技术
"""

import numpy as np
import cv2
from scipy import ndimage, signal
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# 导入新的高级概率预测系统
from advanced_nodule_probability_system import AdvancedNoduleProbabilitySystem

class SureTouchElastographySystem:
    """
    SureTouch风格的弹性成像系统
    实现应力分析、边界检测和结节概率评估
    """
    
    def __init__(self):
        # 系统参数
        self.sensor_resolution = (8, 8)  # 传感器阵列分辨率
        self.pressure_threshold = 0.1  # 压力阈值 (Pa)
        self.elasticity_contrast_threshold = 5.0  # 弹性对比度阈值
        
        # 材料属性参数
        self.normal_tissue_modulus = 10.0  # 正常组织杨氏模量 (kPa)
        self.poisson_ratio = 0.45  # 泊松比
        
        # 结节检测参数
        self.min_nodule_size = 0.5  # 最小结节尺寸 (cm)
        self.max_nodule_size = 5.0  # 最大结节尺寸 (cm)
        self.stiffness_ratio_threshold = 5.0  # 硬度比阈值
        
        # 边界检测参数
        self.boundary_smoothing_sigma = 1.0
        self.contour_min_area = 10
        
        # 概率评估权重
        self.elasticity_weight = 0.4
        self.morphology_weight = 0.3
        self.size_weight = 0.2
        self.boundary_weight = 0.1
        
        # 初始化高级概率预测系统
        self.advanced_probability_system = AdvancedNoduleProbabilitySystem(
            history_length=10,
            ensemble_size=3
        )
        
    def calculate_stress_distribution(self, pressure_matrix: np.ndarray) -> Dict:
        """
        计算应力分布
        
        Args:
            pressure_matrix: 压力传感器数据矩阵
            
        Returns:
            包含应力分析结果的字典
        """
        try:
            # 应力张量计算
            stress_xx = pressure_matrix
            stress_yy = pressure_matrix
            stress_xy = np.gradient(pressure_matrix, axis=0) * np.gradient(pressure_matrix, axis=1)
            
            # 主应力计算
            principal_stress_1 = 0.5 * (stress_xx + stress_yy) + \
                               0.5 * np.sqrt((stress_xx - stress_yy)**2 + 4 * stress_xy**2)
            principal_stress_2 = 0.5 * (stress_xx + stress_yy) - \
                               0.5 * np.sqrt((stress_xx - stress_yy)**2 + 4 * stress_xy**2)
            
            # 等效应力 (von Mises应力)
            von_mises_stress = np.sqrt(principal_stress_1**2 - 
                                     principal_stress_1 * principal_stress_2 + 
                                     principal_stress_2**2)
            
            # 应力梯度
            stress_gradient_x = np.gradient(von_mises_stress, axis=1)
            stress_gradient_y = np.gradient(von_mises_stress, axis=0)
            stress_gradient_magnitude = np.sqrt(stress_gradient_x**2 + stress_gradient_y**2)
            
            return {
                'principal_stress_1': principal_stress_1,
                'principal_stress_2': principal_stress_2,
                'von_mises_stress': von_mises_stress,
                'stress_gradient': stress_gradient_magnitude,
                'stress_tensor': {
                    'xx': stress_xx,
                    'yy': stress_yy,
                    'xy': stress_xy
                }
            }
            
        except Exception as e:
            print(f"应力分布计算错误: {e}")
            return {}
    
    def calculate_youngs_modulus(self, stress_data: Dict, strain_matrix: np.ndarray) -> np.ndarray:
        """
        计算杨氏模量分布
        
        Args:
            stress_data: 应力分析结果
            strain_matrix: 应变矩阵
            
        Returns:
            杨氏模量分布矩阵
        """
        try:
            von_mises_stress = stress_data.get('von_mises_stress', np.zeros_like(strain_matrix))
            
            # 避免除零
            strain_matrix_safe = np.where(np.abs(strain_matrix) < 1e-6, 1e-6, strain_matrix)
            
            # 杨氏模量 E = σ/ε
            youngs_modulus = von_mises_stress / strain_matrix_safe
            
            # 限制合理范围 (1-1000 kPa)
            youngs_modulus = np.clip(youngs_modulus, 1.0, 1000.0)
            
            return youngs_modulus
            
        except Exception as e:
            print(f"杨氏模量计算错误: {e}")
            return np.ones_like(strain_matrix) * self.normal_tissue_modulus
    
    def calculate_elasticity_contrast(self, youngs_modulus: np.ndarray) -> np.ndarray:
        """
        计算弹性对比度
        
        Args:
            youngs_modulus: 杨氏模量分布
            
        Returns:
            弹性对比度矩阵
        """
        try:
            # 计算局部平均杨氏模量
            kernel = np.ones((3, 3)) / 9
            local_mean = ndimage.convolve(youngs_modulus, kernel, mode='reflect')
            
            # 弹性对比度 = 局部模量 / 周围平均模量
            elasticity_contrast = youngs_modulus / (local_mean + 1e-6)
            
            return elasticity_contrast
            
        except Exception as e:
            print(f"弹性对比度计算错误: {e}")
            return np.ones_like(youngs_modulus)
    
    def detect_abnormal_regions(self, elasticity_contrast: np.ndarray, 
                              stress_gradient: np.ndarray) -> List[Dict]:
        """
        检测异常区域
        
        Args:
            elasticity_contrast: 弹性对比度矩阵
            stress_gradient: 应力梯度矩阵
            
        Returns:
            异常区域列表
        """
        try:
            # 异常区域标准：高弹性对比度 + 高应力梯度
            abnormal_mask = (elasticity_contrast > self.elasticity_contrast_threshold) & \
                           (stress_gradient > np.percentile(stress_gradient, 75))
            
            # 形态学处理
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            abnormal_mask = cv2.morphologyEx(abnormal_mask.astype(np.uint8), 
                                           cv2.MORPH_CLOSE, kernel)
            abnormal_mask = cv2.morphologyEx(abnormal_mask, cv2.MORPH_OPEN, kernel)
            
            # 连通域分析
            contours, _ = cv2.findContours(abnormal_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            abnormal_regions = []
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area > self.contour_min_area:
                    # 计算区域属性
                    moments = cv2.moments(contour)
                    if moments['m00'] != 0:
                        cx = int(moments['m10'] / moments['m00'])
                        cy = int(moments['m01'] / moments['m00'])
                        
                        # 边界框
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        # 椭圆拟合
                        if len(contour) >= 5:
                            ellipse = cv2.fitEllipse(contour)
                            major_axis = max(ellipse[1])
                            minor_axis = min(ellipse[1])
                            eccentricity = np.sqrt(1 - (minor_axis/major_axis)**2) if major_axis > 0 else 0
                        else:
                            eccentricity = 0
                            major_axis = w
                            minor_axis = h
                        
                        abnormal_regions.append({
                            'id': i,
                            'contour': contour,
                            'center': (cx, cy),
                            'area': area,
                            'bounding_box': (x, y, w, h),
                            'eccentricity': eccentricity,
                            'major_axis': major_axis,
                            'minor_axis': minor_axis,
                            'perimeter': cv2.arcLength(contour, True)
                        })
            
            return abnormal_regions
            
        except Exception as e:
            print(f"异常区域检测错误: {e}")
            return []
    
    def mark_region_boundaries(self, abnormal_regions: List[Dict], 
                             image_shape: Tuple[int, int],
                             nodule_roundness: float = 0.8,
                             nodule_size_cm: float = 1.0) -> np.ndarray:
        """
        标记异常区域边界
        
        Args:
            abnormal_regions: 异常区域列表
            image_shape: 图像尺寸
            nodule_roundness: 结节原形度 (0-1)
            nodule_size_cm: 结节尺寸 (cm)
            
        Returns:
            边界标记图像
        """
        try:
            boundary_image = np.zeros(image_shape, dtype=np.uint8)
            
            # 像素到厘米的转换比例 (假设8x8传感器覆盖10x10cm区域)
            pixel_to_cm = 10.0 / max(image_shape)
            
            for region in abnormal_regions:
                contour = region['contour']
                
                # 基于输入参数调整边界
                region_size_cm = region['area'] * (pixel_to_cm ** 2)
                size_similarity = 1.0 - abs(region_size_cm - nodule_size_cm) / max(nodule_size_cm, region_size_cm)
                
                # 圆形度计算
                perimeter = region['perimeter']
                area = region['area']
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    roundness_similarity = 1.0 - abs(circularity - nodule_roundness)
                else:
                    roundness_similarity = 0
                
                # 综合相似度
                similarity_score = (size_similarity + roundness_similarity) / 2
                
                # 根据相似度调整边界强度
                boundary_intensity = int(255 * similarity_score)
                
                # 绘制边界
                cv2.drawContours(boundary_image, [contour], -1, boundary_intensity, 2)
                
                # 标记中心点
                center = region['center']
                cv2.circle(boundary_image, center, 3, boundary_intensity, -1)
            
            return boundary_image
            
        except Exception as e:
            print(f"边界标记错误: {e}")
            return np.zeros(image_shape, dtype=np.uint8)
    
    def calculate_equivalent_size(self, region: Dict) -> float:
        """
        计算异常区域的等效尺寸
        
        Args:
            region: 异常区域信息
            
        Returns:
            等效直径 (cm)
        """
        try:
            # 像素到厘米的转换比例 (假设8x8传感器覆盖10x10cm区域)
            pixel_to_cm = 10.0 / 8.0  # 每个像素约1.25cm
            
            # 计算等效直径：基于面积的圆形等效直径
            area_pixels = region['area']
            area_cm2 = area_pixels * (pixel_to_cm ** 2)
            equivalent_diameter_cm = 2 * np.sqrt(area_cm2 / np.pi)
            
            return equivalent_diameter_cm
            
        except Exception as e:
            print(f"等效尺寸计算错误: {e}")
            return 1.0  # 默认值
    
    def calculate_normal_distribution_probability(self, calculated_size: float, 
                                                real_size: float, 
                                                size_std: float = 0.3) -> float:
        """
        基于正态分布计算概率
        
        Args:
            calculated_size: 计算得到的等效尺寸 (cm)
            real_size: 真实结节尺寸 (cm)
            size_std: 尺寸标准差 (cm)
            
        Returns:
            基于正态分布的概率值 (0-1)
        """
        try:
            from scipy.stats import norm
            
            # 计算尺寸差异
            size_difference = abs(calculated_size - real_size)
            
            # 使用正态分布计算概率
            # 当尺寸差异为0时，概率最高；差异越大，概率越低
            probability = norm.pdf(size_difference, loc=0, scale=size_std)
            
            # 归一化到0-1范围
            max_probability = norm.pdf(0, loc=0, scale=size_std)
            normalized_probability = probability / max_probability
            
            return float(normalized_probability)
            
        except Exception as e:
            print(f"正态分布概率计算错误: {e}")
            # 回退到简单的线性概率计算
            size_difference = abs(calculated_size - real_size)
            max_acceptable_diff = 2.0  # 最大可接受差异 (cm)
            probability = max(0, 1 - (size_difference / max_acceptable_diff))
            return probability

    def calculate_nodule_probability(self, abnormal_regions: List[Dict],
                                   elasticity_contrast: np.ndarray,
                                   stress_data: Dict,
                                   nodule_roundness: float = 0.8,
                                   nodule_size_cm: float = 1.0) -> List[Dict]:
        """
        使用基于正态分布的概率模型计算异常区域为结节的概率
        
        Args:
            abnormal_regions: 异常区域列表
            elasticity_contrast: 弹性对比度矩阵
            stress_data: 应力分析数据
            nodule_roundness: 期望的结节原形度
            nodule_size_cm: 期望的结节尺寸 (真实尺寸)
            
        Returns:
            包含概率评估的区域列表
        """
        try:
            nodule_probabilities = []
            
            for region in abnormal_regions:
                # 计算等效尺寸
                calculated_size = self.calculate_equivalent_size(region)
                
                # 基于正态分布计算尺寸匹配概率
                size_probability = self.calculate_normal_distribution_probability(
                    calculated_size, nodule_size_cm, size_std=0.3
                )
                
                # 创建结节掩码用于其他特征计算
                mask = np.zeros_like(elasticity_contrast, dtype=np.uint8)
                cv2.fillPoly(mask, [region['contour']], 1)
                
                # 计算弹性特征
                region_elasticity = elasticity_contrast[mask == 1]
                avg_elasticity_contrast = np.mean(region_elasticity) if len(region_elasticity) > 0 else 1.0
                
                # 计算形态学特征
                perimeter = region.get('perimeter', cv2.arcLength(region['contour'], True))
                area = region['area']
                circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
                
                # 形态学概率：基于与期望原形度的匹配
                roundness_diff = abs(circularity - nodule_roundness)
                morphology_probability = self.calculate_normal_distribution_probability(
                    circularity, nodule_roundness, size_std=0.2
                )
                
                # 弹性成像概率：基于弹性对比度
                elasticity_probability = min(avg_elasticity_contrast / 10.0, 1.0)  # 归一化
                
                # 综合概率计算 (加权平均)
                weights = {
                    'size': 0.5,        # 尺寸匹配权重最高
                    'morphology': 0.3,  # 形态学权重
                    'elasticity': 0.2   # 弹性特征权重
                }
                
                nodule_probability = (
                    weights['size'] * size_probability +
                    weights['morphology'] * morphology_probability +
                    weights['elasticity'] * elasticity_probability
                )
                
                # 确保概率在合理范围内
                nodule_probability = np.clip(nodule_probability, 0.0, 1.0)
                
                # 计算置信度：基于各个概率的一致性
                prob_values = [size_probability, morphology_probability, elasticity_probability]
                confidence = 1.0 - np.std(prob_values)  # 标准差越小，置信度越高
                confidence = np.clip(confidence, 0.0, 1.0)
                
                # 风险分级
                if nodule_probability >= 0.8:
                    risk_level = "高风险"
                elif nodule_probability >= 0.5:
                    risk_level = "中风险"
                else:
                    risk_level = "低风险"
                
                # 构建结果
                region_result = region.copy()
                region_result.update({
                    'nodule_probability': nodule_probability,
                    'confidence': confidence,
                    'risk_level': risk_level,
                    'calculated_size_cm': calculated_size,
                    'real_size_cm': nodule_size_cm,
                    'size_difference_cm': abs(calculated_size - nodule_size_cm),
                    'size_probability': size_probability,
                    'morphology_probability': morphology_probability,
                    'elasticity_probability': elasticity_probability,
                    'avg_elasticity_contrast': avg_elasticity_contrast,
                    'circularity': circularity,
                    # 保持向后兼容性
                    'elasticity_score': elasticity_probability,
                    'morphology_score': morphology_probability,
                    'size_score': size_probability,
                    'boundary_score': confidence
                })
                
                nodule_probabilities.append(region_result)
            
            # 按概率排序
            nodule_probabilities.sort(key=lambda x: x['nodule_probability'], reverse=True)
            
            return nodule_probabilities
            
        except Exception as e:
            print(f"基于正态分布的结节概率计算错误: {e}")
            # 回退到简化的概率计算
            return self._fallback_probability_calculation(abnormal_regions, elasticity_contrast, stress_data)
    
    def _fallback_probability_calculation(self, abnormal_regions: List[Dict],
                                        elasticity_contrast: np.ndarray,
                                        stress_data: Dict) -> List[Dict]:
        """回退的简化概率计算方法"""
        try:
            nodule_probabilities = []
            pixel_to_cm = 10.0 / max(elasticity_contrast.shape)
            
            for region in abnormal_regions:
                mask = np.zeros_like(elasticity_contrast)
                cv2.fillPoly(mask, [region['contour']], 1)
                
                region_elasticity = elasticity_contrast[mask == 1]
                avg_elasticity_contrast = np.mean(region_elasticity) if len(region_elasticity) > 0 else 1.0
                
                # 简化的概率计算
                elasticity_score = min(avg_elasticity_contrast / 22.0, 1.0)
                area_cm2 = region['area'] * (pixel_to_cm ** 2)
                size_cm = np.sqrt(area_cm2 / np.pi) * 2
                
                # 基本概率评估
                nodule_probability = 0.5 + 0.3 * elasticity_score
                if self.min_nodule_size <= size_cm <= self.max_nodule_size:
                    nodule_probability += 0.2
                
                nodule_probability = min(nodule_probability, 1.0)
                
                risk_level = "高风险" if nodule_probability >= 0.8 else "中风险" if nodule_probability >= 0.5 else "低风险"
                
                region_result = region.copy()
                region_result.update({
                    'nodule_probability': nodule_probability,
                    'confidence': 0.5,  # 低置信度
                    'risk_level': risk_level,
                    'size_cm': size_cm,
                    'avg_elasticity_contrast': avg_elasticity_contrast
                })
                
                nodule_probabilities.append(region_result)
            
            return nodule_probabilities
            
        except Exception as e:
            print(f"回退概率计算错误: {e}")
            return []
    
    def analyze_tissue_elasticity(self, pressure_matrix: np.ndarray,
                                nodule_roundness: float = 0.8,
                                nodule_size_cm: float = 1.0) -> Dict:
        """
        完整的组织弹性分析
        
        Args:
            pressure_matrix: 压力传感器数据
            nodule_roundness: 结节原形度
            nodule_size_cm: 结节尺寸
            
        Returns:
            完整的分析结果
        """
        try:
            # 1. 应力分析
            stress_data = self.calculate_stress_distribution(pressure_matrix)
            
            # 2. 应变估算 (简化模型)
            strain_matrix = pressure_matrix / (self.normal_tissue_modulus * 1000)  # 转换为Pa
            
            # 3. 杨氏模量计算
            youngs_modulus = self.calculate_youngs_modulus(stress_data, strain_matrix)
            
            # 4. 弹性对比度
            elasticity_contrast = self.calculate_elasticity_contrast(youngs_modulus)
            
            # 5. 异常区域检测
            abnormal_regions = self.detect_abnormal_regions(
                elasticity_contrast, 
                stress_data.get('stress_gradient', np.zeros_like(pressure_matrix))
            )
            
            # 6. 边界标记
            boundary_image = self.mark_region_boundaries(
                abnormal_regions, 
                pressure_matrix.shape,
                nodule_roundness,
                nodule_size_cm
            )
            
            # 7. 结节概率评估
            nodule_probabilities = self.calculate_nodule_probability(
                abnormal_regions,
                elasticity_contrast,
                stress_data,
                nodule_roundness,
                nodule_size_cm
            )
            
            return {
                'stress_data': stress_data,
                'youngs_modulus': youngs_modulus,
                'elasticity_contrast': elasticity_contrast,
                'abnormal_regions': abnormal_regions,
                'boundary_image': boundary_image,
                'nodule_probabilities': nodule_probabilities,
                'analysis_summary': {
                    'total_regions': len(abnormal_regions),
                    'high_risk_count': len([r for r in nodule_probabilities if r['nodule_probability'] >= 0.8]),
                    'medium_risk_count': len([r for r in nodule_probabilities if 0.5 <= r['nodule_probability'] < 0.8]),
                    'low_risk_count': len([r for r in nodule_probabilities if r['nodule_probability'] < 0.5]),
                    'max_probability': max([r['nodule_probability'] for r in nodule_probabilities]) if nodule_probabilities else 0,
                    'avg_elasticity_contrast': np.mean(elasticity_contrast)
                }
            }
            
        except Exception as e:
            print(f"弹性分析错误: {e}")
            return {
                'stress_data': {},
                'youngs_modulus': np.ones_like(pressure_matrix) * self.normal_tissue_modulus,
                'elasticity_contrast': np.ones_like(pressure_matrix),
                'abnormal_regions': [],
                'boundary_image': np.zeros_like(pressure_matrix, dtype=np.uint8),
                'nodule_probabilities': [],
                'analysis_summary': {
                    'total_regions': 0,
                    'high_risk_count': 0,
                    'medium_risk_count': 0,
                    'low_risk_count': 0,
                    'max_probability': 0,
                    'avg_elasticity_contrast': 1.0
                }
            }

# 测试函数
def test_suretouch_system():
    """测试SureTouch弹性成像系统"""
    system = SureTouchElastographySystem()
    
    # 创建模拟数据
    test_pressure = np.random.rand(8, 8) * 50
    # 添加一个硬结节
    test_pressure[3:5, 3:5] += 100
    
    # 分析
    results = system.analyze_tissue_elasticity(
        test_pressure,
        nodule_roundness=0.8,
        nodule_size_cm=1.2
    )
    
    print("SureTouch弹性成像分析结果:")
    print(f"检测到异常区域: {results['analysis_summary']['total_regions']}")
    print(f"高风险区域: {results['analysis_summary']['high_risk_count']}")
    print(f"最高结节概率: {results['analysis_summary']['max_probability']:.3f}")
    
    return results

if __name__ == "__main__":
    test_suretouch_system()