# -*- coding: utf-8 -*-
"""
Enhanced Stress Sensor Nodule Detection System
基于应力传感器数据的增强结节检测系统

主要功能：
1. 应力分布矩阵提取和预处理
2. 半监督学习异常区域检测
3. 真实结节参数训练数据管理
4. LSTM时序模型捕捉持续结节特征
5. 概率输出和持续跟踪
6. 传感器滑动因素处理
"""

import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.semi_supervised import LabelSpreading
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import cv2
from scipy import ndimage
from scipy.spatial.distance import euclidean
from collections import deque, defaultdict
import time
import json
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class StressMatrixExtractor:
    """应力分布矩阵提取器"""
    
    def __init__(self, matrix_shape=(12, 8)):
        self.matrix_shape = matrix_shape
        self.scaler = StandardScaler()
        self.noise_threshold = 0.1
        self.calibration_data = None
        
    def extract_stress_matrix(self, raw_data, timestamp=None):
        """从原始传感器数据提取应力分布矩阵"""
        try:
            # 数据预处理
            if isinstance(raw_data, (list, tuple)):
                raw_data = np.array(raw_data)
            
            # 确保数据形状正确
            if raw_data.size != np.prod(self.matrix_shape):
                # 尝试重塑或填充数据
                if raw_data.size < np.prod(self.matrix_shape):
                    padded_data = np.zeros(np.prod(self.matrix_shape))
                    padded_data[:raw_data.size] = raw_data.flatten()
                    raw_data = padded_data
                else:
                    raw_data = raw_data.flatten()[:np.prod(self.matrix_shape)]
            
            # 重塑为矩阵
            matrix = raw_data.reshape(self.matrix_shape)
            
            # 噪声滤波
            matrix = self.denoise_matrix(matrix)
            
            # 标准化
            matrix_normalized = self.normalize_matrix(matrix)
            
            # 校准（如果有校准数据）
            if self.calibration_data is not None:
                matrix_normalized = self.apply_calibration(matrix_normalized)
            
            return {
                'raw_matrix': matrix,
                'normalized_matrix': matrix_normalized,
                'timestamp': timestamp or time.time(),
                'quality_score': self.assess_data_quality(matrix)
            }
            
        except Exception as e:
            print(f"矩阵提取错误: {e}")
            return None
    
    def denoise_matrix(self, matrix):
        """矩阵去噪"""
        # 高斯滤波去噪
        denoised = cv2.GaussianBlur(matrix.astype(np.float32), (3, 3), 0.5)
        
        # 中值滤波去除椒盐噪声
        denoised = cv2.medianBlur(denoised.astype(np.float32), 3)
        
        return denoised
    
    def normalize_matrix(self, matrix):
        """矩阵标准化"""
        # Z-score标准化
        matrix_flat = matrix.flatten().reshape(-1, 1)
        normalized_flat = self.scaler.fit_transform(matrix_flat)
        normalized = normalized_flat.reshape(matrix.shape)
        
        # 限制到[0, 1]范围
        min_val, max_val = normalized.min(), normalized.max()
        if max_val > min_val:
            normalized = (normalized - min_val) / (max_val - min_val)
        
        return normalized
    
    def assess_data_quality(self, matrix):
        """评估数据质量"""
        # 计算信噪比
        signal_power = np.var(matrix)
        noise_power = np.var(matrix - cv2.GaussianBlur(matrix.astype(np.float32), (5, 5), 1.0))
        snr = signal_power / (noise_power + 1e-8)
        
        # 计算数据完整性
        completeness = 1.0 - (np.sum(np.isnan(matrix)) / matrix.size)
        
        # 综合质量分数
        quality_score = min(1.0, (snr / 10.0) * completeness)
        
        return quality_score
    
    def apply_calibration(self, matrix):
        """应用校准数据"""
        if self.calibration_data is None:
            return matrix
        
        # 简单的线性校准
        calibrated = matrix * self.calibration_data.get('scale', 1.0) + self.calibration_data.get('offset', 0.0)
        return calibrated
    
    def set_calibration(self, calibration_data):
        """设置校准数据"""
        self.calibration_data = calibration_data

class SemiSupervisedAnomalyDetector:
    """半监督学习异常区域检测器"""
    
    def __init__(self):
        self.unsupervised_model = IsolationForest(contamination=0.1, random_state=42)
        self.supervised_model = LabelSpreading(kernel='rbf', gamma=0.1, alpha=0.2)
        self.pca = PCA(n_components=0.95)  # 保留95%的方差
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def extract_features(self, matrix):
        """从矩阵中提取特征"""
        features = []
        
        # 基本统计特征
        features.extend([
            np.mean(matrix),
            np.std(matrix),
            np.max(matrix),
            np.min(matrix),
            np.median(matrix)
        ])
        
        # 纹理特征（基于灰度共生矩阵）
        matrix_uint8 = (matrix * 255).astype(np.uint8)
        
        # 梯度特征
        grad_x = cv2.Sobel(matrix, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(matrix, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        features.extend([
            np.mean(gradient_magnitude),
            np.std(gradient_magnitude),
            np.max(gradient_magnitude)
        ])
        
        # 形状特征（基于连通组件）
        binary = (matrix > np.mean(matrix) + np.std(matrix)).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            features.extend([area, perimeter, area / (perimeter + 1e-8)])
        else:
            features.extend([0, 0, 0])
        
        # 频域特征
        fft = np.fft.fft2(matrix)
        fft_magnitude = np.abs(fft)
        features.extend([
            np.mean(fft_magnitude),
            np.std(fft_magnitude)
        ])
        
        return np.array(features)
    
    def unsupervised_pretraining(self, matrices):
        """无监督预训练"""
        print("开始无监督预训练...")
        
        # 提取特征
        features_list = []
        for matrix in matrices:
            features = self.extract_features(matrix)
            features_list.append(features)
        
        features_array = np.array(features_list)
        
        # 标准化特征
        features_scaled = self.scaler.fit_transform(features_array)
        
        # PCA降维
        features_pca = self.pca.fit_transform(features_scaled)
        
        # 无监督异常检测
        anomaly_scores = self.unsupervised_model.fit_predict(features_pca)
        
        print(f"无监督预训练完成，检测到 {np.sum(anomaly_scores == -1)} 个异常样本")
        
        return anomaly_scores, features_pca
    
    def supervised_fine_tuning(self, matrices, labels):
        """有监督微调"""
        print("开始有监督微调...")
        
        # 提取特征
        features_list = []
        for matrix in matrices:
            features = self.extract_features(matrix)
            features_list.append(features)
        
        features_array = np.array(features_list)
        
        # 使用已训练的scaler和pca
        features_scaled = self.scaler.transform(features_array)
        features_pca = self.pca.transform(features_scaled)
        
        # 半监督学习
        self.supervised_model.fit(features_pca, labels)
        self.is_trained = True
        
        print("有监督微调完成")
    
    def detect_anomalies(self, matrix):
        """检测异常区域"""
        if not self.is_trained:
            print("模型未训练，使用无监督方法")
            features = self.extract_features(matrix)
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            features_pca = self.pca.transform(features_scaled)
            anomaly_score = self.unsupervised_model.decision_function(features_pca)[0]
            probability = 1 / (1 + np.exp(-anomaly_score))  # sigmoid转换
        else:
            features = self.extract_features(matrix)
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            features_pca = self.pca.transform(features_scaled)
            probability = self.supervised_model.predict_proba(features_pca)[0][1]
        
        return {
            'anomaly_probability': probability,
            'is_anomaly': probability > 0.5,
            'confidence': abs(probability - 0.5) * 2
        }

class NoduleTrainingDataManager:
    """结节训练数据管理器"""
    
    def __init__(self):
        self.training_data = []
        self.nodule_parameters = {
            'area': [],
            'diameter': [],
            'depth': [],
            'density': [],
            'shape_factor': [],
            'position': []
        }
        
    def add_nodule_data(self, matrix, area, diameter, depth, position=None, density=None):
        """添加真实结节数据"""
        # 计算形状因子
        shape_factor = 4 * np.pi * area / (diameter * np.pi)**2 if diameter > 0 else 0
        
        # 估算密度（如果未提供）
        if density is None:
            density = np.mean(matrix[matrix > np.percentile(matrix, 75)])
        
        nodule_data = {
            'matrix': matrix.copy(),
            'area': area,
            'diameter': diameter,
            'depth': depth,
            'density': density,
            'shape_factor': shape_factor,
            'position': position or (matrix.shape[0]//2, matrix.shape[1]//2),
            'timestamp': time.time(),
            'label': 1  # 正样本
        }
        
        self.training_data.append(nodule_data)
        
        # 更新参数统计
        self.nodule_parameters['area'].append(area)
        self.nodule_parameters['diameter'].append(diameter)
        self.nodule_parameters['depth'].append(depth)
        self.nodule_parameters['density'].append(density)
        self.nodule_parameters['shape_factor'].append(shape_factor)
        self.nodule_parameters['position'].append(position)
        
        print(f"添加结节数据: 面积={area:.2f}, 直径={diameter:.2f}, 深度={depth:.2f}")
    
    def add_normal_data(self, matrix):
        """添加正常数据"""
        normal_data = {
            'matrix': matrix.copy(),
            'area': 0,
            'diameter': 0,
            'depth': 0,
            'density': np.mean(matrix),
            'shape_factor': 0,
            'position': None,
            'timestamp': time.time(),
            'label': 0  # 负样本
        }
        
        self.training_data.append(normal_data)
    
    def get_training_data(self):
        """获取训练数据"""
        if not self.training_data:
            return None, None
        
        matrices = [data['matrix'] for data in self.training_data]
        labels = [data['label'] for data in self.training_data]
        
        return matrices, labels
    
    def get_statistics(self):
        """获取结节参数统计信息"""
        if not self.nodule_parameters['area']:
            return None
        
        stats = {}
        for param, values in self.nodule_parameters.items():
            if param != 'position' and values:
                stats[param] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values)
                }
        
        return stats
    
    def save_data(self, filename):
        """保存训练数据"""
        with open(filename, 'wb') as f:
            pickle.dump(self.training_data, f)
        print(f"训练数据已保存到 {filename}")
    
    def load_data(self, filename):
        """加载训练数据"""
        try:
            with open(filename, 'rb') as f:
                self.training_data = pickle.load(f)
            
            # 重建参数统计
            self.nodule_parameters = {
                'area': [],
                'diameter': [],
                'depth': [],
                'density': [],
                'shape_factor': [],
                'position': []
            }
            
            for data in self.training_data:
                if data['label'] == 1:  # 只统计结节数据
                    for param in self.nodule_parameters:
                        if param in data and data[param] is not None:
                            self.nodule_parameters[param].append(data[param])
            
            print(f"训练数据已从 {filename} 加载，共 {len(self.training_data)} 个样本")
        except Exception as e:
            print(f"加载训练数据失败: {e}")

class LSTMTemporalModel:
    """LSTM时序模型用于捕捉持续结节特征"""
    
    def __init__(self, sequence_length=10, matrix_shape=(12, 8)):
        self.sequence_length = sequence_length
        self.matrix_shape = matrix_shape
        self.model = None
        self.scaler = MinMaxScaler()
        self.is_trained = False
        
        # 历史数据缓存
        self.history_buffer = deque(maxlen=sequence_length)
        
    def build_model(self):
        """构建LSTM模型"""
        input_shape = (self.sequence_length, np.prod(self.matrix_shape))
        
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')  # 输出结节概率
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.model = model
        return model
    
    def prepare_sequences(self, matrices, labels=None):
        """准备时序数据"""
        sequences = []
        sequence_labels = []
        
        # 标准化矩阵数据
        matrices_flat = [matrix.flatten() for matrix in matrices]
        matrices_scaled = self.scaler.fit_transform(matrices_flat)
        
        # 创建时序序列
        for i in range(len(matrices_scaled) - self.sequence_length + 1):
            sequence = matrices_scaled[i:i + self.sequence_length]
            sequences.append(sequence)
            
            if labels is not None:
                # 使用序列中最后一个标签作为序列标签
                sequence_labels.append(labels[i + self.sequence_length - 1])
        
        sequences = np.array(sequences)
        sequence_labels = np.array(sequence_labels) if labels is not None else None
        
        return sequences, sequence_labels
    
    def train(self, matrices, labels, validation_split=0.2, epochs=50):
        """训练LSTM模型"""
        print("开始训练LSTM时序模型...")
        
        if self.model is None:
            self.build_model()
        
        # 准备时序数据
        sequences, sequence_labels = self.prepare_sequences(matrices, labels)
        
        if len(sequences) < 10:
            print("训练数据不足，需要至少10个时序样本")
            return False
        
        # 训练回调
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        # 训练模型
        history = self.model.fit(
            sequences, sequence_labels,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=8,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_trained = True
        print("LSTM模型训练完成")
        
        return history
    
    def predict_temporal(self, matrix):
        """基于时序信息预测结节概率"""
        if not self.is_trained or self.model is None:
            return 0.5  # 默认概率
        
        # 添加到历史缓存
        matrix_flat = matrix.flatten()
        matrix_scaled = self.scaler.transform(matrix_flat.reshape(1, -1))[0]
        self.history_buffer.append(matrix_scaled)
        
        # 如果缓存不足，返回默认值
        if len(self.history_buffer) < self.sequence_length:
            return 0.5
        
        # 构建序列并预测
        sequence = np.array(list(self.history_buffer)).reshape(1, self.sequence_length, -1)
        probability = self.model.predict(sequence, verbose=0)[0][0]
        
        return float(probability)
    
    def save_model(self, filepath):
        """保存模型"""
        if self.model is not None:
            # 确保数据文件夹路径
            data_dir = os.path.join(os.path.dirname(__file__), '..', '数据文件')
            os.makedirs(data_dir, exist_ok=True)
            
            model_path = os.path.join(data_dir, os.path.basename(filepath))
            self.model.save(model_path)
            # 保存scaler
            scaler_path = os.path.join(data_dir, os.path.basename(filepath) + '_scaler.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            print(f"LSTM模型已保存到 {model_path}")
    
    def load_model(self, filepath):
        """加载模型"""
        try:
            # 确保数据文件夹路径
            data_dir = os.path.join(os.path.dirname(__file__), '..', '数据文件')
            
            model_path = os.path.join(data_dir, os.path.basename(filepath))
            self.model = tf.keras.models.load_model(model_path)
            # 加载scaler
            scaler_path = os.path.join(data_dir, os.path.basename(filepath) + '_scaler.pkl')
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            self.is_trained = True
            print(f"LSTM模型已从 {model_path} 加载")
        except Exception as e:
            print(f"加载LSTM模型失败: {e}")

class PersistentNoduleTracker:
    """持续结节跟踪器"""
    
    def __init__(self, persistence_threshold=5, probability_threshold=0.7):
        self.persistence_threshold = persistence_threshold  # 持续帧数阈值
        self.probability_threshold = probability_threshold  # 概率阈值
        self.tracked_regions = {}  # 跟踪的区域
        self.region_id_counter = 0
        self.annotation_history = []
        
    def update_tracking(self, matrix, anomaly_result, temporal_probability):
        """更新跟踪信息"""
        current_time = time.time()
        
        # 合并概率
        combined_probability = (anomaly_result['anomaly_probability'] + temporal_probability) / 2
        
        if combined_probability > self.probability_threshold:
            # 寻找高概率区域的中心
            center = self.find_region_center(matrix, combined_probability)
            
            # 检查是否与现有区域匹配
            matched_region_id = self.match_existing_region(center)
            
            if matched_region_id is not None:
                # 更新现有区域
                region = self.tracked_regions[matched_region_id]
                region['last_seen'] = current_time
                region['persistence_count'] += 1
                region['probability_history'].append(combined_probability)
                region['center_history'].append(center)
                
                # 检查是否达到持续阈值
                if (region['persistence_count'] >= self.persistence_threshold and 
                    not region['annotated']):
                    self.annotate_persistent_region(matched_region_id, matrix)
            else:
                # 创建新区域
                self.region_id_counter += 1
                new_region = {
                    'id': self.region_id_counter,
                    'center': center,
                    'first_seen': current_time,
                    'last_seen': current_time,
                    'persistence_count': 1,
                    'probability_history': [combined_probability],
                    'center_history': [center],
                    'annotated': False
                }
                self.tracked_regions[self.region_id_counter] = new_region
        
        # 清理过期区域
        self.cleanup_expired_regions(current_time)
        
        return self.get_current_annotations()
    
    def find_region_center(self, matrix, probability):
        """寻找高概率区域的中心"""
        # 简化实现：使用矩阵最大值位置作为中心
        max_pos = np.unravel_index(np.argmax(matrix), matrix.shape)
        return max_pos
    
    def match_existing_region(self, center, max_distance=2.0):
        """匹配现有区域"""
        for region_id, region in self.tracked_regions.items():
            if region['annotated']:
                continue
            
            # 计算与历史中心的平均距离
            avg_center = np.mean(region['center_history'], axis=0)
            distance = euclidean(center, avg_center)
            
            if distance <= max_distance:
                return region_id
        
        return None
    
    def annotate_persistent_region(self, region_id, matrix):
        """标注持续区域"""
        region = self.tracked_regions[region_id]
        region['annotated'] = True
        
        # 计算区域统计信息
        avg_probability = np.mean(region['probability_history'])
        avg_center = np.mean(region['center_history'], axis=0)
        
        annotation = {
            'region_id': region_id,
            'center': avg_center,
            'probability': avg_probability,
            'persistence_count': region['persistence_count'],
            'first_seen': region['first_seen'],
            'annotated_at': time.time(),
            'matrix_snapshot': matrix.copy()
        }
        
        self.annotation_history.append(annotation)
        
        print(f"标注持续结节区域 {region_id}: 中心={avg_center}, 概率={avg_probability:.3f}, "
              f"持续={region['persistence_count']}帧")
        
        return annotation
    
    def cleanup_expired_regions(self, current_time, expiry_time=10.0):
        """清理过期区域"""
        expired_ids = []
        for region_id, region in self.tracked_regions.items():
            if current_time - region['last_seen'] > expiry_time:
                expired_ids.append(region_id)
        
        for region_id in expired_ids:
            del self.tracked_regions[region_id]
    
    def get_current_annotations(self):
        """获取当前标注"""
        current_annotations = []
        for region in self.tracked_regions.values():
            if region['annotated']:
                current_annotations.append({
                    'id': region['id'],
                    'center': np.mean(region['center_history'], axis=0),
                    'probability': np.mean(region['probability_history']),
                    'persistence': region['persistence_count']
                })
        
        return current_annotations
    
    def get_annotation_history(self):
        """获取标注历史"""
        return self.annotation_history.copy()

class EnhancedStressNoduleDetectionSystem:
    """增强的应力传感器结节检测系统"""
    
    def __init__(self):
        # 初始化各个组件
        self.matrix_extractor = StressMatrixExtractor()
        self.anomaly_detector = SemiSupervisedAnomalyDetector()
        self.training_manager = NoduleTrainingDataManager()
        self.lstm_model = LSTMTemporalModel()
        self.tracker = PersistentNoduleTracker()
        
        # 系统状态
        self.is_trained = False
        self.processing_stats = {
            'total_frames': 0,
            'detected_nodules': 0,
            'annotated_nodules': 0,
            'average_processing_time': 0
        }
    
    def add_training_data(self, raw_data, area=None, diameter=None, depth=None, 
                         position=None, is_nodule=True):
        """添加训练数据"""
        # 提取应力矩阵
        matrix_data = self.matrix_extractor.extract_stress_matrix(raw_data)
        if matrix_data is None:
            return False
        
        matrix = matrix_data['normalized_matrix']
        
        if is_nodule and area is not None and diameter is not None and depth is not None:
            self.training_manager.add_nodule_data(matrix, area, diameter, depth, position)
        else:
            self.training_manager.add_normal_data(matrix)
        
        return True
    
    def train_system(self):
        """训练整个系统"""
        print("开始训练增强检测系统...")
        
        # 获取训练数据
        matrices, labels = self.training_manager.get_training_data()
        if matrices is None or len(matrices) < 10:
            print("训练数据不足，需要至少10个样本")
            return False
        
        # 1. 无监督预训练
        anomaly_scores, _ = self.anomaly_detector.unsupervised_pretraining(matrices)
        
        # 2. 有监督微调
        self.anomaly_detector.supervised_fine_tuning(matrices, labels)
        
        # 3. 训练LSTM时序模型
        if len(matrices) >= 20:  # LSTM需要更多数据
            lstm_history = self.lstm_model.train(matrices, labels)
        
        self.is_trained = True
        print("系统训练完成")
        
        return True
    
    def process_frame(self, raw_data, timestamp=None):
        """处理单帧数据"""
        start_time = time.time()
        
        # 1. 提取应力矩阵
        matrix_data = self.matrix_extractor.extract_stress_matrix(raw_data, timestamp)
        if matrix_data is None:
            return None
        
        matrix = matrix_data['normalized_matrix']
        
        # 2. 异常检测
        anomaly_result = self.anomaly_detector.detect_anomalies(matrix)
        
        # 3. 时序分析
        temporal_probability = self.lstm_model.predict_temporal(matrix)
        
        # 4. 持续跟踪
        annotations = self.tracker.update_tracking(matrix, anomaly_result, temporal_probability)
        
        # 5. 更新统计信息
        processing_time = time.time() - start_time
        self.processing_stats['total_frames'] += 1
        self.processing_stats['average_processing_time'] = (
            (self.processing_stats['average_processing_time'] * 
             (self.processing_stats['total_frames'] - 1) + processing_time) /
            self.processing_stats['total_frames']
        )
        
        if anomaly_result['is_anomaly']:
            self.processing_stats['detected_nodules'] += 1
        
        self.processing_stats['annotated_nodules'] = len(self.tracker.annotation_history)
        
        # 返回检测结果
        result = {
            'matrix_data': matrix_data,
            'anomaly_result': anomaly_result,
            'temporal_probability': temporal_probability,
            'combined_probability': (anomaly_result['anomaly_probability'] + temporal_probability) / 2,
            'annotations': annotations,
            'processing_time': processing_time,
            'timestamp': timestamp or time.time()
        }
        
        return result
    
    def get_system_status(self):
        """获取系统状态"""
        return {
            'is_trained': self.is_trained,
            'training_samples': len(self.training_manager.training_data),
            'tracked_regions': len(self.tracker.tracked_regions),
            'annotation_history': len(self.tracker.annotation_history),
            'processing_stats': self.processing_stats.copy(),
            'nodule_statistics': self.training_manager.get_statistics()
        }
    
    def save_system(self, base_path):
        """保存整个系统"""
        try:
            # 确保数据文件夹路径
            data_dir = os.path.join(os.path.dirname(__file__), '..', '数据文件')
            os.makedirs(data_dir, exist_ok=True)
            
            # 保存训练数据
            training_data_path = os.path.join(data_dir, f"{os.path.basename(base_path)}_training_data.pkl")
            self.training_manager.save_data(training_data_path)
            
            # 保存LSTM模型
            if self.lstm_model.is_trained:
                model_path = os.path.join(data_dir, f"{os.path.basename(base_path)}_lstm_model.h5")
                self.lstm_model.save_model(model_path)
            
            # 保存其他组件
            system_state = {
                'is_trained': self.is_trained,
                'processing_stats': self.processing_stats,
                'anomaly_detector': self.anomaly_detector,
                'tracker_history': self.tracker.annotation_history
            }
            
            system_state_path = os.path.join(data_dir, f"{os.path.basename(base_path)}_system_state.pkl")
            with open(system_state_path, 'wb') as f:
                pickle.dump(system_state, f)
            
            print(f"系统已保存到 {data_dir}")
            return True
            
        except Exception as e:
            print(f"保存系统失败: {e}")
            return False
    
    def load_system(self, base_path):
        """加载整个系统"""
        try:
            # 确保数据文件夹路径
            data_dir = os.path.join(os.path.dirname(__file__), '..', '数据文件')
            
            # 加载训练数据
            training_data_path = os.path.join(data_dir, f"{os.path.basename(base_path)}_training_data.pkl")
            self.training_manager.load_data(training_data_path)
            
            # 加载LSTM模型
            model_path = os.path.join(data_dir, f"{os.path.basename(base_path)}_lstm_model.h5")
            self.lstm_model.load_model(model_path)
            
            # 加载其他组件
            system_state_path = os.path.join(data_dir, f"{os.path.basename(base_path)}_system_state.pkl")
            with open(system_state_path, 'rb') as f:
                system_state = pickle.load(f)
            
            self.is_trained = system_state['is_trained']
            self.processing_stats = system_state['processing_stats']
            self.anomaly_detector = system_state['anomaly_detector']
            self.tracker.annotation_history = system_state['tracker_history']
            
            print(f"系统已从 {base_path} 加载")
            return True
            
        except Exception as e:
            print(f"加载系统失败: {e}")
            return False

# 测试代码
if __name__ == "__main__":
    # 创建检测系统
    detection_system = EnhancedStressNoduleDetectionSystem()
    
    # 生成模拟数据进行测试
    print("生成模拟训练数据...")
    
    # 模拟正常数据
    for i in range(20):
        normal_data = np.random.normal(0.5, 0.1, 96)  # 12*8=96
        detection_system.add_training_data(normal_data, is_nodule=False)
    
    # 模拟结节数据
    for i in range(15):
        nodule_data = np.random.normal(0.8, 0.15, 96)
        # 添加结节特征
        nodule_data[40:50] += 0.3  # 模拟结节区域
        detection_system.add_training_data(
            nodule_data, 
            area=2.5 + i*0.1, 
            diameter=1.8 + i*0.05, 
            depth=0.5 + i*0.02,
            is_nodule=True
        )
    
    # 训练系统
    print("\n训练系统...")
    success = detection_system.train_system()
    
    if success:
        print("\n系统训练成功！")
        
        # 测试实时检测
        print("\n测试实时检测...")
        for i in range(10):
            test_data = np.random.normal(0.6, 0.12, 96)
            if i % 3 == 0:  # 每3帧添加一个疑似结节
                test_data[45:55] += 0.25
            
            result = detection_system.process_frame(test_data)
            if result:
                print(f"帧 {i}: 异常概率={result['anomaly_result']['anomaly_probability']:.3f}, "
                      f"时序概率={result['temporal_probability']:.3f}, "
                      f"综合概率={result['combined_probability']:.3f}")
        
        # 显示系统状态
        print("\n系统状态:")
        status = detection_system.get_system_status()
        for key, value in status.items():
            if key != 'nodule_statistics':
                print(f"{key}: {value}")
    else:
        print("系统训练失败")