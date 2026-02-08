#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
结节概率预测方法性能对比测试
Performance Comparison Test for Nodule Probability Prediction Methods

作者: 生物医学工程师
日期: 2024年
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# 导入系统模块
from advanced_nodule_probability_system import AdvancedNoduleProbabilitySystem
from suretouch_elastography_system import SureTouchElastographySystem
from enhanced_detection_system import EnhancedNoduleDetectionSystem

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class PerformanceComparisonTest:
    """结节概率预测方法性能对比测试类"""
    
    def __init__(self):
        """初始化测试环境"""
        self.advanced_system = AdvancedNoduleProbabilitySystem(history_length=10, ensemble_size=3)
        self.elastography_system = SureTouchElastographySystem()
        self.enhanced_system = EnhancedNoduleDetectionSystem()
        
        # 测试结果存储
        self.test_results = {
            'advanced_method': {'predictions': [], 'times': [], 'confidence': []},
            'traditional_method': {'predictions': [], 'times': [], 'confidence': []},
            'elastography_method': {'predictions': [], 'times': [], 'confidence': []}
        }
        
        # 性能指标
        self.performance_metrics = {}
        
    def generate_synthetic_test_data(self, n_samples=200):
        """生成合成测试数据"""
        np.random.seed(42)
        
        test_data = []
        ground_truth = []
        
        for i in range(n_samples):
            # 生成12x8的应力网格数据
            if i < n_samples // 2:  # 前一半为正常组织
                stress_grid = np.random.normal(0.3, 0.1, (12, 8))
                stress_grid = np.clip(stress_grid, 0, 1)
                label = 0  # 正常
            else:  # 后一半为异常结节
                # 创建具有结节特征的应力分布
                stress_grid = np.random.normal(0.4, 0.15, (12, 8))
                
                # 在随机位置添加高应力区域（模拟结节）
                center_x = np.random.randint(2, 10)
                center_y = np.random.randint(2, 6)
                
                # 创建结节掩码
                nodule_mask = np.zeros((12, 8))
                for x in range(max(0, center_x-2), min(12, center_x+3)):
                    for y in range(max(0, center_y-2), min(8, center_y+3)):
                        distance = np.sqrt((x-center_x)**2 + (y-center_y)**2)
                        if distance <= 2:
                            stress_grid[x, y] += np.random.normal(0.3, 0.1)
                            nodule_mask[x, y] = 1
                
                stress_grid = np.clip(stress_grid, 0, 1)
                label = 1  # 异常结节
            
            # 创建对应的结节掩码
            if label == 1:
                # 基于高应力区域创建掩码
                threshold = np.percentile(stress_grid, 75)
                nodule_mask = (stress_grid > threshold).astype(int)
            else:
                nodule_mask = np.zeros((12, 8))
            
            test_data.append({
                'stress_grid': stress_grid,
                'nodule_mask': nodule_mask,
                'timestamp': i
            })
            ground_truth.append(label)
        
        return test_data, np.array(ground_truth)
    
    def test_advanced_method(self, test_data):
        """测试高级概率预测方法"""
        predictions = []
        times = []
        confidences = []
        
        print("测试高级概率预测方法...")
        for i, data in enumerate(test_data):
            start_time = time.time()
            
            try:
                # 使用高级系统预测
                result = self.advanced_system.predict_probability(
                    matrix=data['stress_grid'],
                    nodule_mask=data['nodule_mask']
                )
                
                probability = result['probability']
                confidence = result['confidence']
                
                predictions.append(probability)
                confidences.append(confidence)
                
            except Exception as e:
                print(f"高级方法预测失败 (样本 {i}): {e}")
                predictions.append(0.5)  # 默认值
                confidences.append(0.5)
            
            end_time = time.time()
            times.append(end_time - start_time)
        
        return predictions, times, confidences
    
    def test_traditional_method(self, test_data):
        """测试传统概率预测方法"""
        predictions = []
        times = []
        confidences = []
        
        print("测试传统概率预测方法...")
        for i, data in enumerate(test_data):
            start_time = time.time()
            
            try:
                # 计算基本特征
                stress_grid = data['stress_grid']
                nodule_mask = data['nodule_mask']
                
                # 计算面积、圆形度和强度
                area = np.sum(nodule_mask)
                if area > 0:
                    # 计算周长（简化方法）
                    perimeter = np.sum(np.abs(np.diff(nodule_mask, axis=0))) + np.sum(np.abs(np.diff(nodule_mask, axis=1)))
                    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
                    intensity = np.mean(stress_grid[nodule_mask > 0])
                else:
                    circularity = 1.0
                    intensity = np.mean(stress_grid)
                
                # 传统风险评分
                area_score = min(area / 20.0, 1.0)
                shape_score = 1.0 - circularity
                intensity_score = intensity
                
                probability = 0.4 * area_score + 0.3 * shape_score + 0.3 * intensity_score
                probability = min(probability, 1.0)
                
                predictions.append(probability)
                confidences.append(0.7)  # 固定置信度
                
            except Exception as e:
                print(f"传统方法预测失败 (样本 {i}): {e}")
                predictions.append(0.5)
                confidences.append(0.5)
            
            end_time = time.time()
            times.append(end_time - start_time)
        
        return predictions, times, confidences
    
    def test_elastography_method(self, test_data):
        """测试弹性成像方法"""
        predictions = []
        times = []
        confidences = []
        
        print("测试弹性成像概率预测方法...")
        for i, data in enumerate(test_data):
            start_time = time.time()
            
            try:
                # 使用弹性成像系统的备用方法
                stress_grid = data['stress_grid']
                nodule_mask = data['nodule_mask']
                
                # 计算弹性特征
                elasticity_ratio = np.mean(stress_grid) / (np.std(stress_grid) + 1e-6)
                morphology_score = np.sum(nodule_mask) / (12 * 8)  # 归一化面积
                
                # 简化的弹性成像评分
                probability = 0.6 * elasticity_ratio + 0.4 * morphology_score
                probability = min(probability, 1.0)
                
                predictions.append(probability)
                confidences.append(0.8)  # 固定置信度
                
            except Exception as e:
                print(f"弹性成像方法预测失败 (样本 {i}): {e}")
                predictions.append(0.5)
                confidences.append(0.5)
            
            end_time = time.time()
            times.append(end_time - start_time)
        
        return predictions, times, confidences
    
    def calculate_performance_metrics(self, y_true, y_pred, method_name):
        """计算性能指标"""
        # 将概率转换为二分类预测
        y_pred_binary = (np.array(y_pred) > 0.5).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred_binary),
            'precision': precision_score(y_true, y_pred_binary, zero_division=0),
            'recall': recall_score(y_true, y_pred_binary, zero_division=0),
            'f1_score': f1_score(y_true, y_pred_binary, zero_division=0),
            'auc_score': roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.5
        }
        
        return metrics
    
    def run_comprehensive_test(self):
        """运行综合性能测试"""
        print("=" * 60)
        print("结节概率预测方法性能对比测试")
        print("=" * 60)
        
        # 生成测试数据
        print("\n1. 生成测试数据...")
        test_data, ground_truth = self.generate_synthetic_test_data(n_samples=100)
        print(f"生成了 {len(test_data)} 个测试样本")
        print(f"正常样本: {np.sum(ground_truth == 0)}, 异常样本: {np.sum(ground_truth == 1)}")
        
        # 测试各种方法
        print("\n2. 执行性能测试...")
        
        # 高级方法
        adv_pred, adv_times, adv_conf = self.test_advanced_method(test_data)
        self.test_results['advanced_method'] = {
            'predictions': adv_pred,
            'times': adv_times,
            'confidence': adv_conf
        }
        
        # 传统方法
        trad_pred, trad_times, trad_conf = self.test_traditional_method(test_data)
        self.test_results['traditional_method'] = {
            'predictions': trad_pred,
            'times': trad_times,
            'confidence': trad_conf
        }
        
        # 弹性成像方法
        elas_pred, elas_times, elas_conf = self.test_elastography_method(test_data)
        self.test_results['elastography_method'] = {
            'predictions': elas_pred,
            'times': elas_times,
            'confidence': elas_conf
        }
        
        # 计算性能指标
        print("\n3. 计算性能指标...")
        methods = ['advanced_method', 'traditional_method', 'elastography_method']
        method_names = ['高级多模态方法', '传统风险评分', '弹性成像方法']
        
        for method, name in zip(methods, method_names):
            predictions = self.test_results[method]['predictions']
            metrics = self.calculate_performance_metrics(ground_truth, predictions, name)
            self.performance_metrics[method] = metrics
        
        # 生成报告
        self.generate_performance_report()
        self.create_visualization()
        
        return self.performance_metrics
    
    def generate_performance_report(self):
        """生成性能报告"""
        print("\n" + "=" * 60)
        print("性能测试结果报告")
        print("=" * 60)
        
        methods = ['advanced_method', 'traditional_method', 'elastography_method']
        method_names = ['高级多模态方法', '传统风险评分', '弹性成像方法']
        
        # 创建结果表格
        results_df = pd.DataFrame()
        
        for method, name in zip(methods, method_names):
            metrics = self.performance_metrics[method]
            times = self.test_results[method]['times']
            
            row_data = {
                '方法': name,
                '准确率': f"{metrics['accuracy']:.3f}",
                '精确率': f"{metrics['precision']:.3f}",
                '召回率': f"{metrics['recall']:.3f}",
                'F1分数': f"{metrics['f1_score']:.3f}",
                'AUC分数': f"{metrics['auc_score']:.3f}",
                '平均耗时(ms)': f"{np.mean(times)*1000:.2f}",
                '标准差(ms)': f"{np.std(times)*1000:.2f}"
            }
            results_df = pd.concat([results_df, pd.DataFrame([row_data])], ignore_index=True)
        
        print("\n性能指标对比:")
        print(results_df.to_string(index=False))
        
        # 分析结果
        print("\n" + "-" * 40)
        print("结果分析:")
        print("-" * 40)
        
        best_accuracy = max([self.performance_metrics[m]['accuracy'] for m in methods])
        best_f1 = max([self.performance_metrics[m]['f1_score'] for m in methods])
        fastest_method = min(methods, key=lambda m: np.mean(self.test_results[m]['times']))
        
        for method, name in zip(methods, method_names):
            metrics = self.performance_metrics[method]
            times = self.test_results[method]['times']
            
            print(f"\n{name}:")
            if metrics['accuracy'] == best_accuracy:
                print("  ✓ 最高准确率")
            if metrics['f1_score'] == best_f1:
                print("  ✓ 最高F1分数")
            if method == fastest_method:
                print("  ✓ 最快执行速度")
            
            print(f"  - 综合性能评分: {(metrics['accuracy'] + metrics['f1_score'] + metrics['auc_score'])/3:.3f}")
            print(f"  - 计算效率: {1000/np.mean(times):.1f} 预测/秒")
        
        # 保存结果到文件
        results_df.to_csv('performance_comparison_results.csv', index=False, encoding='utf-8-sig')
        print(f"\n结果已保存到: performance_comparison_results.csv")
    
    def create_visualization(self):
        """创建可视化图表"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('结节概率预测方法性能对比', fontsize=16, fontweight='bold')
        
        methods = ['advanced_method', 'traditional_method', 'elastography_method']
        method_names = ['高级多模态', '传统评分', '弹性成像']
        colors = ['#2E86AB', '#A23B72', '#F18F01']
        
        # 1. 性能指标雷达图
        ax1 = axes[0, 0]
        metrics_names = ['准确率', '精确率', '召回率', 'F1分数', 'AUC分数']
        
        angles = np.linspace(0, 2*np.pi, len(metrics_names), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形
        
        for i, (method, name, color) in enumerate(zip(methods, method_names, colors)):
            metrics = self.performance_metrics[method]
            values = [metrics['accuracy'], metrics['precision'], metrics['recall'], 
                     metrics['f1_score'], metrics['auc_score']]
            values += values[:1]  # 闭合图形
            
            ax1.plot(angles, values, 'o-', linewidth=2, label=name, color=color)
            ax1.fill(angles, values, alpha=0.25, color=color)
        
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(metrics_names)
        ax1.set_ylim(0, 1)
        ax1.set_title('性能指标对比')
        ax1.legend()
        ax1.grid(True)
        
        # 2. 执行时间对比
        ax2 = axes[0, 1]
        times_data = [np.array(self.test_results[method]['times']) * 1000 for method in methods]
        
        bp = ax2.boxplot(times_data, labels=method_names, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_title('执行时间分布 (毫秒)')
        ax2.set_ylabel('时间 (ms)')
        ax2.grid(True, alpha=0.3)
        
        # 3. 预测概率分布
        ax3 = axes[1, 0]
        for i, (method, name, color) in enumerate(zip(methods, method_names, colors)):
            predictions = self.test_results[method]['predictions']
            ax3.hist(predictions, bins=20, alpha=0.6, label=name, color=color, density=True)
        
        ax3.set_title('预测概率分布')
        ax3.set_xlabel('预测概率')
        ax3.set_ylabel('密度')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 综合性能评分
        ax4 = axes[1, 1]
        composite_scores = []
        for method in methods:
            metrics = self.performance_metrics[method]
            score = (metrics['accuracy'] + metrics['f1_score'] + metrics['auc_score']) / 3
            composite_scores.append(score)
        
        bars = ax4.bar(method_names, composite_scores, color=colors, alpha=0.8)
        ax4.set_title('综合性能评分')
        ax4.set_ylabel('评分')
        ax4.set_ylim(0, 1)
        
        # 添加数值标签
        for bar, score in zip(bars, composite_scores):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('performance_comparison_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"可视化图表已保存到: performance_comparison_visualization.png")

def main():
    """主函数"""
    # 创建测试实例
    tester = PerformanceComparisonTest()
    
    # 运行综合测试
    results = tester.run_comprehensive_test()
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
    
    return results

if __name__ == "__main__":
    main()