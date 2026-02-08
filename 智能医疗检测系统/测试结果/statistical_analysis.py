import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy import stats
from scipy.signal import find_peaks, savgol_filter
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class StatisticalAnalyzer:
    def __init__(self):
        """初始化统计分析器"""
        self.analysis_results = {}
        self.anomaly_threshold = 2.0  # 异常检测阈值（标准差倍数）
        
    def analyze_nodule_trends(self, nodule_history):
        """分析结节特征变化趋势"""
        if not nodule_history['timestamps']:
            return None
        
        # 转换为DataFrame便于分析
        df = pd.DataFrame({
            'timestamp': nodule_history['timestamps'],
            'area': nodule_history['areas'],
            'circularity': nodule_history['circularities'],
            'intensity': nodule_history['intensities'],
            'risk_score': nodule_history['risk_scores'],
            'count': nodule_history['count']
        })
        
        # 基本统计信息
        basic_stats = {
            'total_frames': len(df),
            'detection_rate': (df['count'] > 0).mean(),
            'avg_area': df[df['area'] > 0]['area'].mean() if any(df['area'] > 0) else 0,
            'max_area': df['area'].max(),
            'avg_risk': df['risk_score'].mean(),
            'max_risk': df['risk_score'].max(),
            'high_risk_frames': (df['risk_score'] > 0.7).sum(),
            'medium_risk_frames': ((df['risk_score'] > 0.4) & (df['risk_score'] <= 0.7)).sum()
        }
        
        # 趋势分析
        trend_analysis = self._analyze_trends(df)
        
        # 异常检测
        anomalies = self._detect_anomalies(df)
        
        # 周期性分析
        periodicity = self._analyze_periodicity(df)
        
        # 相关性分析
        correlations = self._analyze_correlations(df)
        
        self.analysis_results = {
            'basic_stats': basic_stats,
            'trends': trend_analysis,
            'anomalies': anomalies,
            'periodicity': periodicity,
            'correlations': correlations,
            'dataframe': df
        }
        
        return self.analysis_results
    
    def _analyze_trends(self, df):
        """分析数据趋势"""
        trends = {}
        
        for column in ['area', 'circularity', 'intensity', 'risk_score']:
            if column in df.columns:
                values = df[column].values
                
                # 线性趋势
                if len(values) > 1:
                    x = np.arange(len(values))
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
                    
                    trends[column] = {
                        'slope': slope,
                        'r_squared': r_value**2,
                        'p_value': p_value,
                        'trend_direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable',
                        'significance': 'significant' if p_value < 0.05 else 'not_significant'
                    }
                    
                    # 平滑趋势（使用Savitzky-Golay滤波）
                    if len(values) >= 5:
                        smoothed = savgol_filter(values, min(len(values)//3*2+1, 11), 3)
                        trends[column]['smoothed'] = smoothed
        
        return trends
    
    def _detect_anomalies(self, df):
        """检测异常值"""
        anomalies = {}
        
        # 基于统计的异常检测
        for column in ['area', 'intensity', 'risk_score']:
            if column in df.columns and df[column].std() > 0:
                values = df[column].values
                mean_val = np.mean(values)
                std_val = np.std(values)
                
                # Z-score异常检测
                z_scores = np.abs((values - mean_val) / std_val)
                anomaly_indices = np.where(z_scores > self.anomaly_threshold)[0]
                
                anomalies[f'{column}_statistical'] = {
                    'indices': anomaly_indices.tolist(),
                    'values': values[anomaly_indices].tolist(),
                    'z_scores': z_scores[anomaly_indices].tolist()
                }
        
        # 基于机器学习的异常检测
        feature_columns = ['area', 'circularity', 'intensity', 'risk_score']
        available_columns = [col for col in feature_columns if col in df.columns]
        
        if len(available_columns) >= 2:
            features = df[available_columns].values
            
            # 标准化特征
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Isolation Forest异常检测
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomaly_labels = iso_forest.fit_predict(features_scaled)
            anomaly_indices = np.where(anomaly_labels == -1)[0]
            
            anomalies['isolation_forest'] = {
                'indices': anomaly_indices.tolist(),
                'scores': iso_forest.decision_function(features_scaled)[anomaly_indices].tolist()
            }
        
        return anomalies
    
    def _analyze_periodicity(self, df):
        """分析周期性模式"""
        periodicity = {}
        
        for column in ['area', 'risk_score']:
            if column in df.columns and len(df) > 10:
                values = df[column].values
                
                # 寻找峰值
                peaks, properties = find_peaks(values, height=np.mean(values))
                
                if len(peaks) > 1:
                    # 计算峰值间距
                    peak_intervals = np.diff(peaks)
                    avg_interval = np.mean(peak_intervals)
                    
                    periodicity[column] = {
                        'peak_count': len(peaks),
                        'peak_indices': peaks.tolist(),
                        'avg_interval': avg_interval,
                        'interval_std': np.std(peak_intervals),
                        'regularity': 'regular' if np.std(peak_intervals) < avg_interval * 0.3 else 'irregular'
                    }
        
        return periodicity
    
    def _analyze_correlations(self, df):
        """分析特征间相关性"""
        feature_columns = ['area', 'circularity', 'intensity', 'risk_score', 'count']
        available_columns = [col for col in feature_columns if col in df.columns]
        
        if len(available_columns) >= 2:
            corr_matrix = df[available_columns].corr()
            
            # 找出强相关性（|r| > 0.7）
            strong_correlations = []
            for i in range(len(available_columns)):
                for j in range(i+1, len(available_columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        strong_correlations.append({
                            'feature1': available_columns[i],
                            'feature2': available_columns[j],
                            'correlation': corr_val,
                            'strength': 'very_strong' if abs(corr_val) > 0.9 else 'strong'
                        })
            
            return {
                'correlation_matrix': corr_matrix.to_dict(),
                'strong_correlations': strong_correlations
            }
        
        return {}
    
    def generate_comprehensive_report(self):
        """生成综合分析报告"""
        if not self.analysis_results:
            return "暂无分析数据，请先运行分析。"
        
        report = []
        report.append("=" * 60)
        report.append("结节检测统计分析报告")
        report.append("=" * 60)
        
        # 基本统计
        basic = self.analysis_results['basic_stats']
        report.append("\n【基本统计信息】")
        report.append(f"总帧数: {basic['total_frames']}")
        report.append(f"结节检出率: {basic['detection_rate']:.1%}")
        report.append(f"平均结节面积: {basic['avg_area']:.2f}")
        report.append(f"最大结节面积: {basic['max_area']:.2f}")
        report.append(f"平均风险评分: {basic['avg_risk']:.3f}")
        report.append(f"最高风险评分: {basic['max_risk']:.3f}")
        report.append(f"高风险帧数: {basic['high_risk_frames']} ({basic['high_risk_frames']/basic['total_frames']:.1%})")
        report.append(f"中风险帧数: {basic['medium_risk_frames']} ({basic['medium_risk_frames']/basic['total_frames']:.1%})")
        
        # 趋势分析
        trends = self.analysis_results['trends']
        report.append("\n【趋势分析】")
        for feature, trend_data in trends.items():
            direction = trend_data['trend_direction']
            significance = trend_data['significance']
            r_squared = trend_data['r_squared']
            
            report.append(f"{feature}:")
            report.append(f"  - 趋势方向: {direction}")
            report.append(f"  - 拟合度(R²): {r_squared:.3f}")
            report.append(f"  - 统计显著性: {significance}")
        
        # 异常检测
        anomalies = self.analysis_results['anomalies']
        report.append("\n【异常检测】")
        total_anomalies = 0
        for anomaly_type, anomaly_data in anomalies.items():
            if 'indices' in anomaly_data:
                count = len(anomaly_data['indices'])
                total_anomalies += count
                report.append(f"{anomaly_type}: 检测到 {count} 个异常点")
        
        if total_anomalies == 0:
            report.append("未检测到显著异常。")
        
        # 周期性分析
        periodicity = self.analysis_results['periodicity']
        report.append("\n【周期性分析】")
        if periodicity:
            for feature, period_data in periodicity.items():
                report.append(f"{feature}:")
                report.append(f"  - 峰值数量: {period_data['peak_count']}")
                report.append(f"  - 平均间隔: {period_data['avg_interval']:.1f} 帧")
                report.append(f"  - 规律性: {period_data['regularity']}")
        else:
            report.append("未发现明显的周期性模式。")
        
        # 相关性分析
        correlations = self.analysis_results['correlations']
        report.append("\n【特征相关性】")
        if 'strong_correlations' in correlations and correlations['strong_correlations']:
            for corr in correlations['strong_correlations']:
                report.append(f"{corr['feature1']} vs {corr['feature2']}: {corr['correlation']:.3f} ({corr['strength']})")
        else:
            report.append("未发现强相关性特征。")
        
        # 风险评估
        report.append("\n【风险评估】")
        high_risk_rate = basic['high_risk_frames'] / basic['total_frames']
        if high_risk_rate > 0.3:
            risk_level = "高风险"
        elif high_risk_rate > 0.1:
            risk_level = "中等风险"
        else:
            risk_level = "低风险"
        
        report.append(f"整体风险等级: {risk_level}")
        report.append(f"建议: ", end="")
        
        if risk_level == "高风险":
            report.append("需要密切监控，建议增加检测频率。")
        elif risk_level == "中等风险":
            report.append("保持定期监控，注意异常变化。")
        else:
            report.append("维持当前监控频率即可。")
        
        return "\n".join(report)
    
    def create_statistical_plots(self, save_path=None):
        """创建统计分析图表"""
        if not self.analysis_results:
            return None
        
        df = self.analysis_results['dataframe']
        
        # 创建多子图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('结节检测统计分析图表', fontsize=16, fontweight='bold')
        
        # 1. 特征分布直方图
        ax1 = axes[0, 0]
        features = ['area', 'risk_score', 'intensity']
        colors = ['blue', 'red', 'green']
        
        for i, (feature, color) in enumerate(zip(features, colors)):
            if feature in df.columns:
                values = df[df[feature] > 0][feature]  # 排除零值
                if len(values) > 0:
                    ax1.hist(values, alpha=0.6, label=feature, color=color, bins=15)
        
        ax1.set_title('特征分布')
        ax1.set_xlabel('特征值')
        ax1.set_ylabel('频次')
        ax1.legend()
        
        # 2. 时间序列趋势
        ax2 = axes[0, 1]
        ax2.plot(df.index, df['area'], 'b-', label='面积', alpha=0.7)
        ax2_twin = ax2.twinx()
        ax2_twin.plot(df.index, df['risk_score'], 'r-', label='风险评分', alpha=0.7)
        
        # 添加趋势线
        if 'area' in self.analysis_results['trends']:
            trend_data = self.analysis_results['trends']['area']
            if 'smoothed' in trend_data:
                ax2.plot(df.index, trend_data['smoothed'], 'b--', linewidth=2, label='面积趋势')
        
        ax2.set_title('时间序列趋势')
        ax2.set_xlabel('帧数')
        ax2.set_ylabel('面积', color='blue')
        ax2_twin.set_ylabel('风险评分', color='red')
        
        # 合并图例
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # 3. 相关性热力图
        ax3 = axes[0, 2]
        if 'correlation_matrix' in self.analysis_results['correlations']:
            corr_data = pd.DataFrame(self.analysis_results['correlations']['correlation_matrix'])
            sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, ax=ax3)
            ax3.set_title('特征相关性矩阵')
        else:
            ax3.text(0.5, 0.5, '相关性数据不足', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('特征相关性矩阵')
        
        # 4. 异常检测可视化
        ax4 = axes[1, 0]
        ax4.plot(df.index, df['risk_score'], 'b-', alpha=0.6, label='风险评分')
        
        # 标记异常点
        anomalies = self.analysis_results['anomalies']
        for anomaly_type, anomaly_data in anomalies.items():
            if 'indices' in anomaly_data and anomaly_data['indices']:
                indices = anomaly_data['indices']
                ax4.scatter(indices, df.iloc[indices]['risk_score'], 
                           color='red', s=50, alpha=0.8, label=f'异常点({anomaly_type})')
        
        ax4.set_title('异常检测结果')
        ax4.set_xlabel('帧数')
        ax4.set_ylabel('风险评分')
        ax4.legend()
        
        # 5. 检出率统计
        ax5 = axes[1, 1]
        detection_counts = df['count'].value_counts().sort_index()
        ax5.bar(detection_counts.index, detection_counts.values, alpha=0.7)
        ax5.set_title('结节检出数量分布')
        ax5.set_xlabel('检出结节数')
        ax5.set_ylabel('帧数')
        
        # 6. 风险等级分布
        ax6 = axes[1, 2]
        risk_categories = pd.cut(df['risk_score'], 
                                bins=[0, 0.4, 0.7, 1.0], 
                                labels=['低风险', '中风险', '高风险'])
        risk_counts = risk_categories.value_counts()
        
        colors_risk = ['green', 'orange', 'red']
        ax6.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%', 
                colors=colors_risk, startangle=90)
        ax6.set_title('风险等级分布')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"统计图表已保存到: {save_path}")
        
        return fig
    
    def export_detailed_data(self, save_path):
        """导出详细分析数据"""
        if not self.analysis_results:
            return False
        
        try:
            with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
                # 原始数据
                self.analysis_results['dataframe'].to_excel(writer, sheet_name='原始数据', index=False)
                
                # 基本统计
                basic_df = pd.DataFrame([self.analysis_results['basic_stats']]).T
                basic_df.columns = ['值']
                basic_df.to_excel(writer, sheet_name='基本统计')
                
                # 趋势分析
                if self.analysis_results['trends']:
                    trend_data = []
                    for feature, trend in self.analysis_results['trends'].items():
                        trend_data.append({
                            '特征': feature,
                            '斜率': trend['slope'],
                            'R平方': trend['r_squared'],
                            'P值': trend['p_value'],
                            '趋势方向': trend['trend_direction'],
                            '显著性': trend['significance']
                        })
                    trend_df = pd.DataFrame(trend_data)
                    trend_df.to_excel(writer, sheet_name='趋势分析', index=False)
                
                # 异常检测结果
                anomaly_data = []
                for anomaly_type, anomaly_info in self.analysis_results['anomalies'].items():
                    if 'indices' in anomaly_info:
                        for idx, val in zip(anomaly_info['indices'], anomaly_info.get('values', [])):
                            anomaly_data.append({
                                '异常类型': anomaly_type,
                                '帧索引': idx,
                                '异常值': val
                            })
                
                if anomaly_data:
                    anomaly_df = pd.DataFrame(anomaly_data)
                    anomaly_df.to_excel(writer, sheet_name='异常检测', index=False)
            
            print(f"详细数据已导出到: {save_path}")
            return True
            
        except Exception as e:
            print(f"导出失败: {e}")
            return False

# 使用示例
if __name__ == '__main__':
    # 创建示例数据进行测试
    np.random.seed(42)
    n_frames = 100
    
    # 模拟结节历史数据
    timestamps = np.linspace(0, 10, n_frames)
    areas = np.random.exponential(5, n_frames) + np.sin(timestamps) * 2
    areas[areas < 0] = 0
    
    risk_scores = np.random.beta(2, 5, n_frames)
    risk_scores += 0.3 * np.sin(timestamps * 2)  # 添加周期性
    risk_scores = np.clip(risk_scores, 0, 1)
    
    # 添加一些异常值
    anomaly_indices = [20, 45, 78]
    risk_scores[anomaly_indices] += 0.4
    areas[anomaly_indices] *= 3
    
    nodule_history = {
        'timestamps': timestamps.tolist(),
        'areas': areas.tolist(),
        'circularities': np.random.uniform(0.3, 0.9, n_frames).tolist(),
        'intensities': np.random.uniform(0.4, 1.0, n_frames).tolist(),
        'risk_scores': risk_scores.tolist(),
        'count': np.random.poisson(1.5, n_frames).tolist(),
        'positions': [(np.random.uniform(0, 8), np.random.uniform(0, 12)) for _ in range(n_frames)]
    }
    
    # 创建分析器并运行分析
    analyzer = StatisticalAnalyzer()
    results = analyzer.analyze_nodule_trends(nodule_history)
    
    # 生成报告
    report = analyzer.generate_comprehensive_report()
    print(report)
    
    # 创建图表
    fig = analyzer.create_statistical_plots('statistical_analysis.png')
    plt.show()
    
    # 导出详细数据
    analyzer.export_detailed_data('detailed_analysis.xlsx')