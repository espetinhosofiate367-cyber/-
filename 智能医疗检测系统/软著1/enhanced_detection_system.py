import os
os.environ['OMP_NUM_THREADS'] = '1'
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from skimage.measure import regionprops, label
from skimage.morphology import binary_dilation, binary_erosion, closing, disk, opening
from skimage.filters import gaussian
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.fftpack import fft
from scipy.ndimage import binary_fill_holes
from PIL import Image
import io
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 300
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']

class EnhancedNoduleDetectionSystem:
    def __init__(self):
        """初始化增强版结节检测系统"""
        self.medical_colors = ['black', 'navy', 'blue', 'cyan', 'yellow', 'orange', 'red']
        self.medical_cmap = LinearSegmentedColormap.from_list('medical', self.medical_colors)
        
        # 检测参数
        self.detection_params = {
            'gmm_components': 3,
            'smoothing_sigma': 0.8,
            'morphology_disk_size': 2,
            'min_nodule_area': 3,
            'sensitivity_threshold': 0.7
        }
        
        # 统计数据存储
        self.nodule_history = {
            'timestamps': [],
            'areas': [],
            'circularities': [],
            'intensities': [],
            'positions': [],
            'count': [],
            'risk_scores': []
        }
    
    def preprocess_stress_data(self, stress_data):
        """预处理应力数据"""
        # 处理NaN值
        col_mean_stress = np.nanmean(stress_data, axis=0)
        inds_stress = np.where(np.isnan(stress_data))
        stress_data[inds_stress] = np.take(col_mean_stress, inds_stress[1])
        
        return stress_data
    
    def advanced_nodule_detection(self, stress_grid, timestamp):
        """高级结节检测算法"""
        # 1. 数据平滑
        smoothed = gaussian(stress_grid, sigma=self.detection_params['smoothing_sigma'])
        
        # 2. 归一化
        min_val = np.nanmin(smoothed)
        max_val = np.nanmax(smoothed)
        if max_val != min_val:
            normalized = (smoothed - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(smoothed)
        
        # 3. 多组件高斯混合模型检测
        try:
            gmm = GaussianMixture(
                n_components=self.detection_params['gmm_components'], 
                random_state=42,
                covariance_type='full'
            )
            gmm.fit(normalized.reshape(-1, 1))
            labels = gmm.predict(normalized.reshape(-1, 1))
            probabilities = gmm.predict_proba(normalized.reshape(-1, 1))
            
            # 确定异常类别（应力值最高的类别）
            means = gmm.means_.flatten()
            abnormal_class = np.argmax(means)
            
            # 基于概率的软分割
            prob_map = probabilities[:, abnormal_class].reshape(12, 8)
            nodule_mask = (prob_map > self.detection_params['sensitivity_threshold']).astype(int)
            
        except Exception as e:
            print(f"GMM检测失败: {e}")
            # 备用检测方法：基于阈值
            threshold = np.percentile(normalized, 85)
            nodule_mask = (normalized > threshold).astype(int)
            prob_map = normalized
        
        # 4. 形态学后处理
        selem = disk(self.detection_params['morphology_disk_size'])
        nodule_mask = closing(nodule_mask, selem)
        nodule_mask = opening(nodule_mask, disk(1))  # 去除小噪点
        nodule_mask = binary_fill_holes(nodule_mask)  # 填充空洞
        
        # 5. 连通域分析
        labeled_mask = label(nodule_mask)
        props = regionprops(labeled_mask, intensity_image=normalized)
        
        # 6. 筛选有效结节
        valid_nodules = []
        for prop in props:
            if prop.area >= self.detection_params['min_nodule_area']:
                # 计算结节特征
                area = prop.area
                perimeter = prop.perimeter
                circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
                mean_intensity = prop.mean_intensity
                centroid = prop.centroid
                
                # 计算风险评分
                risk_score = self.calculate_risk_score(area, circularity, mean_intensity)
                
                valid_nodules.append({
                    'area': area,
                    'circularity': circularity,
                    'intensity': mean_intensity,
                    'centroid': centroid,
                    'risk_score': risk_score,
                    'bbox': prop.bbox
                })
        
        # 7. 更新历史记录
        self.update_nodule_history(timestamp, valid_nodules)
        
        return normalized, nodule_mask, valid_nodules, prob_map
    
    def calculate_risk_score(self, area, circularity, intensity):
        """计算结节风险评分"""
        # 基于面积、圆形度和强度的综合评分
        area_score = min(area / 20.0, 1.0)  # 面积越大风险越高
        shape_score = 1.0 - circularity  # 不规则形状风险更高
        intensity_score = intensity  # 强度越高风险越高
        
        # 加权平均
        risk_score = 0.4 * area_score + 0.3 * shape_score + 0.3 * intensity_score
        return min(risk_score, 1.0)
    
    def update_nodule_history(self, timestamp, nodules):
        """更新结节历史记录"""
        self.nodule_history['timestamps'].append(timestamp)
        
        if nodules:
            # 选择最大的结节作为主要结节
            main_nodule = max(nodules, key=lambda x: x['area'])
            self.nodule_history['areas'].append(main_nodule['area'])
            self.nodule_history['circularities'].append(main_nodule['circularity'])
            self.nodule_history['intensities'].append(main_nodule['intensity'])
            self.nodule_history['positions'].append(main_nodule['centroid'])
            self.nodule_history['risk_scores'].append(main_nodule['risk_score'])
        else:
            # 无结节检测到
            self.nodule_history['areas'].append(0)
            self.nodule_history['circularities'].append(0)
            self.nodule_history['intensities'].append(0)
            self.nodule_history['positions'].append((0, 0))
            self.nodule_history['risk_scores'].append(0)
        
        self.nodule_history['count'].append(len(nodules))
    
    def create_enhanced_visualization(self, df, output_path='enhanced_nodule_detection.gif', max_frames=50):
        """创建增强版可视化"""
        try:
            print("开始生成增强版动态结节检测分析...")
            
            # 数据预处理
            stress_columns = [f'MAT_{i}' for i in range(96)]
            if not all(col in df.columns for col in stress_columns):
                raise ValueError("数据文件缺少必要的应力数据列")
            
            stress_data = df[stress_columns].values
            time_points = df['SN'].values
            
            stress_data = self.preprocess_stress_data(stress_data)
            
            # 限制帧数
            total_frames = min(max_frames, len(time_points))
            images = []
            
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            for idx in tqdm(range(total_frames), desc="生成动画帧"):
                try:
                    # 创建四联图布局
                    fig = plt.figure(figsize=(20, 12))
                    gs = plt.GridSpec(2, 4, height_ratios=[3, 2], width_ratios=[1, 1, 1, 1])
                    
                    # 当前应力数据
                    current_stress = stress_data[idx:idx+1, :]
                    stress_grid = current_stress.reshape(12, 8)
                    
                    # 执行结节检测
                    normalized, nodule_mask, nodules, prob_map = self.advanced_nodule_detection(
                        stress_grid, time_points[idx]
                    )
                    
                    # 1. 等高线图 (左上)
                    ax1 = plt.subplot(gs[0, 0])
                    contour = ax1.contour(normalized, levels=15, colors='white', alpha=0.6, linewidths=0.8)
                    contourf = ax1.contourf(normalized, levels=20, cmap=self.medical_cmap, alpha=0.8)
                    ax1.set_title(f'等高线图 - 帧 {idx}', fontsize=12, color='white')
                    ax1.set_facecolor('black')
                    plt.colorbar(contourf, ax=ax1, label='应力值')
                    
                    # 2. 热力图 (右上)
                    ax2 = plt.subplot(gs[0, 1])
                    im2 = ax2.imshow(normalized, cmap=self.medical_cmap, origin='lower')
                    ax2.set_title(f'热力图 - 帧 {idx}', fontsize=12)
                    plt.colorbar(im2, ax=ax2, label='归一化应力')
                    
                    # 3. 结节检测图 (左中)
                    ax3 = plt.subplot(gs[0, 2])
                    ax3.imshow(normalized, cmap='gray', origin='lower', alpha=0.7)
                    ax3.imshow(nodule_mask, cmap='Reds', alpha=0.6, origin='lower')
                    
                    # 标记检测到的结节
                    for nodule in nodules:
                        centroid = nodule['centroid']
                        risk = nodule['risk_score']
                        color = 'red' if risk > 0.7 else 'yellow' if risk > 0.4 else 'green'
                        ax3.plot(centroid[1], centroid[0], '*', color=color, markersize=15)
                        ax3.text(centroid[1]+0.5, centroid[0]+0.5, f'{risk:.2f}', 
                                color=color, fontsize=10, fontweight='bold')
                    
                    ax3.set_title(f'结节检测 - 帧 {idx}', fontsize=12)
                    
                    # 4. 统计趋势图 (右中)
                    ax4 = plt.subplot(gs[0, 3])
                    if len(self.nodule_history['areas']) > 1:
                        frames = range(len(self.nodule_history['areas']))
                        ax4.plot(frames, self.nodule_history['areas'], 'b-', label='面积', linewidth=2)
                        ax4_twin = ax4.twinx()
                        ax4_twin.plot(frames, self.nodule_history['risk_scores'], 'r-', label='风险评分', linewidth=2)
                        
                        ax4.set_xlabel('帧数')
                        ax4.set_ylabel('面积', color='blue')
                        ax4_twin.set_ylabel('风险评分', color='red')
                        ax4.set_title('特征趋势', fontsize=12)
                        
                        # 图例
                        lines1, labels1 = ax4.get_legend_handles_labels()
                        lines2, labels2 = ax4_twin.get_legend_handles_labels()
                        ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
                    
                    # 5. 详细统计信息 (下方)
                    ax5 = plt.subplot(gs[1, :])
                    ax5.axis('off')
                    
                    # 当前帧统计
                    current_stats = f"""
                    时间戳: {time_points[idx]:.3f}s  |  检测到结节数: {len(nodules)}  |  
                    """
                    
                    if nodules:
                        main_nodule = max(nodules, key=lambda x: x['area'])
                        current_stats += f"""主要结节 - 面积: {main_nodule['area']:.1f}  圆形度: {main_nodule['circularity']:.3f}  
                        强度: {main_nodule['intensity']:.3f}  风险评分: {main_nodule['risk_score']:.3f}  
                        位置: ({main_nodule['centroid'][1]:.1f}, {main_nodule['centroid'][0]:.1f})"""
                    else:
                        current_stats += "未检测到显著结节"
                    
                    # 历史统计
                    if len(self.nodule_history['areas']) > 0:
                        avg_area = np.mean([a for a in self.nodule_history['areas'] if a > 0])
                        max_risk = max(self.nodule_history['risk_scores']) if self.nodule_history['risk_scores'] else 0
                        detection_rate = sum(1 for c in self.nodule_history['count'] if c > 0) / len(self.nodule_history['count']) if self.nodule_history['count'] else 0
                        
                        history_stats = f"""
                        历史统计 - 平均面积: {avg_area:.1f}  最高风险: {max_risk:.3f}  检出率: {detection_rate:.1%}
                        """
                    else:
                        history_stats = "历史统计 - 暂无数据"
                    
                    ax5.text(0.02, 0.7, current_stats, transform=ax5.transAxes, fontsize=11,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
                    ax5.text(0.02, 0.3, history_stats, transform=ax5.transAxes, fontsize=11,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
                    
                    # 总标题
                    plt.suptitle(f'动态应力分析 - 帧 {idx}/{total_frames-1} - 时间: {time_points[idx]:.3f}s', 
                                fontsize=16, y=0.95)
                    
                    plt.tight_layout()
                    
                    # 保存帧到内存
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', 
                               facecolor='white', edgecolor='none')
                    buf.seek(0)
                    
                    # 转换为PIL图像
                    img = Image.open(buf)
                    # 转换为RGB模式以确保GIF兼容性
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    images.append(img)
                    
                    # 清理内存
                    buf.close()
                    plt.close(fig)
                    
                except Exception as e:
                    print(f"生成第{idx}帧时出错: {str(e)}")
                    plt.close('all')  # 确保清理所有图形
                    continue
            
            # 保存GIF
            if images:
                print(f"保存动画到: {output_path}")
                try:
                    # 使用更稳定的GIF保存参数
                    images[0].save(
                        output_path,
                        save_all=True,
                        append_images=images[1:],
                        duration=500,  # 每帧500ms，稍慢一些更稳定
                        loop=0,
                        optimize=False,  # 关闭优化以避免某些兼容性问题
                        disposal=2  # 清除前一帧
                    )
                    print("增强版结节检测分析完成！")
                    
                    # 清理图像内存
                    for img in images:
                        img.close()
                    
                    return True
                except Exception as e:
                    print(f"保存GIF时出错: {str(e)}")
                    return False
            else:
                print("没有生成任何图像帧")
                return False
                
        except Exception as e:
            print(f"创建可视化时出错: {str(e)}")
            plt.close('all')  # 清理所有matplotlib图形
            return False
    
    def generate_analysis_report(self):
        """生成分析报告"""
        if not self.nodule_history['timestamps']:
            return "暂无分析数据"
        
        report = f"""
        === 结节检测分析报告 ===
        
        总帧数: {len(self.nodule_history['timestamps'])}
        检出率: {sum(1 for c in self.nodule_history['count'] if c > 0) / len(self.nodule_history['count']):.1%}
        
        结节特征统计:
        - 平均面积: {np.mean([a for a in self.nodule_history['areas'] if a > 0]):.2f}
        - 最大面积: {max(self.nodule_history['areas']):.2f}
        - 平均风险评分: {np.mean(self.nodule_history['risk_scores']):.3f}
        - 最高风险评分: {max(self.nodule_history['risk_scores']):.3f}
        
        异常检测:
        - 高风险帧数: {sum(1 for r in self.nodule_history['risk_scores'] if r > 0.7)}
        - 中风险帧数: {sum(1 for r in self.nodule_history['risk_scores'] if 0.4 < r <= 0.7)}
        """
        
        return report

# 使用示例
if __name__ == '__main__':
    import tkinter as tk
    from tkinter import filedialog
    
    # 创建检测系统
    detector = EnhancedNoduleDetectionSystem()
    
    # 文件选择
    root = tk.Tk()
    root.withdraw()
    
    input_path = filedialog.askopenfilename(
        title='选择输入CSV文件',
        filetypes=[('CSV文件', '*.csv'), ('所有文件', '*.*')]
    )
    
    if input_path:
        output_path = filedialog.asksaveasfilename(
            title='保存结果GIF文件',
            defaultextension='.gif',
            filetypes=[('GIF图像', '*.gif')]
        )
        
        if output_path:
            # 读取数据并生成可视化
            df = pd.read_csv(input_path)
            success = detector.create_enhanced_visualization(df, output_path, max_frames=30)
            
            if success:
                print("\n" + detector.generate_analysis_report())
        else:
            print("未指定输出路径")
    else:
        print("未选择输入文件")
    
    root.destroy()