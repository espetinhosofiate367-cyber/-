import os
os.environ['OMP_NUM_THREADS'] = '1'  # 必须在导入numpy和sklearn之前设置
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from skimage.measure import regionprops
from skimage.morphology import binary_dilation, binary_erosion, closing, disk
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from PIL import Image
import io
from tqdm import tqdm
from matplotlib.colors import LinearSegmentedColormap
plt.rcParams['figure.dpi'] = 300
plt.rcParams['axes.unicode_minus'] = False

def create_nodule_evolution_gif(df, output_path='nodule_detection_evolution.gif', frame_count=50):
    """Create dynamic visualization for nodule detection"""
    # 设置matplotlib样式

    plt.rcParams.update({
        'figure.dpi': 300,
        'axes.unicode_minus': False,
    })
    stress_columns = [f'MAT_{i}' for i in range(96)]
    stress_data = df[stress_columns].values
    time_points = df['SN'].values  # 提前定义time_points
    
    # 新增帧数选择（移到time_points定义之后）
    while True:
        try:
            max_frames = int(input("请输入要处理的帧数（1-{}）：".format(len(time_points))))
            if 1 <= max_frames <= len(time_points):
                break
            print("输入超出范围，请重新输入！")
        except ValueError:
            print("请输入有效的数字！")
    
    # 处理NaN值
    col_mean_stress = np.nanmean(stress_data, axis=0)
    inds_stress = np.where(np.isnan(stress_data))
    stress_data[inds_stress] = np.take(col_mean_stress, inds_stress[1])
    
    images = []
    print("Generating dynamic analysis for nodule detection...")
    
    medical_colors = ['black', 'navy', 'blue', 'cyan', 'yellow', 'red']
    medical_cmap = LinearSegmentedColormap.from_list('medical', medical_colors)
    
    # 记录结节特征随时间的变化
    nodule_features = {
        'area': [],
        'circularity': [],
        'intensity': []
    }
    
    for idx in tqdm(range(min(frame_count, len(time_points)))):  # 修改此处
        # 创建三联图布局
        fig = plt.figure(figsize=(18, 10))
        gs = plt.GridSpec(2, 3, height_ratios=[3, 1])
        
        # 1. 原始应力分布图
        ax_orig = plt.subplot(gs[0, 0])
        current_stress = stress_data[idx:idx+1, :]
        stress_grid = current_stress.reshape(12, 8)
        
        # 归一化处理
        min_val = np.nanmin(stress_grid)
        max_val = np.nanmax(stress_grid)
        if max_val != min_val:
            stress_normalized = (stress_grid - min_val) / (max_val - min_val)
        else:
            stress_normalized = np.zeros_like(stress_grid)
        
        im_orig = ax_orig.imshow(stress_normalized, cmap=medical_cmap, origin='lower')
        ax_orig.set_title('Original Stress Distribution', fontsize=12)
        plt.colorbar(im_orig, ax=ax_orig, label='Normalized Stress Value')
        
        # 2. 结节检测图
        ax_detect = plt.subplot(gs[0, 1])
        
        # 使用改进的高斯混合模型进行结节检测
        try:
            gmm = GaussianMixture(n_components=2, random_state=42)
            gmm.fit(stress_normalized.reshape(-1, 1))
            labels = gmm.predict(stress_normalized.reshape(-1, 1))
            
            # 确定异常类别（应力值较高的类别）
            means = gmm.means_.flatten()
            abnormal_class = np.argmax(means)
            nodule_mask = (labels.reshape(12, 8) == abnormal_class).astype(int)
            
            # 形态学处理，优化结节区域
            selem = disk(2)  # 使用较小的结构元素，避免过度平滑
            nodule_mask = closing(nodule_mask, selem)
            
            # 计算结节特征
            props = regionprops(nodule_mask.astype(int))
            
            if props:
                largest_nodule = max(props, key=lambda x: x.area)
                area = largest_nodule.area
                perimeter = largest_nodule.perimeter
                circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
                centroid = largest_nodule.centroid
                
                # 记录特征
                nodule_features['area'].append(area)
                nodule_features['circularity'].append(circularity)
                nodule_features['intensity'].append(np.mean(stress_normalized[nodule_mask > 0]))
                
                # 显示结节检测结果
                ax_detect.imshow(stress_normalized, cmap='gray', origin='lower')
                ax_detect.imshow(nodule_mask, cmap='Reds', alpha=0.4, origin='lower')
                ax_detect.plot(centroid[1], centroid[0], 'y*', markersize=15)
                
                # 添加结节特征信息
                stats_text = (
                    f'Nodule Features:\n'
                    f'Area: {area:.1f} units^2\n'
                    f'Circularity: {circularity:.3f}\n'
                    f'Mean Intensity: {nodule_features["intensity"][-1]:.3f}\n'
                    f'Position: ({centroid[1]:.1f}, {centroid[0]:.1f})'
                )
                ax_detect.text(1.05, 0.95, stats_text,
                                transform=ax_detect.transAxes,
                                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
                                verticalalignment='top',
                                fontsize=10)
            
        except Exception as e:
            print(f"Warning: Nodule detection failed at timestamp {time_points[idx]}")
            nodule_mask = np.zeros_like(stress_normalized)
        
        ax_detect.set_title('Nodule Detection Result', fontsize=12)
        
        # 3. 特征趋势图
        ax_trend = plt.subplot(gs[0, 2])
        if len(nodule_features['area']) > 1:
            ax_trend.plot(nodule_features['area'], label='Area', color='blue')
            ax_trend2 = ax_trend.twinx()
            ax_trend2.plot(nodule_features['circularity'], label='Circularity', color='red')
            ax_trend.set_xlabel('Time Series')
            ax_trend.set_ylabel('Area', color='blue')
            ax_trend2.set_ylabel('Circularity', color='red')
            ax_trend.set_title('Feature Trends', fontsize=12)
            
            # 合并图例
            lines1, labels1 = ax_trend.get_legend_handles_labels()
            lines2, labels2 = ax_trend2.get_legend_handles_labels()
            ax_trend2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        # 添加总标题
        plt.suptitle(f'Nodule Detection Analysis - Timestamp: {time_points[idx]}', 
                    fontsize=14, y=1.02)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存当前帧
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        images.append(Image.open(buf))
        plt.close()
    
    print(f"Saving analysis results to {output_path}")
    # 保存GIF
    # 删除嵌套的函数定义（原160行附近）
    # 原错误：在函数内部重复定义了同名的函数
    # 将保存GIF的逻辑移到主函数体内
    
    # 修改保存逻辑
    if images:
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:frame_count],  # 应用帧数限制
            duration=300,
            loop=0,
            optimize=True,
            quality=95
        )
    print("Nodule detection analysis completed! Check the output file:", output_path)
# 修改原文件的main部分
if __name__ == '__main__':
    # 原Tkinter代码可以删除，已整合到GUI中
    pass
    import tkinter as tk
    from tkinter import filedialog
    
    # 初始化Tkinter
    root = tk.Tk()
    root.withdraw()
    
    # 选择输入文件
    input_path = filedialog.askopenfilename(
        title='选择输入CSV文件',
        filetypes=[('CSV文件', '*.csv'), ('所有文件', '*.*')]
    )
    if not input_path:
        print("未选择输入文件，程序终止")
        exit()
    
    # 选择输出文件
    output_path = filedialog.asksaveasfilename(
        title='保存结果GIF文件',
        defaultextension='.gif',
        filetypes=[('GIF图像', '*.gif')]
    )
    if not output_path:
        print("未指定输出路径，程序终止")
        exit()
    
    # 销毁Tkinter窗口
    root.destroy()
    
    df = pd.read_csv(input_path)  
    create_nodule_evolution_gif(df, output_path)
