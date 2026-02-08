#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动态逐帧检测系统主程序
Enhanced Dynamic Frame-by-Frame Detection System

集成功能：
- 高级结节检测算法
- 实时可视化界面
- 统计分析和报告生成
- 多种导出格式支持

作者: AI Assistant
版本: 2.0
"""

import sys
import os
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import traceback

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from enhanced_detection_system import EnhancedNoduleDetectionSystem
    from modern_detection_gui import ModernDetectionGUI
    from statistical_analysis import StatisticalAnalyzer
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保所有必要的文件都在同一目录下")
    sys.exit(1)

class MainDetectionApp:
    def __init__(self):
        """初始化主应用程序"""
        self.root = tk.Tk()
        self.root.title("动态逐帧检测系统 - 主界面")
        self.root.geometry("800x600")
        self.root.configure(bg='#f0f0f0')
        
        # 设置应用图标和样式
        self.setup_styles()
        
        # 创建启动界面
        self.create_startup_interface()
        
        # 检查依赖
        self.check_dependencies()
    
    def setup_styles(self):
        """设置应用样式"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # 自定义样式
        style.configure('Title.TLabel', 
                       font=('Arial', 24, 'bold'), 
                       background='#f0f0f0',
                       foreground='#2c3e50')
        
        style.configure('Subtitle.TLabel', 
                       font=('Arial', 14), 
                       background='#f0f0f0',
                       foreground='#34495e')
        
        style.configure('Info.TLabel', 
                       font=('Arial', 11), 
                       background='#f0f0f0',
                       foreground='#7f8c8d')
        
        style.configure('Launch.TButton', 
                       font=('Arial', 12, 'bold'),
                       padding=(20, 10))
        
        style.configure('Feature.TButton', 
                       font=('Arial', 10),
                       padding=(15, 8))
    
    def create_startup_interface(self):
        """创建启动界面"""
        # 主容器
        main_frame = ttk.Frame(self.root)
        main_frame.pack(expand=True, fill='both', padx=40, pady=30)
        
        # 标题区域
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill='x', pady=(0, 30))
        
        ttk.Label(title_frame, 
                 text="动态逐帧检测系统", 
                 style='Title.TLabel').pack()
        
        ttk.Label(title_frame, 
                 text="Enhanced Dynamic Frame-by-Frame Detection System", 
                 style='Subtitle.TLabel').pack(pady=(5, 0))
        
        ttk.Label(title_frame, 
                 text="基于机器学习的智能结节检测与分析平台", 
                 style='Info.TLabel').pack(pady=(10, 0))
        
        # 功能介绍区域
        features_frame = ttk.LabelFrame(main_frame, text="系统功能", padding=20)
        features_frame.pack(fill='both', expand=True, pady=(0, 20))
        
        features = [
            "🔍 高精度结节检测算法",
            "📊 实时动态可视化分析",
            "📈 智能统计趋势分析",
            "⚡ 多线程并行处理",
            "📋 详细分析报告生成",
            "🎬 高质量GIF动画导出",
            "⚙️ 灵活参数调整界面",
            "📁 多格式数据导出"
        ]
        
        # 创建两列布局显示功能
        left_frame = ttk.Frame(features_frame)
        right_frame = ttk.Frame(features_frame)
        left_frame.pack(side='left', fill='both', expand=True, padx=(0, 10))
        right_frame.pack(side='right', fill='both', expand=True, padx=(10, 0))
        
        for i, feature in enumerate(features):
            frame = left_frame if i < len(features)//2 else right_frame
            ttk.Label(frame, text=feature, font=('Arial', 11)).pack(anchor='w', pady=2)
        
        # 启动按钮区域
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill='x')
        
        # 主启动按钮
        launch_button = ttk.Button(button_frame, 
                                  text="启动检测系统", 
                                  style='Launch.TButton',
                                  command=self.launch_main_system)
        launch_button.pack(pady=10)
        
        # 功能按钮行
        feature_buttons_frame = ttk.Frame(button_frame)
        feature_buttons_frame.pack(fill='x', pady=(10, 0))
        
        ttk.Button(feature_buttons_frame, 
                  text="快速检测", 
                  style='Feature.TButton',
                  command=self.quick_detection).pack(side='left', padx=5)
        
        ttk.Button(feature_buttons_frame, 
                  text="批量分析", 
                  style='Feature.TButton',
                  command=self.batch_analysis).pack(side='left', padx=5)
        
        ttk.Button(feature_buttons_frame, 
                  text="系统设置", 
                  style='Feature.TButton',
                  command=self.show_settings).pack(side='left', padx=5)
        
        ttk.Button(feature_buttons_frame, 
                  text="帮助文档", 
                  style='Feature.TButton',
                  command=self.show_help).pack(side='left', padx=5)
        
        # 状态栏
        self.status_frame = ttk.Frame(main_frame)
        self.status_frame.pack(fill='x', pady=(20, 0))
        
        self.status_var = tk.StringVar(value="系统就绪")
        self.status_label = ttk.Label(self.status_frame, 
                                     textvariable=self.status_var,
                                     style='Info.TLabel')
        self.status_label.pack(side='left')
        
        # 版本信息
        version_label = ttk.Label(self.status_frame, 
                                 text="v2.0", 
                                 style='Info.TLabel')
        version_label.pack(side='right')
    
    def check_dependencies(self):
        """检查系统依赖"""
        try:
            import numpy
            import pandas
            import matplotlib
            import sklearn
            import scipy
            import PIL
            import seaborn
            
            self.status_var.set("系统依赖检查完成 ✓")
            
        except ImportError as e:
            missing_module = str(e).split("'")[1] if "'" in str(e) else "未知模块"
            self.status_var.set(f"缺少依赖: {missing_module}")
            messagebox.showerror("依赖错误", 
                               f"缺少必要的Python模块: {missing_module}\n"
                               f"请使用 pip install {missing_module} 安装")
    
    def launch_main_system(self):
        """启动主检测系统"""
        try:
            self.status_var.set("正在启动主系统...")
            
            # 在新线程中启动GUI，避免阻塞
            def start_gui():
                try:
                    # 创建新的Tk实例
                    gui_root = tk.Tk()
                    app = ModernDetectionGUI(gui_root)
                    
                    # 设置关闭回调
                    def on_gui_closing():
                        app.on_closing()
                        self.status_var.set("主系统已关闭")
                    
                    gui_root.protocol("WM_DELETE_WINDOW", on_gui_closing)
                    gui_root.mainloop()
                    
                except Exception as e:
                    error_msg = f"启动主系统失败: {str(e)}"
                    print(error_msg)
                    print(traceback.format_exc())
                    messagebox.showerror("启动错误", error_msg)
                    self.status_var.set("启动失败")
            
            # 启动GUI线程
            gui_thread = threading.Thread(target=start_gui, daemon=True)
            gui_thread.start()
            
            self.status_var.set("主系统已启动")
            
        except Exception as e:
            error_msg = f"启动失败: {str(e)}"
            messagebox.showerror("错误", error_msg)
            self.status_var.set("启动失败")
    
    def quick_detection(self):
        """快速检测功能"""
        from tkinter import filedialog
        
        # 选择文件
        file_path = filedialog.askopenfilename(
            title="选择CSV数据文件",
            filetypes=[("CSV文件", "*.csv"), ("所有文件", "*.*")]
        )
        
        if not file_path:
            return
        
        # 选择输出路径
        output_path = filedialog.asksaveasfilename(
            title="保存检测结果",
            defaultextension=".gif",
            filetypes=[("GIF动画", "*.gif"), ("所有文件", "*.*")]
        )
        
        if not output_path:
            return
        
        def run_quick_detection():
            try:
                self.status_var.set("正在执行快速检测...")
                
                import pandas as pd
                
                # 读取数据
                df = pd.read_csv(file_path)
                
                # 创建检测器
                detector = EnhancedNoduleDetectionSystem()
                
                # 执行检测（限制帧数以提高速度）
                success = detector.create_enhanced_visualization(
                    df, output_path, max_frames=20
                )
                
                if success:
                    self.status_var.set("快速检测完成 ✓")
                    messagebox.showinfo("完成", f"检测结果已保存到:\n{output_path}")
                else:
                    self.status_var.set("快速检测失败")
                    messagebox.showerror("错误", "检测过程中出现错误")
                    
            except Exception as e:
                error_msg = f"快速检测失败: {str(e)}"
                self.status_var.set("检测失败")
                messagebox.showerror("错误", error_msg)
        
        # 在后台线程执行
        detection_thread = threading.Thread(target=run_quick_detection, daemon=True)
        detection_thread.start()
    
    def batch_analysis(self):
        """批量分析功能"""
        from tkinter import filedialog
        
        # 选择多个文件
        file_paths = filedialog.askopenfilenames(
            title="选择多个CSV数据文件",
            filetypes=[("CSV文件", "*.csv"), ("所有文件", "*.*")]
        )
        
        if not file_paths:
            return
        
        # 选择输出目录
        output_dir = filedialog.askdirectory(title="选择输出目录")
        
        if not output_dir:
            return
        
        def run_batch_analysis():
            try:
                self.status_var.set(f"正在批量分析 {len(file_paths)} 个文件...")
                
                import pandas as pd
                
                for i, file_path in enumerate(file_paths):
                    filename = os.path.splitext(os.path.basename(file_path))[0]
                    
                    # 读取数据
                    df = pd.read_csv(file_path)
                    
                    # 创建检测器
                    detector = EnhancedNoduleDetectionSystem()
                    
                    # 生成GIF
                    gif_path = os.path.join(output_dir, f"{filename}_detection.gif")
                    detector.create_enhanced_visualization(df, gif_path, max_frames=30)
                    
                    # 生成报告
                    report_path = os.path.join(output_dir, f"{filename}_report.txt")
                    with open(report_path, 'w', encoding='utf-8') as f:
                        f.write(detector.generate_analysis_report())
                    
                    # 更新进度
                    progress = (i + 1) / len(file_paths) * 100
                    self.status_var.set(f"批量分析进度: {progress:.1f}%")
                
                self.status_var.set("批量分析完成 ✓")
                messagebox.showinfo("完成", f"批量分析完成！\n结果保存在: {output_dir}")
                
            except Exception as e:
                error_msg = f"批量分析失败: {str(e)}"
                self.status_var.set("分析失败")
                messagebox.showerror("错误", error_msg)
        
        # 在后台线程执行
        batch_thread = threading.Thread(target=run_batch_analysis, daemon=True)
        batch_thread.start()
    
    def show_settings(self):
        """显示系统设置"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("系统设置")
        settings_window.geometry("500x400")
        settings_window.configure(bg='#f0f0f0')
        
        # 设置内容
        ttk.Label(settings_window, text="系统设置", font=('Arial', 16, 'bold')).pack(pady=20)
        
        # 检测参数设置
        params_frame = ttk.LabelFrame(settings_window, text="默认检测参数", padding=20)
        params_frame.pack(fill='x', padx=20, pady=10)
        
        # GMM组件数
        ttk.Label(params_frame, text="GMM组件数:").pack(anchor='w')
        gmm_var = tk.IntVar(value=3)
        ttk.Scale(params_frame, from_=2, to=5, variable=gmm_var, orient='horizontal').pack(fill='x')
        
        # 平滑参数
        ttk.Label(params_frame, text="平滑参数:").pack(anchor='w')
        smooth_var = tk.DoubleVar(value=0.8)
        ttk.Scale(params_frame, from_=0.1, to=2.0, variable=smooth_var, orient='horizontal').pack(fill='x')
        
        # 性能设置
        perf_frame = ttk.LabelFrame(settings_window, text="性能设置", padding=20)
        perf_frame.pack(fill='x', padx=20, pady=10)
        
        ttk.Label(perf_frame, text="最大处理帧数:").pack(anchor='w')
        max_frames_var = tk.IntVar(value=50)
        ttk.Scale(perf_frame, from_=10, to=200, variable=max_frames_var, orient='horizontal').pack(fill='x')
        
        # 按钮
        button_frame = ttk.Frame(settings_window)
        button_frame.pack(fill='x', padx=20, pady=20)
        
        ttk.Button(button_frame, text="保存设置", command=settings_window.destroy).pack(side='right', padx=5)
        ttk.Button(button_frame, text="恢复默认", command=lambda: None).pack(side='right', padx=5)
    
    def show_help(self):
        """显示帮助文档"""
        help_window = tk.Toplevel(self.root)
        help_window.title("帮助文档")
        help_window.geometry("700x500")
        help_window.configure(bg='#f0f0f0')
        
        # 创建文本框和滚动条
        text_frame = ttk.Frame(help_window)
        text_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        text_widget = tk.Text(text_frame, wrap='word', font=('Arial', 11))
        scrollbar = ttk.Scrollbar(text_frame, orient='vertical', command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # 帮助内容
        help_content = """
动态逐帧检测系统 - 使用指南

=== 系统概述 ===
本系统是一个基于机器学习的智能结节检测与分析平台，专门用于处理时序应力数据，
实现结节的自动检测、跟踪和分析。

=== 主要功能 ===

1. 高精度结节检测
   - 基于高斯混合模型的智能检测算法
   - 支持多种形态学后处理方法
   - 可调节的敏感度和参数设置

2. 实时动态可视化
   - 多种可视化模式（热力图、等高线、3D视图）
   - 实时播放控制和帧跳转
   - 结节特征实时显示

3. 统计分析功能
   - 趋势分析和异常检测
   - 相关性分析和周期性检测
   - 详细的统计报告生成

4. 数据导出功能
   - 高质量GIF动画导出
   - Excel格式详细数据导出
   - 文本格式分析报告

=== 使用步骤 ===

1. 数据准备
   - 准备CSV格式的应力数据文件
   - 确保数据包含MAT_0到MAT_95列（96个应力点）
   - 确保包含SN列作为时间戳

2. 启动系统
   - 点击"启动检测系统"按钮
   - 在主界面中加载CSV数据文件

3. 参数调整
   - 根据数据特点调整检测参数
   - 实时预览检测效果
   - 优化检测精度

4. 分析和导出
   - 查看实时检测结果
   - 分析统计趋势
   - 导出结果和报告

=== 技术参数 ===

- 支持的数据格式: CSV
- 最大处理帧数: 200帧
- 检测精度: 亚像素级别
- 支持的导出格式: GIF, PNG, TXT, XLSX

=== 注意事项 ===

1. 确保数据质量良好，避免过多缺失值
2. 根据实际需求调整检测参数
3. 大数据集处理时请耐心等待
4. 定期保存分析结果

=== 技术支持 ===

如遇到问题，请检查：
1. Python环境和依赖包是否完整
2. 数据格式是否正确
3. 系统资源是否充足

更多技术细节请参考源代码注释。
        """
        
        text_widget.insert('1.0', help_content)
        text_widget.config(state='disabled')  # 只读模式
    
    def run(self):
        """运行主应用程序"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            print("\n程序被用户中断")
        except Exception as e:
            print(f"程序运行错误: {e}")
            traceback.print_exc()

def main():
    """主函数"""
    print("=" * 60)
    print("动态逐帧检测系统 v2.0")
    print("Enhanced Dynamic Frame-by-Frame Detection System")
    print("=" * 60)
    print("正在启动系统...")
    
    try:
        app = MainDetectionApp()
        app.run()
    except Exception as e:
        print(f"系统启动失败: {e}")
        traceback.print_exc()
        input("按回车键退出...")

if __name__ == '__main__':
    main()