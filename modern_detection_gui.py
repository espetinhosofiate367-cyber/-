#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动态逐帧检测系统 - 主界面程序
Enhanced Dynamic Frame-by-Frame Detection System - Main GUI

本模块实现了系统的主要图形用户界面，包括：
1. 现代化的GUI界面设计
2. 实时数据可视化功能
3. 检测参数交互式调整
4. 多视图同步显示
5. 串口数据采集支持
6. 文件导入导出功能

技术特点：
- 基于Tkinter的现代化界面设计
- 集成Matplotlib实现数据可视化
- 多线程处理确保界面响应性
- 支持实时参数调整和效果预览

作者: 四川大学 生物医学工程学院 靳天乐
版本: v2.1
创建日期: 2025.10.1
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import LinearSegmentedColormap
import threading
import json
from datetime import datetime
import serial
import serial.tools.list_ports
import queue
import time
from enhanced_detection_system import EnhancedNoduleDetectionSystem

class ModernDetectionGUI:
    """
    现代化检测系统图形用户界面类
    
    主要功能：
    1. 提供直观的用户操作界面
    2. 实时显示检测结果和统计信息
    3. 支持多种可视化模式
    4. 集成串口数据采集功能
    5. 提供丰富的导出选项
    """
    
    def __init__(self, root):
        """
        初始化GUI界面
        
        Args:
            root: Tkinter根窗口对象
        """
        self.root = root
        self.root.title("动态逐帧检测系统 v2.1")
        self.root.geometry("1600x1000")
        self.root.configure(bg='#f0f0f0')
        
        # 初始化检测系统
        self.detector = EnhancedNoduleDetectionSystem()
        self.data = None
        self.current_frame = 0
        self.is_playing = False
        self.play_thread = None
        
        # 串口相关变量
        self.serial_port = None
        self.serial_thread = None
        self.serial_data_queue = queue.Queue()
        self.is_serial_connected = False
        
        # 设置界面样式和颜色方案
        self.setup_color_schemes()
        self.setup_styles()
        
        # 创建界面组件
        self.create_widgets()
        
        # 加载配置文件
        self.load_config()
    
    def setup_color_schemes(self):
        """
        设置医学专业的颜色方案
        
        功能：
        1. 定义医学影像专用配色
        2. 创建自定义颜色映射
        3. 注册matplotlib颜色方案
        """
        # 医学专业配色方案
        self.medical_colors = {
            'primary': '#2E86AB',      # 医学蓝
            'secondary': '#A23B72',    # 深粉红
            'accent': '#F18F01',       # 橙色
            'success': '#C73E1D',      # 红色（高风险）
            'warning': '#F79824',      # 橙色（中风险）
            'safe': '#4CAF50',         # 绿色（低风险）
            'background': '#F8F9FA',   # 浅灰背景
            'surface': '#FFFFFF'       # 白色表面
        }
        
        # 创建自定义颜色映射
        self.nodule_cmap = LinearSegmentedColormap.from_list(
            'nodule', ['#E3F2FD', '#1976D2', '#0D47A1'], N=256
        )
        
        self.risk_cmap = LinearSegmentedColormap.from_list(
            'risk', ['#4CAF50', '#FF9800', '#F44336'], N=256
        )
        
        self.brightness_cmap = LinearSegmentedColormap.from_list(
            'brightness', ['#000000', '#424242', '#FFFFFF'], N=256
        )
        
        # 注册颜色映射到matplotlib
        plt.register_cmap(cmap=self.nodule_cmap)
        plt.register_cmap(cmap=self.risk_cmap)
        plt.register_cmap(cmap=self.brightness_cmap)
    
    def setup_styles(self):
        """
        设置ttk组件的自定义样式
        
        功能：
        1. 配置按钮样式
        2. 设置标签字体
        3. 定义颜色主题
        """
        style = ttk.Style()
        style.theme_use('clam')
        
        # 自定义样式配置
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'), background='#f0f0f0')
        style.configure('Header.TLabel', font=('Arial', 12, 'bold'), background='#f0f0f0')
        style.configure('Info.TLabel', font=('Arial', 10), background='#f0f0f0')
        style.configure('Success.TButton', background='#4CAF50')
        style.configure('Warning.TButton', background='#FF9800')
        style.configure('Error.TButton', background='#F44336')
    
    def create_widgets(self):
        """
        创建主界面的所有组件
        
        功能：
        1. 创建标题栏
        2. 构建控制面板
        3. 设置可视化区域
        4. 添加状态栏
        """
        # 主标题区域
        title_frame = ttk.Frame(self.root)
        title_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(title_frame, text="动态逐帧检测系统", style='Title.TLabel').pack()
        ttk.Label(title_frame, text="Enhanced Nodule Detection System", style='Info.TLabel').pack()
        
        # 创建主要布局框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # 左侧控制面板
        self.create_control_panel(main_frame)
        
        # 右侧可视化面板
        self.create_visualization_panel(main_frame)
        
        # 底部状态栏
        self.create_status_bar()
    
    def create_control_panel(self, parent):
        """
        创建左侧控制面板
        
        Args:
            parent: 父容器组件
            
        功能：
        1. 文件操作控制
        2. 检测参数设置
        3. 播放控制
        4. 统计信息显示
        5. 串口数据控制
        """
        control_frame = ttk.Frame(parent)
        control_frame.pack(side='left', fill='y', padx=(0, 10))
        
        # 文件操作区域
        file_frame = ttk.LabelFrame(control_frame, text="文件操作", padding=10)
        file_frame.pack(fill='x', pady=5)
        
        ttk.Button(file_frame, text="选择CSV文件", command=self.load_file, 
                  style='Success.TButton').pack(fill='x', pady=2)
        
        self.file_label = ttk.Label(file_frame, text="未选择文件", style='Info.TLabel')
        self.file_label.pack(fill='x', pady=2)
        
        ttk.Button(file_frame, text="导出GIF动画", command=self.export_gif).pack(fill='x', pady=2)
        ttk.Button(file_frame, text="导出分析报告", command=self.export_report).pack(fill='x', pady=2)
        
        # 检测参数设置区域
        param_frame = ttk.LabelFrame(control_frame, text="检测参数", padding=10)
        param_frame.pack(fill='x', pady=5)
        
        # GMM组件数设置
        ttk.Label(param_frame, text="GMM组件数:").pack(anchor='w')
        self.gmm_var = tk.IntVar(value=3)
        gmm_scale = ttk.Scale(param_frame, from_=2, to=5, variable=self.gmm_var, 
                             orient='horizontal', command=self.update_params)
        gmm_scale.pack(fill='x')
        self.gmm_label = ttk.Label(param_frame, text="3")
        self.gmm_label.pack(anchor='w')
        
        # 平滑参数设置
        ttk.Label(param_frame, text="平滑参数:").pack(anchor='w')
        self.smooth_var = tk.DoubleVar(value=0.8)
        smooth_scale = ttk.Scale(param_frame, from_=0.1, to=2.0, variable=self.smooth_var,
                                orient='horizontal', command=self.update_params)
        smooth_scale.pack(fill='x')
        self.smooth_label = ttk.Label(param_frame, text="0.8")
        self.smooth_label.pack(anchor='w')
        
        # 敏感度阈值设置
        ttk.Label(param_frame, text="敏感度阈值:").pack(anchor='w')
        self.sensitivity_var = tk.DoubleVar(value=0.7)
        sens_scale = ttk.Scale(param_frame, from_=0.1, to=1.0, variable=self.sensitivity_var,
                              orient='horizontal', command=self.update_params)
        sens_scale.pack(fill='x')
        self.sens_label = ttk.Label(param_frame, text="0.7")
        self.sens_label.pack(anchor='w')
        
        # 最小结节面积设置
        ttk.Label(param_frame, text="最小结节面积:").pack(anchor='w')
        self.area_var = tk.IntVar(value=3)
        area_scale = ttk.Scale(param_frame, from_=1, to=10, variable=self.area_var,
                              orient='horizontal', command=self.update_params)
        area_scale.pack(fill='x')
        self.area_label = ttk.Label(param_frame, text="3")
        self.area_label.pack(anchor='w')
        
        # 播放控制区域
        play_frame = ttk.LabelFrame(control_frame, text="播放控制", padding=10)
        play_frame.pack(fill='x', pady=5)
        
        # 帧控制按钮
        frame_control = ttk.Frame(play_frame)
        frame_control.pack(fill='x')
        
        ttk.Button(frame_control, text="◀◀", command=self.first_frame).pack(side='left')
        ttk.Button(frame_control, text="◀", command=self.prev_frame).pack(side='left')
        self.play_button = ttk.Button(frame_control, text="▶", command=self.toggle_play)
        self.play_button.pack(side='left')
        ttk.Button(frame_control, text="▶", command=self.next_frame).pack(side='left')
        ttk.Button(frame_control, text="▶▶", command=self.last_frame).pack(side='left')
        
        # 帧数滑块和显示
        self.frame_var = tk.IntVar(value=0)
        self.frame_scale = ttk.Scale(play_frame, from_=0, to=100, variable=self.frame_var,
                                    orient='horizontal', command=self.goto_frame)
        self.frame_scale.pack(fill='x', pady=5)
        
        self.frame_info = ttk.Label(play_frame, text="帧: 0/0", style='Info.TLabel')
        self.frame_info.pack()
        
        # 播放速度控制
        ttk.Label(play_frame, text="播放速度(ms):").pack(anchor='w')
        self.speed_var = tk.IntVar(value=500)
        speed_scale = ttk.Scale(play_frame, from_=100, to=2000, variable=self.speed_var,
                               orient='horizontal')
        speed_scale.pack(fill='x')
        
        # 统计信息显示区域
        stats_frame = ttk.LabelFrame(control_frame, text="实时统计", padding=10)
        stats_frame.pack(fill='x', pady=5)
        
        self.stats_text = tk.Text(stats_frame, height=6, width=30, font=('Courier', 9))
        stats_scroll = ttk.Scrollbar(stats_frame, orient='vertical', command=self.stats_text.yview)
        self.stats_text.configure(yscrollcommand=stats_scroll.set)
        
        self.stats_text.pack(side='left', fill='both', expand=True)
        stats_scroll.pack(side='right', fill='y')
        
        # 串口控制面板
        self.create_serial_panel(control_frame)
    
    def create_serial_panel(self, parent):
        """
        创建串口数据采集控制面板
        
        Args:
            parent: 父容器组件
            
        功能：
        1. 串口选择和配置
        2. 连接状态管理
        3. 实时数据显示
        4. 波特率设置
        """
        serial_frame = ttk.LabelFrame(parent, text="串口数据", padding=10)
        serial_frame.pack(fill='x', pady=5)
        
        # 串口选择区域
        port_frame = ttk.Frame(serial_frame)
        port_frame.pack(fill='x', pady=2)
        
        ttk.Label(port_frame, text="串口:").pack(side='left')
        self.port_var = tk.StringVar()
        self.port_combo = ttk.Combobox(port_frame, textvariable=self.port_var, width=15)
        self.port_combo.pack(side='left', padx=5)
        
        ttk.Button(port_frame, text="刷新", command=self.refresh_ports).pack(side='left', padx=2)
        
        # 波特率选择区域
        baud_frame = ttk.Frame(serial_frame)
        baud_frame.pack(fill='x', pady=2)
        
        ttk.Label(baud_frame, text="波特率:").pack(side='left')
        self.baud_var = tk.StringVar(value="9600")
        baud_combo = ttk.Combobox(baud_frame, textvariable=self.baud_var, 
                                 values=["9600", "19200", "38400", "57600", "115200"], width=10)
        baud_combo.pack(side='left', padx=5)
        
        # 连接控制区域
        connect_frame = ttk.Frame(serial_frame)
        connect_frame.pack(fill='x', pady=2)
        
        self.connect_button = ttk.Button(connect_frame, text="连接", command=self.toggle_serial_connection)
        self.connect_button.pack(side='left')
        
        self.serial_status = ttk.Label(connect_frame, text="未连接", foreground='red')
        self.serial_status.pack(side='left', padx=10)
        
        # 串口数据显示区域
        self.serial_text = tk.Text(serial_frame, height=4, width=30, font=('Courier', 8))
        serial_scroll = ttk.Scrollbar(serial_frame, orient='vertical', command=self.serial_text.yview)
        self.serial_text.configure(yscrollcommand=serial_scroll.set)
        
        self.serial_text.pack(side='left', fill='both', expand=True)
        serial_scroll.pack(side='right', fill='y')
        
        # 初始化串口列表
        self.refresh_ports()
    
    def refresh_ports(self):
        """
        刷新可用串口列表
        
        功能：
        1. 扫描系统可用串口
        2. 更新串口选择下拉框
        3. 自动选择第一个可用串口
        """
        ports = [port.device for port in serial.tools.list_ports.comports()]
        self.port_combo['values'] = ports
        if ports and not self.port_var.get():
            self.port_var.set(ports[0])
    
    def toggle_serial_connection(self):
        """
        切换串口连接状态
        
        功能：
        1. 连接或断开串口
        2. 更新界面状态显示
        3. 启动或停止数据读取线程
        """
        if not self.is_serial_connected:
            self.connect_serial()
        else:
            self.disconnect_serial()
    
    def connect_serial(self):
        """
        建立串口连接
        
        功能：
        1. 验证串口参数
        2. 建立串口连接
        3. 启动数据读取线程
        4. 更新界面状态
        """
        try:
            port = self.port_var.get()
            baud = int(self.baud_var.get())
            
            if not port:
                messagebox.showwarning("警告", "请选择串口")
                return
            
            self.serial_port = serial.Serial(port, baud, timeout=1)
            self.is_serial_connected = True
            
            # 启动串口读取线程
            self.serial_thread = threading.Thread(target=self.serial_read_loop)
            self.serial_thread.daemon = True
            self.serial_thread.start()
            
            self.connect_button.config(text="断开")
            self.serial_status.config(text="已连接", foreground='green')
            self.status_var.set(f"串口已连接: {port}")
            
        except Exception as e:
            messagebox.showerror("错误", f"串口连接失败: {str(e)}")
            self.status_var.set(f"串口连接失败: {str(e)}")
    
    def disconnect_serial(self):
        """
        断开串口连接
        
        功能：
        1. 关闭串口连接
        2. 停止数据读取线程
        3. 更新界面状态
        """
        try:
            self.is_serial_connected = False
            
            if self.serial_port and self.serial_port.is_open:
                self.serial_port.close()
            
            self.connect_button.config(text="连接")
            self.serial_status.config(text="未连接", foreground='red')
            self.status_var.set("串口已断开")
            
        except Exception as e:
            messagebox.showerror("错误", f"断开串口失败: {str(e)}")
    
    def serial_read_loop(self):
        """
        串口数据读取循环（后台线程）
        
        功能：
        1. 持续读取串口数据
        2. 将数据放入队列
        3. 触发界面更新
        """
        while self.is_serial_connected and self.serial_port and self.serial_port.is_open:
            try:
                if self.serial_port.in_waiting > 0:
                    data = self.serial_port.readline().decode('utf-8', errors='ignore').strip()
                    if data:
                        # 将数据放入队列
                        self.serial_data_queue.put(data)
                        # 在主线程中更新显示
                        self.root.after(0, self.update_serial_display)
                
                time.sleep(0.1)  # 避免过度占用CPU
                
            except Exception as e:
                print(f"串口读取错误: {e}")
                break
    
    def update_serial_display(self):
        """
        更新串口数据显示（主线程）
        
        功能：
        1. 从队列获取数据
        2. 在文本框中显示
        3. 限制显示行数
        4. 自动滚动到最新数据
        """
        try:
            while not self.serial_data_queue.empty():
                data = self.serial_data_queue.get_nowait()
                timestamp = datetime.now().strftime("%H:%M:%S")
                
                # 在串口文本框中显示数据
                self.serial_text.insert(tk.END, f"[{timestamp}] {data}\n")
                self.serial_text.see(tk.END)
                
                # 限制显示行数，避免内存占用过多
                lines = self.serial_text.get("1.0", tk.END).split('\n')
                if len(lines) > 50:
                    self.serial_text.delete("1.0", "2.0")
                
        except queue.Empty:
            pass
        except Exception as e:
            print(f"串口显示更新错误: {e}")

    def create_visualization_panel(self, parent):
        """
        创建右侧可视化面板
        
        Args:
            parent: 父容器组件
            
        功能：
        1. 创建选项卡界面
        2. 设置实时检测视图
        3. 配置趋势分析视图
        4. 添加3D可视化视图
        """
        viz_frame = ttk.Frame(parent)
        viz_frame.pack(side='right', fill='both', expand=True)
        
        # 创建选项卡控件
        self.notebook = ttk.Notebook(viz_frame)
        self.notebook.pack(fill='both', expand=True)
        
        # 实时检测选项卡
        self.create_realtime_tab()
        
        # 趋势分析选项卡
        self.create_trend_tab()
        
        # 3D可视化选项卡
        self.create_3d_tab()
    
    def create_realtime_tab(self):
        """
        创建实时检测选项卡
        
        功能：
        1. 原始图像显示
        2. 检测结果显示
        3. 热力图显示
        4. 3D可视化显示
        """
        realtime_frame = ttk.Frame(self.notebook)
        self.notebook.add(realtime_frame, text="实时检测")
        
        # 创建matplotlib图形对象
        self.fig_realtime = Figure(figsize=(14, 10), dpi=100)
        self.canvas_realtime = FigureCanvasTkAgg(self.fig_realtime, realtime_frame)
        self.canvas_realtime.get_tk_widget().pack(fill='both', expand=True)
        
        # 创建2×2子图布局
        self.ax_original = self.fig_realtime.add_subplot(221)
        self.ax_detection = self.fig_realtime.add_subplot(222)
        self.ax_contour = self.fig_realtime.add_subplot(223)
        self.ax_3d = self.fig_realtime.add_subplot(224, projection='3d')
        
        # 设置子图标题
        self.ax_original.set_title('原始图像', fontsize=10, fontweight='bold')
        self.ax_detection.set_title('结节检测', fontsize=10, fontweight='bold')
        self.ax_contour.set_title('热力图', fontsize=10, fontweight='bold')
        self.ax_3d.set_title('3D可视化', fontsize=10, fontweight='bold')
        
        self.fig_realtime.tight_layout(pad=2.0)
    
    def create_trend_tab(self):
        """
        创建趋势分析选项卡
        
        功能：
        1. 面积变化趋势
        2. 风险评分趋势
        3. 结节数量趋势
        4. 强度变化趋势
        """
        trend_frame = ttk.Frame(self.notebook)
        self.notebook.add(trend_frame, text="趋势分析")
        
        self.fig_trend = Figure(figsize=(12, 8), dpi=100)
        self.canvas_trend = FigureCanvasTkAgg(self.fig_trend, trend_frame)
        self.canvas_trend.get_tk_widget().pack(fill='both', expand=True)
        
        # 创建趋势图子图
        self.ax_area_trend = self.fig_trend.add_subplot(221)
        self.ax_risk_trend = self.fig_trend.add_subplot(222)
        self.ax_count_trend = self.fig_trend.add_subplot(223)
        self.ax_intensity_trend = self.fig_trend.add_subplot(224)
        
        self.fig_trend.tight_layout()
    
    def create_3d_tab(self):
        """
        创建3D可视化选项卡
        
        功能：
        1. 3D数据表面显示
        2. 结节区域立体显示
        3. 交互式视角控制
        """
        viz_3d_frame = ttk.Frame(self.notebook)
        self.notebook.add(viz_3d_frame, text="3D可视化")
        
        self.fig_3d = Figure(figsize=(12, 8), dpi=100)
        self.canvas_3d = FigureCanvasTkAgg(self.fig_3d, viz_3d_frame)
        self.canvas_3d.get_tk_widget().pack(fill='both', expand=True)
        
        self.ax_3d_main = self.fig_3d.add_subplot(111, projection='3d')
        self.fig_3d.tight_layout()
    
    def create_status_bar(self):
        """
        创建底部状态栏
        
        功能：
        1. 显示系统状态信息
        2. 显示处理进度
        3. 显示错误和警告信息
        """
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill='x', side='bottom')
        
        self.status_var = tk.StringVar(value="就绪")
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var, 
                                     relief='sunken', anchor='w')
        self.status_label.pack(fill='x', side='left')
        
        # 进度条
        self.progress = ttk.Progressbar(status_frame, mode='determinate')
        self.progress.pack(side='right', padx=5)
    
    def load_file(self):
        """
        加载CSV数据文件
        
        功能：
        1. 打开文件选择对话框
        2. 读取CSV数据
        3. 初始化检测系统
        4. 更新界面显示
        """
        file_path = filedialog.askopenfilename(
            title="选择CSV数据文件",
            filetypes=[("CSV文件", "*.csv"), ("所有文件", "*.*")]
        )
        
        if file_path:
            try:
                self.data = pd.read_csv(file_path)
                self.file_label.config(text=f"已加载: {os.path.basename(file_path)}")
                
                # 更新帧数范围
                max_frames = len(self.data)
                self.frame_scale.config(to=max_frames-1)
                self.frame_info.config(text=f"帧: 0/{max_frames-1}")
                
                # 重置检测系统
                self.detector = EnhancedNoduleDetectionSystem()
                self.current_frame = 0
                
                # 显示第一帧
                self.update_visualization()
                
                self.status_var.set(f"成功加载 {max_frames} 帧数据")
                
            except Exception as e:
                messagebox.showerror("错误", f"加载文件失败: {str(e)}")
                self.status_var.set("加载失败")
    
    def update_params(self, *args):
        """
        更新检测参数
        
        功能：
        1. 获取界面参数值
        2. 更新检测器参数
        3. 更新参数显示标签
        4. 重新计算当前帧
        """
        if hasattr(self, 'detector'):
            self.detector.detection_params.update({
                'gmm_components': int(self.gmm_var.get()),
                'smoothing_sigma': float(self.smooth_var.get()),
                'sensitivity_threshold': float(self.sensitivity_var.get()),
                'min_nodule_area': int(self.area_var.get())
            })
            
            # 更新标签显示
            self.gmm_label.config(text=str(int(self.gmm_var.get())))
            self.smooth_label.config(text=f"{self.smooth_var.get():.1f}")
            self.sens_label.config(text=f"{self.sensitivity_var.get():.1f}")
            self.area_label.config(text=str(int(self.area_var.get())))
            
            # 如果有数据，重新计算当前帧
            if self.data is not None:
                self.update_visualization()
    
    def update_visualization(self):
        """
        更新可视化显示
        
        功能：
        1. 获取当前帧数据
        2. 执行结节检测
        3. 更新所有视图
        4. 更新统计信息
        """
        if self.data is None:
            return
        
        try:
            # 获取当前帧的MAT数据
            mat_columns = [col for col in self.data.columns if col.startswith('MAT_')]
            if not mat_columns:
                self.status_var.set("数据格式错误：未找到MAT列")
                return
            
            current_data = self.data.iloc[self.current_frame][mat_columns].values
            stress_grid = current_data.reshape(12, 8)  # 假设12x8网格
            
            # 执行检测
            timestamp = self.data.iloc[self.current_frame].get('SN', self.current_frame)
            normalized, nodule_mask, nodules, prob_map = self.detector.advanced_nodule_detection(
                stress_grid, timestamp
            )
            
            # 更新实时显示
            self.update_realtime_plots(normalized, nodule_mask, nodules, prob_map)
            
            # 更新趋势图
            self.update_trend_plots()
            
            # 更新3D图
            self.update_3d_plot(normalized, nodule_mask)
            
            # 更新统计信息
            self.update_statistics(nodules, timestamp)
            
            # 更新帧信息
            max_frames = len(self.data)
            self.frame_info.config(text=f"帧: {self.current_frame}/{max_frames-1}")
            self.frame_var.set(self.current_frame)
            
        except Exception as e:
            self.status_var.set(f"可视化更新失败: {str(e)}")
            print(f"可视化更新错误: {e}")
    
    def update_realtime_plots(self, normalized, nodule_mask, nodules, prob_map):
        """
        更新实时检测图表
        
        Args:
            normalized: 归一化数据
            nodule_mask: 结节掩码
            nodules: 检测到的结节列表
            prob_map: 概率图
            
        功能：
        1. 显示原始数据
        2. 显示检测结果
        3. 显示热力图
        4. 显示3D视图
        """
        # 清除之前的图像
        self.ax_original.clear()
        self.ax_detection.clear()
        self.ax_contour.clear()
        self.ax_3d.clear()
        
        # 1. 原始图像
        im1 = self.ax_original.imshow(normalized, cmap='viridis', aspect='auto')
        self.ax_original.set_title('原始数据', fontsize=10, fontweight='bold')
        self.ax_original.set_xlabel('X坐标')
        self.ax_original.set_ylabel('Y坐标')
        
        # 2. 检测结果
        # 创建彩色检测结果
        detection_result = np.zeros((*normalized.shape, 3))
        detection_result[:, :, 0] = normalized  # 红色通道显示原始数据
        detection_result[:, :, 1] = normalized  # 绿色通道显示原始数据
        detection_result[:, :, 2] = normalized  # 蓝色通道显示原始数据
        
        # 在检测到的结节位置添加红色标记
        detection_result[nodule_mask == 1, 0] = 1.0  # 红色
        detection_result[nodule_mask == 1, 1] = 0.0  # 绿色
        detection_result[nodule_mask == 1, 2] = 0.0  # 蓝色
        
        self.ax_detection.imshow(detection_result, aspect='auto')
        self.ax_detection.set_title(f'检测结果 (结节数: {len(nodules)})', fontsize=10, fontweight='bold')
        self.ax_detection.set_xlabel('X坐标')
        self.ax_detection.set_ylabel('Y坐标')
        
        # 添加结节标注
        for i, nodule in enumerate(nodules):
            y, x = nodule['centroid']
            self.ax_detection.plot(x, y, 'wo', markersize=8)
            self.ax_detection.text(x, y, str(i+1), ha='center', va='center', 
                                 color='white', fontweight='bold', fontsize=8)
        
        # 3. 热力图（概率图）
        im3 = self.ax_contour.imshow(prob_map, cmap=self.risk_cmap, aspect='auto')
        self.ax_contour.set_title('风险热力图', fontsize=10, fontweight='bold')
        self.ax_contour.set_xlabel('X坐标')
        self.ax_contour.set_ylabel('Y坐标')
        
        # 添加等高线
        contours = self.ax_contour.contour(prob_map, levels=5, colors='white', alpha=0.6, linewidths=1)
        self.ax_contour.clabel(contours, inline=True, fontsize=8)
        
        # 4. 3D可视化
        x = np.arange(normalized.shape[1])
        y = np.arange(normalized.shape[0])
        X, Y = np.meshgrid(x, y)
        
        # 3D表面图
        surf = self.ax_3d.plot_surface(X, Y, normalized, cmap='viridis', alpha=0.8)
        
        # 在结节位置添加3D标记
        for nodule in nodules:
            ny, nx = nodule['centroid']
            nz = normalized[int(ny), int(nx)]
            self.ax_3d.scatter([nx], [ny], [nz], color='red', s=100, alpha=1.0)
        
        self.ax_3d.set_title('3D数据表面', fontsize=10, fontweight='bold')
        self.ax_3d.set_xlabel('X坐标')
        self.ax_3d.set_ylabel('Y坐标')
        self.ax_3d.set_zlabel('数值')
        
        # 调整布局并刷新
        self.fig_realtime.tight_layout(pad=2.0)
        self.canvas_realtime.draw()
    
    def calculate_brightness_map(self, data):
        """
        计算亮度分布图
        
        Args:
            data: 输入数据矩阵
            
        Returns:
            brightness_map: 亮度分布图
            
        功能：
        1. 计算局部亮度统计
        2. 生成亮度分布图
        3. 标识异常亮度区域
        """
        # 计算局部统计特征
        from scipy.ndimage import uniform_filter
        
        # 局部均值
        local_mean = uniform_filter(data.astype(float), size=3)
        
        # 局部标准差
        local_var = uniform_filter(data.astype(float)**2, size=3) - local_mean**2
        local_std = np.sqrt(np.maximum(local_var, 0))
        
        # 亮度特征组合
        brightness_map = local_mean + 0.5 * local_std
        
        return brightness_map
    
    def update_trend_plots(self):
        """
        更新趋势分析图表
        
        功能：
        1. 绘制面积变化趋势
        2. 绘制风险评分趋势
        3. 绘制结节数量趋势
        4. 绘制强度变化趋势
        """
        if not self.detector.nodule_history['timestamps']:
            return
        
        # 清除之前的图表
        self.ax_area_trend.clear()
        self.ax_risk_trend.clear()
        self.ax_count_trend.clear()
        self.ax_intensity_trend.clear()
        
        timestamps = self.detector.nodule_history['timestamps']
        
        # 1. 面积趋势
        if self.detector.nodule_history['areas']:
            areas = [np.mean(area_list) if area_list else 0 
                    for area_list in self.detector.nodule_history['areas']]
            self.ax_area_trend.plot(timestamps, areas, 'b-o', linewidth=2, markersize=4)
            self.ax_area_trend.set_title('平均结节面积趋势', fontweight='bold')
            self.ax_area_trend.set_ylabel('面积')
            self.ax_area_trend.grid(True, alpha=0.3)
        
        # 2. 风险评分趋势
        if self.detector.nodule_history['risk_scores']:
            risk_scores = [np.mean(score_list) if score_list else 0 
                          for score_list in self.detector.nodule_history['risk_scores']]
            self.ax_risk_trend.plot(timestamps, risk_scores, 'r-o', linewidth=2, markersize=4)
            self.ax_risk_trend.set_title('平均风险评分趋势', fontweight='bold')
            self.ax_risk_trend.set_ylabel('风险评分')
            self.ax_risk_trend.grid(True, alpha=0.3)
        
        # 3. 结节数量趋势
        counts = self.detector.nodule_history['count']
        self.ax_count_trend.plot(timestamps, counts, 'g-o', linewidth=2, markersize=4)
        self.ax_count_trend.set_title('结节数量趋势', fontweight='bold')
        self.ax_count_trend.set_ylabel('数量')
        self.ax_count_trend.grid(True, alpha=0.3)
        
        # 4. 强度趋势
        if self.detector.nodule_history['intensities']:
            intensities = [np.mean(intensity_list) if intensity_list else 0 
                          for intensity_list in self.detector.nodule_history['intensities']]
            self.ax_intensity_trend.plot(timestamps, intensities, 'm-o', linewidth=2, markersize=4)
            self.ax_intensity_trend.set_title('平均强度趋势', fontweight='bold')
            self.ax_intensity_trend.set_ylabel('强度')
            self.ax_intensity_trend.grid(True, alpha=0.3)
        
        # 设置x轴标签
        for ax in [self.ax_area_trend, self.ax_risk_trend, self.ax_count_trend, self.ax_intensity_trend]:
            ax.set_xlabel('时间/帧数')
        
        self.fig_trend.tight_layout()
        self.canvas_trend.draw()
    
    def update_3d_plot(self, normalized, nodule_mask):
        """
        更新3D可视化图表
        
        Args:
            normalized: 归一化数据
            nodule_mask: 结节掩码
            
        功能：
        1. 创建3D数据表面
        2. 标记结节位置
        3. 设置视角和颜色
        """
        self.ax_3d_main.clear()
        
        # 创建网格
        x = np.arange(normalized.shape[1])
        y = np.arange(normalized.shape[0])
        X, Y = np.meshgrid(x, y)
        
        # 3D表面图
        surf = self.ax_3d_main.plot_surface(X, Y, normalized, 
                                           cmap='viridis', alpha=0.7, 
                                           linewidth=0, antialiased=True)
        
        # 添加结节位置的3D标记
        nodule_positions = np.where(nodule_mask == 1)
        if len(nodule_positions[0]) > 0:
            nodule_x = nodule_positions[1]
            nodule_y = nodule_positions[0]
            nodule_z = normalized[nodule_positions]
            
            self.ax_3d_main.scatter(nodule_x, nodule_y, nodule_z, 
                                   color='red', s=50, alpha=1.0, 
                                   label='检测结节')
        
        # 设置标签和标题
        self.ax_3d_main.set_title('3D数据可视化', fontweight='bold')
        self.ax_3d_main.set_xlabel('X坐标')
        self.ax_3d_main.set_ylabel('Y坐标')
        self.ax_3d_main.set_zlabel('数值强度')
        
        # 设置视角
        self.ax_3d_main.view_init(elev=30, azim=45)
        
        self.fig_3d.tight_layout()
        self.canvas_3d.draw()
    
    def update_statistics(self, nodules, timestamp):
        """
        更新统计信息显示
        
        Args:
            nodules: 检测到的结节列表
            timestamp: 时间戳
            
        功能：
        1. 计算当前帧统计
        2. 更新历史统计
        3. 显示实时统计信息
        """
        # 清除统计文本
        self.stats_text.delete(1.0, tk.END)
        
        # 当前帧统计
        stats_info = []
        stats_info.append(f"=== 帧 {self.current_frame} 统计 ===")
        stats_info.append(f"时间戳: {timestamp}")
        stats_info.append(f"检测结节数: {len(nodules)}")
        
        if nodules:
            areas = [nodule['area'] for nodule in nodules]
            intensities = [nodule['mean_intensity'] for nodule in nodules]
            risk_scores = [nodule.get('risk_score', 0) for nodule in nodules]
            
            stats_info.append(f"平均面积: {np.mean(areas):.2f}")
            stats_info.append(f"最大面积: {np.max(areas):.2f}")
            stats_info.append(f"平均强度: {np.mean(intensities):.3f}")
            stats_info.append(f"平均风险: {np.mean(risk_scores):.3f}")
            
            # 风险等级统计
            high_risk = sum(1 for score in risk_scores if score > 0.7)
            medium_risk = sum(1 for score in risk_scores if 0.3 < score <= 0.7)
            low_risk = sum(1 for score in risk_scores if score <= 0.3)
            
            stats_info.append(f"高风险: {high_risk}")
            stats_info.append(f"中风险: {medium_risk}")
            stats_info.append(f"低风险: {low_risk}")
        
        # 历史统计
        if self.detector.nodule_history['timestamps']:
            total_frames = len(self.detector.nodule_history['timestamps'])
            total_detections = sum(self.detector.nodule_history['count'])
            avg_per_frame = total_detections / total_frames if total_frames > 0 else 0
            
            stats_info.append(f"\n=== 总体统计 ===")
            stats_info.append(f"已处理帧数: {total_frames}")
            stats_info.append(f"总检测数: {total_detections}")
            stats_info.append(f"平均每帧: {avg_per_frame:.2f}")
        
        # 显示统计信息
        stats_text = "\n".join(stats_info)
        self.stats_text.insert(tk.END, stats_text)
    
    def toggle_play(self):
        """
        切换播放/暂停状态
        
        功能：
        1. 启动或停止自动播放
        2. 更新播放按钮状态
        3. 管理播放线程
        """
        if self.is_playing:
            self.stop_playback()
        else:
            self.start_playback()
    
    def start_playback(self):
        """
        开始自动播放
        
        功能：
        1. 启动播放线程
        2. 更新按钮状态
        3. 设置播放标志
        """
        if self.data is None:
            return
        
        self.is_playing = True
        self.play_button.config(text="⏸")
        
        self.play_thread = threading.Thread(target=self.playback_loop)
        self.play_thread.daemon = True
        self.play_thread.start()
    
    def stop_playback(self):
        """
        停止自动播放
        
        功能：
        1. 停止播放线程
        2. 更新按钮状态
        3. 清除播放标志
        """
        self.is_playing = False
        self.play_button.config(text="▶")
    
    def playback_loop(self):
        """
        播放循环（后台线程）
        
        功能：
        1. 自动切换帧
        2. 控制播放速度
        3. 处理播放边界
        """
        while self.is_playing and self.data is not None:
            if self.current_frame < len(self.data) - 1:
                self.current_frame += 1
                self.root.after(0, self.update_visualization)
            else:
                # 到达末尾，停止播放
                self.root.after(0, self.stop_playback)
                break
            
            # 根据速度设置延时
            time.sleep(self.speed_var.get() / 1000.0)
    
    def goto_frame(self, value):
        """
        跳转到指定帧
        
        Args:
            value: 目标帧数
            
        功能：
        1. 设置当前帧
        2. 更新可视化
        3. 验证帧数范围
        """
        if self.data is None:
            return
        
        frame_num = int(float(value))
        if 0 <= frame_num < len(self.data):
            self.current_frame = frame_num
            self.update_visualization()
    
    def first_frame(self):
        """跳转到第一帧"""
        if self.data is not None:
            self.current_frame = 0
            self.update_visualization()
    
    def last_frame(self):
        """跳转到最后一帧"""
        if self.data is not None:
            self.current_frame = len(self.data) - 1
            self.update_visualization()
    
    def prev_frame(self):
        """跳转到上一帧"""
        if self.data is not None and self.current_frame > 0:
            self.current_frame -= 1
            self.update_visualization()
    
    def next_frame(self):
        """跳转到下一帧"""
        if self.data is not None and self.current_frame < len(self.data) - 1:
            self.current_frame += 1
            self.update_visualization()
    
    def export_gif(self):
        """
        导出GIF动画
        
        功能：
        1. 选择保存路径
        2. 生成动画帧
        3. 保存GIF文件
        4. 显示进度
        """
        if self.data is None:
            messagebox.showwarning("警告", "请先加载数据文件")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="保存GIF动画",
            defaultextension=".gif",
            filetypes=[("GIF文件", "*.gif"), ("所有文件", "*.*")]
        )
        
        if file_path:
            try:
                self.status_var.set("正在生成GIF动画...")
                self.progress.config(mode='indeterminate')
                self.progress.start()
                
                # 在后台线程中生成GIF
                def generate_gif():
                    success = self.detector.create_enhanced_visualization(
                        self.data, file_path, max_frames=50
                    )
                    
                    self.root.after(0, lambda: self.gif_export_complete(success, file_path))
                
                gif_thread = threading.Thread(target=generate_gif)
                gif_thread.daemon = True
                gif_thread.start()
                
            except Exception as e:
                self.progress.stop()
                messagebox.showerror("错误", f"导出GIF失败: {str(e)}")
                self.status_var.set("导出失败")
    
    def gif_export_complete(self, success, file_path):
        """
        GIF导出完成回调
        
        Args:
            success: 是否成功
            file_path: 文件路径
        """
        self.progress.stop()
        if success:
            self.status_var.set("GIF导出完成")
            messagebox.showinfo("完成", f"GIF动画已保存至:\n{file_path}")
        else:
            self.status_var.set("GIF导出失败")
            messagebox.showerror("错误", "GIF导出失败")
    
    def export_report(self):
        """
        导出分析报告
        
        功能：
        1. 生成详细分析报告
        2. 保存为文本文件
        3. 包含统计信息和图表
        """
        if self.data is None:
            messagebox.showwarning("警告", "请先加载数据文件")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="保存分析报告",
            defaultextension=".txt",
            filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")]
        )
        
        if file_path:
            try:
                report = self.detector.generate_analysis_report()
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(report)
                
                self.status_var.set("报告导出完成")
                messagebox.showinfo("完成", f"分析报告已保存至:\n{file_path}")
                
            except Exception as e:
                messagebox.showerror("错误", f"导出报告失败: {str(e)}")
                self.status_var.set("导出失败")
    
    def save_config(self):
        """
        保存当前配置
        
        功能：
        1. 保存检测参数
        2. 保存界面设置
        3. 写入配置文件
        """
        config = {
            'detection_params': self.detector.detection_params,
            'gui_params': {
                'gmm_components': self.gmm_var.get(),
                'smoothing_sigma': self.smooth_var.get(),
                'sensitivity_threshold': self.sensitivity_var.get(),
                'min_nodule_area': self.area_var.get(),
                'play_speed': self.speed_var.get()
            }
        }
        
        try:
            with open('detection_config.json', 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"保存配置失败: {e}")
    
    def load_config(self):
        """
        加载配置文件
        
        功能：
        1. 读取配置文件
        2. 恢复检测参数
        3. 恢复界面设置
        """
        try:
            if os.path.exists('detection_config.json'):
                with open('detection_config.json', 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # 恢复检测参数
                if 'detection_params' in config:
                    self.detector.detection_params.update(config['detection_params'])
                
                # 恢复GUI参数
                if 'gui_params' in config:
                    gui_params = config['gui_params']
                    self.gmm_var.set(gui_params.get('gmm_components', 3))
                    self.smooth_var.set(gui_params.get('smoothing_sigma', 0.8))
                    self.sensitivity_var.set(gui_params.get('sensitivity_threshold', 0.7))
                    self.area_var.set(gui_params.get('min_nodule_area', 3))
                    self.speed_var.set(gui_params.get('play_speed', 500))
                    
        except Exception as e:
            print(f"加载配置失败: {e}")
    
    def on_closing(self):
        """
        程序关闭时的清理工作
        
        功能：
        1. 停止播放线程
        2. 断开串口连接
        3. 保存配置
        4. 清理资源
        """
        # 停止播放
        self.is_playing = False
        
        # 断开串口
        if self.is_serial_connected:
            self.disconnect_serial()
        
        # 保存配置
        self.save_config()
        
        # 关闭窗口
        self.root.destroy()

# 程序入口点
if __name__ == '__main__':
    root = tk.Tk()
    app = ModernDetectionGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()