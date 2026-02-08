# -*- coding: utf-8 -*-
"""
现代化结节检测GUI - 集成版本
整合了 fusion_real_time_detection.py 的颜色映射、图像处理方法和实时串口数据读取功能
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from datetime import datetime
import json
import os
import threading
import time
import queue
import serial
import serial.tools.list_ports
from collections import deque
from enhanced_detection_system import EnhancedNoduleDetectionSystem

# 高性能协议解析器 - 从 fusion_real_time_detection.py 集成
class FastProtocolParser:
    """高性能协议解析器 - 精简版，与 optimized_serial_monitor 中保持一致"""

    def __init__(self):
        self.buffer = bytearray()
        self.frame_header = b"\xA5\x5A"
        self.expected_frame_size = 104
        self.latest_frame = None
        self.last_parse_time = 0
        self.parse_interval = 0.005  # 5 ms

    def add_data(self, data: bytes):
        self.buffer.extend(data)
        # 限制缓冲区大小，防止内存积压
        if len(self.buffer) > 1024:
            self.buffer = self.buffer[-512:]

        # 节流解析
        now = time.time()
        if now - self.last_parse_time < self.parse_interval:
            return
        self.last_parse_time = now
        self._parse()

    def _parse(self):
        while len(self.buffer) >= self.expected_frame_size:
            # 查找帧头
            idx = self.buffer.find(self.frame_header)
            if idx == -1:
                # 未找到帧头，保留最后 10 字节
                self.buffer = self.buffer[-10:]
                break
            if idx > 0:
                self.buffer = self.buffer[idx:]
            if len(self.buffer) < self.expected_frame_size:
                break
            frame_data = self.buffer[: self.expected_frame_size]
            matrix = np.frombuffer(frame_data[6:102], dtype=np.uint8).reshape(12, 8)
            self.latest_frame = {
                "timestamp": time.time(),
                "matrix": matrix,
            }
            # 丢弃已处理数据
            self.buffer = self.buffer[self.expected_frame_size :]

    def get_latest(self):
        return self.latest_frame

class ModernDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("现代化结节检测系统 - 集成版本")
        self.root.geometry("1400x900")
        
        # 数据相关
        self.data = []
        self.current_frame = 0
        self.detector = EnhancedNoduleDetectionSystem()
        
        # 串口相关 - 集成 fusion_real_time_detection.py 的功能
        self.serial_port = None
        self.is_serial_connected = False
        self.serial_thread = None
        self.data_queue = queue.Queue(maxsize=100)
        self.parser = FastProtocolParser()
        self.is_realtime_processing = False
        self.plot_interval = 0.03  # 约 33 fps
        
        # 颜色映射设置 - 集成 fusion_real_time_detection.py 的颜色方案
        self.setup_color_schemes()
        
        # 统计数据
        self.nodule_history = deque(maxlen=1000)
        self.risk_history = deque(maxlen=1000)
        self.time_history = deque(maxlen=1000)
        
        # 创建GUI
        self.create_gui()
        
    def setup_color_schemes(self):
        """设置颜色方案 - 集成 fusion_real_time_detection.py 的颜色映射"""
        # 医学专用颜色
        medical_colors = ['#000080', '#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF8000', '#FF0000']
        
        # 创建自定义颜色映射
        self.nodule_cmap = LinearSegmentedColormap.from_list('nodule', medical_colors)
        self.risk_cmap = LinearSegmentedColormap.from_list('risk', ['green', 'yellow', 'red'])
        self.brightness_cmap = LinearSegmentedColormap.from_list('brightness', ['black', 'blue', 'cyan', 'yellow', 'red'])
        
        # 集成 fusion_real_time_detection.py 的 turbo 颜色映射
        try:
            self.turbo_cmap = plt.get_cmap("turbo")
        except Exception:
            self.turbo_cmap = plt.get_cmap("viridis")
    
    def create_gui(self):
        """创建主界面"""
        # 主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # 左侧控制面板
        self.create_control_panel(main_frame)
        
        # 右侧可视化面板
        self.create_visualization_panel(main_frame)
    
    def create_control_panel(self, parent):
        """创建左侧控制面板"""
        control_frame = ttk.Frame(parent, width=300)
        control_frame.pack(side='left', fill='y', padx=(0, 10))
        control_frame.pack_propagate(False)
        
        # 数据加载区域
        data_frame = ttk.LabelFrame(control_frame, text="数据加载", padding=10)
        data_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Button(data_frame, text="加载CSV文件", command=self.load_csv_file).pack(fill='x', pady=2)
        
        # 串口控制区域 - 集成 fusion_real_time_detection.py 的串口功能
        serial_frame = ttk.LabelFrame(control_frame, text="串口控制", padding=10)
        serial_frame.pack(fill='x', pady=(0, 10))
        
        # 串口选择
        ttk.Label(serial_frame, text="串口:").pack(anchor='w')
        self.port_var = tk.StringVar()
        self.port_combo = ttk.Combobox(serial_frame, textvariable=self.port_var, width=25)
        self.port_combo.pack(fill='x', pady=2)
        
        # 波特率选择
        ttk.Label(serial_frame, text="波特率:").pack(anchor='w', pady=(5, 0))
        self.baud_var = tk.StringVar(value="115200")
        self.baud_combo = ttk.Combobox(serial_frame, textvariable=self.baud_var, 
                                     values=["9600", "19200", "38400", "57600", "115200"], width=25)
        self.baud_combo.pack(fill='x', pady=2)
        
        # 串口连接按钮
        self.serial_btn = ttk.Button(serial_frame, text="连接串口", command=self.toggle_serial_connection)
        self.serial_btn.pack(fill='x', pady=5)
        
        # 刷新串口按钮
        ttk.Button(serial_frame, text="刷新串口", command=self.refresh_ports).pack(fill='x', pady=2)
        
        # 帧控制区域
        frame_control = ttk.LabelFrame(control_frame, text="帧控制", padding=10)
        frame_control.pack(fill='x', pady=(0, 10))
        
        # 帧滑块
        self.frame_var = tk.IntVar()
        self.frame_scale = ttk.Scale(frame_control, from_=0, to=100, orient='horizontal', 
                                   variable=self.frame_var, command=self.on_frame_change)
        self.frame_scale.pack(fill='x', pady=5)
        
        # 帧信息
        self.frame_info = ttk.Label(frame_control, text="帧: 0/0")
        self.frame_info.pack()
        
        # 播放控制
        play_frame = ttk.Frame(frame_control)
        play_frame.pack(fill='x', pady=5)
        
        ttk.Button(play_frame, text="◀◀", command=self.first_frame, width=6).pack(side='left', padx=2)
        ttk.Button(play_frame, text="◀", command=self.prev_frame, width=6).pack(side='left', padx=2)
        ttk.Button(play_frame, text="▶", command=self.next_frame, width=6).pack(side='left', padx=2)
        ttk.Button(play_frame, text="▶▶", command=self.last_frame, width=6).pack(side='left', padx=2)
        
        # 检测参数
        param_frame = ttk.LabelFrame(control_frame, text="检测参数", padding=10)
        param_frame.pack(fill='x', pady=(0, 10))
        
        # 阈值设置
        ttk.Label(param_frame, text="检测阈值:").pack(anchor='w')
        self.threshold_var = tk.DoubleVar(value=0.3)
        ttk.Scale(param_frame, from_=0.1, to=0.9, orient='horizontal', 
                variable=self.threshold_var, command=self.update_detection).pack(fill='x')
        
        # 状态显示
        status_frame = ttk.LabelFrame(control_frame, text="状态信息", padding=10)
        status_frame.pack(fill='x', pady=(0, 10))
        
        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(status_frame, textvariable=self.status_var, wraplength=250).pack(anchor='w')
        
        # 串口数据显示
        self.serial_text = tk.Text(status_frame, height=8, width=30, font=('Consolas', 8))
        self.serial_text.pack(fill='both', expand=True, pady=5)
        
        scrollbar = ttk.Scrollbar(status_frame, orient="vertical", command=self.serial_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.serial_text.configure(yscrollcommand=scrollbar.set)
    
    def create_visualization_panel(self, parent):
        """创建右侧可视化面板"""
        viz_frame = ttk.Frame(parent)
        viz_frame.pack(side='right', fill='both', expand=True)
        
        # 创建选项卡
        self.notebook = ttk.Notebook(viz_frame)
        self.notebook.pack(fill='both', expand=True)
        
        # 实时检测选项卡
        self.create_realtime_tab()
        
        # 趋势分析选项卡
        self.create_trend_tab()
        
        # 3D可视化选项卡
        self.create_3d_tab()
    
    def create_realtime_tab(self):
        """创建实时检测选项卡"""
        realtime_frame = ttk.Frame(self.notebook)
        self.notebook.add(realtime_frame, text="实时检测")
        
        # 创建matplotlib图形
        self.fig = Figure(figsize=(12, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=realtime_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # 创建子图
        self.ax_original = self.fig.add_subplot(221)
        self.ax_detection = self.fig.add_subplot(222)
        self.ax_contour = self.fig.add_subplot(223)
        self.ax_3d = self.fig.add_subplot(224, projection='3d')
        
        self.fig.tight_layout()
        
        # 初始化空图
        self.init_empty_plots()
    
    def create_trend_tab(self):
        """创建趋势分析选项卡"""
        trend_frame = ttk.Frame(self.notebook)
        self.notebook.add(trend_frame, text="趋势分析")
        
        # 趋势图
        self.trend_fig = Figure(figsize=(12, 6), dpi=100)
        self.trend_canvas = FigureCanvasTkAgg(self.trend_fig, master=trend_frame)
        self.trend_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        self.ax_trend1 = self.trend_fig.add_subplot(211)
        self.ax_trend2 = self.trend_fig.add_subplot(212)
        
        self.trend_fig.tight_layout()
    
    def create_3d_tab(self):
        """创建3D可视化选项卡"""
        viz_3d_frame = ttk.Frame(self.notebook)
        self.notebook.add(viz_3d_frame, text="3D可视化")
        
        # 3D图
        self.viz_3d_fig = Figure(figsize=(10, 8), dpi=100)
        self.viz_3d_canvas = FigureCanvasTkAgg(self.viz_3d_fig, master=viz_3d_frame)
        self.viz_3d_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        self.ax_3d_viz = self.viz_3d_fig.add_subplot(111, projection='3d')
    
    def init_empty_plots(self):
        """初始化空图表"""
        empty_data = np.zeros((12, 8))
        
        self.ax_original.imshow(empty_data, cmap=self.turbo_cmap)
        self.ax_original.set_title('原始数据')
        
        self.ax_detection.imshow(empty_data, cmap=self.turbo_cmap)
        self.ax_detection.set_title('检测结果')
        
        self.ax_contour.contourf(empty_data, cmap=self.turbo_cmap)
        self.ax_contour.set_title('热力图')
        
        x = np.arange(8)
        y = np.arange(12)
        X, Y = np.meshgrid(x, y)
        self.ax_3d.plot_surface(X, Y, empty_data, cmap=self.turbo_cmap)
        self.ax_3d.set_title('3D可视化')
        
        self.canvas.draw()
    
    # 串口相关方法 - 集成 fusion_real_time_detection.py 的功能
    def refresh_ports(self):
        """刷新可用串口列表"""
        ports = [port.device for port in serial.tools.list_ports.comports()]
        self.port_combo['values'] = ports
        if ports and not self.port_var.get():
            self.port_var.set(ports[0])
    
    def toggle_serial_connection(self):
        """切换串口连接状态"""
        if self.is_serial_connected:
            self.disconnect_serial()
        else:
            self.connect_serial()
    
    def connect_serial(self):
        """连接串口 - 集成 fusion_real_time_detection.py 的低延迟配置"""
        port = self.port_var.get()
        if not port:
            messagebox.showerror("错误", "请选择串口")
            return
        
        try:
            baud = int(self.baud_var.get())
            self.serial_port = serial.Serial(
                port=port,
                baudrate=baud,
                bytesize=8,
                stopbits=1,
                parity=serial.PARITY_NONE,
                timeout=0.001,  # 低延迟设置
                write_timeout=0.001,
            )
            
            self.is_serial_connected = True
            self.serial_btn.config(text="断开串口")
            self.status_var.set(f"已连接到 {port}@{baud}")
            
            # 启动串口读取线程
            self.serial_thread = threading.Thread(target=self.serial_read_loop, daemon=True)
            self.serial_thread.start()
            
            # 启动实时处理
            self.start_realtime_processing()
            
        except Exception as e:
            messagebox.showerror("连接失败", f"无法连接到串口: {str(e)}")
    
    def disconnect_serial(self):
        """断开串口连接"""
        self.is_serial_connected = False
        self.is_realtime_processing = False
        
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
        
        self.serial_btn.config(text="连接串口")
        self.status_var.set("已断开连接")
    
    def serial_read_loop(self):
        """串口读取循环 - 集成 fusion_real_time_detection.py 的高性能读取"""
        while self.is_serial_connected and self.serial_port and self.serial_port.is_open:
            try:
                # 高性能二进制数据读取
                avail = self.serial_port.in_waiting
                if avail:
                    data = self.serial_port.read(min(avail, 2048))
                    
                    # 添加到解析器
                    self.parser.add_data(data)
                    
                    # 同时保留原有的文本显示功能
                    try:
                        text_data = data.decode('utf-8', errors='ignore')
                        if text_data.strip():
                            self.root.after(0, lambda: self.update_serial_display(text_data))
                    except:
                        pass
                    
                    # 将数据放入队列供实时处理使用
                    try:
                        self.data_queue.put_nowait(data)
                    except queue.Full:
                        # 丢弃最旧数据，保证低延迟
                        try:
                            self.data_queue.get_nowait()
                            self.data_queue.put_nowait(data)
                        except queue.Empty:
                            pass
                
                time.sleep(0.001)  # 低延迟睡眠
                
            except Exception as e:
                print(f"串口读取错误: {e}")
                break
    
    def start_realtime_processing(self):
        """启动实时数据处理循环 - 集成 fusion_real_time_detection.py 的实时处理功能"""
        self.is_realtime_processing = True
        self.root.after(int(self.plot_interval * 1000), self.process_realtime_data)
    
    def process_realtime_data(self):
        """处理实时数据 - 集成 fusion_real_time_detection.py 的数据处理逻辑"""
        if not self.is_realtime_processing or not self.is_serial_connected:
            return
        
        try:
            # 获取最新解析的帧数据
            latest_frame = self.parser.get_latest()
            
            if latest_frame is not None:
                # 获取矩阵数据
                matrix = latest_frame['matrix']
                timestamp = latest_frame['timestamp']
                
                # 执行结节检测
                normalized, nodule_mask, nodules, prob_map = self.detector.advanced_nodule_detection(
                    matrix, timestamp
                )
                
                # 更新实时可视化
                self.update_realtime_plots(normalized, nodule_mask, nodules, prob_map)
                
                # 更新统计信息
                self.update_statistics(nodules, timestamp)
                
                # 更新趋势图
                self.update_trend_plots()
                
                # 更新3D图
                self.update_3d_plot(normalized, nodule_mask)
                
                # 更新状态
                self.status_var.set(f"实时处理中 - 检出结节: {len(nodules)}个")
                
        except Exception as e:
            print(f"实时数据处理错误: {e}")
            self.status_var.set(f"实时处理错误: {str(e)}")
        
        # 继续处理循环
        if self.is_realtime_processing:
            self.root.after(int(self.plot_interval * 1000), self.process_realtime_data)
    
    def update_serial_display(self, data):
        """更新串口数据显示"""
        self.serial_text.insert(tk.END, data)
        self.serial_text.see(tk.END)
        
        # 限制文本长度
        if self.serial_text.index(tk.END).split('.')[0] > '1000':
            self.serial_text.delete('1.0', '500.0')
    
    def update_statistics(self, nodules, timestamp):
        """更新统计信息显示"""
        try:
            # 更新结节计数
            total_nodules = len(nodules)
            high_risk = len([n for n in nodules if n['risk_score'] > 0.8])
            medium_risk = len([n for n in nodules if 0.6 < n['risk_score'] <= 0.8])
            low_risk = len([n for n in nodules if n['risk_score'] <= 0.6])
            
            # 更新统计标签
            stats_text = f"""实时统计信息:
总结节数: {total_nodules}
高风险: {high_risk}
中风险: {medium_risk}
低风险: {low_risk}
更新时间: {timestamp:.2f}s"""
            
            # 如果有统计标签，更新它
            if hasattr(self, 'stats_label'):
                self.stats_label.config(text=stats_text)
            
        except Exception as e:
            print(f"统计信息更新错误: {e}")
    
    def update_realtime_plots(self, normalized, nodule_mask, nodules, prob_map):
        """更新实时可视化图表 - 集成 fusion_real_time_detection.py 的可视化功能"""
        try:
            # 清除现有图表
            self.ax_original.clear()
            self.ax_detection.clear()
            self.ax_contour.clear()
            
            # 显示原始数据（归一化后）
            im1 = self.ax_original.imshow(normalized, cmap=self.turbo_cmap, aspect='auto')
            self.ax_original.set_title('实时原始数据', fontsize=10, fontweight='bold')
            self.ax_original.set_xlabel('列索引')
            self.ax_original.set_ylabel('行索引')
            
            # 显示检测结果
            im2 = self.ax_detection.imshow(prob_map, cmap=self.turbo_cmap, aspect='auto')
            self.ax_detection.set_title('实时结节检测', fontsize=10, fontweight='bold')
            self.ax_detection.set_xlabel('列索引')
            self.ax_detection.set_ylabel('行索引')
            
            # 显示热力图（等高线图）
            contour = self.ax_contour.contourf(normalized, levels=20, cmap=self.turbo_cmap)
            self.ax_contour.set_title('实时热力图', fontsize=10, fontweight='bold')
            self.ax_contour.set_xlabel('列索引')
            self.ax_contour.set_ylabel('行索引')
            
            # 在检测图上标记结节
            for nodule in nodules:
                x, y = nodule['position']
                risk = nodule['risk_score']
                
                # 根据风险等级选择颜色和大小
                if risk > 0.8:
                    color = 'red'
                    size = 100
                    marker = 'X'
                elif risk > 0.6:
                    color = 'orange'
                    size = 80
                    marker = 's'
                else:
                    color = 'yellow'
                    size = 60
                    marker = 'o'
                
                self.ax_detection.scatter(y, x, c=color, s=size, marker=marker, 
                                       edgecolors='black', linewidth=1, alpha=0.8)
                
                # 添加风险评分标签
                self.ax_detection.annotate(f'{risk:.2f}', (y, x), 
                                        xytext=(5, 5), textcoords='offset points',
                                        fontsize=8, color='white', fontweight='bold',
                                        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
            
            # 更新3D可视化
            self.ax_3d.clear()
            x = np.arange(normalized.shape[1])
            y = np.arange(normalized.shape[0])
            X, Y = np.meshgrid(x, y)
            surf = self.ax_3d.plot_surface(X, Y, normalized, cmap=self.turbo_cmap, alpha=0.8)
            self.ax_3d.set_title('3D实时可视化', fontsize=10, fontweight='bold')
            self.ax_3d.set_xlabel('X')
            self.ax_3d.set_ylabel('Y')
            self.ax_3d.set_zlabel('强度')
            
            # 更新颜色条
            if hasattr(self, 'cbar_realtime1'):
                self.cbar_realtime1.remove()
            if hasattr(self, 'cbar_realtime2'):
                self.cbar_realtime2.remove()
                
            self.cbar_realtime1 = self.fig.colorbar(im1, ax=self.ax_original, shrink=0.8)
            self.cbar_realtime1.set_label('强度值', rotation=270, labelpad=15)
            
            self.cbar_realtime2 = self.fig.colorbar(im2, ax=self.ax_detection, shrink=0.8)
            self.cbar_realtime2.set_label('检测概率', rotation=270, labelpad=15)
            
            # 刷新画布
            self.canvas.draw()
            
        except Exception as e:
            print(f"实时可视化更新错误: {e}")
    
    # 其他方法保持不变...
    def load_csv_file(self):
        """加载CSV文件"""
        file_path = filedialog.askopenfilename(
            title="选择CSV文件",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.data = pd.read_csv(file_path).values
                self.current_frame = 0
                self.frame_scale.config(to=len(self.data)-1)
                self.frame_var.set(0)
                self.update_display()
                self.status_var.set(f"已加载 {len(self.data)} 帧数据")
            except Exception as e:
                messagebox.showerror("错误", f"加载文件失败: {str(e)}")
    
    def on_frame_change(self, value):
        """帧变化回调"""
        self.current_frame = int(float(value))
        self.update_display()
    
    def update_display(self):
        """更新显示"""
        if not self.data:
            return
        
        try:
            frame_data = self.data[self.current_frame]
            matrix = frame_data.reshape(12, 8)
            
            # 执行检测
            normalized, nodule_mask, nodules, prob_map = self.detector.advanced_nodule_detection(
                matrix, self.current_frame
            )
            
            # 更新可视化
            self.update_realtime_plots(normalized, nodule_mask, nodules, prob_map)
            
            # 更新帧信息
            max_frames = len(self.data)
            self.frame_info.config(text=f"帧: {self.current_frame}/{max_frames-1}")
            
        except Exception as e:
            print(f"显示更新错误: {e}")
    
    def first_frame(self):
        """跳转到第一帧"""
        self.frame_var.set(0)
    
    def prev_frame(self):
        """上一帧"""
        if self.current_frame > 0:
            self.frame_var.set(self.current_frame - 1)
    
    def next_frame(self):
        """下一帧"""
        if self.current_frame < len(self.data) - 1:
            self.frame_var.set(self.current_frame + 1)
    
    def last_frame(self):
        """跳转到最后一帧"""
        if self.data:
            self.frame_var.set(len(self.data) - 1)
    
    def update_detection(self, value):
        """更新检测参数"""
        self.update_display()
    
    def update_trend_plots(self):
        """更新趋势图"""
        # 实现趋势图更新逻辑
        pass
    
    def update_3d_plot(self, normalized, nodule_mask):
        """更新3D图"""
        # 实现3D图更新逻辑
        pass

if __name__ == "__main__":
    root = tk.Tk()
    app = ModernDetectionGUI(root)
    root.mainloop()