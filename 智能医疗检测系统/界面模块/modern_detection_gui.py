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
from collections import deque  # 新增

class FastProtocolParser:
    """高性能协议解析器 - 精简版，与 fusion_real_time_detection 中保持一致"""

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
        self.root.title("现代化结节检测系统")
        self.root.geometry("1400x900")
        
        # 数据相关
        self.data = None
        self.current_frame = 0
        self.is_playing = False
        self.playback_thread = None
        
        # 串口相关
        self.serial_port = None
        self.is_serial_connected = False
        self.serial_thread = None
        self.serial_data_queue = queue.Queue()
        
        # 实时数据处理相关 - 集成 fusion_real_time_detection.py 的功能
        self.data_q = queue.Queue(maxsize=100)  # 限制队列大小以保证低延迟
        self.last_plot_update = 0
        self.plot_interval = 1.0 / 30.0  # 30 FPS
        self.is_realtime_processing = False
        
        # 解析器和检测系统
        self.parser = FastProtocolParser()
        self.detector = EnhancedNoduleDetectionSystem()
        
        # 设置样式和颜色方案
        self.setup_color_schemes()
        self.setup_styles()
        
        # 创建界面
        self.create_widgets()
        
        # 加载配置
        self.load_config()
    
    def setup_color_schemes(self):
        """设置美化的颜色方案"""
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
        
        # 集成 fusion_real_time_detection.py 的颜色映射方案
        # 使用好看的 colormap，优先使用 turbo，不可用时退化为 viridis
        try:
            self.medical_cmap = plt.get_cmap("turbo")
        except Exception:
            self.medical_cmap = plt.get_cmap("viridis")
        
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
        
        # 注册颜色映射
        plt.register_cmap(cmap=self.nodule_cmap)
        plt.register_cmap(cmap=self.risk_cmap)
        plt.register_cmap(cmap=self.brightness_cmap)
    
    def setup_styles(self):
        """设置界面样式"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # 自定义样式
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'), background='#f0f0f0')
        style.configure('Header.TLabel', font=('Arial', 12, 'bold'), background='#f0f0f0')
        style.configure('Info.TLabel', font=('Arial', 10), background='#f0f0f0')
        style.configure('Success.TButton', background='#4CAF50')
        style.configure('Warning.TButton', background='#FF9800')
        style.configure('Error.TButton', background='#F44336')
    
    def create_widgets(self):
        """创建界面组件"""
        # 主标题
        title_frame = ttk.Frame(self.root)
        title_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(title_frame, text="动态逐帧检测系统", style='Title.TLabel').pack()
        ttk.Label(title_frame, text="Enhanced Nodule Detection System", style='Info.TLabel').pack()
        
        # 创建主要布局
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # 左侧控制面板
        self.create_control_panel(main_frame)
        
        # 右侧可视化面板
        self.create_visualization_panel(main_frame)
        
        # 底部状态栏
        self.create_status_bar()
    
    def create_control_panel(self, parent):
        """创建左侧控制面板"""
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
        
        # 检测参数设置
        param_frame = ttk.LabelFrame(control_frame, text="检测参数", padding=10)
        param_frame.pack(fill='x', pady=5)
        
        # GMM组件数
        ttk.Label(param_frame, text="GMM组件数:").pack(anchor='w')
        self.gmm_var = tk.IntVar(value=3)
        gmm_scale = ttk.Scale(param_frame, from_=2, to=5, variable=self.gmm_var, 
                             orient='horizontal', command=self.update_params)
        gmm_scale.pack(fill='x')
        self.gmm_label = ttk.Label(param_frame, text="3")
        self.gmm_label.pack(anchor='w')
        
        # 平滑参数
        ttk.Label(param_frame, text="平滑参数:").pack(anchor='w')
        self.smooth_var = tk.DoubleVar(value=0.8)
        smooth_scale = ttk.Scale(param_frame, from_=0.1, to=2.0, variable=self.smooth_var,
                                orient='horizontal', command=self.update_params)
        smooth_scale.pack(fill='x')
        self.smooth_label = ttk.Label(param_frame, text="0.8")
        self.smooth_label.pack(anchor='w')
        
        # 敏感度阈值
        ttk.Label(param_frame, text="敏感度阈值:").pack(anchor='w')
        self.sensitivity_var = tk.DoubleVar(value=0.7)
        sens_scale = ttk.Scale(param_frame, from_=0.1, to=1.0, variable=self.sensitivity_var,
                              orient='horizontal', command=self.update_params)
        sens_scale.pack(fill='x')
        self.sens_label = ttk.Label(param_frame, text="0.7")
        self.sens_label.pack(anchor='w')
        
        # 最小结节面积
        ttk.Label(param_frame, text="最小结节面积:").pack(anchor='w')
        self.area_var = tk.IntVar(value=3)
        area_scale = ttk.Scale(param_frame, from_=1, to=10, variable=self.area_var,
                              orient='horizontal', command=self.update_params)
        area_scale.pack(fill='x')
        self.area_label = ttk.Label(param_frame, text="3")
        self.area_label.pack(anchor='w')
        
        # 播放控制
        play_frame = ttk.LabelFrame(control_frame, text="播放控制", padding=10)
        play_frame.pack(fill='x', pady=5)
        
        # 帧控制
        frame_control = ttk.Frame(play_frame)
        frame_control.pack(fill='x')
        
        ttk.Button(frame_control, text="◀◀", command=self.first_frame).pack(side='left')
        ttk.Button(frame_control, text="◀", command=self.prev_frame).pack(side='left')
        self.play_button = ttk.Button(frame_control, text="▶", command=self.toggle_play)
        self.play_button.pack(side='left')
        ttk.Button(frame_control, text="▶", command=self.next_frame).pack(side='left')
        ttk.Button(frame_control, text="▶▶", command=self.last_frame).pack(side='left')
        
        # 帧数显示和滑块
        self.frame_var = tk.IntVar(value=0)
        self.frame_scale = ttk.Scale(play_frame, from_=0, to=100, variable=self.frame_var,
                                    orient='horizontal', command=self.goto_frame)
        self.frame_scale.pack(fill='x', pady=5)
        
        self.frame_info = ttk.Label(play_frame, text="帧: 0/0", style='Info.TLabel')
        self.frame_info.pack()
        
        # 播放速度
        ttk.Label(play_frame, text="播放速度(ms):").pack(anchor='w')
        self.speed_var = tk.IntVar(value=500)
        speed_scale = ttk.Scale(play_frame, from_=100, to=2000, variable=self.speed_var,
                               orient='horizontal')
        speed_scale.pack(fill='x')
        
        # 统计信息
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
        """创建串口控制面板"""
        serial_frame = ttk.LabelFrame(parent, text="串口数据", padding=10)
        serial_frame.pack(fill='x', pady=5)
        
        # 串口选择
        port_frame = ttk.Frame(serial_frame)
        port_frame.pack(fill='x', pady=2)
        
        ttk.Label(port_frame, text="串口:").pack(side='left')
        self.port_var = tk.StringVar()
        self.port_combo = ttk.Combobox(port_frame, textvariable=self.port_var, width=15)
        self.port_combo.pack(side='left', padx=5)
        
        ttk.Button(port_frame, text="刷新", command=self.refresh_ports).pack(side='left', padx=2)
        
        # 波特率选择
        baud_frame = ttk.Frame(serial_frame)
        baud_frame.pack(fill='x', pady=2)
        
        ttk.Label(baud_frame, text="波特率:").pack(side='left')
        self.baud_var = tk.StringVar(value="9600")
        baud_combo = ttk.Combobox(baud_frame, textvariable=self.baud_var, 
                                 values=["9600", "19200", "38400", "57600", "115200"], width=10)
        baud_combo.pack(side='left', padx=5)
        
        # 连接控制
        connect_frame = ttk.Frame(serial_frame)
        connect_frame.pack(fill='x', pady=2)
        
        self.connect_button = ttk.Button(connect_frame, text="连接", command=self.toggle_serial_connection)
        self.connect_button.pack(side='left')
        
        self.serial_status = ttk.Label(connect_frame, text="未连接", foreground='red')
        self.serial_status.pack(side='left', padx=10)
        
        # 串口数据显示
        self.serial_text = tk.Text(serial_frame, height=4, width=30, font=('Courier', 8))
        serial_scroll = ttk.Scrollbar(serial_frame, orient='vertical', command=self.serial_text.yview)
        self.serial_text.configure(yscrollcommand=serial_scroll.set)
        
        self.serial_text.pack(side='left', fill='both', expand=True)
        serial_scroll.pack(side='right', fill='y')
        
        # 初始化串口列表
        self.refresh_ports()
    
    def refresh_ports(self):
        """刷新可用串口列表"""
        ports = [port.device for port in serial.tools.list_ports.comports()]
        self.port_combo['values'] = ports
        if ports and not self.port_var.get():
            self.port_var.set(ports[0])
    
    def toggle_serial_connection(self):
        """切换串口连接状态"""
        if not self.is_serial_connected:
            self.connect_serial()
        else:
            self.disconnect_serial()
    
    def connect_serial(self):
        """连接串口"""
        try:
            port = self.port_var.get()
            baud = int(self.baud_var.get())
            
            if not port:
                messagebox.showwarning("警告", "请选择串口")
                return
            
            # 使用与 fusion_real_time_detection.py 相同的串口配置
            self.serial_port = serial.Serial(
                port=port,
                baudrate=baud,
                bytesize=8,
                stopbits=1,
                parity=serial.PARITY_NONE,
                timeout=0.001,  # 低延迟配置
                write_timeout=0.001,
            )
            self.is_serial_connected = True
            
            # 启动串口读取线程
            self.serial_thread = threading.Thread(target=self.serial_read_loop)
            self.serial_thread.daemon = True
            self.serial_thread.start()
            
            # 启动实时数据处理循环
            self.start_realtime_processing()
            
            self.connect_button.config(text="断开")
            self.serial_status.config(text="已连接", foreground='green')
            self.status_var.set(f"串口已连接: {port}")
            
        except Exception as e:
            messagebox.showerror("错误", f"串口连接失败: {str(e)}")
            self.status_var.set(f"串口连接失败: {str(e)}")
    
    def disconnect_serial(self):
        """断开串口连接"""
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
        """串口数据读取循环 - 集成 fusion_real_time_detection.py 的高性能读取方法"""
        while self.is_serial_connected and self.serial_port and self.serial_port.is_open:
            try:
                # 使用高性能读取方法
                avail = self.serial_port.in_waiting
                if avail:
                    data = self.serial_port.read(min(avail, 2048))
                    # 将二进制数据传递给解析器
                    self.parser.add_data(data)
                    
                    # 同时保持原有的文本显示功能
                    try:
                        text_data = data.decode('utf-8', errors='ignore').strip()
                        if text_data:
                            self.serial_data_queue.put(text_data)
                            self.root.after(0, self.update_serial_display)
                    except:
                        pass  # 忽略解码错误
                
                time.sleep(0.001)  # 低延迟配置
                
            except Exception as e:
                print(f"串口读取错误: {e}")
                break
    
    def update_serial_display(self):
        """更新串口数据显示"""
        try:
            while not self.serial_data_queue.empty():
                data = self.serial_data_queue.get_nowait()
                timestamp = datetime.now().strftime("%H:%M:%S")
                
                # 在串口文本框中显示数据
                self.serial_text.insert(tk.END, f"[{timestamp}] {data}\n")
                self.serial_text.see(tk.END)
                
                # 限制显示行数
                lines = self.serial_text.get("1.0", tk.END).split('\n')
                if len(lines) > 50:
                    self.serial_text.delete("1.0", "2.0")
                
        except queue.Empty:
            pass
        except Exception as e:
            print(f"串口显示更新错误: {e}")
    
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
        self.fig_realtime = Figure(figsize=(14, 10), dpi=100)
        self.canvas_realtime = FigureCanvasTkAgg(self.fig_realtime, realtime_frame)
        self.canvas_realtime.get_tk_widget().pack(fill='both', expand=True)
        
        # 创建子图 - 使用2×2布局（去除亮度分布和直方图）
        self.ax_original = self.fig_realtime.add_subplot(221)
        self.ax_detection = self.fig_realtime.add_subplot(222)
        self.ax_contour = self.fig_realtime.add_subplot(223)  # 热力图位置
        self.ax_3d = self.fig_realtime.add_subplot(224, projection='3d')
        
        # 设置子图标题
        self.ax_original.set_title('实时图像', fontsize=10, fontweight='bold')
        self.ax_detection.set_title('结节检测', fontsize=10, fontweight='bold')
        self.ax_contour.set_title('热力图', fontsize=10, fontweight='bold')
        self.ax_3d.set_title('3D可视化', fontsize=10, fontweight='bold')
        
        self.fig_realtime.tight_layout(pad=2.0)
    
    def create_trend_tab(self):
        """创建趋势分析选项卡"""
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
        """创建3D可视化选项卡"""
        viz_3d_frame = ttk.Frame(self.notebook)
        self.notebook.add(viz_3d_frame, text="3D可视化")
        
        self.fig_3d = Figure(figsize=(12, 8), dpi=100)
        self.canvas_3d = FigureCanvasTkAgg(self.fig_3d, viz_3d_frame)
        self.canvas_3d.get_tk_widget().pack(fill='both', expand=True)
        
        self.ax_3d_main = self.fig_3d.add_subplot(111, projection='3d')
        self.fig_3d.tight_layout()
    
    def create_status_bar(self):
        """创建底部状态栏"""
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
        """加载CSV文件"""
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
        """更新检测参数"""
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
            
            # 如果有数据，重新分析当前帧
            if self.data is not None:
                self.update_visualization()
    
    def update_visualization(self):
        """更新可视化显示"""
        if self.data is None:
            return
        
        try:
            # 获取当前帧数据
            stress_columns = [f'MAT_{i}' for i in range(96)]
            current_stress = self.data[stress_columns].iloc[self.current_frame].values
            stress_grid = current_stress.reshape(12, 8)
            timestamp = self.data['SN'].iloc[self.current_frame]
            
            # 执行检测
            normalized, nodule_mask, nodules, prob_map = self.detector.advanced_nodule_detection(
                stress_grid, timestamp
            )
            
            # 更新实时可视化图表 - 集成 fusion_real_time_detection.py 的可视化功能
            try:
                # 清除现有图表
                self.ax_raw.clear()
                self.ax_detected.clear()
                
                # 显示原始数据（归一化后）
                im1 = self.ax_raw.imshow(normalized, cmap=self.turbo_cmap, aspect='auto')
                self.ax_raw.set_title('实时原始数据', fontsize=10, fontweight='bold')
                self.ax_raw.set_xlabel('列索引')
                self.ax_raw.set_ylabel('行索引')
                
                # 显示检测结果
                im2 = self.ax_detected.imshow(prob_map, cmap=self.turbo_cmap, aspect='auto')
                self.ax_detected.set_title('实时结节检测', fontsize=10, fontweight='bold')
                self.ax_detected.set_xlabel('列索引')
                self.ax_detected.set_ylabel('行索引')
                
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
                    
                    self.ax_detected.scatter(y, x, c=color, s=size, marker=marker, 
                                           edgecolors='black', linewidth=1, alpha=0.8)
                    
                    # 添加风险评分标签
                    self.ax_detected.annotate(f'{risk:.2f}', (y, x), 
                                            xytext=(5, 5), textcoords='offset points',
                                            fontsize=8, color='white', fontweight='bold',
                                            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
                
                # 更新颜色条
                if hasattr(self, 'cbar1'):
                    self.cbar1.remove()
                if hasattr(self, 'cbar2'):
                    self.cbar2.remove()
                    
                self.cbar1 = self.fig.colorbar(im1, ax=self.ax_raw, shrink=0.8)
                self.cbar1.set_label('强度值', rotation=270, labelpad=15)
                
                self.cbar2 = self.fig.colorbar(im2, ax=self.ax_detected, shrink=0.8)
                self.cbar2.set_label('检测概率', rotation=270, labelpad=15)
                
                # 刷新画布
                self.canvas.draw()
                
            except Exception as e:
                print(f"实时可视化更新错误: {e}")
    
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
            if hasattr(self, 'cbar_realtime3'):
                self.cbar_realtime3.remove()
                
            self.cbar_realtime1 = self.fig_realtime.colorbar(im1, ax=self.ax_original, shrink=0.8)
            self.cbar_realtime1.set_label('强度值', rotation=270, labelpad=15)
            
            self.cbar_realtime2 = self.fig_realtime.colorbar(im2, ax=self.ax_detection, shrink=0.8)
            self.cbar_realtime2.set_label('检测概率', rotation=270, labelpad=15)
            
            self.cbar_realtime3 = self.fig_realtime.colorbar(contour, ax=self.ax_contour, shrink=0.8)
            self.cbar_realtime3.set_label('热力强度', rotation=270, labelpad=15)
            
            # 刷新画布
            self.canvas_realtime.draw()
            
        except Exception as e:
            print(f"实时可视化更新错误: {e}")
            
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
                
                # 更新趋势图
                self.update_trend_plots()
                
                # 更新3D图
                self.update_3d_plot(normalized, nodule_mask)
                
                # 更新状态
                self.status_var.set(f"实时处理中 - 检出结节: {len(nodules)}个")
                
                # 更新帧信息
                max_frames = len(self.data)
                self.frame_info.config(text=f"帧: {self.current_frame}/{max_frames-1}")
                self.frame_var.set(self.current_frame)
                
            except Exception as e:
                print(f"可视化更新错误: {e}")
                self.status_var.set(f"可视化错误: {str(e)}")
    
    def calculate_brightness_map(self, data):
        """计算亮度分布图"""
        # 将数据标准化到0-1范围
        normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
        
        # 应用高斯滤波平滑处理
        from scipy import ndimage
        smoothed = ndimage.gaussian_filter(normalized_data, sigma=0.8)
        
        # 计算局部对比度增强
        brightness_map = np.zeros_like(smoothed)
        for i in range(1, smoothed.shape[0]-1):
            for j in range(1, smoothed.shape[1]-1):
                # 计算局部梯度
                local_region = smoothed[i-1:i+2, j-1:j+2]
                local_std = np.std(local_region)
                local_mean = np.mean(local_region)
                
                # 结合原始值和局部对比度
                brightness_map[i, j] = smoothed[i, j] * (1 + local_std) + local_mean * 0.1
        
        # 边界处理
        brightness_map[0, :] = smoothed[0, :]
        brightness_map[-1, :] = smoothed[-1, :]
        brightness_map[:, 0] = smoothed[:, 0]
        brightness_map[:, -1] = smoothed[:, -1]
        
        return brightness_map
    
    def update_trend_plots(self):
        """更新趋势分析图"""
        if not self.detector.nodule_history['timestamps']:
            return
        
        # 清除图形
        self.ax_area_trend.clear()
        self.ax_risk_trend.clear()
        self.ax_count_trend.clear()
        self.ax_intensity_trend.clear()
        
        frames = range(len(self.detector.nodule_history['areas']))
        
        # 面积趋势
        self.ax_area_trend.plot(frames, self.detector.nodule_history['areas'], 'b-', linewidth=2)
        self.ax_area_trend.set_title('结节面积变化')
        self.ax_area_trend.set_ylabel('面积')
        
        # 风险评分趋势
        self.ax_risk_trend.plot(frames, self.detector.nodule_history['risk_scores'], 'r-', linewidth=2)
        self.ax_risk_trend.axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='高风险线')
        self.ax_risk_trend.axhline(y=0.4, color='orange', linestyle='--', alpha=0.7, label='中风险线')
        self.ax_risk_trend.set_title('风险评分变化')
        self.ax_risk_trend.set_ylabel('风险评分')
        self.ax_risk_trend.legend()
        
        # 结节数量趋势
        self.ax_count_trend.plot(frames, self.detector.nodule_history['count'], 'g-', linewidth=2)
        self.ax_count_trend.set_title('检出结节数量')
        self.ax_count_trend.set_ylabel('数量')
        
        # 强度趋势
        self.ax_intensity_trend.plot(frames, self.detector.nodule_history['intensities'], 'm-', linewidth=2)
        self.ax_intensity_trend.set_title('结节强度变化')
        self.ax_intensity_trend.set_ylabel('强度')
        
        self.canvas_trend.draw()
    
    def update_3d_plot(self, normalized, nodule_mask):
        """更新3D可视化"""
        self.ax_3d_main.clear()
        
        x, y = np.meshgrid(range(8), range(12))
        
        # 绘制应力表面
        surf = self.ax_3d_main.plot_surface(x, y, normalized, 
                                           cmap=self.detector.medical_cmap, 
                                           alpha=0.7, linewidth=0)
        
        # 绘制结节区域
        if np.any(nodule_mask):
            nodule_z = np.where(nodule_mask, normalized + 0.1, np.nan)
            self.ax_3d_main.plot_surface(x, y, nodule_z, color='red', alpha=0.8)
        
        self.ax_3d_main.set_title('3D应力与结节分布')
        self.ax_3d_main.set_xlabel('X')
        self.ax_3d_main.set_ylabel('Y')
        self.ax_3d_main.set_zlabel('应力值')
        
        self.canvas_3d.draw()
    
    def update_statistics(self, nodules, timestamp):
        """更新统计信息显示"""
        self.stats_text.delete(1.0, tk.END)
        
        stats_info = f"""=== 当前帧统计 ===
时间戳: {timestamp:.3f}s
检出结节: {len(nodules)}个

"""
        
        if nodules:
            main_nodule = max(nodules, key=lambda x: x['area'])
            stats_info += f"""主要结节信息:
面积: {main_nodule['area']:.1f}
圆形度: {main_nodule['circularity']:.3f}
强度: {main_nodule['intensity']:.3f}
风险评分: {main_nodule['risk_score']:.3f}
位置: ({main_nodule['centroid'][1]:.1f}, {main_nodule['centroid'][0]:.1f})

"""
        
        # 历史统计
        if self.detector.nodule_history['timestamps']:
            areas = [a for a in self.detector.nodule_history['areas'] if a > 0]
            if areas:
                stats_info += f"""=== 历史统计 ===
总帧数: {len(self.detector.nodule_history['timestamps'])}
检出率: {len(areas)/len(self.detector.nodule_history['timestamps']):.1%}
平均面积: {np.mean(areas):.2f}
最大面积: {max(areas):.2f}
最高风险: {max(self.detector.nodule_history['risk_scores']):.3f}
"""
        
        self.stats_text.insert(tk.END, stats_info)
    
    def toggle_play(self):
        """切换播放状态"""
        if self.data is None:
            return
        
        self.is_playing = not self.is_playing
        
        if self.is_playing:
            self.play_button.config(text="⏸")
            self.start_playback()
        else:
            self.play_button.config(text="▶")
            self.stop_playback()
    
    def start_playback(self):
        """开始播放"""
        if self.play_thread is None or not self.play_thread.is_alive():
            self.play_thread = threading.Thread(target=self.playback_loop)
            self.play_thread.daemon = True
            self.play_thread.start()
    
    def stop_playback(self):
        """停止播放"""
        self.is_playing = False
    
    def playback_loop(self):
        """播放循环"""
        while self.is_playing and self.data is not None:
            if self.current_frame < len(self.data) - 1:
                self.current_frame += 1
            else:
                self.current_frame = 0  # 循环播放
            
            # 在主线程中更新UI
            self.root.after(0, self.update_visualization)
            
            # 等待指定时间
            import time
            time.sleep(self.speed_var.get() / 1000.0)
    
    def goto_frame(self, value):
        """跳转到指定帧"""
        if self.data is not None:
            self.current_frame = int(float(value))
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
        """上一帧"""
        if self.data is not None and self.current_frame > 0:
            self.current_frame -= 1
            self.update_visualization()
    
    def next_frame(self):
        """下一帧"""
        if self.data is not None and self.current_frame < len(self.data) - 1:
            self.current_frame += 1
            self.update_visualization()
    
    def export_gif(self):
        """导出GIF动画"""
        if self.data is None:
            messagebox.showwarning("警告", "请先加载数据文件")
            return
        
        output_path = filedialog.asksaveasfilename(
            title="保存GIF动画",
            defaultextension=".gif",
            filetypes=[("GIF文件", "*.gif")]
        )
        
        if output_path:
            # 获取帧数
            max_frames = min(50, len(self.data))  # 限制最大帧数
            
            def export_thread():
                try:
                    self.progress.config(maximum=max_frames)
                    self.status_var.set("正在生成GIF动画...")
                    
                    success = self.detector.create_enhanced_visualization(
                        self.data, output_path, max_frames
                    )
                    
                    if success:
                        self.status_var.set(f"GIF动画已保存: {output_path}")
                        messagebox.showinfo("成功", f"GIF动画已保存到:\n{output_path}")
                    else:
                        self.status_var.set("GIF生成失败")
                        messagebox.showerror("错误", "GIF生成失败")
                        
                except Exception as e:
                    self.status_var.set(f"导出错误: {str(e)}")
                    messagebox.showerror("错误", f"导出失败: {str(e)}")
                finally:
                    self.progress.config(value=0)
            
            # 在后台线程中执行导出
            export_thread = threading.Thread(target=export_thread)
            export_thread.daemon = True
            export_thread.start()
    
    def export_report(self):
        """导出分析报告"""
        if not self.detector.nodule_history['timestamps']:
            messagebox.showwarning("警告", "暂无分析数据")
            return
        
        output_path = filedialog.asksaveasfilename(
            title="保存分析报告",
            defaultextension=".txt",
            filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")]
        )
        
        if output_path:
            try:
                report = self.detector.generate_analysis_report()
                report += f"\n\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(report)
                
                self.status_var.set(f"报告已保存: {output_path}")
                messagebox.showinfo("成功", f"分析报告已保存到:\n{output_path}")
                
            except Exception as e:
                messagebox.showerror("错误", f"保存报告失败: {str(e)}")
    
    def save_config(self):
        """保存配置"""
        config = {
            'gmm_components': self.gmm_var.get(),
            'smoothing_sigma': self.smooth_var.get(),
            'sensitivity_threshold': self.sensitivity_var.get(),
            'min_nodule_area': self.area_var.get(),
            'play_speed': self.speed_var.get()
        }
        
        try:
            with open('detection_config.json', 'w') as f:
                json.dump(config, f, indent=2)
        except:
            pass
    
    def load_config(self):
        """加载配置"""
        try:
            with open('detection_config.json', 'r') as f:
                config = json.load(f)
            
            self.gmm_var.set(config.get('gmm_components', 3))
            self.smooth_var.set(config.get('smoothing_sigma', 0.8))
            self.sensitivity_var.set(config.get('sensitivity_threshold', 0.7))
            self.area_var.set(config.get('min_nodule_area', 3))
            self.speed_var.set(config.get('play_speed', 500))
            
        except:
            pass  # 使用默认值
    
    def on_closing(self):
        """程序关闭时的处理"""
        self.is_playing = False
        
        # 断开串口连接
        if hasattr(self, 'is_serial_connected') and self.is_serial_connected:
            self.disconnect_serial()
        
        self.save_config()
        self.root.destroy()

if __name__ == '__main__':
    root = tk.Tk()
    app = ModernDetectionGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
