# -*- coding: utf-8 -*-
"""
优化版串口监控工具 - 专门解决延迟问题
实时读取和显示串口数据，支持协议解析和结节检测
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import serial
import serial.tools.list_ports
import threading
import time
from datetime import datetime
import queue
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from collections import deque
import gc

# 尝试导入检测系统，如果不存在则创建简化版本
try:
    from enhanced_detection_system import EnhancedNoduleDetectionSystem
except ImportError:
    # 超轻量级检测系统类
    class EnhancedNoduleDetectionSystem:
        def __init__(self):
            self.medical_cmap = plt.cm.hot
            self.nodule_history = deque(maxlen=100)  # 使用deque限制历史记录
        
        def advanced_nodule_detection(self, stress_grid, timestamp):
            # 极简化的检测逻辑，减少计算开销
            max_val = stress_grid.max()
            if max_val == 0:
                normalized = stress_grid
                nodule_mask = np.zeros_like(stress_grid, dtype=bool)
                nodules = []
            else:
                normalized = stress_grid / max_val
                threshold = 0.3
                nodule_mask = normalized > threshold
                
                # 简单的结节信息
                nodules = []
                if np.any(nodule_mask):
                    area = np.sum(nodule_mask)
                    if area > 2:  # 只处理较大的区域
                        y_coords, x_coords = np.where(nodule_mask)
                        centroid = (np.mean(y_coords), np.mean(x_coords))
                        intensity = np.mean(normalized[nodule_mask])
                        nodules.append({
                            'area': area,
                            'centroid': centroid,
                            'intensity': intensity,
                            'risk_score': min(intensity * 1.5, 1.0),
                            'circularity': 0.8
                        })
            
            # 更新历史记录（使用deque自动限制大小）
            self.nodule_history.append({
                'timestamp': timestamp,
                'area': nodules[0]['area'] if nodules else 0,
                'risk_score': nodules[0]['risk_score'] if nodules else 0,
                'count': len(nodules),
                'intensity': nodules[0]['intensity'] if nodules else 0
            })
            
            return normalized, nodule_mask, nodules, normalized

class FastProtocolParser:
    """高性能协议解析器 - 优化版本"""
    
    def __init__(self):
        self.buffer = bytearray()
        self.frame_header = b'\xA5\x5A'  # 直接使用字节模式
        self.expected_frame_size = 104
        self.parsed_frames = deque(maxlen=50)  # 限制缓存帧数
        self.last_parse_time = 0
        self.parse_interval = 0.01  # 最小解析间隔10ms
        
    def add_data(self, data):
        """添加新的串口数据到缓冲区"""
        self.buffer.extend(data)
        
        # 限制解析频率，避免过度处理
        current_time = time.time()
        if current_time - self.last_parse_time < self.parse_interval:
            return
        
        self.last_parse_time = current_time
        self._parse_frames()
    
    def _parse_frames(self):
        """从缓冲区解析完整帧 - 优化版本"""
        # 限制缓冲区大小，避免内存积压
        if len(self.buffer) > 2048:
            self.buffer = self.buffer[-1024:]  # 保留最后1KB数据
        
        while len(self.buffer) >= self.expected_frame_size:
            # 快速查找帧头
            frame_start = self.buffer.find(self.frame_header)
            if frame_start == -1:
                # 没找到帧头，保留最后几个字节防止帧头被截断
                if len(self.buffer) > 10:
                    self.buffer = self.buffer[-10:]
                break
            
            # 移除帧头之前的数据
            if frame_start > 0:
                self.buffer = self.buffer[frame_start:]
            
            # 检查是否有完整帧
            if len(self.buffer) < self.expected_frame_size:
                break
            
            # 快速解析帧
            frame_data = self.buffer[:self.expected_frame_size]
            parsed_frame = self._parse_single_frame_fast(frame_data)
            
            if parsed_frame:
                self.parsed_frames.append(parsed_frame)
            
            # 移除已处理的帧
            self.buffer = self.buffer[self.expected_frame_size:]
    
    def _parse_single_frame_fast(self, frame_data):
        """快速解析单个帧 - 减少验证步骤"""
        try:
            # 最小验证
            if len(frame_data) < 102:
                return None
            
            # 跳过详细验证，直接提取数据
            pressure_data = frame_data[6:102]
            pressure_matrix = np.frombuffer(pressure_data, dtype=np.uint8).reshape(12, 8)
            
            return {
                'timestamp': time.time(),
                'matrix': pressure_matrix,
                'checksum_valid': True  # 简化验证
            }
            
        except Exception:
            return None
    
    def get_latest_frame(self):
        """获取最新解析的帧"""
        return self.parsed_frames[-1] if self.parsed_frames else None
    
    def get_frame_count(self):
        """获取已解析的帧数量"""
        return len(self.parsed_frames)

class OptimizedSerialMonitor:
    def __init__(self, root):
        self.root = root
        self.root.title("优化版串口监控工具 - 低延迟")
        self.root.geometry("1200x800")
        
        # 串口相关变量
        self.serial_port = None
        self.is_connected = False
        self.is_monitoring = False
        self.data_queue = queue.Queue(maxsize=100)  # 限制队列大小
        
        # 高性能解析器
        self.protocol_parser = FastProtocolParser()
        
        # 轻量级检测系统
        self.detector = EnhancedNoduleDetectionSystem()
        
        # 优化的数据缓存
        self.realtime_data_buffer = deque(maxlen=500)  # 使用deque，自动限制大小
        
        # 统计变量
        self.bytes_received = 0
        self.frames_parsed = 0
        self.start_time = None
        
        # 性能优化参数
        self.max_process_bytes_per_tick = 4096  # 减少单次处理量
        self.max_queue_chunks_per_tick = 4      # 减少队列处理量
        self.max_display_bytes_per_line = 256   # 减少显示量
        self.enable_truncate_display = True
        self.displayed_bytes = 0
        self.truncated_bytes = 0
        
        # 显示更新控制
        self.last_display_update = 0
        self.display_update_interval = 0.05  # 20fps最大更新频率
        self.last_plot_update = 0
        self.plot_update_interval = 0.1  # 10fps绘图更新
        
        # 数据保存
        self.save_file = None
        self.auto_save = False
        
        # 显示模式
        self.display_mode = tk.StringVar(value="protocol")
        
        # 性能监控
        self.frame_times = deque(maxlen=100)
        self.processing_times = deque(maxlen=100)
        
        self.setup_gui()
        self.update_ports()
        self.start_data_processor()
        
        # 启用垃圾回收优化
        gc.set_threshold(700, 10, 10)
        
    def setup_gui(self):
        """设置GUI界面 - 简化版本"""
        # 创建主要布局
        main_paned = ttk.PanedWindow(self.root, orient='horizontal')
        main_paned.pack(fill='both', expand=True, padx=5, pady=5)
        
        # 左侧控制面板
        self.create_control_panel(main_paned)
        
        # 右侧显示面板
        self.create_display_panel(main_paned)
        
        # 底部状态栏
        self.create_status_bar()
    
    def create_control_panel(self, parent):
        """创建左侧控制面板 - 简化版本"""
        control_frame = ttk.Frame(parent, width=300)
        parent.add(control_frame, weight=0)
        
        # 连接配置区域
        config_frame = ttk.LabelFrame(control_frame, text="连接配置", padding="5")
        config_frame.pack(fill='x', pady=5)
        
        # 串口选择
        port_frame = ttk.Frame(config_frame)
        port_frame.pack(fill='x', pady=2)
        ttk.Label(port_frame, text="串口:").pack(side='left')
        self.port_var = tk.StringVar()
        self.port_combo = ttk.Combobox(port_frame, textvariable=self.port_var, width=12)
        self.port_combo.pack(side='left', padx=5)
        
        # 波特率选择
        baud_frame = ttk.Frame(config_frame)
        baud_frame.pack(fill='x', pady=2)
        ttk.Label(baud_frame, text="波特率:").pack(side='left')
        self.baudrate_var = tk.StringVar(value="115200")
        baudrate_combo = ttk.Combobox(baud_frame, textvariable=self.baudrate_var, 
                                     values=["115200", "57600", "38400", "19200", "9600"], width=8)
        baudrate_combo.pack(side='left', padx=5)
        
        # 控制按钮
        button_frame = ttk.Frame(config_frame)
        button_frame.pack(fill='x', pady=5)
        
        self.connect_btn = ttk.Button(button_frame, text="连接", command=self.toggle_connection)
        self.connect_btn.pack(side='left', padx=2)
        
        ttk.Button(button_frame, text="刷新", command=self.update_ports).pack(side='left', padx=2)
        
        # 性能优化选项
        perf_frame = ttk.LabelFrame(control_frame, text="性能设置", padding="5")
        perf_frame.pack(fill='x', pady=5)
        
        # 更新频率控制
        ttk.Label(perf_frame, text="显示频率(fps):").pack(anchor='w')
        self.fps_var = tk.IntVar(value=20)
        fps_scale = ttk.Scale(perf_frame, from_=5, to=60, variable=self.fps_var, 
                             orient='horizontal', command=self.update_fps)
        fps_scale.pack(fill='x')
        self.fps_label = ttk.Label(perf_frame, text="20 fps")
        self.fps_label.pack(anchor='w')
        
        # 缓存大小控制
        ttk.Label(perf_frame, text="缓存大小:").pack(anchor='w')
        self.buffer_size_var = tk.IntVar(value=500)
        buffer_scale = ttk.Scale(perf_frame, from_=100, to=1000, variable=self.buffer_size_var,
                                orient='horizontal', command=self.update_buffer_size)
        buffer_scale.pack(fill='x')
        self.buffer_label = ttk.Label(perf_frame, text="500 帧")
        self.buffer_label.pack(anchor='w')
        
        # 显示模式选择
        mode_frame = ttk.LabelFrame(control_frame, text="显示模式", padding="5")
        mode_frame.pack(fill='x', pady=5)
        
        ttk.Radiobutton(mode_frame, text="协议解析", variable=self.display_mode, 
                       value="protocol", command=self.change_display_mode).pack(anchor='w')
        ttk.Radiobutton(mode_frame, text="实时检测", variable=self.display_mode, 
                       value="detection", command=self.change_display_mode).pack(anchor='w')
        ttk.Radiobutton(mode_frame, text="性能监控", variable=self.display_mode, 
                       value="performance", command=self.change_display_mode).pack(anchor='w')
        
        # 数据操作
        data_frame = ttk.LabelFrame(control_frame, text="数据操作", padding="5")
        data_frame.pack(fill='x', pady=5)
        
        ttk.Button(data_frame, text="清空显示", command=self.clear_display).pack(fill='x', pady=1)
        ttk.Button(data_frame, text="导出CSV", command=self.export_csv).pack(fill='x', pady=1)
        
        # 性能统计
        stats_frame = ttk.LabelFrame(control_frame, text="性能统计", padding="5")
        stats_frame.pack(fill='x', pady=5)
        
        self.perf_text = tk.Text(stats_frame, height=8, width=30, font=('Courier', 8))
        self.perf_text.pack(fill='both', expand=True)
    
    def create_display_panel(self, parent):
        """创建右侧显示面板"""
        display_frame = ttk.Frame(parent)
        parent.add(display_frame, weight=1)
        
        # 创建选项卡
        self.notebook = ttk.Notebook(display_frame)
        self.notebook.pack(fill='both', expand=True)
        
        # 协议解析选项卡
        self.create_protocol_tab()
        
        # 实时检测选项卡
        self.create_detection_tab()
        
        # 性能监控选项卡
        self.create_performance_tab()
    
    def create_protocol_tab(self):
        """创建协议解析选项卡"""
        protocol_frame = ttk.Frame(self.notebook)
        self.notebook.add(protocol_frame, text="协议解析")
        
        # 压力矩阵显示
        self.matrix_display = tk.Text(protocol_frame, font=('Courier', 8))
        matrix_scroll = ttk.Scrollbar(protocol_frame, orient='vertical', command=self.matrix_display.yview)
        self.matrix_display.configure(yscrollcommand=matrix_scroll.set)
        
        self.matrix_display.pack(side='left', fill='both', expand=True)
        matrix_scroll.pack(side='right', fill='y')
    
    def create_detection_tab(self):
        """创建实时检测选项卡"""
        detection_frame = ttk.Frame(self.notebook)
        self.notebook.add(detection_frame, text="实时检测")
        
        # 创建matplotlib图形 - 减少子图数量
        self.fig = Figure(figsize=(10, 6), dpi=80)  # 降低DPI提高性能
        self.canvas = FigureCanvasTkAgg(self.fig, detection_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # 只创建两个关键子图
        self.ax_original = self.fig.add_subplot(121)
        self.ax_detection = self.fig.add_subplot(122)
        
        self.fig.tight_layout()
        
        # 预创建图像对象以提高性能
        self._init_detection_plots()
    
    def create_performance_tab(self):
        """创建性能监控选项卡"""
        perf_frame = ttk.Frame(self.notebook)
        self.notebook.add(perf_frame, text="性能监控")
        
        # 性能图表
        self.perf_fig = Figure(figsize=(10, 6), dpi=80)
        self.perf_canvas = FigureCanvasTkAgg(self.perf_fig, perf_frame)
        self.perf_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        self.ax_fps = self.perf_fig.add_subplot(211)
        self.ax_latency = self.perf_fig.add_subplot(212)
        
        self.perf_fig.tight_layout()
    
    def _init_detection_plots(self):
        """初始化检测绘图对象"""
        empty_data = np.zeros((12, 8))
        
        # 预创建图像对象
        self.im_original = self.ax_original.imshow(empty_data, cmap='hot', origin='lower', 
                                                  vmin=0, vmax=255, animated=True)
        self.ax_original.set_title('压力分布')
        
        self.im_detection = self.ax_detection.imshow(empty_data, cmap='gray', origin='lower',
                                                    vmin=0, vmax=1, animated=True)
        self.ax_detection.set_title('检测结果')
        
        # 结节标记列表
        self.nodule_markers = []
    
    def create_status_bar(self):
        """创建底部状态栏"""
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill='x', side='bottom')
        
        self.status_var = tk.StringVar(value="就绪")
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var, 
                                     relief='sunken', anchor='w')
        self.status_label.pack(fill='x', side='left')
        
        # 性能指标
        self.perf_label = ttk.Label(status_frame, text="FPS: 0 | 延迟: 0ms")
        self.perf_label.pack(side='right', padx=5)
    
    def update_fps(self, value):
        """更新显示频率"""
        fps = int(float(value))
        self.display_update_interval = 1.0 / fps
        self.fps_label.config(text=f"{fps} fps")
    
    def update_buffer_size(self, value):
        """更新缓存大小"""
        size = int(float(value))
        self.realtime_data_buffer = deque(self.realtime_data_buffer, maxlen=size)
        self.buffer_label.config(text=f"{size} 帧")
    
    def update_ports(self):
        """更新可用串口列表"""
        ports = [port.device for port in serial.tools.list_ports.comports()]
        self.port_combo['values'] = ports
        if ports and not self.port_var.get():
            self.port_var.set(ports[0])
    
    def toggle_connection(self):
        """切换连接状态"""
        if not self.is_connected:
            self.connect_serial()
        else:
            self.disconnect_serial()
    
    def connect_serial(self):
        """连接串口 - 优化版本"""
        try:
            port = self.port_var.get()
            baudrate = int(self.baudrate_var.get())
            
            if not port:
                messagebox.showerror("错误", "请选择串口")
                return
            
            # 优化串口参数以减少延迟
            self.serial_port = serial.Serial(
                port=port,
                baudrate=baudrate,
                bytesize=8,
                stopbits=1,
                parity=serial.PARITY_NONE,
                timeout=0.001,  # 极短超时
                write_timeout=0.001,
                inter_byte_timeout=None,
                exclusive=True
            )
            
            # 设置缓冲区大小
            if hasattr(self.serial_port, 'set_buffer_size'):
                self.serial_port.set_buffer_size(rx_size=4096, tx_size=4096)
            
            self.is_connected = True
            self.is_monitoring = True
            self.bytes_received = 0
            self.frames_parsed = 0
            self.start_time = time.time()
            
            # 重置解析器
            self.protocol_parser = FastProtocolParser()
            
            # 启动高优先级数据读取线程
            self.read_thread = threading.Thread(target=self.read_serial_data_fast, daemon=True)
            self.read_thread.start()
            
            # 更新UI
            self.connect_btn.config(text="断开")
            self.status_var.set(f"已连接 {port} @ {baudrate}bps")
            
        except Exception as e:
            messagebox.showerror("连接错误", f"无法连接串口: {str(e)}")
    
    def read_serial_data_fast(self):
        """高性能串口数据读取"""
        while self.is_monitoring and self.serial_port and self.serial_port.is_open:
            try:
                # 批量读取以提高效率
                bytes_available = self.serial_port.in_waiting
                if bytes_available > 0:
                    # 限制单次读取量，避免阻塞
                    read_size = min(bytes_available, 2048)
                    data = self.serial_port.read(read_size)
                    if data:
                        # 非阻塞放入队列
                        try:
                            self.data_queue.put_nowait(data)
                            self.bytes_received += len(data)
                        except queue.Full:
                            # 队列满时丢弃最旧的数据
                            try:
                                self.data_queue.get_nowait()
                                self.data_queue.put_nowait(data)
                            except queue.Empty:
                                pass
                
                # 极短延迟
                time.sleep(0.001)
                
            except Exception as e:
                print(f"串口读取错误: {e}")
                break
    
    def disconnect_serial(self):
        """断开串口连接"""
        self.is_monitoring = False
        self.is_connected = False
        
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
        
        self.connect_btn.config(text="连接")
        self.status_var.set("已断开")
    
    def start_data_processor(self):
        """启动数据处理循环"""
        self.process_data()
    
    def process_data(self):
        """处理数据队列 - 优化版本"""
        try:
            process_start = time.time()
            chunks_processed = 0
            
            # 限制单次处理的数据块数量
            while not self.data_queue.empty() and chunks_processed < self.max_queue_chunks_per_tick:
                try:
                    data_chunk = self.data_queue.get_nowait()
                    self.protocol_parser.add_data(data_chunk)
                    chunks_processed += 1
                except queue.Empty:
                    break
            
            # 处理解析后的帧
            self.process_parsed_frames()
            
            # 记录处理时间
            process_time = (time.time() - process_start) * 1000
            self.processing_times.append(process_time)
            
            # 更新性能统计
            if time.time() - self.last_display_update > self.display_update_interval:
                self.update_performance_stats()
                self.last_display_update = time.time()
            
        except Exception as e:
            print(f"数据处理错误: {e}")
        
        # 继续处理循环
        self.root.after(5, self.process_data)  # 5ms间隔
    
    def process_parsed_frames(self):
        """处理解析后的帧数据"""
        latest_frame = self.protocol_parser.get_latest_frame()
        if latest_frame:
            self.frames_parsed += 1
            
            # 添加到实时缓存
            self.realtime_data_buffer.append(latest_frame)
            
            # 根据显示模式更新界面
            current_time = time.time()
            if current_time - self.last_plot_update > self.plot_update_interval:
                mode = self.display_mode.get()
                if mode == "protocol":
                    self.display_protocol_data_fast(latest_frame)
                elif mode == "detection":
                    self.display_detection_data_fast(latest_frame)
                
                self.last_plot_update = current_time
    
    def display_protocol_data_fast(self, frame):
        """快速显示协议数据"""
        try:
            matrix = frame['matrix']
            timestamp = frame['timestamp']
            
            # 格式化矩阵显示
            display_text = f"时间戳: {timestamp:.3f}\n"
            display_text += "压力矩阵 (12x8):\n"
            for row in matrix:
                display_text += " ".join(f"{val:3d}" for val in row) + "\n"
            display_text += "-" * 50 + "\n"
            
            # 更新显示（限制显示长度）
            self.matrix_display.insert(tk.END, display_text)
            if self.matrix_display.index(tk.END).split('.')[0] > '100':
                self.matrix_display.delete('1.0', '50.0')
            
            self.matrix_display.see(tk.END)
            
        except Exception as e:
            print(f"协议显示错误: {e}")
    
    def display_detection_data_fast(self, frame):
        """快速显示检测数据"""
        try:
            matrix = frame['matrix'].astype(float)
            timestamp = frame['timestamp']
            
            # 执行检测
            normalized, nodule_mask, nodules, _ = self.detector.advanced_nodule_detection(
                matrix, timestamp
            )
            
            # 更新图像数据
            self.im_original.set_array(normalized)
            self.im_original.set_clim(vmin=normalized.min(), vmax=normalized.max())
            
            self.im_detection.set_array(nodule_mask)
            
            # 清除旧的结节标记
            for marker in self.nodule_markers:
                marker.remove()
            self.nodule_markers.clear()
            
            # 添加新的结节标记
            for nodule in nodules:
                centroid = nodule['centroid']
                risk_score = nodule['risk_score']
                
                # 根据风险评分选择颜色
                color = 'red' if risk_score > 0.7 else 'yellow' if risk_score > 0.4 else 'green'
                
                marker = self.ax_detection.plot(centroid[1], centroid[0], '*', 
                                              color=color, markersize=15)[0]
                self.nodule_markers.append(marker)
                
                # 添加风险评分标签
                text = self.ax_detection.text(centroid[1]+0.5, centroid[0]+0.5, 
                                            f'{risk_score:.2f}', 
                                            color=color, fontsize=10, fontweight='bold')
                self.nodule_markers.append(text)
            
            # 更新画布
            self.canvas.draw_idle()
            
        except Exception as e:
            print(f"检测显示错误: {e}")
    
    def update_performance_stats(self):
        """更新性能统计"""
        try:
            current_time = time.time()
            
            # 计算FPS
            if self.start_time:
                elapsed = current_time - self.start_time
                fps = self.frames_parsed / elapsed if elapsed > 0 else 0
            else:
                fps = 0
            
            # 计算平均处理时间
            avg_process_time = np.mean(self.processing_times) if self.processing_times else 0
            
            # 计算数据速率
            if self.start_time:
                elapsed = current_time - self.start_time
                data_rate = self.bytes_received / elapsed if elapsed > 0 else 0
            else:
                data_rate = 0
            
            # 更新状态栏
            self.perf_label.config(text=f"FPS: {fps:.1f} | 延迟: {avg_process_time:.1f}ms | 速率: {data_rate:.0f}B/s")
            
            # 更新详细统计
            stats_text = f"""性能统计:
FPS: {fps:.2f}
处理延迟: {avg_process_time:.2f}ms
数据速率: {data_rate:.0f} B/s
接收字节: {self.bytes_received}
解析帧数: {self.frames_parsed}
缓存帧数: {len(self.realtime_data_buffer)}
队列大小: {self.data_queue.qsize()}

内存使用:
解析缓存: {len(self.protocol_parser.parsed_frames)}
历史记录: {len(self.detector.nodule_history)}
处理时间: {len(self.processing_times)}
"""
            
            self.perf_text.delete('1.0', tk.END)
            self.perf_text.insert('1.0', stats_text)
            
            # 更新性能图表
            if self.display_mode.get() == "performance":
                self.update_performance_plots(fps, avg_process_time)
            
        except Exception as e:
            print(f"性能统计更新错误: {e}")
    
    def update_performance_plots(self, fps, latency):
        """更新性能图表"""
        try:
            # 记录帧时间
            self.frame_times.append(time.time())
            
            # 计算最近的FPS
            if len(self.frame_times) > 10:
                recent_fps = []
                for i in range(1, min(len(self.frame_times), 50)):
                    dt = self.frame_times[-i] - self.frame_times[-i-1]
                    if dt > 0:
                        recent_fps.append(1.0 / dt)
                
                # 更新FPS图表
                self.ax_fps.clear()
                self.ax_fps.plot(recent_fps[-30:], 'b-', linewidth=2)
                self.ax_fps.set_title('实时FPS')
                self.ax_fps.set_ylabel('FPS')
                self.ax_fps.grid(True, alpha=0.3)
                
                # 更新延迟图表
                self.ax_latency.clear()
                recent_latency = list(self.processing_times)[-30:]
                self.ax_latency.plot(recent_latency, 'r-', linewidth=2)
                self.ax_latency.set_title('处理延迟')
                self.ax_latency.set_ylabel('延迟 (ms)')
                self.ax_latency.set_xlabel('样本')
                self.ax_latency.grid(True, alpha=0.3)
                
                self.perf_fig.tight_layout()
                self.perf_canvas.draw_idle()
            
        except Exception as e:
            print(f"性能图表更新错误: {e}")
    
    def change_display_mode(self):
        """切换显示模式"""
        mode = self.display_mode.get()
        if mode == "protocol":
            self.notebook.select(0)
        elif mode == "detection":
            self.notebook.select(1)
        elif mode == "performance":
            self.notebook.select(2)
    
    def clear_display(self):
        """清空显示"""
        try:
            # 清空文本显示
            self.matrix_display.delete('1.0', tk.END)
            
            # 重置统计
            self.bytes_received = 0
            self.frames_parsed = 0
            self.start_time = time.time()
            
            # 清空缓存
            self.realtime_data_buffer.clear()
            self.protocol_parser.parsed_frames.clear()
            self.detector.nodule_history.clear()
            self.frame_times.clear()
            self.processing_times.clear()
            
            # 强制垃圾回收
            gc.collect()
            
        except Exception as e:
            print(f"清空显示错误: {e}")
    
    def export_csv(self):
        """导出CSV数据"""
        try:
            if not self.realtime_data_buffer:
                messagebox.showwarning("警告", "没有数据可导出")
                return
            
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV文件", "*.csv"), ("所有文件", "*.*")],
                title="导出数据"
            )
            
            if filename:
                # 准备数据
                data_rows = []
                for i, frame in enumerate(self.realtime_data_buffer):
                    row = {'SN': i, 'timestamp': frame['timestamp']}
                    matrix = frame['matrix'].flatten()
                    for j, val in enumerate(matrix):
                        row[f'MAT_{j}'] = val
                    data_rows.append(row)
                
                # 保存CSV
                df = pd.DataFrame(data_rows)
                df.to_csv(filename, index=False)
                
                messagebox.showinfo("成功", f"数据已导出到: {filename}")
                
        except Exception as e:
            messagebox.showerror("导出错误", f"导出失败: {str(e)}")
    
    def on_closing(self):
        """程序关闭处理"""
        self.disconnect_serial()
        self.root.destroy()

def main():
    """主函数"""
    root = tk.Tk()
    app = OptimizedSerialMonitor(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()