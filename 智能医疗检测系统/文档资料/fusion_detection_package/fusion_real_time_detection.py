# -*- coding: utf-8 -*-
"""
Fusion Real-Time Detection Monitor
整合 OptimizedSerialMonitor 的高性能串口读取与 ModernDetectionGUI 的结节检测算法，
使用实时串口/传感器数据流直接驱动检测可视化，避免 CSV 文件加载造成的延迟。

特点：
1. 低延迟：<1ms 读取超时 + 解析节流，目标 30-50 fps 更新。
2. 好看的色彩映射：默认使用 matplotlib 的 "turbo" colormap（≥3.4），若不可用则退化为 "viridis"。
3. 极简 GUI：左侧串口控制，右侧两幅图（原始热力图 + 检测结果）。
4. 自动平滑帧率：根据 fps_scale 调整 after() 调度间隔。
"""

import tkinter as tk
from tkinter import ttk, messagebox
import serial
import serial.tools.list_ports
import threading
import time
from collections import deque
from datetime import datetime
import queue
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# ---------- 复用已有类（直接从 optimized_serial_monitor.py 复制） ----------

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


# 超轻量级增强检测系统，同 optimized_serial_monitor 中定义
class EnhancedNoduleDetectionSystem:
    def __init__(self):
        # 使用好看的 colormap
        try:
            self.medical_cmap = plt.get_cmap("turbo")
        except Exception:
            self.medical_cmap = plt.get_cmap("viridis")

    def advanced_nodule_detection(self, stress_grid, timestamp):
        max_val = stress_grid.max()
        normalized = stress_grid / max_val if max_val > 0 else stress_grid
        threshold = 0.3
        nodule_mask = normalized > threshold
        nodules = []
        if np.any(nodule_mask):
            area = np.sum(nodule_mask)
            if area > 2:
                y, x = np.where(nodule_mask)
                centroid = (np.mean(y), np.mean(x))
                intensity = np.mean(normalized[nodule_mask])
                nodules.append(
                    {
                        "area": area,
                        "centroid": centroid,
                        "intensity": intensity,
                        "risk_score": min(intensity * 1.5, 1.0),
                    }
                )
        return normalized, nodule_mask, nodules


# ---------- GUI 主体 ----------
class FusionMonitorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Fusion Real-Time Detection Monitor")
        self.root.geometry("1000x700")

        # 串口变量
        self.serial_port = None
        self.is_running = False
        self.data_q = queue.Queue(maxsize=100)
        self.parser = FastProtocolParser()
        self.detector = EnhancedNoduleDetectionSystem()
        self.last_plot_update = 0
        self.plot_interval = 0.03  # 约 33 fps

        self.init_gui()
        self.update_ports()
        self.process_data_loop()

    # ---------- GUI ----------
    def init_gui(self):
        main = ttk.Frame(self.root)
        main.pack(fill="both", expand=True, padx=5, pady=5)
        left = ttk.Frame(main, width=200)
        left.pack(side="left", fill="y")
        right = ttk.Frame(main)
        right.pack(side="right", fill="both", expand=True)

        # --- 控制面板 ---
        ttk.Label(left, text="串口:").pack(anchor="w")
        self.port_var = tk.StringVar()
        self.port_combo = ttk.Combobox(left, textvariable=self.port_var, width=12)
        self.port_combo.pack(fill="x")

        ttk.Label(left, text="波特率:").pack(anchor="w", pady=(5, 0))
        self.baud_var = tk.StringVar(value="115200")
        self.baud_combo = ttk.Combobox(
            left,
            textvariable=self.baud_var,
            values=["9600", "19200", "38400", "57600", "115200"],
            width=12,
        )
        self.baud_combo.pack(fill="x")

        self.connect_btn = ttk.Button(left, text="连接", command=self.toggle_connection)
        self.connect_btn.pack(fill="x", pady=5)
        ttk.Button(left, text="刷新", command=self.update_ports).pack(fill="x")

        self.status_var = tk.StringVar(value="未连接")
        ttk.Label(left, textvariable=self.status_var).pack(anchor="w", pady=5)

        # --- FPS 调节 ---
        ttk.Label(left, text="显示 FPS:").pack(anchor="w", pady=(10, 0))
        self.fps_var = tk.IntVar(value=30)
        ttk.Scale(
            left,
            from_=5,
            to=60,
            orient="horizontal",
            variable=self.fps_var,
            command=self.update_fps,
        ).pack(fill="x")

        # --- matplotlib 区域 ---
        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.ax_raw = self.fig.add_subplot(121)
        self.ax_det = self.fig.add_subplot(122)
        self.fig.tight_layout()

        # 初始化图像对象
        empty = np.zeros((12, 8))
        self.im_raw = self.ax_raw.imshow(empty, cmap=self.detector.medical_cmap, origin="lower")
        self.im_det_bg = self.ax_det.imshow(empty, cmap=self.detector.medical_cmap, origin="lower")
        self.im_det_mask = self.ax_det.imshow(empty, cmap="Reds", alpha=0.4, origin="lower")
        self.ax_raw.set_title("压力分布")
        self.ax_det.set_title("结节检测")

    def update_fps(self, val):
        fps = int(float(val))
        self.plot_interval = 1.0 / fps

    # ---------- 串口处理 ----------
    def update_ports(self):
        ports = [p.device for p in serial.tools.list_ports.comports()]
        self.port_combo["values"] = ports
        if ports and not self.port_var.get():
            self.port_var.set(ports[0])

    def toggle_connection(self):
        if self.serial_port and self.serial_port.is_open:
            self.disconnect()
        else:
            self.connect()

    def connect(self):
        port = self.port_var.get()
        if not port:
            messagebox.showerror("错误", "请选择串口")
            return
        baud = int(self.baud_var.get())
        try:
            self.serial_port = serial.Serial(
                port=port,
                baudrate=baud,
                bytesize=8,
                stopbits=1,
                parity=serial.PARITY_NONE,
                timeout=0.001,
                write_timeout=0.001,
            )
            self.is_running = True
            threading.Thread(target=self.read_thread, daemon=True).start()
            self.connect_btn.config(text="断开")
            self.status_var.set(f"已连接 {port}@{baud}")
        except Exception as e:
            messagebox.showerror("连接失败", str(e))

    def disconnect(self):
        self.is_running = False
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
        self.connect_btn.config(text="连接")
        self.status_var.set("未连接")

    def read_thread(self):
        while self.is_running and self.serial_port and self.serial_port.is_open:
            try:
                avail = self.serial_port.in_waiting
                if avail:
                    data = self.serial_port.read(min(avail, 2048))
                    try:
                        self.data_q.put_nowait(data)
                    except queue.Full:
                        # 丢弃最旧数据，保证低延迟
                        try:
                            self.data_q.get_nowait()
                            self.data_q.put_nowait(data)
                        except queue.Empty:
                            pass
                time.sleep(0.001)
            except Exception:
                break

    # ---------- 数据处理 & 绘图循环 ----------
    def process_data_loop(self):
        try:
            # 取队列数据
            while not self.data_q.empty():
                data = self.data_q.get_nowait()
                self.parser.add_data(data)
            frame = self.parser.get_latest()
            if frame:
                now = time.time()
                if now - self.last_plot_update >= self.plot_interval:
                    self.last_plot_update = now
                    self.update_plots(frame)
        except Exception as e:
            print("数据处理错误", e)
        finally:
            self.root.after(5, self.process_data_loop)

    def update_plots(self, frame):
        matrix = frame["matrix"].astype(float)
        norm, mask, nodules = self.detector.advanced_nodule_detection(matrix, frame["timestamp"])
        # 更新图像
        self.im_raw.set_array(norm)
        self.im_raw.set_clim(vmin=norm.min(), vmax=norm.max())
        self.im_det_bg.set_array(norm)
        self.im_det_bg.set_clim(vmin=norm.min(), vmax=norm.max())
        self.im_det_mask.set_array(mask)
        # 清除旧标记
        if hasattr(self, "_markers"):
            for m in self._markers:
                m.remove()
        self._markers = []
        for n in nodules:
            cy, cx = n["centroid"]
            color = "red" if n["risk_score"] > 0.7 else "yellow"
            marker, = self.ax_det.plot(cx, cy, "*", color=color, markersize=12)
            self._markers.append(marker)
        self.canvas.draw_idle()


# ---------- 入口 ----------
if __name__ == "__main__":
    root = tk.Tk()
    FusionMonitorGUI(root)
    root.mainloop()