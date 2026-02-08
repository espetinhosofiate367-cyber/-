#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
智能医疗检测系统启动程序
用于启动重新组织后的医疗检测系统
"""

import sys
import os
import tkinter as tk
from tkinter import messagebox

def main():
    try:
        # 添加核心程序路径到系统路径
        core_path = os.path.join(os.path.dirname(__file__), '核心程序')
        sys.path.insert(0, core_path)
        
        # 导入主程序
        from modern_detection_gui_optimized import OptimizedDetectionGUI
        
        # 创建主窗口
        root = tk.Tk()
        
        # 设置窗口图标和标题
        root.title("智能医疗检测系统 - 重新组织版")
        root.geometry("1400x900")
        
        # 创建应用实例
        app = OptimizedDetectionGUI(root)
        
        print("智能医疗检测系统启动成功！")
        print("文件已重新组织到以下结构：")
        print("├── 核心程序/        # 主程序文件")
        print("├── 检测算法/        # 检测算法模块")
        print("├── 界面模块/        # GUI界面模块")
        print("├── 配置文件/        # 配置和设置文件")
        print("├── 数据文件/        # 数据和模型文件")
        print("├── 测试结果/        # 测试结果和图表")
        print("├── 文档资料/        # 文档和说明")
        print("├── 实验数据/        # 实验数据和媒体文件")
        print("└── 缓存文件/        # 缓存和临时文件")
        
        # 启动主循环
        root.mainloop()
        
    except ImportError as e:
        error_msg = f"导入模块失败: {e}\n请检查文件路径是否正确。"
        print(error_msg)
        messagebox.showerror("导入错误", error_msg)
        
    except Exception as e:
        error_msg = f"程序启动失败: {e}"
        print(error_msg)
        messagebox.showerror("启动错误", error_msg)

if __name__ == "__main__":
    main()