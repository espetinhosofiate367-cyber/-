# 智能医疗检测系统 - 重新组织版

## 项目概述

本项目是一个高性能的肺结节检测系统，基于多传感器融合技术，提供实时检测、可视化和分析功能。系统已重新组织文件结构，提高了代码的可维护性和可扩展性。

## 文件结构

```
智能医疗检测系统/
├── 启动程序.py          # 系统启动脚本
├── README.md            # 本说明文档
├── 核心程序/            # 主程序文件
│   └── modern_detection_gui_optimized.py
├── 检测算法/            # 检测算法模块
│   ├── fusion_real_time_detection.py      # 融合实时检测
│   ├── enhanced_stress_detection_system.py # 增强应力检测系统
│   ├── suretouch_elastography_system.py   # SureTouch弹性成像系统
│   ├── enhanced_detection_system.py       # 增强检测系统
│   ├── advanced_nodule_probability_system.py # 高级结节概率系统
│   ├── algorithms.py                      # 基础算法
│   └── realtime_correction_visualization.py # 实时校正可视化
├── 界面模块/            # GUI界面模块
│   ├── modern_detection_gui.py           # 现代检测GUI
│   ├── modern_detection_gui_fixed.py     # 修复版GUI
│   ├── integrated_gui.py                 # 集成GUI
│   ├── main_detection_app.py             # 主检测应用
│   └── optimized_serial_monitor.py       # 优化串口监视器
├── 配置文件/            # 配置和设置文件
│   ├── detection_config.json             # 检测配置
│   ├── optimization_cache.json           # 优化缓存
│   ├── requirements.txt                  # Python依赖
│   └── requirements_enhanced.txt         # 增强版依赖
├── 数据文件/            # 数据和模型文件
│   ├── pulmonary_nodule_training_system.pkl # 训练系统数据
│   └── performance_comparison_results.csv   # 性能对比结果
├── 测试结果/            # 测试结果和图表
│   ├── *.png                             # 测试结果图表
│   ├── *.json                            # 测试报告
│   ├── performance_comparison_test.py    # 性能对比测试
│   └── statistical_analysis.py          # 统计分析
├── 文档资料/            # 文档和说明
│   ├── README.md                         # 原始说明文档
│   ├── README_Enhanced_System.md         # 增强系统说明
│   ├── performance_analysis_report.md    # 性能分析报告
│   └── fusion_detection_package/         # 融合检测包文档
├── 实验数据/            # 实验数据和媒体文件
│   └── 视频图片/                         # 实验视频和图片
└── 缓存文件/            # 缓存和临时文件
    └── __pycache__/                      # Python缓存
```

## 快速启动

### 方法1：使用启动脚本（推荐）
```bash
cd 智能医疗检测系统
python 启动程序.py
```

### 方法2：直接运行主程序
```bash
cd 智能医疗检测系统/核心程序
python modern_detection_gui_optimized.py
```

## 系统特性

### 核心功能
- **实时数据采集**：支持串口通信，实时获取传感器数据
- **多模态检测**：融合应力传感器、弹性成像等多种检测方式
- **智能算法**：基于机器学习的结节检测和分类
- **可视化界面**：直观的GUI界面，支持多种视图模式
- **性能优化**：高效的数据处理和缓存机制

### 检测算法
1. **融合实时检测**：多传感器数据融合处理
2. **增强应力检测**：基于LSTM的时序分析
3. **弹性成像系统**：SureTouch技术集成
4. **概率评估系统**：智能概率计算和风险评估

### 界面特性
- 现代化GUI设计
- 实时数据可视化
- 多视图切换（原始数据、检测结果、3D视图、热力图）
- 参数调节和系统配置
- 性能监控和统计分析

## 依赖环境

### 基础依赖
- Python 3.7+
- tkinter（GUI框架）
- numpy（数值计算）
- pandas（数据处理）
- matplotlib（数据可视化）
- scikit-learn（机器学习）
- serial（串口通信）

### 增强功能依赖
- tensorflow（深度学习）
- opencv-python（图像处理）
- scipy（科学计算）

详细依赖列表请参考 `配置文件/requirements.txt`

## 使用说明

1. **系统启动**：运行启动脚本或主程序
2. **串口连接**：选择正确的串口并连接设备
3. **参数配置**：根据需要调整检测参数
4. **开始检测**：点击开始按钮进行实时检测
5. **结果分析**：查看检测结果和统计数据

## 开发说明

### 文件组织原则
- **功能分离**：按功能模块组织文件
- **层次清晰**：核心程序、算法、界面分层管理
- **易于维护**：相关文件集中存放，便于维护和更新

### 路径管理
- 所有模块使用相对路径引用
- 自动处理跨文件夹的模块导入
- 数据文件统一存放在指定目录

### 扩展开发
- 新算法添加到 `检测算法/` 目录
- 新界面模块添加到 `界面模块/` 目录
- 配置文件统一管理在 `配置文件/` 目录

## 版本信息

- **版本**：2.0（重新组织版）
- **更新日期**：2025年1月
- **主要改进**：
  - 重新组织文件结构
  - 优化模块导入机制
  - 改进路径管理
  - 增强系统可维护性

## 技术支持

如有问题或建议，请参考：
- `文档资料/` 目录下的详细文档
- `测试结果/` 目录下的测试报告
- 系统日志和错误信息

---

**注意**：本系统为医疗辅助检测工具，检测结果仅供参考，不能替代专业医疗诊断。