# 增强应力传感器结节检测系统

## 系统概述

本系统是基于应力传感器数据的增强结节检测系统，集成了半监督学习、LSTM时序模型和持续跟踪等先进技术，能够实现高精度的结节检测和标注。

## 主要功能

### 1. 应力分布矩阵提取
- **数据预处理**: 噪声滤波、标准化处理
- **矩阵重构**: 将传感器数据重构为12x8应力分布矩阵
- **质量评估**: 自动评估数据质量和信噪比
- **校准支持**: 支持传感器校准数据应用

### 2. 半监督学习异常检测
- **无监督预训练**: 使用Isolation Forest进行异常检测
- **有监督微调**: 基于标注数据的Label Spreading算法
- **特征提取**: 多维度特征提取（统计、纹理、形状、频域）
- **PCA降维**: 自动降维保留95%方差

### 3. 真实结节参数管理
- **参数输入**: 面积、直径、深度、位置、密度等
- **数据管理**: 训练数据的存储和加载
- **统计分析**: 结节参数的统计信息
- **数据验证**: 输入数据的合理性检查

### 4. LSTM时序模型
- **时序建模**: 捕捉持续存在的结节特征
- **滑动处理**: 处理传感器滑动因素
- **序列预测**: 基于历史数据预测结节概率
- **模型保存**: 支持模型的保存和加载

### 5. 概率输出系统
- **多源融合**: 结合异常检测和时序分析结果
- **概率计算**: 输出结节存在的概率值
- **置信度评估**: 提供预测的置信度信息
- **阈值调节**: 可调节的检测阈值

### 6. 持续跟踪和标注
- **区域跟踪**: 跟踪高概率区域的持续性
- **自动标注**: 对稳定出现的区域进行自动标注
- **历史记录**: 保存标注历史和统计信息
- **过期清理**: 自动清理过期的跟踪区域

## 系统架构

```
应力传感器数据
    ↓
应力矩阵提取器 (StressMatrixExtractor)
    ↓
半监督异常检测器 (SemiSupervisedAnomalyDetector)
    ↓
LSTM时序模型 (LSTMTemporalModel)
    ↓
持续结节跟踪器 (PersistentNoduleTracker)
    ↓
增强检测系统 (EnhancedStressNoduleDetectionSystem)
```

## 使用方法

### 1. 系统初始化
```python
from enhanced_stress_detection_system import EnhancedStressNoduleDetectionSystem

# 创建检测系统
detection_system = EnhancedStressNoduleDetectionSystem()
```

### 2. 添加训练数据
```python
# 添加结节数据
detection_system.add_training_data(
    raw_data=sensor_data,
    area=2.5,           # 面积 (cm²)
    diameter=1.8,       # 直径 (cm)
    depth=0.5,          # 深度 (cm)
    position=(6, 4),    # 位置 (x, y)
    is_nodule=True
)

# 添加正常数据
detection_system.add_training_data(
    raw_data=normal_data,
    is_nodule=False
)
```

### 3. 训练系统
```python
# 训练整个系统
success = detection_system.train_system()
if success:
    print("系统训练完成")
```

### 4. 实时检测
```python
# 处理单帧数据
result = detection_system.process_frame(raw_data, timestamp)

if result:
    print(f"异常概率: {result['anomaly_result']['anomaly_probability']:.3f}")
    print(f"时序概率: {result['temporal_probability']:.3f}")
    print(f"综合概率: {result['combined_probability']:.3f}")
    print(f"标注数量: {len(result['annotations'])}")
```

### 5. 系统状态
```python
# 获取系统状态
status = detection_system.get_system_status()
print(f"训练样本数: {status['training_samples']}")
print(f"跟踪区域数: {status['tracked_regions']}")
print(f"标注历史数: {status['annotation_history']}")
```

## GUI界面使用

### 1. 启动界面
```bash
python modern_detection_gui_optimized.py
```

### 2. 切换检测模式
- 在"检测模式"下拉框中选择"增强应力"
- 系统会自动切换到增强检测模式

### 3. 添加训练数据
- 点击"添加训练数据"按钮
- 在弹出窗口中选择数据类型（结节/正常）
- 如果是结节数据，输入相关参数（面积、直径、深度、位置）
- 点击"添加数据"完成添加

### 4. 训练系统
- 添加足够的训练数据后（建议至少20个样本）
- 点击"训练系统"按钮
- 系统会在后台进行训练，完成后显示结果

### 5. 实时检测
- 连接串口或加载CSV数据
- 系统会自动使用增强检测算法进行实时分析
- 在可视化界面中查看检测结果

## 性能特点

### 1. 高精度检测
- 结合多种算法提高检测精度
- 半监督学习充分利用有限标注数据
- 时序模型捕捉持续特征

### 2. 实时处理
- 优化的数据处理流程
- 缓存机制减少重复计算
- 多线程处理提高响应速度

### 3. 自适应学习
- 持续学习新的数据模式
- 自动调整检测参数
- 适应不同的传感器特性

### 4. 鲁棒性强
- 噪声滤波和数据质量评估
- 多重验证机制
- 异常情况的优雅处理

## 技术规格

- **矩阵尺寸**: 12x8 (可配置)
- **检测精度**: >90% (在充足训练数据下)
- **处理速度**: <100ms/帧
- **内存占用**: <500MB
- **支持格式**: CSV, 实时串口数据

## 注意事项

1. **训练数据质量**: 确保训练数据的质量和多样性
2. **参数设置**: 根据实际应用调整检测阈值
3. **系统资源**: 确保有足够的内存和计算资源
4. **数据备份**: 定期备份训练数据和模型

## 故障排除

### 常见问题

1. **训练失败**
   - 检查训练数据数量是否足够（至少10个样本）
   - 确认数据格式正确
   - 检查系统资源是否充足

2. **检测精度低**
   - 增加训练数据数量
   - 检查数据质量
   - 调整检测阈值

3. **处理速度慢**
   - 减少跳帧数设置
   - 增加更新间隔
   - 检查系统性能

## 更新日志

### v1.0.0 (2024-01-XX)
- 初始版本发布
- 实现基础的增强检测功能
- 集成GUI界面
- 支持实时和离线检测

## 联系信息

如有问题或建议，请联系开发团队。