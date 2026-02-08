#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
导出功能测试脚本
测试GIF导出和报告导出功能是否正常工作
"""

import os
import sys
import numpy as np
import pandas as pd
from enhanced_detection_system import EnhancedNoduleDetectionSystem

def create_test_data():
    """创建测试数据"""
    print("创建测试数据...")
    
    # 创建模拟的应力数据，符合系统要求的格式
    n_frames = 20
    
    # 创建测试数据
    test_data = []
    for frame in range(n_frames):
        # 创建一行数据，包含SN列和96个MAT列
        row_data = {'SN': frame}
        
        # 生成96个应力传感器的数据
        for i in range(96):
            # 模拟应力值，包含一些异常点（结节）
            base_stress = np.sin(i * 0.1 + frame * 0.05) * np.cos(i * 0.05 + frame * 0.1)
            
            # 在某些传感器位置添加异常值（模拟结节）
            if i in [20, 35, 60, 75]:  # 模拟结节位置
                stress = base_stress + 2.0 + np.random.normal(0, 0.1)
            else:
                stress = base_stress + np.random.normal(0, 0.1)
            
            row_data[f'MAT_{i}'] = stress
        
        test_data.append(row_data)
    
    return pd.DataFrame(test_data)

def test_gif_export():
    """测试GIF导出功能"""
    print("\n=== 测试GIF导出功能 ===")
    
    try:
        # 创建检测系统实例
        detector = EnhancedNoduleDetectionSystem()
        
        # 创建测试数据
        test_data = create_test_data()
        
        # 设置输出路径
        output_path = "test_export.gif"
        max_frames = 10
        
        print(f"开始生成GIF动画，输出路径: {output_path}")
        print(f"数据形状: {test_data.shape}")
        print(f"最大帧数: {max_frames}")
        
        # 调用GIF导出功能
        success = detector.create_enhanced_visualization(
            test_data, output_path, max_frames
        )
        
        if success:
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                print(f"✓ GIF导出成功！")
                print(f"  文件路径: {os.path.abspath(output_path)}")
                print(f"  文件大小: {file_size / 1024:.1f} KB")
                return True
            else:
                print("✗ GIF导出失败：文件未创建")
                return False
        else:
            print("✗ GIF导出失败：函数返回False")
            return False
            
    except Exception as e:
        print(f"✗ GIF导出测试出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_report_export():
    """测试报告导出功能"""
    print("\n=== 测试报告导出功能 ===")
    
    try:
        # 创建检测系统实例
        detector = EnhancedNoduleDetectionSystem()
        
        # 创建测试数据
        test_data = create_test_data()
        
        # 模拟检测结果 - 修复数据结构
        detector.nodule_history = {
            'timestamps': ['2024-01-01 10:00:00', '2024-01-01 10:00:05'],
            'count': [1, 1],
            'areas': [15.5, 12.3],
            'risk_scores': [0.8, 0.6]
        }
        
        # 测试TXT报告导出
        txt_output = "test_report.txt"
        print(f"测试TXT报告导出: {txt_output}")
        
        # 模拟报告内容
        report_content = detector.generate_analysis_report()
        
        with open(txt_output, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        if os.path.exists(txt_output):
            file_size = os.path.getsize(txt_output)
            print(f"✓ TXT报告导出成功！")
            print(f"  文件路径: {os.path.abspath(txt_output)}")
            print(f"  文件大小: {file_size} bytes")
        else:
            print("✗ TXT报告导出失败")
            return False
        
        # 测试CSV报告导出
        csv_output = "test_report.csv"
        print(f"测试CSV报告导出: {csv_output}")
        
        # 创建CSV数据 - 使用正确的数据结构
        csv_data = []
        for i, timestamp in enumerate(detector.nodule_history['timestamps']):
            csv_data.append({
                '序号': i + 1,
                '检测时间': timestamp,
                '结节数量': detector.nodule_history['count'][i],
                '结节面积': detector.nodule_history['areas'][i],
                '风险评分': detector.nodule_history['risk_scores'][i]
            })
        
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_output, index=False, encoding='utf-8-sig')
        
        if os.path.exists(csv_output):
            file_size = os.path.getsize(csv_output)
            print(f"✓ CSV报告导出成功！")
            print(f"  文件路径: {os.path.abspath(csv_output)}")
            print(f"  文件大小: {file_size} bytes")
            return True
        else:
            print("✗ CSV报告导出失败")
            return False
            
    except Exception as e:
        print(f"✗ 报告导出测试出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("开始导出功能测试...")
    print("=" * 50)
    
    # 测试结果
    gif_success = test_gif_export()
    report_success = test_report_export()
    
    # 总结
    print("\n" + "=" * 50)
    print("测试结果总结:")
    print(f"GIF导出功能: {'✓ 通过' if gif_success else '✗ 失败'}")
    print(f"报告导出功能: {'✓ 通过' if report_success else '✗ 失败'}")
    
    if gif_success and report_success:
        print("\n🎉 所有导出功能测试通过！")
        return True
    else:
        print("\n⚠️  部分导出功能存在问题，需要进一步检查")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)