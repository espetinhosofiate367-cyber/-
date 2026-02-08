import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
from 察结 import StressAnalyzer
from algorithms import create_nodule_evolution_gif
from 统计 import main as stats_main


class IntegratedAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("多功能分析平台")
        self.root.geometry("1200x800")

        # 创建选项卡容器
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(expand=True, fill="both")

        # 初始化三个模块的Frame
        self.create_stress_tab()
        self.create_nodule_tab()
        self.create_stats_tab()

        # 数据存储
        self.stress_df = None
        self.nodule_df = None
        self.stats_folder = None

    def create_stress_tab(self):
        """应力分析模块"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="应力分析")

        # 文件选择区域
        file_frame = ttk.LabelFrame(frame, text="数据文件")
        file_frame.pack(pady=10, padx=10, fill="x")

        ttk.Button(file_frame, text="选择CSV", command=self.load_stress_file).pack(side="left", padx=5)
        self.stress_file_label = ttk.Label(file_frame, text="未选择文件")
        self.stress_file_label.pack(side="left", padx=5)

        # 参数设置区域
        param_frame = ttk.LabelFrame(frame, text="动画参数")
        param_frame.pack(pady=10, padx=10, fill="x")

        ttk.Label(param_frame, text="帧数:").pack(side="left", padx=5)
        self.stress_frame_entry = ttk.Entry(param_frame, width=10)
        self.stress_frame_entry.insert(0, "50")
        self.stress_frame_entry.pack(side="left", padx=5)

        # 手动参数输入区域
        manual_frame = ttk.LabelFrame(frame, text="手动参数")
        manual_frame.pack(pady=10, padx=10, fill="x")

        ttk.Label(manual_frame, text="MAT列数:").pack(side="left", padx=5)
        self.mat_cols_entry = ttk.Entry(manual_frame, width=10)
        self.mat_cols_entry.pack(side="left", padx=5)

        ttk.Label(manual_frame, text="SN列名:").pack(side="left", padx=5)
        self.sn_col_entry = ttk.Entry(manual_frame, width=10)
        self.sn_col_entry.pack(side="left", padx=5)

        # 执行按钮
        ttk.Button(frame, text="生成应力动画", command=self.generate_stress_animation).pack(pady=10)

    def create_nodule_tab(self):
        """结节检测模块"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="结节检测")

        # 文件控制区域
        file_frame = ttk.LabelFrame(frame, text="检测文件")
        file_frame.pack(pady=10, padx=10, fill="x")

        ttk.Button(file_frame, text="选择数据文件", command=self.load_nodule_file).pack(side="left", padx=5)
        self.nodule_file_label = ttk.Label(file_frame, text="未选择文件")
        self.nodule_file_label.pack(side="left", padx=5)

        # 添加重置按钮
        ttk.Button(file_frame, text="重置", command=self.reset_nodule_data).pack(side="left", padx=5)

        # 检测参数区域
        param_frame = ttk.LabelFrame(frame, text="检测参数")
        param_frame.pack(pady=10, padx=10, fill="x")

        ttk.Label(param_frame, text="检测帧数:").pack(side="left", padx=5)
        self.nodule_frame_entry = ttk.Entry(param_frame, width=10)
        self.nodule_frame_entry.insert(0, "30")
        self.nodule_frame_entry.pack(side="left", padx=5)

        # 执行按钮
        ttk.Button(frame, text="开始结节检测", command=self.run_nodule_detection).pack(pady=10)

    def reset_nodule_data(self):
        """重置结节检测数据"""
        self.nodule_df = None
        self.nodule_file_label.config(text="未选择文件")
        messagebox.showinfo("重置", "结节检测数据已重置")

    def run_nodule_detection(self):
        if self.nodule_df is None:
            messagebox.showwarning("警告", "请先选择数据文件")
            return

        try:
            frame_count = int(self.nodule_frame_entry.get() or 30)
            if frame_count <= 0 or frame_count > len(self.nodule_df):
                raise ValueError(f"请输入1-{len(self.nodule_df)}之间的有效帧数")

            output_path = filedialog.asksaveasfilename(
                defaultextension=".gif",
                filetypes=[("GIF动画", "*.gif")]
            )

            if output_path:
                # 检查数据有效性
                if len(self.nodule_df) == 0:
                    raise ValueError("数据为空，请检查输入文件")

                create_nodule_evolution_gif(self.nodule_df, output_path, frame_count)
                messagebox.showinfo("成功", f"结节检测完成！结果已保存至:\n{output_path}")

        except ValueError as e:
            messagebox.showerror("错误", str(e))
        except Exception as e:
            messagebox.showerror("错误", f"结节检测失败: {str(e)}")

    def create_stats_tab(self):
        """统计分析模块"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="统计分析")

        # 文件夹选择区域
        folder_frame = ttk.LabelFrame(frame, text="数据目录")
        folder_frame.pack(pady=10, padx=10, fill="x")

        ttk.Button(folder_frame, text="选择文件夹", command=self.select_stats_folder).pack(side="left", padx=5)
        self.stats_folder_label = ttk.Label(folder_frame, text="未选择目录")
        self.stats_folder_label.pack(side="left", padx=5)

        # 执行按钮
        ttk.Button(frame, text="执行统计分析", command=self.run_statistical_analysis).pack(pady=10)

    def load_stress_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV文件", "*.csv")])
        if file_path:
            try:
                self.stress_df = pd.read_csv(file_path)
                self.stress_file_label.config(text=os.path.basename(file_path))

                # 自动检测列名并填充到手动参数区域
                mat_cols = [col for col in self.stress_df.columns if col.startswith('MAT_')]
                if mat_cols:
                    self.mat_cols_entry.delete(0, tk.END)
                    self.mat_cols_entry.insert(0, str(len(mat_cols)))

                if 'SN' in self.stress_df.columns:
                    self.sn_col_entry.delete(0, tk.END)
                    self.sn_col_entry.insert(0, "SN")

                messagebox.showinfo("成功", "文件加载成功！")
            except Exception as e:
                messagebox.showerror("错误", f"加载文件失败: {str(e)}")
                self.stress_df = None

    def generate_stress_animation(self):
        if self.stress_df is None:
            messagebox.showwarning("警告", "请先选择数据文件")
            return

        try:
            frame_count = int(self.stress_frame_entry.get() or 50)
            if frame_count <= 0 or frame_count > len(self.stress_df):
                raise ValueError("无效帧数")

            # 获取手动参数
            mat_cols = int(self.mat_cols_entry.get() or 96)
            sn_col = self.sn_col_entry.get() or "SN"

            # 验证手动参数
            if not all(col in self.stress_df.columns for col in [f"MAT_{i}" for i in range(mat_cols)] + [sn_col]):
                raise ValueError("手动参数与数据列不匹配")

            output_path = filedialog.asksaveasfilename(
                defaultextension=".gif",
                filetypes=[("GIF动画", "*.gif")]
            )

            if output_path:
                analyzer = StressAnalyzer()
                analyzer.df = self.stress_df
                analyzer.create_stress_animation(output_path, frame_count)
                messagebox.showinfo("成功", f"应力动画已保存至:\n{output_path}")

        except ValueError as e:
            messagebox.showerror("错误", f"参数错误: {str(e)}")
        except Exception as e:
            messagebox.showerror("错误", f"生成动画失败: {str(e)}")

    def load_nodule_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV文件", "*.csv")])
        if file_path:
            try:
                # 重置当前数据
                self.nodule_df = None

                # 重新读取文件
                self.nodule_df = pd.read_csv(file_path)
                self.nodule_file_label.config(text=os.path.basename(file_path))
                messagebox.showinfo("成功", "文件加载成功！")
            except Exception as e:
                messagebox.showerror("错误", f"加载文件失败: {str(e)}")

    def select_stats_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.stats_folder = folder_path
            self.stats_folder_label.config(text=os.path.basename(folder_path))

    def run_statistical_analysis(self):
        if not self.stats_folder:
            messagebox.showwarning("警告", "请先选择数据目录")
            return

        try:
            # 检查目录有效性
            if not os.path.exists(self.stats_folder):
                raise ValueError("目录不存在，请重新选择")

            if not os.listdir(self.stats_folder):
                raise ValueError("所选目录为空，请检查数据文件")

            # 传递当前目录给统计模块
            from 统计 import main as stats_main
            stats_main(self.stats_folder)  # 直接传递文件夹路径

            messagebox.showinfo("成功", "统计分析完成！结果保存在analysis_results目录")

        except Exception as e:
            messagebox.showerror("错误", f"统计分析失败: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = IntegratedAnalysisApp(root)
    root.mainloop()
