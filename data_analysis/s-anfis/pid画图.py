import numpy as np
import pandas as pd
# import torch # Not needed for this plotting script
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, AutoMinorLocator
import matplotlib.colors as mcolors
import copy
import os
from datetime import datetime
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# from sanfis import SANFIS # Not needed for plotting example
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# Set English font
import matplotlib as mpl

# ===== Matplotlib Configuration =====
mpl.rcParams['font.sans-serif'] = ['Helvetica', 'Helvetica World', 'Arial', 'Arial Unicode MS']
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
plt.rcParams['path.simplify'] = True
plt.rcParams['path.snap'] = True

# Create save directories
pic_dir = './pic'
if not os.path.exists(pic_dir):
    os.makedirs(pic_dir)

# #############################################################################
# # MODIFIED FUNCTION: plot_control_comparison
# # The rest of the functions (train_sanfis_model, controllers, etc.)
# # would be here in the full script.
# #############################################################################

def plot_control_comparison(comparison_results):
    """
    绘制PID控制对比图，应用最终的样式和标签要求。
    """

    results = comparison_results

    # 创建一个 1x2 的子图网格
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), dpi=300)

    # 2. 左图: H2S 浓度追踪对比
    ax1 = axes[0]
    ax1.plot(results['fuzzy_pid']['time_steps'], results['fuzzy_pid']['predicted_outputs'],
             'b-', linewidth=2, label='Fuzzy Supervised PID')
    ax1.plot(results['traditional_pid']['time_steps'], results['traditional_pid']['predicted_outputs'],
             'g-', linewidth=2, label='Traditional PID')
    ax1.axhline(y=results['setpoint'], color='r', linestyle='--',
                linewidth=2, label=f'Setpoint')

    # --- 应用最终风格 (左图) ---
    ax1.set_xlabel('Time Step', fontsize=22)
    # 1. 更换Y轴标签
    ax1.set_ylabel(r'$C_{\mathrm{out}} \, (\mathrm{mg/m}^3)$', fontsize=22)
    ax1.tick_params(axis='both', which='major', direction='out', length=6, color='black', labelsize=18)
    for spine in ax1.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.2)
    # 2. 恢复图例到原始设定
    ax1.legend(fontsize=18)
    ax1.grid(False)

    # 3. 右图: 转速控制对比
    ax2 = axes[1]
    ax2.plot(results['fuzzy_pid']['time_steps'], results['fuzzy_pid']['rotation_speeds'],
             'b-', linewidth=2, label='Fuzzy Supervised PID')
    ax2.plot(results['traditional_pid']['time_steps'], results['traditional_pid']['rotation_speeds'],
             'g-', linewidth=2, label='Traditional PID')

    # --- 应用最终风格 (右图) ---
    ax2.set_xlabel('Time Step', fontsize=22)
    ax2.set_ylabel('Rotation Speed (rpm)', fontsize=22)
    ax2.tick_params(axis='both', which='major', direction='out', length=6, color='black', labelsize=18)
    for spine in ax2.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.2)
    # 2. 恢复图例到原始设定 (即不显示右图的图例)
    ax2.grid(False)

    # 保留您满意的宽松间距设置
    plt.tight_layout(pad=3.0)

    # 保存为 PDF 和 PNG 格式
    plt.savefig("./pic/pid_control_comparison_final_tweaked.pdf", format='pdf', bbox_inches='tight')
    save_path = './pic/pid_control_comparison_final_tweaked.png'
    plt.savefig(save_path, format='png', bbox_inches='tight', dpi=300)
    plt.close()

    print(f"\n已应用最终微调的PID对比图已保存至: {save_path}")


def main():
    """Main function to generate the plot with sample data."""

    print("Generating plot with sample data that mimics the original image...")

    # Create dummy data for plotting
    time_steps = np.arange(50)
    setpoint = 1.0

    # Fuzzy PID data (blue line) - stable with small, quick corrections
    fuzzy_peak1 = 1.5 * np.exp(-0.2 * (time_steps - 28) ** 2)
    fuzzy_peak2 = 1.5 * np.exp(-0.2 * (time_steps - 47) ** 2)
    fuzzy_outputs = setpoint + fuzzy_peak1 + fuzzy_peak2 + np.random.normal(0, 0.05, 50)
    fuzzy_outputs[:23] = setpoint + np.random.normal(0, 0.02, 23)
    fuzzy_speeds = 55.0 - 0.2 * np.cumsum(np.exp(-0.1 * (time_steps - 26) ** 2))
    fuzzy_speeds -= 0.3 * np.cumsum(np.exp(-0.1 * (time_steps - 44) ** 2))
    fuzzy_speeds = np.clip(fuzzy_speeds, 54.7, 55.0)

    # Traditional PID data (green line) - larger overshoot and slower response
    trad_peak1 = 1.2 * np.exp(-0.2 * (time_steps - 28) ** 2)
    trad_peak2 = 6.5 * np.exp(-0.05 * (time_steps - 42) ** 2)
    trad_outputs = setpoint + trad_peak1 + trad_peak2 + np.random.normal(0, 0.1, 50)
    trad_outputs[:24] = setpoint + np.random.normal(0, 0.02, 24)
    trad_speeds = 55.0 - 1.8 * np.cumsum(np.exp(-0.1 * (time_steps - 35) ** 2))
    trad_speeds = np.clip(trad_speeds, 53.25, 55.0)

    dummy_results = {
        'fuzzy_pid': {
            'time_steps': time_steps,
            'predicted_outputs': fuzzy_outputs,
            'rotation_speeds': fuzzy_speeds
        },
        'traditional_pid': {
            'time_steps': time_steps,
            'predicted_outputs': trad_outputs,
            'rotation_speeds': trad_speeds
        },
        'setpoint': setpoint
    }

    plot_control_comparison(dummy_results)

    print(f"\n{'=' * 80}")
    print(f"Program execution complete!")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
