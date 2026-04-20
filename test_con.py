import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------- 1. 配置全局字体（Times New Roman） --------------------------
plt.rcParams['font.family'] = 'Times New Roman'  # 设置全局字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题（若系统无中文字体）


# -------------------------- 2. 定义类别标签和混淆矩阵数据 --------------------------
labels = [
    'pedestrian', 'people', 'bicycle', 'car', 'van',
    'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor', 'background'
]

# 混淆矩阵（行：Predicted；列：True）
conf_matrix = np.array([
    # pedestrian  people  bicycle   car    van   truck  tricycle  awning-tricycle   bus   motor  background
    [0.40,       0.06,    0.02,    0.00,  0.00,  0.00,    0.00,         0.00,      0.00,  0.02,      0.21],  # pedestrian
    [0.03,       0.31,    0.00,    0.00,  0.00,  0.00,    0.00,         0.00,      0.00,  0.02,      0.11],  # people
    [0.00,       0.00,    0.14,    0.00,  0.00,  0.00,    0.00,         0.00,      0.00,  0.02,      0.03],  # bicycle
    [0.00,       0.00,    0.00,    0.79,  0.37,  0.15,    0.06,         0.13,      0.03,  0.00,      0.40],  # car
    [0.00,       0.00,    0.00,    0.02,  0.37,  0.06,    0.00,         0.02,      0.03,  0.00,      0.07],  # van
    [0.00,       0.00,    0.00,    0.00,  0.01,  0.27,    0.00,         0.00,      0.08,  0.00,      0.02],  # truck
    [0.00,       0.00,    0.00,    0.00,  0.00,  0.01,    0.18,         0.09,      0.00,  0.00,      0.02],  # tricycle
    [0.00,       0.00,    0.00,    0.00,  0.00,  0.01,    0.04,         0.11,      0.00,  0.00,      0.01],  # awning-tricycle
    [0.00,       0.00,    0.00,    0.00,  0.00,  0.02,    0.00,         0.00,      0.44,  0.00,      0.01],  # bus
    [0.00,       0.01,    0.09,    0.00,  0.00,  0.00,    0.08,         0.03,      0.00,  0.41,      0.12],  # motor
    [0.57,       0.62,    0.74,    0.19,  0.24,  0.47,    0.62,         0.61,      0.42,  0.53,      0.00]   # background
])


# -------------------------- 3. 绘制混淆矩阵（放大字号） --------------------------
plt.figure(figsize=(14, 12))  # 放大画布防止标签拥挤

# 绘制热力图：annot显示数值，annot_kws放大注释字号，xtick/ytick调整标签字号
ax = sns.heatmap(
    conf_matrix,
    annot=True,          # 显示单元格数值
    fmt='.2f',           # 数值格式（保留2位小数）
    cmap='Blues',        # 配色（与原图一致）
    xticklabels=labels,  # 横轴（True Label）
    yticklabels=labels,  # 纵轴（Predicted Label）
    annot_kws={'size': 12},  # 放大数值字号
    cbar_kws={'label': 'Normalized Value'}  # 颜色条标签
)

# 调整坐标轴标签字号
ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)

# 调整标题、坐标轴名称字号
ax.set_title('Confusion Matrix Normalized', fontsize=14)
ax.set_xlabel('True Label', fontsize=12)
ax.set_ylabel('Predicted Label', fontsize=12)

# 调整颜色条的标签和刻度字号
cbar = ax.collections[0].colorbar
cbar.ax.set_ylabel('Normalized Value', fontsize=12)
cbar.ax.tick_params(labelsize=12)

plt.tight_layout()  # 自动调整布局（防止标签重叠）
plt.show()