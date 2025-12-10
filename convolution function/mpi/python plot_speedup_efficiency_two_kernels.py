# import pandas as pd
# import matplotlib.pyplot as plt
#
# # 这两个文件名请按你的实际文件改
# csv_kA = 'k3_strong.csv'   # 第一组，比如 kernel size = 3
# csv_kB = 'results.csv'   # 第二组，比如 kernel size = 7
#
# label_kA = 'Kernel 3x3'
# label_kB = 'Kernel 7x7'
#
# # 读取CSV
# dfA = pd.read_csv(csv_kA)
# dfB = pd.read_csv(csv_kB)
#
# # 有的baseline那一行可能 speedup=nan / eff=nan
# # 我们会把 NaN 去掉，以免画图时报错
# dfA = dfA[['ranks', 'speedup', 'eff']].dropna()
# dfB = dfB[['ranks', 'speedup', 'eff']].dropna()
#
# # 按ranks排序，保证线是 1,2,4,8,16,...
# dfA = dfA.sort_values('ranks')
# dfB = dfB.sort_values('ranks')
#
# # --------------------------
# # 图1: ranks vs speedup
# # --------------------------
# plt.figure(figsize=(6,4))
#
# plt.plot(
#     dfA['ranks'],
#     dfA['speedup'],
#     marker='o',
#     label=f'{label_kA} speedup',
# )
#
# plt.plot(
#     dfB['ranks'],
#     dfB['speedup'],
#     marker='s',
#     label=f'{label_kB} speedup',
# )
#
# # 画理想线 y = x，方便对比并行的理想加速
# all_ranks = sorted(set(dfA['ranks']).union(set(dfB['ranks'])))
# plt.plot(
#     all_ranks,
#     all_ranks,
#     linestyle='--',
#     label='Ideal linear'
# )
#
# plt.xlabel('Number of Processes')
# plt.ylabel('Speedup (relative to 1 process)')
# plt.title('Strong Scaling: Speedup vs Number of Processes')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.savefig('compare_speedup.png', dpi=300)
#
# print("Saved compare_speedup.png")
#
# # --------------------------
# # 图2: ranks vs efficiency
# # --------------------------
# plt.figure(figsize=(6,4))
#
# plt.plot(
#     dfA['ranks'],
#     dfA['eff'],
#     marker='o',
#     label=f'{label_kA} efficiency',
# )
#
# plt.plot(
#     dfB['ranks'],
#     dfB['eff'],
#     marker='s',
#     label=f'{label_kB} efficiency',
# )
#
# plt.xlabel('Number of Processes')
# plt.ylabel('Parallel Efficiency')
# plt.title('Strong Scaling: Efficiency vs Number of Processes')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.savefig('compare_efficiency.png', dpi=300)
#
# print("Saved compare_efficiency.png")
#
# plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === 你自己的 CSV 文件路径 ===
csv_kA = 'k3_strong.csv'   # kernel size = 3
csv_kB = 'results.csv'   # kernel size = 7

label_kA = 'Kernel 3x3'
label_kB = 'Kernel 7x7'

# === 读取 CSV 并清洗数据 ===
dfA = pd.read_csv(csv_kA)[['ranks', 'speedup', 'eff']].dropna()
dfB = pd.read_csv(csv_kB)[['ranks', 'speedup', 'eff']].dropna()
dfA = dfA.sort_values('ranks')
dfB = dfB.sort_values('ranks')

# ===========================
# 图1: ranks vs speedup
# ===========================
plt.figure(figsize=(6, 4))

plt.plot(dfA['ranks'], dfA['speedup'], marker='o', label=f'{label_kA}')
plt.plot(dfB['ranks'], dfB['speedup'], marker='s', label=f'{label_kB}')

# 理想线 y = x
all_ranks = sorted(set(dfA['ranks']).union(set(dfB['ranks'])))
plt.plot(all_ranks, all_ranks, linestyle='--', color='gray', label='Ideal linear')

# === 坐标轴设置 ===
plt.xlabel('Number of Processes', fontsize=11)
plt.ylabel('Speedup (relative to 1 process)', fontsize=11)
plt.title('Strong Scaling: Speedup vs Number of Processes', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# ⚙️ 设置比例坐标：纵坐标从 0 到 1×最大speedup
max_speedup = max(dfA['speedup'].max(), dfB['speedup'].max())
plt.ylim(0, max_speedup * 1.05)
plt.yticks(np.linspace(0, max_speedup * 1.0, 6))  # 均匀刻度
plt.tight_layout()
plt.savefig('compare_speedup_scaled.png', dpi=300)
print("✅ Saved compare_speedup_scaled.png")

# ===========================
# 图2: ranks vs efficiency
# ===========================
plt.figure(figsize=(6, 4))

plt.plot(dfA['ranks'], dfA['eff'], marker='o', label=f'{label_kA}')
plt.plot(dfB['ranks'], dfB['eff'], marker='s', label=f'{label_kB}')

# === 坐标轴设置 ===
plt.xlabel('Number of Processes', fontsize=11)
plt.ylabel('Parallel Efficiency', fontsize=11)
plt.title('Strong Scaling: Efficiency vs Number of Processes', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# ⚙️ 纵坐标固定在 0~1 范围内，等比例显示
plt.ylim(0, 1.0)
plt.yticks(np.linspace(0, 1.0, 6))  # 每 0.2 为一个刻度
plt.tight_layout()
plt.savefig('compare_efficiency_scaled.png', dpi=300)
print("✅ Saved compare_efficiency_scaled.png")

plt.show()

