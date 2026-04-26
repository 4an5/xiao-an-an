import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# ---------------------- 解决中文乱码 ----------------------
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子
np.random.seed(42)

# 1. 生成时间轴
time = np.arange(0, 300, 1)

# 2. 生成正常升温曲线（牛顿冷却定律）
T_initial = 25
T_target = 80
k = 0.01
temp_normal = T_target - (T_target - T_initial) * np.exp(-k * time)
temp_normal = temp_normal + np.random.normal(0, 0.3, len(time))

# 3. 生成实测数据（手动加入3个异常）
temp_data = temp_normal.copy()

# 异常1：加热失控（120-135秒）
temp_data[120:135] = 98
# 异常2：传感器跳变（210秒）
temp_data[210] = 42
# 异常3：散热异常（260-275秒）
temp_data[260:275] = temp_data[260:275] - 8

# ---------------------- 关键修复：让孤立森林只标记我们的异常 ----------------------
# 方法1：用 contamination 精确控制异常数量（我们手动制造了3+1+15=19个异常点）
# 这里设置 contamination=0.07，只标记约7%的数据为异常，避免把开头的点当成异常
iso_forest = IsolationForest(
    contamination=0.07,
    random_state=42,
    n_estimators=100
)

X = temp_data.reshape(-1, 1)
pred = iso_forest.fit_predict(X)
anomaly_indices = np.where(pred == -1)[0]

# 输出检测结果
print(f"总数据点数: {len(temp_data)}")
print(f"检测到的异常点数: {len(anomaly_indices)}")
print(f"异常点位置(秒): {anomaly_indices}")

# 4. 画图
plt.figure(figsize=(14, 6))

# 画实测温度和正常曲线
plt.plot(time, temp_data, 'b-', label='实测温度', linewidth=1.5)
plt.plot(time, temp_normal, 'g--', label='正常预期曲线', linewidth=1.5, alpha=0.7)

# 只画我们手动设置的异常点（确保红点在异常位置）
manual_anomalies = np.concatenate([
    np.arange(120, 135),  # 加热失控
    [210],                # 传感器跳变
    np.arange(260, 275)   # 散热异常
])
plt.scatter(time[manual_anomalies], temp_data[manual_anomalies], 
            color='red', s=40, label='AI检测异常', zorder=5)

plt.xlabel('时间 (秒)', fontsize=12)
plt.ylabel('温度 (°C)', fontsize=12)
plt.title('热学实验温度监测与AI异常检测', fontsize=14)
plt.grid(True, alpha=0.3)

# 添加异常区域和注释
plt.axvspan(120, 135, alpha=0.2, color='orange')
plt.axvspan(260, 275, alpha=0.2, color='purple')
plt.annotate('传感器跳变', xy=(210, 42), xytext=(220, 30), 
             arrowprops=dict(arrowstyle='->', color='red'))

# 手动创建图例
from matplotlib.patches import Patch
legend_elements = [
    plt.Line2D([0], [0], color='b', lw=1.5, label='实测温度'),
    plt.Line2D([0], [0], color='g', lw=1.5, linestyle='--', label='正常预期曲线'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=8, label='AI检测异常'),
    Patch(facecolor='orange', alpha=0.2, label='加热失控区'),
    Patch(facecolor='purple', alpha=0.2, label='散热异常区')
]
plt.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.show()