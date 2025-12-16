import json
import math
import matplotlib.pyplot as plt
# ========== 参数区 ==========
log_path = "/home/xjh/RSVG-xjh/OPT-RSVG-main/output/5.14-14:28/log.txt"
save_path = "/home/xjh/RSVG-xjh/OPT-RSVG-main/output/log_subplots.png"
fields = [
    "train_loss_bbox",
    "train_loss_giou",
    "validation_precision",
    "validation_recall",
    "validation_f1_score"
]
# ===========================
# 1. 读取日志并收集数据
epoch_list = []
field_vals = {k: [] for k in fields}
with open(log_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            d = json.loads(line)
            epoch_list.append(d['epoch'])
            for k in fields:
                field_vals[k].append(d.get(k, None))
        except Exception as e:
            print("skip error:", line)
            continue
# 2. 计算subplot布局（每行俩）
n_fields = len(fields)
n_cols = 2
n_rows = math.ceil(n_fields / n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3*n_rows))
if n_rows == 1:
    axes = axes.reshape((1, -1))  # 保证二维
for idx, name in enumerate(fields):
    row = idx // n_cols
    col = idx % n_cols
    ax = axes[row, col]
    ax.plot(epoch_list, field_vals[name], label=name, color='C0')
    ax.set_title(name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.grid(True)
    ax.legend()
# 若最后一行有空白子图，隐藏
if n_fields % n_cols != 0:
    for empty in range(n_fields, n_rows*n_cols):
        row = empty // n_cols
        col = empty % n_cols
        axes[row, col].set_visible(False)
plt.tight_layout()
plt.savefig(save_path)
plt.close()
print(f"多子图文件已保存到: {save_path}")