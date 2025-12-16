import os
import glob
import random

# ========== 配置区 ==========
xml_dir = "/home/xjh/RSVG-xjh/OPT-RSVG-main/data/opt_rsvg/Annotations_multi-target_v1/"       # 你的xml文件夹路径
train_txt = "/home/xjh/RSVG-xjh/OPT-RSVG-main/data/opt_rsvg/split/split/train.txt"
val_txt = "/home/xjh/RSVG-xjh/OPT-RSVG-main/data/opt_rsvg/split/split/val.txt"
train_ratio = 0.8
# ===========================

# 1. 收集所有xml文件
xml_files = sorted(glob.glob(os.path.join(xml_dir, "*.xml")))

# 2. 提取编号（去掉xml后缀, 要去0前缀）
indices = []
for f in xml_files:
    base = os.path.basename(f)
    idx = os.path.splitext(base)[0]
    idx_noleading = str(int(idx))  # 去除前导0
    indices.append(idx_noleading)

# 3. 打乱
random.shuffle(indices)

# 4. 按8:2划分
n_train = int(len(indices) * train_ratio)
train_indices = indices[:n_train]
val_indices = indices[n_train:]

# 5. 写入文件
with open(train_txt, "w") as f:
    for idx in train_indices:
        f.write(f"{idx}\n")

with open(val_txt, "w") as f:
    for idx in val_indices:
        f.write(f"{idx}\n")

print(f"总数：{len(indices)}，训练集：{len(train_indices)}，验证集：{len(val_indices)}")
print("train.txt, val.txt 保存完成。")
