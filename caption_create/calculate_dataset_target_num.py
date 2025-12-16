import os
import xml.etree.ElementTree as ET
from collections import Counter

def count_data_pairs(xml_dir):
    """
    统计所有xml里，各种目标数的数据对出现次数
    :param xml_dir: xml文件夹路径
    :return: dict {目标数n: n目标数据对数量}
    """
    per_n_counter = Counter()
    total = 0
    for fname in os.listdir(xml_dir):
        if not fname.endswith('.xml'):
            continue
        xml_path = os.path.join(xml_dir, fname)
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            # 1. 每个object都算1目标数据对
            obj_nodes = root.findall('./object')
            per_n_counter[1] += len(obj_nodes)
            total += len(obj_nodes)

            # 2. 每个group算作其object_ref个数目标的“n目标数据对”
            group_nodes = root.findall('.//group')
            for group in group_nodes:
                obj_refs = group.findall('./object_ref')
                n_objs = len(obj_refs)
                if n_objs > 0:
                    per_n_counter[n_objs] += 1
                    total += 1
        except ET.ParseError as e:
            print(f'ParseError in {fname}: {e}')
    return dict(per_n_counter), total

if __name__ == '__main__':
    xml_dir = '/home/xjh/RSVG-xjh/OPT-RSVG-main/data/opt_rsvg/Annotations_multi-target_v1/'  # 替换为你自己的路径
    res_dict, total = count_data_pairs(xml_dir)
    print(f"【总数据对数】:{total}")
    print("【每个数据对包含的目标数分布】:")
    for k in sorted(res_dict):
        print(f"{k}目标:\t{res_dict[k]} 个数据对")
