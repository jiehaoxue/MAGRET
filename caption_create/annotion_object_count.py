import os
import xml.etree.ElementTree as ET
from collections import defaultdict

def count_objects(xml_folder):
    """
    统计XML文件夹中各个object数量的分布
    :param xml_folder: 包含VOC格式XML文件的文件夹路径
    :return: 包含数量统计的字典（key: object数量，value: 对应的文件数）
    """
    count_dict = defaultdict(int)
    
    # 遍历文件夹中的所有文件
    for filename in os.listdir(xml_folder):
        if not filename.endswith('.xml'):
            continue
            
        file_path = os.path.join(xml_folder, filename)
        
        try:
            # 解析XML文件
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # 计算object数量
            objects = root.findall('object')
            num_objects = len(objects)
            
            # 更新统计字典
            count_dict[num_objects] += 1
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue
            
    return count_dict

def print_statistics(count_dict):
    """
    打印统计结果
    :param count_dict: 统计字典
    """
    print("XML文件object数量统计：")
    print("========================")
    total_files = sum(count_dict.values())
    print(f"总文件数：{total_files}\n")
    
    # 按object数量排序输出
    for count in sorted(count_dict.keys()):
        files = count_dict[count]
        percentage = (files / total_files) * 100
        print(f"包含 {count} 个object的文件: {files} 个 ({percentage:.1f}%)")

if __name__ == "__main__":
    # 修改为你的XML文件夹路径
    XML_FOLDER = "/home/xjh/RSVG-xjh/OPT-RSVG-main/data/opt_rsvg/Annotations/Annotations"   #opt-RSVG
    #XML_FOLDER = "/home/xjh/RSVG-xjh/RSVG-dataset/DIOR-RSVG/Annotations"  #DIOR-RSVG
    # 执行统计
    statistics = count_objects(XML_FOLDER)
    
    # 打印结果
    print_statistics(statistics)