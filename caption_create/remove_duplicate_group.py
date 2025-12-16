import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
from collections import defaultdict

def remove_duplicate_groups(xml_path, output_path=None):
    """
    移除XML文件中包含完全相同对象的重复组
    
    参数:
        xml_path: 输入XML文件路径
        output_path: 输出XML文件路径，如果为None则覆盖原文件
    
    返回:
        bool: 处理是否成功
    """
    try:
        # 解析XML文件
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # 查找groups元素
        groups_elem = root.find('groups')
        if groups_elem is None:
            print(f"No groups found in {xml_path}")
            return True  # 没有组，视为成功处理
        
        # 存储每个组包含的对象引用
        group_objects = {}
        # 存储要保留的组
        groups_to_keep = []
        # 存储组ID到组元素的映射
        group_elements = {}
        
        # 首先收集所有组及其对象
        for group in groups_elem.findall('group'):
            group_id = group.get('id')
            if group_id is None:
                continue
                
            # 收集该组引用的所有对象ID
            object_refs = []
            for obj_ref in group.findall('object_ref'):
                obj_id = obj_ref.text
                if obj_id:
                    object_refs.append(obj_id)
            
            # 对对象引用排序以便比较
            object_refs.sort()
            
            # 将对象引用列表转换为元组，以便用作字典键
            object_refs_tuple = tuple(object_refs)
            group_objects[group_id] = object_refs_tuple
            group_elements[group_id] = group
        
        # 找出要保留的组（去除重复）
        seen_object_sets = {}
        for group_id, objects in group_objects.items():
            if objects not in seen_object_sets:
                seen_object_sets[objects] = group_id
                groups_to_keep.append(group_id)
        
        # 统计要移除的组
        groups_to_remove = set(group_objects.keys()) - set(groups_to_keep)
        if groups_to_remove:
            print(f"Removing {len(groups_to_remove)} duplicate groups from {xml_path}")
            for group_id in groups_to_remove:
                groups_elem.remove(group_elements[group_id])
        
        for group_id in groups_to_keep:
            group = group_elements[group_id]
            # 移除id属性
            if 'id' in group.attrib:
                del group.attrib['id']
        # 保存修改后的XML
        if output_path is None:
            output_path = xml_path
            
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
        return True
        
    except Exception as e:
        print(f"Error processing {xml_path}: {e}")
        return False

def process_directory(input_dir, output_dir=None, start_from=None):
    """
    处理目录中的所有XML文件，移除重复组
    
    参数:
        input_dir: 输入目录
        output_dir: 输出目录，如果为None则覆盖原文件
        start_from: 可选，指定从哪个文件名开始处理（包含该文件）
    """
    if not os.path.exists(input_dir):
        print(f"Input directory {input_dir} does not exist")
        return
        
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 获取所有XML文件
    xml_files = [f for f in os.listdir(input_dir) if f.endswith('.xml')]
    xml_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    
    # 如果指定了起始文件，找到对应的索引位置
    start_index = 0
    if start_from:
        try:
            start_index = xml_files.index(start_from)
            print(f"Starting from file {start_from} (index {start_index})")
        except ValueError:
            print(f"Warning: File {start_from} not found. Starting from the beginning.")
    
    # 只处理起始索引之后的文件
    files_to_process = xml_files[start_index:]
    print(f"Found {len(xml_files)} total XML files, will process {len(files_to_process)} files")
    
    success_count = 0
    duplicate_groups_removed = 0
    
    for xml_file in tqdm(files_to_process):
        input_path = os.path.join(input_dir, xml_file)
        
        if output_dir:
            output_path = os.path.join(output_dir, xml_file)
        else:
            output_path = None  # 覆盖原文件
            
        # 处理前统计组数
        tree_before = ET.parse(input_path)
        root_before = tree_before.getroot()
        groups_before = root_before.find('groups')
        group_count_before = len(groups_before.findall('group')) if groups_before is not None else 0
        
        # 处理文件
        if remove_duplicate_groups(input_path, output_path):
            success_count += 1
            
            # 处理后统计组数
            if output_path is None:
                output_path = input_path
            tree_after = ET.parse(output_path)
            root_after = tree_after.getroot()
            groups_after = root_after.find('groups')
            group_count_after = len(groups_after.findall('group')) if groups_after is not None else 0
            
            # 计算移除的组数
            groups_removed = group_count_before - group_count_after
            duplicate_groups_removed += groups_removed
            
            if groups_removed > 0:
                print(f"Removed {groups_removed} duplicate groups from {xml_file}")
    
    print(f"Successfully processed {success_count} out of {len(files_to_process)} files")
    print(f"Total duplicate groups removed: {duplicate_groups_removed}")

# 使用示例
if __name__ == "__main__":
    # 处理单个文件
    # remove_duplicate_groups("path/to/file.xml")
    
    # 处理整个目录
    process_directory(
        input_dir="/home/xjh/RSVG-xjh/OPT-RSVG-main/data/opt_rsvg/Annotations_modified",
        output_dir="/home/xjh/RSVG-xjh/OPT-RSVG-main/data/opt_rsvg/Annotations_modified",  # 如果不想覆盖原文件，指定输出目录
        start_from=None  # 可选，指定从哪个文件开始处理
    )
