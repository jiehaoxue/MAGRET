import os
import xml.etree.ElementTree as ET
import random
from itertools import combinations
import requests
import json
from tqdm import tqdm
from collections import defaultdict
import os
from openai import OpenAI
import base64
    
import time
import random
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# 添加这个全局变量来跟踪最近的API调用时间
_last_api_call_time = 0

def generate_description_with_ai(objects_info, image_path=None, image_width=None, image_height=None):
    """
    使用百度AI Studio API生成对象组的描述
    """
    try:
        # 准备提示信息
        objects_list = []
        positions_list = []
        
        for obj in objects_info:
            objects_list.append(obj['name'])
            positions_list.append(f"{obj['name']}: ({obj['xmin']}, {obj['ymin']}, {obj['xmax']}, {obj['ymax']})")
        
        # # # 检查图像路径
        # if image_path:
        #     if os.path.exists(image_path):
        #         print(f"Image exists: {image_path}")
        #     else:
        #         print(f"Image not found: {image_path}")
        
        # 构建提示文本

        prompt_text = f"""
        This image size is [{image_width}x{image_height}]
        Generate a concise description for a group of objects in a satellite/aerial image.
        
        Below I provide the name and position information , corresponding as follows:
        Objects in this group: {', '.join(objects_list)}
        Object positions: {', '.join(positions_list)}
        The quantity and coordinates of the objects first depend on the list I provide you.for example,I send you 3 objects,you can't say there are 4 objects.make sure the number of objects in the description is consistent with the number I provide.
        The description should be brief and focus on the common characteristics.no more than 30 words.
        Do not include specific distance descriptions, and keep the language relatively coarse-grained.
        The Objects must be the subject of the sentence.
        there is only one object in this description,you can use "a" or "one" to express the quantity.
        """
        # 设置API客户端  
        # 百度api：bce-v3/ALTAK-QqBbIIzV7vb53KUBz9uMs/0d3a94f1218a83c0137f4711554006d9ee12faa1    
        # ernie-4.5-8k-preview  
        #https://qianfan.baidubce.com/v2
        #阿里api：sk-472810fbb2f1461ca496f7e612d1d8d6
        #阿里base：https://dashscope.aliyuncs.com/compatible-mode/v1
        #阿里：qwen-vl-max
        try:
            client = OpenAI(
                api_key="sk-472810fbb2f1461ca496f7e612d1d8d6",
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",#https://aistudio.baidu.com/llm/lmapi/v3
            )
        except Exception as e:
            print(f"Error initializing OpenAI client: {e}")
            return "A group of objects in the image."
        
        messages = []
        
        # 如果有图像路径，添加图像
        try:
            if image_path and os.path.exists(image_path):
                try:
                    base64_image = encode_image(image_path)
                    messages.append({
                        'role': 'user', 
                        'content': [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt_text
                            }
                        ]   
                    })
                except Exception as e:
                    print(f"Error encoding image: {e}")
                    # 如果图像编码失败，回退到纯文本
                    messages.append({
                        'role': 'user',
                        'content': prompt_text
                    })
            else:
                # 如果没有图像，只使用文本
                messages.append({
                    'role': 'user',
                    'content': prompt_text
                })
        except Exception as e:
            print(f"Error preparing messages: {e}")
            return "A group of objects in the image."
        
        # 调用API
        try:
            print(f"Sending request to AI with {len(objects_info)} objects")
            completion = client.chat.completions.create(
                model="qwen-vl-max",
                messages=messages,
                stream=False,
                timeout=30,  # 添加超时设置
            )
            
            # 获取生成的描述
            description = completion.choices[0].message.content
            print(f"AI response: {description[:90]}...")  # 打印部分响应
            return description
        except Exception as e:
            print(f"Error during API call: {e}")
            # print(f"Request payload: {messages}")
            return "A group of objects in the image."
        
    except Exception as e:
        print(f"Unexpected error in generate_description_with_ai: {e}")
        import traceback
        traceback.print_exc()  # 打印完整的堆栈跟踪
        return "A group of objects in the image."


def process_xml_file(xml_path, output_path):
    """处理单个XML文件，添加groups属性"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # 获取图像尺寸
        image_width = int(root.find('./size/width').text)
        image_height = int(root.find('./size/height').text)
        
        # 从XML文件中获取图像文件名
        filename_elem = root.find('./filename')
        if filename_elem is not None:
            image_filename = filename_elem.text
            # 构建完整的图像路径（假设图像在JPEGImages目录下）
            image_path = os.path.join('/home/xjh/RSVG-xjh/OPT-RSVG-main/data/opt_rsvg/Image/Image', image_filename)
        else:
            image_path = None
        
        # 1. 为每个object添加ID
        objects_info = []
        for i, obj in enumerate(root.findall('./object')):
            obj_id = f"obj_{i+1}"
            obj.set('id', obj_id)
            
            # 收集对象信息用于后续生成描述
            name = obj.find('name').text
            bbox = obj.find('bndbox')
            obj_info = {
                'id': obj_id,
                'name': name,
                'xmin': bbox.find('xmin').text,
                'ymin': bbox.find('ymin').text,
                'xmax': bbox.find('xmax').text,
                'ymax': bbox.find('ymax').text
            }
            objects_info.append(obj_info)
        
        # 2. 创建groups元素
        groups_elem = ET.SubElement(root, 'groups')
        
        # 3. 创建包含所有对象的组
        if len(objects_info) > 0:
            all_group = ET.SubElement(groups_elem, 'group')
            all_group.set('id', 'group_all')
            
            for obj_info in objects_info:
                obj_ref = ET.SubElement(all_group, 'object_ref')
                obj_ref.text = obj_info['id']
            
            # 为"all"组生成描述
            description = generate_description_with_ai(objects_info, image_path, image_width, image_height)
            desc_elem = ET.SubElement(all_group, 'description')
            desc_elem.text = description

        # 4. 按对象类型创建组
        # 将对象按类型分组
        objects_by_type = defaultdict(list)
        for obj_info in objects_info:
            objects_by_type[obj_info['name']].append(obj_info)
        
        # 为每种类型创建一个组（如果该类型有多个对象）
        for obj_type, type_objects in objects_by_type.items():
            if len(type_objects) > 1:  # 只有当有多个相同类型的对象时才创建组
                type_group = ET.SubElement(groups_elem, 'group')
                type_group.set('id', f'group_{obj_type.replace(" ", "_")}')
                
                for obj_info in type_objects:
                    obj_ref = ET.SubElement(type_group, 'object_ref')
                    obj_ref.text = obj_info['id']
                
                description = generate_description_with_ai(type_objects, image_path, image_width, image_height)
                desc_elem = ET.SubElement(type_group, 'description')
                desc_elem.text = description
        
        # 5. 基于空间关系创建组
        for obj in objects_info:
            obj['center_x'] = (int(obj['xmin']) + int(obj['xmax'])) / 2
            obj['center_y'] = (int(obj['ymin']) + int(obj['ymax'])) / 2
            
        # 定义区域边界
        # 左右侧：
        left_right_boundary = image_width / 2

        # 上下面：
        top_bottom_boundary = image_height / 2

        # 创建四个区域的对象组
        left_objects = []    # 左侧对象
        right_objects = []   # 右侧对象
        top_objects = []     # 上面对象
        bottom_objects = []  # 下面对象
        
        # 将对象分配到对应区域
        for obj in objects_info:
            x = obj['center_x']
            y = obj['center_y']
            
            # 左侧区域
            if x < left_right_boundary:
                left_objects.append(obj)
            
            # 右侧区域
            if x > left_right_boundary:
                right_objects.append(obj)
            
            # 上面区域
            if y < top_bottom_boundary:
                top_objects.append(obj)
            
            # 下面区域
            if y > top_bottom_boundary:
                bottom_objects.append(obj)
                
        # 为每个非空区域创建组（至少有1个对象）
        # 左侧组
        if len(left_objects) > 1:
            left_group = ET.SubElement(groups_elem, 'group')
            left_group.set('id', 'group_left')
            
            for obj_info in left_objects:
                obj_ref = ET.SubElement(left_group, 'object_ref')
                obj_ref.text = obj_info['id']
            
            description = generate_description_with_ai(left_objects, image_path, image_width, image_height)
            desc_elem = ET.SubElement(left_group, 'description')
            desc_elem.text = description
            
        # 右侧组
        if len(right_objects) > 1:
            right_group = ET.SubElement(groups_elem, 'group')
            right_group.set('id', 'group_right')
            
            for obj_info in right_objects:
                obj_ref = ET.SubElement(right_group, 'object_ref')
                obj_ref.text = obj_info['id']
            
            description = generate_description_with_ai(right_objects, image_path, image_width, image_height)
            desc_elem = ET.SubElement(right_group, 'description')
            desc_elem.text = description
            
        # 上面组
        if len(top_objects) > 1:
            top_group = ET.SubElement(groups_elem, 'group')
            top_group.set('id', 'group_top')
            
            for obj_info in top_objects:
                obj_ref = ET.SubElement(top_group, 'object_ref')
                obj_ref.text = obj_info['id']
            
            description = generate_description_with_ai(top_objects, image_path, image_width, image_height)
            desc_elem = ET.SubElement(top_group, 'description')
            desc_elem.text = description
            
        # 下面组
        if len(bottom_objects) > 1:
            bottom_group = ET.SubElement(groups_elem, 'group')
            bottom_group.set('id', 'group_bottom')
            
            for obj_info in bottom_objects:
                obj_ref = ET.SubElement(bottom_group, 'object_ref')
                obj_ref.text = obj_info['id']
            
            description = generate_description_with_ai(bottom_objects, image_path, image_width, image_height)
            desc_elem = ET.SubElement(bottom_group, 'description')
            desc_elem.text = description

        # 保存修改后的XML
        def indent(elem, level=0):
            """为XML元素添加缩进"""
            i = "\n" + level*"  "
            if len(elem):
                if not elem.text or not elem.text.strip():
                    elem.text = i + "  "
                if not elem.tail or not elem.tail.strip():
                    elem.tail = i
                for child in elem:
                    indent(child, level+1)
                if not child.tail or not child.tail.strip():
                    child.tail = i
            else:
                if level and (not elem.tail or not elem.tail.strip()):
                    elem.tail = i
        
        # 添加缩进
        indent(root)
        # 写入文件
        tree.write(output_path, encoding='utf-8', xml_declaration=False)
        return True
        
    except Exception as e:
        print(f"Error processing {xml_path}: {e}")
        return False

# ...（上面的 import 和函数都不用动）...

def process_dataset(input_dir, output_dir, xml_list_txt=None, start_from=None):
    """
    只处理txt指定的xml文件
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 加载 txt 里的文件名
    if xml_list_txt:
        with open(xml_list_txt, 'r', encoding='utf-8') as fr:
            # 去掉换行和空格
            xml_files = [line.strip() for line in fr if line.strip()]
    else:
        xml_files = [f for f in os.listdir(input_dir) if f.endswith('.xml')]
        xml_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    
    # 可选起始
    start_index = 0
    if start_from:
        try:
            start_index = xml_files.index(start_from)
            print(f"Starting from file {start_from} (index {start_index})")
        except ValueError:
            print(f"Warning: File {start_from} not found in txt list. Starting from the beginning.")

    files_to_process = xml_files[start_index:]
    print(f"Found {len(xml_files)} XML files to process, will process {len(files_to_process)} files")
    
    success_count = 0
    for xml_file in tqdm(files_to_process):
        input_path = os.path.join(input_dir, xml_file)
        output_path = os.path.join(output_dir, xml_file)
        
        if not os.path.exists(input_path):
            print(f"[WARN] {input_path} not found, skip")
            continue
        
        if process_xml_file(input_path, output_path):
            success_count += 1
    
    print(f"Successfully processed {success_count} out of {len(files_to_process)} files")

if __name__ == "__main__":
    # 设置输入/输出目录
    input_directory = "/home/xjh/RSVG-xjh/OPT-RSVG-main/data/opt_rsvg/Annotations/Annotations"
    output_directory = "/home/xjh/RSVG-xjh/OPT-RSVG-main/data/opt_rsvg/Annotations_modified"
    # txt路径
    xml_txt = "/home/xjh/RSVG-xjh/OPT-RSVG-main/wrong_descriptions.txt"

    process_dataset(input_directory, output_directory, xml_list_txt=xml_txt, start_from=None)
