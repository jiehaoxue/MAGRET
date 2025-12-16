import os
import re
import xml.etree.ElementTree as ET

# 数字英文单词映射
NUM_WORDS = {
    'one': 1, 'two': 2, 'three': 3, 'four':4, 'five':5,
    'six':6, 'seven':7, 'eight':8, 'nine':9, 'ten':10
}

def get_first_number_word(desc):
    """提取首个单词或数字（首位）"""
    desc = desc.strip().lower()
    # 取首个单词
    match_word = re.match(r'^([a-z]+)', desc)
    if match_word:
        word = match_word.group(1)
        if word in NUM_WORDS:
            return NUM_WORDS[word]
    # 取首个数字
    match_num = re.match(r'^(\d+)', desc)
    if match_num:
        return int(match_num.group(1))
    return None

def find_wrong_xmls(input_dir, output_txt="wrong_descriptions.txt"):
    xml_files = [f for f in os.listdir(input_dir) if f.endswith('.xml')]
    wrong_files = set()
    for fname in xml_files:
        fpath = os.path.join(input_dir, fname)
        try:
            tree = ET.parse(fpath)
            root = tree.getroot()
            for groups in root.findall('groups'):
                for group in groups.findall('group'):
                    object_num = len(group.findall('object_ref'))
                    desc_elem = group.find('description')
                    if desc_elem is not None and desc_elem.text:
                        num = get_first_number_word(desc_elem.text)
                        if num is not None and num > object_num:
                            wrong_files.add(fname)
                            break # 已记录即可
        except Exception as e:
            print(f"Error parsing {fname}: {e}")
            continue

    # 保存到txt
    with open(output_txt, 'w', encoding='utf-8') as fw:
        for fname in sorted(wrong_files):
            fw.write(fname+'\n')
    print(f"检测完成, 共发现 {len(wrong_files)} 个有严重描述错误的xml，结果已保存至 {output_txt}")

if __name__ == "__main__":
    # 替换为你的xml目录
    xml_dir = "/home/xjh/RSVG-xjh/OPT-RSVG-main/data/opt_rsvg/Annotations_multi-target_v1"
    find_wrong_xmls(xml_dir)
