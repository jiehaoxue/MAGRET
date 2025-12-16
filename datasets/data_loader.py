
import xml.etree.ElementTree as ET
import os
import re
# import cv2
import sys
import json
import torch
import numpy as np
import os.path as osp
import scipy.io as sio
import torch.utils.data as data
sys.path.append('.')

from PIL import Image
from pytorch_pretrained_bert.tokenization import BertTokenizer
from utils.word_utils import Corpus



def read_examples(input_line, unique_id):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    # unique_id = 0
    line = input_line #reader.readline()
    # if not line:
    #     break
    line = line.strip()
    text_a = None
    text_b = None
    m = re.match(r"^(.*) \|\|\| (.*)$", line)
    if m is None:
        text_a = line
    else:
        text_a = m.group(1)
        text_b = m.group(2)
    examples.append(
        InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
    # unique_id += 1
    return examples

## Bert text encoding
class InputExample(object):
    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids

def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length
        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
    return features

class DatasetNotFoundError(Exception):
    pass
def filelist(root, file_type):
    return [os.path.join(directory_path, f) for directory_path, directory_name, files in os.walk(root) for f in files if f.endswith(file_type)]


def get_labels_index_selfdata(label):
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['airplane','ground track field', 'tennis court', 'bridge', 'basketball court',
                             'storage tank','ship','baseball diamond','T junction','crossroad','parking lot',
                             'harbor','vehicle','swimming pool']
    return text_labels.index(label)


class LPVADataset(data.Dataset):
    SUPPORTED_DATASETS = {
        'referit': {'splits': ('train', 'val', 'trainval', 'test')},
        'unc': {
            'splits': ('train', 'val', 'trainval', 'testA', 'testB'),
            'params': {'dataset': 'refcoco', 'split_by': 'unc'}
        },
        'unc+': {
            'splits': ('train', 'val', 'trainval', 'testA', 'testB'),
            'params': {'dataset': 'refcoco+', 'split_by': 'unc'}
        },
        'gref': {
            'splits': ('train', 'val'),
            'params': {'dataset': 'refcocog', 'split_by': 'google'}
        },
        'gref_umd': {
            'splits': ('train', 'val', 'test'),
            'params': {'dataset': 'refcocog', 'split_by': 'umd'}
        },
        'flickr': {
            'splits': ('train', 'val', 'test')}
    }

    def __init__(self, data_root, split_root='data', dataset='opt_rsvg', 
                 transform=None, return_idx=False, testmode=False,
                 split='train', max_query_len=128, lstm=False, 
                 bert_model='bert-base-uncased'):
        self.images = []
        self.data_root = data_root
        self.split_root = split_root
        self.dataset = dataset
        self.query_len = max_query_len
        self.lstm = lstm
        self.transform = transform
        self.testmode = testmode
        self.split = split

        self.samples = []


        # 加载自定义路径下的BERT模型tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('/home/xjh/RSVG-xjh/OPT-RSVG-main/bert')
        #self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)
        self.return_idx=return_idx

        assert self.transform is not None

        if split == 'train':
            self.augment = True
        else:
            self.augment = False

        file = open('/home/xjh/RSVG-xjh/OPT-RSVG-main/data/opt_rsvg/split/split/' + split + '.txt', "r").readlines()
        Index = [int(index.strip('\n')) for index in file]
        count = 0
        annotations = filelist('/home/xjh/RSVG-xjh/OPT-RSVG-main/data/opt_rsvg/Annotations_multi-target_v1', '.xml')
        # for anno_path in annotations:
        #     root = ET.parse(anno_path).getroot()
        #     for member in root.findall('object'):
        #         if count in Index:
        #                 name = member[0].text
        #                 label = get_labels_index_selfdata(name)
        #                 imageFile = str('/home/xjh/RSVG-xjh/OPT-RSVG-main/data/opt_rsvg/Image/Image') + '/' + root.find(
        #                     "./filename").text
        #                 box = np.array([float(member[2][0].text),float(member[2][1].text), float(member[2][2].text),
        #                                     float(member[2][3].text)], dtype=np.float32)
        #                 text = member[3].text
        #                 self.images.append((imageFile, box, text,label))
        #         count += 1
        for anno_path in annotations:
            xml_file_name = os.path.basename(anno_path)
            root = ET.parse(anno_path).getroot()
            img_file = root.findtext("filename")
            img_path = f'/home/xjh/RSVG-xjh/OPT-RSVG-main/data/opt_rsvg/Image/Image/{img_file}'
            # 1. 先解析object，存到dict，方便ID引用
            obj_dict = dict()
            for obj in root.findall('object'):
                obj_id = obj.attrib.get('id')
                name = obj.findtext("name")
                label = get_labels_index_selfdata(name)
                bbox = [
                    float(obj.find("bndbox/xmin").text),
                    float(obj.find("bndbox/ymin").text),
                    float(obj.find("bndbox/xmax").text),
                    float(obj.find("bndbox/ymax").text),
                ]
                desc = obj.findtext("description", default="")
                obj_dict[obj_id] = {'bbox': bbox, 'label': label, 'desc': desc}
            # 2. group分组，每个group为一条数据（描述+一组object）
            groups = root.find("groups")
            if groups is not None:
                for group in groups.findall("group"):
                    group_desc = group.findtext("description", default="")
                    object_ids = [ref.text for ref in group.findall("object_ref")]
                    boxes, labels = [], []
                    for obj_id in object_ids:
                        boxes.append(obj_dict[obj_id]['bbox'])
                        labels.append(obj_dict[obj_id]['label'])
                    # 用count筛选（只保留Index指定的）: 你可以灵活地是对group计数，也可以不要Index过滤（直接全收）
                    if count in Index:
                        self.samples.append((img_path, group_desc, boxes, labels,xml_file_name))
                    # count += 1
            # 3. 每个object单独也作为一条（object单目标情况）
            for obj_id, obj_info in obj_dict.items():
                desc = obj_info['desc']
                box = obj_info['bbox']
                label = obj_info['label']
                if count in Index:
                    self.samples.append((img_path, desc, [box], [label],xml_file_name))
                # count += 1
            count += 1

    def exists_dataset(self):
        path = osp.join(self.split_root, self.dataset)
        return osp.exists(path)

    def pull_item(self, idx):
        if self.dataset != 'rsvgd' and self.dataset!='opt_rsvg':
            if self.dataset == 'flickr':
                img_file, bbox, phrase = self.images[idx]
            else:
                img_file, _, bbox, phrase, attri = self.images[idx]
            ## box format: to x1y1x2y2
            if not (self.dataset == 'referit' or self.dataset == 'flickr'):
                bbox = np.array(bbox, dtype=int)
                bbox[2], bbox[3] = bbox[0]+bbox[2], bbox[1]+bbox[3]
            else:
                bbox = np.array(bbox, dtype=int)

            img_path = osp.join(self.im_dir, img_file)

            img_path = './ln_data/referit/images/27053.jpg'
            img = Image.open(img_path).convert("RGB")
        if self.dataset == 'rsvgd':
            img_path, bbox, phrase,label = self.images[idx]
            bbox = np.array(bbox, dtype=int)  # box format: to x1 y1 x2 y2
            img = Image.open(img_path).convert("RGB")
            bbox = torch.tensor(bbox)
            bbox = bbox.float()
            return img, phrase, bbox,label
        else:
            # img_path, bbox, phrase ,label = self.images[idx]
            # bbox = np.array(bbox, dtype=float)  # box format: to x1 y1 x2 y2
            # img = Image.open(img_path).convert("RGB")
            # bbox = torch.tensor(bbox)
            # bbox = bbox.float()
            # return img, phrase, bbox,label

            img_path, phrase, boxes, labels ,xml_file_name= self.samples[idx]

            img = Image.open(img_path).convert("RGB")


            # try:
            #     # 尽可能早地打印
            #     # print("loading",img_path, flush=True)
            #     img = Image.open(img_path).convert("RGB") # 装载读取图片的操作
            # except Exception as e:
            #     print("[异常图片]",img_path,flush=True)
            #     raise e

            boxes = np.array(boxes, dtype=np.float32)        # (num_queries, 4)
            labels = np.array(labels, dtype=int)             # (num_queries,)
            return img, phrase, boxes, labels,xml_file_name

    def tokenize_phrase(self, phrase):
        return self.corpus.tokenize(phrase, self.query_len)

    def untokenize_word_vector(self, words):
        return self.corpus.dictionary[words]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, phrase, bbox,label,xml_file_name = self.pull_item(idx)
        # phrase = phrase.decode("utf-8").encode().lower()
        phrase = phrase.lower()
        input_dict = {'img': img, 'box': bbox, 'text': phrase}

        input_dict = self.transform(input_dict)

        img = input_dict['img']
        bbox = input_dict['box']
        phrase = input_dict['text']
        # 填充掩码
        img_mask = input_dict['mask']
        
        if self.lstm:
            phrase = self.tokenize_phrase(phrase)
            word_id = phrase
            word_mask = np.array(word_id>0, dtype=int)
        else:
            ## encode phrase to bert input
            examples = read_examples(phrase, idx)
            features = convert_examples_to_features(
                examples=examples, seq_length=self.query_len, tokenizer=self.tokenizer)
            word_id = features[0].input_ids
            word_mask = features[0].input_mask
        
        if self.testmode:
            return img, np.array(word_id, dtype=int), np.array(word_mask, dtype=int), \
                np.array(bbox, dtype=np.float32), np.array(ratio, dtype=np.float32), \
                np.array(dw, dtype=np.float32), np.array(dh, dtype=np.float32), self.images[idx][0]
        else:
            # print(img.shape)
            return img, np.array(img_mask), np.array(word_id, dtype=int), np.array(word_mask, dtype=int), np.array(bbox, dtype=np.float32),np.array(label, dtype=int),xml_file_name
        