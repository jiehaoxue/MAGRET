import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import argparse
import datasets
import utils.misc as utils
from utils.misc import collate_fn
from models import build_model
from datasets import build_dataset
def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)

    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_bert', default=1e-5, type=float)
    parser.add_argument('--lr_visu_cnn', default=1e-5, type=float)
    parser.add_argument('--lr_visu_tra', default=1e-5, type=float)

    parser.add_argument('--batch_size', default=16 ,type=int)#默认 16
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=250,type=int)
    parser.add_argument('--lr_power', default=0.9, type=float, help='lr poly power')
    parser.add_argument('--clip_max_norm', default=1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--eval', dest='eval', default=False, action='store_true', help='if evaluation only')
    parser.add_argument('--optimizer', default='adamw', type=str)
    parser.add_argument('--hgd_optimizer', default='rmsprop', type=str)
    parser.add_argument('--lr_scheduler', default='step', type=str)
    parser.add_argument('--lr_drop', default=100, type=int)
    
    # Augmentation options
    parser.add_argument('--aug_blur', default='',action='store_true',
                        help="If true, use gaussian blur augmentation")
    parser.add_argument('--aug_crop', default='',action='store_true',
                        help="If true, use random crop augmentation")
    parser.add_argument('--aug_scale', default='',action='store_true',
                        help="If true, use multi-scale augmentation")
    parser.add_argument('--aug_translate',default='', action='store_true',
                        help="If true, use random translate augmentation")

    # Model parameters
    parser.add_argument('--model_name', type=str, default='LPVA',
                        help="Name of model to be exploited.")
    
    # DETR parameters
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'), help="Type of positional embedding to use on top of the image features")
    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=0, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    parser.add_argument('--imsize', default=640, type=int, help='image size')
    parser.add_argument('--emb_size', default=512, type=int,
                        help='fusion module embedding dimensions')

    # Transformers in two branches
    parser.add_argument('--bert_enc_num', default=12, type=int)
    parser.add_argument('--detr_enc_num', default=6, type=int)


    # Vision-Language Transformer
    parser.add_argument('--vl_dropout', default=0.1, type=float,
                        help="Dropout applied in the vision-language transformer")
    parser.add_argument('--vl_nheads', default=8, type=int,
                        help="Number of attention heads inside the vision-language transformer's attentions")
    parser.add_argument('--vl_hidden_dim', default=256, type=int,
                        help='Size of the embeddings (dimension of the vision-language transformer)')
    parser.add_argument('--vl_dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the vision-language transformer blocks")
    parser.add_argument('--vl_enc_layers', default=6, type=int,
                        help='Number of encoders in the vision-language transformer')

    # Dataset parameters
    parser.add_argument('--data_root', type=str, default='/home/xjh/RSVG-xjh/OPT-RSVG-main/data/',
                        help='path to ReferIt splits data folder')
    parser.add_argument('--split_root', type=str, default='data',
                        help='location of pre-parsed dataset info')
    parser.add_argument('--dataset', default='opt_rsvg', type=str,
                        help='opt_rsvg/rsvgd')
    parser.add_argument('--max_query_len', default=40, type=int,
                        help='maximum time steps (lang length) per batch')
    
    # dataset parameters
    parser.add_argument('--output_dir', default='/home/xjh/RSVG-xjh/OPT-RSVG-main/output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=13, type=int)
    # parser.add_argument('--resume', default='/home/xjh/RSVG-xjh/OPT-RSVG-main/output/checkpoint.pth', help='resume from checkpoint')
    parser.add_argument('--resume', default='/home/xjh/RSVG-xjh/OPT-RSVG-main/output/6-30_exp2-with-conf-改loss/best_checkpoint.pth', help='resume from checkpoint')
    parser.add_argument('--detr_model', default='/home/xjh/RSVG-xjh/OPT-RSVG-main/pretrained/detr-r50-e632da11.pth', type=str, help='detr model')
    parser.add_argument('--bert_model', default='bert-base-uncased', type=str, help='bert model')
    parser.add_argument('--light', dest='light', default=False, action='store_true', help='if use smaller model')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--cls_loss',default=True,type=str)
    parser.add_argument('--name', type=str, default='6-30_exp2-with-conf-改loss', help='name of the experiment')

    return parser


def normxywh_to_xyxy(boxes, img_w, img_h):
    """
    将归一化[cx, cy, w, h]坐标转为像素[x1, y1, x2, y2]
    输入: boxes [N,4], img_w, img_h
    返回: boxes_xyxy [N,4]
    """
    boxes = np.asarray(boxes)
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return np.stack([x1, y1, x2, y2], axis=1)

def plot_img_pred_gt(img_tensor, pred_boxes_xyxy, pred_scores, gt_boxes_xyxy, save_path):
    """
    img_tensor: torch.Tensor C,H,W, 未归一化到[0,1]
    pred_boxes_xyxy: [N,4]
    pred_scores: [N]
    gt_boxes_xyxy: [M,4]
    """
    img = img_tensor.cpu()
    
    mean = torch.tensor([0.485,0.456,0.406])[:,None,None]
    std = torch.tensor([0.229,0.224,0.225])[:,None,None]
    img = img * std + mean
    img = torch.clip(img, 0, 1)
    img = to_pil_image(img)
    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.imshow(img)
    # Predict
    for box, score in zip(pred_boxes_xyxy, pred_scores):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='lime', facecolor='none',linestyle='--')
        ax.add_patch(rect)
        # ax.text(x1, y1-8, f"{score:.2f}", fontsize=13, color='lime', bbox=dict(facecolor='black', alpha=0.5))
    # GT
    for box in gt_boxes_xyxy:
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
    ax.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=160)
    plt.close()
    print(f"保存到 {save_path}")

@torch.no_grad()
# def visualize_on_val(args, val_idx=5, save_dir="visual_results", conf_thr=0.3):
#     device = torch.device(args.device if torch.cuda.is_available() else "cpu")
#     # 1. 模型
#     model = build_model(args)
#     checkpoint = torch.load(args.resume, map_location=device)
#     model.load_state_dict(checkpoint['model'], strict=False)
#     model.to(device)
#     model.eval()
#     # 2. 数据
#     dataset_val = build_dataset('train_remove_single-target_xmls', args)#val_remove_single-target_xmls
#     # 正确流程，用collate_fn包装
#     datum = dataset_val[val_idx]
#     batch = collate_fn([datum])
#     img_data, text_data, batch_target, batch_labels,xml_names = batch
#     img_data = img_data.to(device)
#     text_data = text_data.to(device)
#     batch_target = batch_target.to(device)
#     with torch.no_grad():
#         outputs = model(img_data, text_data)
#     outputs = outputs[0].cpu().numpy()  # [num_queries, 5]
#     pred_boxes_cxcywh = outputs[:, :4]
#     pred_confs = outputs[:, 4]
#     # ...（后续可视化保证与训练完全一致即可）
#     img = img_data.tensors[0]   # [C,H,W]
#     _, H, W = img.shape
#     pred_boxes_xyxy = normxywh_to_xyxy(pred_boxes_cxcywh, W, H)
#     keep = pred_confs >= conf_thr
#     pred_boxes_xyxy = pred_boxes_xyxy[keep]
#     pred_confs = pred_confs[keep]
#     gt_boxes_xyxy = normxywh_to_xyxy(batch_target[0, :, :4].cpu().numpy(), W, H)
#     # 只可视化非pad位置gt
#     gt_labels = batch_labels[0].cpu().numpy()
#     valid_gt_mask = gt_labels != -1
#     gt_boxes_xyxy = gt_boxes_xyxy[valid_gt_mask]
#     # 5. 保存
#     os.makedirs(save_dir, exist_ok=True)
#     xml_name = os.path.splitext(xml_names[0])[0] + ".png"  # 用xml文件名
#     save_path = os.path.join(save_dir, xml_name)
#     plot_img_pred_gt(img, pred_boxes_xyxy, pred_confs, gt_boxes_xyxy, save_path)
def visualize_on_val(args, indices=None, save_dir="visual_results", conf_thr=0.3):
    from collections import defaultdict
    save_counts = defaultdict(int)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_model(args)
    checkpoint = torch.load(args.resume, map_location=device)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.to(device)
    model.eval()
    dataset_val = build_dataset('train_remove_single-target_xmls', args)
    if indices is None:
        indices = range(len(dataset_val))
    print(f"可视化图片数量: {len(indices)}")
    os.makedirs(save_dir, exist_ok=True)
    for val_idx in indices:
        datum = dataset_val[val_idx]
        batch = collate_fn([datum])
        img_data, text_data, batch_target, batch_labels, xml_names = batch
        img_data = img_data.to(device)
        text_data = text_data.to(device)
        batch_target = batch_target.to(device)
        with torch.no_grad():
            outputs = model(img_data, text_data)
        outputs = outputs[0].cpu().numpy()  # [num_queries, 5]
        pred_boxes_cxcywh = outputs[:, :4]
        pred_confs = outputs[:, 4]
        img = img_data.tensors[0]
        _, H, W = img.shape
        pred_boxes_xyxy = normxywh_to_xyxy(pred_boxes_cxcywh, W, H)
        keep = pred_confs >= conf_thr
        pred_boxes_xyxy = pred_boxes_xyxy[keep]
        pred_confs = pred_confs[keep]
        gt_boxes_xyxy = normxywh_to_xyxy(batch_target[0, :, :4].cpu().numpy(), W, H)
        gt_labels = batch_labels[0].cpu().numpy()
        valid_gt_mask = gt_labels != -1
        gt_boxes_xyxy = gt_boxes_xyxy[valid_gt_mask]
        
        # ===== 新增：防止同图覆盖，用序号 =====
        xml_base = os.path.splitext(xml_names[0])[0]   # 如22928
        save_counts[xml_base] += 1
        save_name = f"{xml_base}_{save_counts[xml_base]}.png"
        save_path = os.path.join(save_dir, save_name)
        plot_img_pred_gt(img, pred_boxes_xyxy, pred_confs, gt_boxes_xyxy, save_path)
if __name__=="__main__":
    # # 用你的train脚本参数接口
    # parser = argparse.ArgumentParser('LPVA training script', parents=[get_args_parser()])
    # args = parser.parse_args()
    # # ==== [1] 指定可视化哪张图片：改为你想要的下标 ====
    # val_idx =3199   # 若改2表示可视化val集第3张
    # # ==== [2] 输出目录 可自定义 ====
    # save_dir = "/home/xjh/RSVG-xjh/OPT-RSVG-main/output/visualized_results"
    # # ==== [3] 置信度阈值 ====
    # conf_thr = 0.3
    # visualize_on_val(args, val_idx, save_dir, conf_thr)

    parser = argparse.ArgumentParser('LPVA training script', parents=[get_args_parser()])
    args = parser.parse_args()
    save_dir = "/media/csl/8T/xjh/MTVG/visualized_results"
    conf_thr = 0.3
    # 可视化train_remove_single-target_xmls所有图片
    visualize_on_val(args, indices=None, save_dir=save_dir, conf_thr=conf_thr)

