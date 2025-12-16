import torch
import numpy as np

from utils.box_utils import bbox_iou, xywh2xyxy,bbox_iou_pairwise

import numpy as np

def boxes_to_mask(boxes, mask_shape):
    """
    boxes: [N, 4], coords in [0, 1], format [x1, y1, x2, y2]
    mask_shape: (H, W)
    return: bool array, 1 where covered by any box
    """
    mask = np.zeros(mask_shape, dtype=bool)
    H, W = mask_shape
    for box in boxes:
        x1, y1, x2, y2 = box
        x1 = int(np.floor(x1 * W))
        y1 = int(np.floor(y1 * H))
        x2 = int(np.ceil(x2 * W))
        y2 = int(np.ceil(y2 * H))
        # 保证在图像范围内
        x1 = np.clip(x1, 0, W)
        y1 = np.clip(y1, 0, H)
        x2 = np.clip(x2, 0, W)
        y2 = np.clip(y2, 0, H)
        # mask打上
        mask[y1:y2, x1:x2] = True
    return mask



#这是引入了置信度之后的eval函数
def compute_metrics_multi_iou_conf(
    batch_pred,           # [B, N, 5], 预测（最后一维为(x, y, w, h, conf)）
    gt_boxes,             # [B, N_gt, 5]，真实框（最后一维为(x, y, w, h, conf)）
    pad_val=-1,            # 你gt的pad方式。
    iou_thresholds=(0.5, 0.6, 0.7, 0.8, 0.9),
    conf_threshold=0.3    # 预测框置信度筛选阈值，可自由调整
):
    """
    筛掉置信度低的框，只用置信度高的那些作评估
    """
    #数据类型转换
    if isinstance(batch_pred, torch.Tensor):
        batch_pred = batch_pred.detach().cpu()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.detach().cpu()

    B, N, C = batch_pred.shape
    #分出预测框和置信度 并且box格式转换
    pred_boxes_raw = batch_pred[..., :4]
    pred_conf = batch_pred[..., 4]
    gt_boxes_raw = gt_boxes[..., :4]
    gt_conf = gt_boxes[..., 4]  # [B, N_gt]
    pred_boxes = xywh2xyxy(pred_boxes_raw)
    gt_boxes = xywh2xyxy(gt_boxes_raw)
    pred_boxes = np.clip(pred_boxes, 0, 1)
    gt_boxes = np.clip(gt_boxes, 0, 1)




    # pred_boxes, pred_conf, gt_boxes: 全部都是numpy
    pred_boxes = pred_boxes.numpy()
    pred_conf = pred_conf.numpy()
    gt_boxes = gt_boxes.numpy()
    gt_conf= gt_conf.numpy()  # [B, N_gt]

    # （可选）置信度激活    若conf没过sigmoid，则需要sigmoid：
    # pred_conf = 1 / (1 + np.exp(-pred_conf))

    #初始化统计量prf_dict: 每个IoU阈值分别统计TP/FP/FN
    # iou_list: 记录所有图像平均IoU
    # union_total/inter_total: 累计交并比
    prf_dict = {thr: {'TP':0, 'FP':0, 'FN':0} for thr in iou_thresholds}
    iou_list = []
    union_total = 0.0
    inter_total = 0.0
    for b in range(B):
        pboxes = pred_boxes[b]
        pconfs = pred_conf[b]
        gboxes = gt_boxes[b]
        gconfs = gt_conf[b]
        
        # 先pad判定，防止某些地方确实是pad（比如值为-1或者0）
        gmask = (gconfs > 0.5)
        valid_gt = gboxes[gmask]   # [M_gt, 4]
        # 只保留置信度高的预测框
        keep_mask = pconfs > conf_threshold
        valid_pred = pboxes[keep_mask]       # [M_pred, 4]
        # 可以按需要进一步加NMS处理
        num_pred = valid_pred.shape[0]
        num_gt   = valid_gt.shape[0]

        #无目标场景补充:“无目标”/“空图”情况不用计入；只有gt没有pred→全部漏检，只有pred没有gt→全部假警
        if num_pred == 0 and num_gt == 0:
            continue
        if num_pred == 0:
            for thr in iou_thresholds:
                prf_dict[thr]['FN'] += num_gt
            continue
        if num_gt == 0:
            for thr in iou_thresholds:
                prf_dict[thr]['FP'] += num_pred
            continue
        # 计算IoU矩阵
        ious = bbox_iou_pairwise(
            torch.from_numpy(valid_pred), torch.from_numpy(valid_gt)
        ).numpy()
        #### meanIoU（最大贪心匹配)
        # 采用贪心的方法，将最高IoU的预测和gt配对，每个框最多与一个gt配对。
        # gt没匹配到的补0（意思是漏检了）。
        # 所有图片的匹配IoU加到iou_list，最终计算meanIoU。
        matched_gt = set()
        matched_pred = set()
        match_ious = []
        num_pred = valid_pred.shape[0]
        num_gt = valid_gt.shape[0]
        iou_flat = ious.reshape(-1)
        indices = np.argsort(-iou_flat)
        pred_inds = indices // num_gt
        gt_inds = indices % num_gt
        for idx in range(len(iou_flat)):
            i = pred_inds[idx].item()
            j = gt_inds[idx].item()
            if i in matched_pred or j in matched_gt:
                continue
            match_ious.append(ious[i, j].item())
            matched_pred.add(i)
            matched_gt.add(j)
        match_ious.extend([0.0] * (num_gt - len(match_ious)))
        iou_list.extend(match_ious)
        # cumIoU统计
        MASK_SHAPE = (512, 512)
        if num_pred > 0 and num_gt > 0:
            pred_mask = boxes_to_mask(valid_pred, MASK_SHAPE)
            gt_mask   = boxes_to_mask(valid_gt, MASK_SHAPE)
            inter = np.logical_and(pred_mask, gt_mask).sum()
            union = np.logical_or(pred_mask, gt_mask).sum()
            inter_total += inter
            union_total += union
        elif num_pred > 0:
            pred_mask = boxes_to_mask(valid_pred, MASK_SHAPE)
            union_total += pred_mask.sum()
        elif num_gt > 0:
            gt_mask = boxes_to_mask(valid_gt, MASK_SHAPE)
            union_total += gt_mask.sum()
        #  PRF指标计算
        for thr in iou_thresholds:
            matched_gt = set()
            matched_pred = set()
            indices = np.argsort(-iou_flat)
            pred_inds = indices // num_gt
            gt_inds = indices % num_gt
            for idx in range(len(iou_flat)):
                i = pred_inds[idx].item()
                j = gt_inds[idx].item()
                if i in matched_pred or j in matched_gt:
                    continue
                if ious[i, j] >= thr:
                    matched_pred.add(i)
                    matched_gt.add(j)
            TP = len(matched_pred)
            FP = num_pred - TP
            FN = num_gt - TP
            prf_dict[thr]['TP'] += TP
            prf_dict[thr]['FP'] += FP
            prf_dict[thr]['FN'] += FN
    # 结果整理 统计总体指标
    results = {}
    for thr in iou_thresholds:
        TP = prf_dict[thr]['TP']
        FP = prf_dict[thr]['FP']
        FN = prf_dict[thr]['FN']
        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        results[thr] = (precision, recall, f1)
    meanIoU = sum(iou_list) / (len(iou_list) + 1e-8)
    cumIoU = inter_total / (union_total + 1e-8)
    return results, meanIoU, cumIoU




#这是之前没有加入置信度的eval函数
def compute_metrics_multi_iou(pred_boxes, gt_boxes, pad_val=-1, iou_thresholds=(0.5, 0.6, 0.7, 0.8, 0.9)):
    """
    pred_boxes: [B, N, 4]  归一化xywh
    gt_boxes:   [B, N, 4]  归一化xywh
    返回：
        prf_dict: {iou_thresh: (precision, recall, f1) for iou_thresh in iou_thresholds}
        meanIoU: float
        cumIoU: float
    """
    B, N, _ = pred_boxes.shape
    pred_boxes = xywh2xyxy(pred_boxes)
    gt_boxes = xywh2xyxy(gt_boxes)
    pred_boxes = torch.clamp(pred_boxes, 0, 1)
    gt_boxes = torch.clamp(gt_boxes, 0, 1)
    
    # 初始化IoU统计
    iou_list = []
    union_total = 0.0
    inter_total = 0.0
    prf_dict = {thr: {'TP':0, 'FP':0, 'FN':0} for thr in iou_thresholds}
    
    for b in range(B):
        pred_b = pred_boxes[b]
        gt_b = gt_boxes[b]
        gt_valid_mask = (gt_b != 0).any(1)
        pred_valid = pred_b[gt_valid_mask]   # [M_pred, 4]
        gt_valid   = gt_b[gt_valid_mask]     # [M_gt, 4]
        num_pred = pred_valid.shape[0]
        num_gt   = gt_valid.shape[0]
        if num_pred == 0 and num_gt == 0:
            continue  # 没有目标，不计入指标
        if num_pred == 0:
            for thr in iou_thresholds:
                prf_dict[thr]['FN'] += num_gt
            continue
        if num_gt == 0:
            for thr in iou_thresholds:
                prf_dict[thr]['FP'] += num_pred
            continue
        # 计算IoU矩阵
        ious = bbox_iou_pairwise(pred_valid, gt_valid)  # [N_pred, N_gt]
        # meanIoU统计
        # 匹配方法: 贪心最大IoU匹配
        matched_gt = set()
        matched_pred = set()
        match_ious = []
        iou_flat = ious.reshape(-1)
        indices = torch.argsort(iou_flat, descending=True)
        pred_inds = indices // num_gt
        gt_inds = indices % num_gt
        for idx in range(len(iou_flat)):
            i = pred_inds[idx].item()
            j = gt_inds[idx].item()
            if i in matched_pred or j in matched_gt:
                continue
            match_ious.append(ious[i, j].item())
            matched_pred.add(i)
            matched_gt.add(j)
        # 如果gt里还有未匹配的object, meanIoU按0计
        match_ious.extend([0.0]*(num_gt-len(match_ious)))
        iou_list.extend(match_ious)
        # cumIoU统计
        MASK_SHAPE = (512, 512)
        if num_pred > 0 and num_gt > 0:
            pred_mask = boxes_to_mask(pred_valid.cpu().numpy(), MASK_SHAPE)
            gt_mask   = boxes_to_mask(gt_valid.cpu().numpy(), MASK_SHAPE)
            inter = np.logical_and(pred_mask, gt_mask).sum()
            union = np.logical_or(pred_mask, gt_mask).sum()
            inter_total += inter
            union_total += union
        elif num_pred > 0:
            pred_mask = boxes_to_mask(pred_valid.cpu().numpy(), MASK_SHAPE)
            union_total += pred_mask.sum()
        elif num_gt > 0:
            gt_mask = boxes_to_mask(gt_valid.cpu().numpy(), MASK_SHAPE)
            union_total += gt_mask.sum()

        # 针对多个IoU阈值循环贪心匹配
        for iou_thr in iou_thresholds:
            matched_gt = set()
            matched_pred = set()
            indices = torch.argsort(iou_flat, descending=True)
            pred_inds = indices // num_gt
            gt_inds = indices % num_gt
            for idx in range(len(iou_flat)):
                i = pred_inds[idx].item()
                j = gt_inds[idx].item()
                if i in matched_pred or j in matched_gt:
                    continue
                if ious[i, j] >= iou_thr:
                    matched_pred.add(i)
                    matched_gt.add(j)
            TP = len(matched_pred)
            FP = num_pred - TP
            FN = num_gt - TP
            prf_dict[iou_thr]['TP'] += TP
            prf_dict[iou_thr]['FP'] += FP
            prf_dict[iou_thr]['FN'] += FN
    # 返回各阈值的precision, recall, f1
    results = {}
    for thr in iou_thresholds:
        TP = prf_dict[thr]['TP']
        FP = prf_dict[thr]['FP']
        FN = prf_dict[thr]['FN']
        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        results[thr] = (precision, recall, f1)
    meanIoU = sum(iou_list) / (len(iou_list) + 1e-8)
    cumIoU = inter_total / (union_total + 1e-8)
    return results, meanIoU, cumIoU



def compute_metrics(pred_boxes, gt_boxes, pad_val=-1,iou_thresh=0.5):
    """
    pred_boxes: [B, N, 4]  归一化xywh
    gt_boxes:   [B, N, 4]  归一化xywh
    返回：全batch的precision, recall, f1
    """
    B, N, _ = pred_boxes.shape

    pred_boxes = xywh2xyxy(pred_boxes)
    gt_boxes = xywh2xyxy(gt_boxes)
    pred_boxes = torch.clamp(pred_boxes, 0, 1)
    gt_boxes = torch.clamp(gt_boxes, 0, 1)
    total_TP, total_FP, total_FN = 0, 0, 0
    for b in range(B):
        pred_b = pred_boxes[b]
        gt_b = gt_boxes[b]
        # print('初始pred_b内容:', pred_b)
        # print('初始gt_b内容:', gt_b)
        # 去除pad
        # pred_valid_mask = (pred_b.sum(-1) != pad_val * 4)
        gt_valid_mask = (gt_b != 0).any(1)
        pred_valid = pred_b[gt_valid_mask]   # [M_pred, 4]
        gt_valid   = gt_b[gt_valid_mask]       # [M_gt, 4]
        # print('最终pred_boxes内容:', pred_valid)
        # print('最终gt_boxes内容:', gt_valid)
        num_pred = pred_valid.shape[0]
        num_gt   = gt_valid.shape[0]
        if num_pred == 0 and num_gt == 0:
            continue  # 没有目标，不计入指标
        if num_pred == 0:
            total_FN += num_gt
            continue
        if num_gt == 0:
            total_FP += num_pred
            continue
        # 计算IoU，得到 [N_pred, N_gt] 大小的矩阵
        ious = bbox_iou_pairwise(pred_valid, gt_valid)  # 两两比较计算iou
        matched_gt = set()
        matched_pred = set()
        # 贪心匹配算法
        iou_flat = ious.reshape(-1)
        indices = torch.argsort(iou_flat, descending=True)  # 按IoU从高到低
        pred_inds = indices // num_gt
        gt_inds = indices % num_gt
        for idx in range(len(iou_flat)):
            i = pred_inds[idx].item()
            j = gt_inds[idx].item()
            if i in matched_pred or j in matched_gt:
                continue
            if ious[i, j] >= iou_thresh:
                matched_pred.add(i)
                matched_gt.add(j)
        TP = len(matched_pred)   # True Positive
        FP = num_pred - TP       # False Positive
        FN = num_gt - TP         # False Negative
        total_TP += TP
        total_FP += FP
        total_FN += FN
    print('最终pred_boxes内容:', pred_valid)
    print('最终gt_boxes内容:', gt_valid)
    precision = total_TP / (total_TP + total_FP + 1e-8)
    recall    = total_TP / (total_TP + total_FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1

# def trans_vg_eval_val(pred_boxes, gt_boxes):
#     batch_size = pred_boxes.shape[0]
#     pred_boxes = xywh2xyxy(pred_boxes)
#     pred_boxes = torch.clamp(pred_boxes, 0, 1)
#     gt_boxes = xywh2xyxy(gt_boxes)
#     iou,inter_area,u_area = bbox_iou(pred_boxes, gt_boxes)
#     accu = torch.sum(iou >= 0.5) / float(batch_size)

#     return iou, accu
def trans_vg_eval_val(pred_boxes, gt_boxes, gt_mask=None, iou_threshold=0.5):
    """
    pred_boxes: tensor, shape [B, N_pred, 4], xywh
    gt_boxes:   tensor, shape [B, N_gt, 4], xywh
    gt_mask:    tensor, shape [B, N_gt], bool or (0/1), 有效gt标记，若无则默认全部有效
    iou_threshold: float, 一般0.5
    返回:
        avg_recall, recall_stats (dict)
    """
    B, N_pred, _ = pred_boxes.shape
    _, N_gt, _ = gt_boxes.shape
    pred_boxes = xywh2xyxy(pred_boxes)
    pred_boxes = torch.clamp(pred_boxes, 0, 1)
    gt_boxes = xywh2xyxy(gt_boxes)
    gt_boxes = torch.clamp(gt_boxes, 0, 1)
    total_gt = 0
    detected_gt = 0
    recall_each_img = []
    for b in range(B):
        preds = pred_boxes[b]     # [N_pred, 4]
        gts   = gt_boxes[b]       # [N_gt, 4]
        # 只取有效gt
        if gt_mask is not None:
            mask = gt_mask[b].bool()
            gts = gts[mask]
        
        n_gt = gts.shape[0]
        if n_gt == 0:
            continue
        
        # [N_pred, n_gt]
        ious, _, _ = bbox_iou(preds, gts)
        # 对于每个gt，找到最大IoU的预测
        max_iou_pred_per_gt, _ = ious.max(dim=0)
        n_detected = (max_iou_pred_per_gt >= iou_threshold).sum().item()
        
        total_gt += n_gt
        detected_gt += n_detected
        recall_each_img.append(n_detected / n_gt)
    # 返回所有gt整体recall，也可以返回每张图的recall
    avg_recall = detected_gt / total_gt if total_gt > 0 else 0
    return avg_recall



def trans_vg_eval_test(pred_boxes, gt_boxes):
    bath_size = pred_boxes.shape[0]
    pred_boxes = xywh2xyxy(pred_boxes)
    pred_boxes = torch.clamp(pred_boxes, 0, 1)
    gt_boxes = xywh2xyxy(gt_boxes)
    iou,inter_area,u_area = bbox_iou(pred_boxes, gt_boxes)
    accu_num = torch.sum(iou >= 0.5)
    accu_num_6 = torch.sum(iou >= 0.6)
    accu_num_7 = torch.sum(iou >= 0.7)
    accu_num_8 = torch.sum(iou >= 0.8)
    accu_num_9 = torch.sum(iou >= 0.9)
    cumInterArea = np.sum(np.array(inter_area.data.cpu().numpy()))
    cumUnionArea = np.sum(np.array(u_area.data.cpu().numpy()))
    meaniou = torch.mean(iou).item()
    return accu_num,accu_num_6,accu_num_7,accu_num_8,accu_num_9,meaniou,cumInterArea/cumUnionArea
if __name__ == '__main__':

    b = torch.randn((16,20))

    d = torch.randn((16, 20))
    accu_cls = torch.sum(np.argmax(b)==d)/float(16)
    print(accu_cls)
