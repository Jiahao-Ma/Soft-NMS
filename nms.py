import numpy as np

def iou(bx1, bx2):
    bx1_x1, bx1_y1, bx1_x2, bx1_y2 = bx1[:, 0], bx1[:, 1], bx1[:, 2], bx1[:, 3]
    bx2_x1, bx2_y1, bx2_x2, bx2_y2 = bx2[:, 0], bx2[:, 1], bx2[:, 2], bx2[:, 3]

    xmin = np.maximum(bx1_x1, bx2_x1)
    xmax = np.minimum(bx1_x2, bx2_x2)
    ymin = np.maximum(bx1_y1, bx2_y1)
    ymax = np.minimum(bx1_y2, bx2_y2)

    inter_area = np.maximum((xmax - xmin), 0) * np.maximum((ymax - ymin), 0)

    bx1_area = (bx1_x2 - bx1_x1) * (bx1_y2 - bx1_y1)
    bx2_area = (bx2_x2 - bx2_x1) * (bx2_y2 - bx2_y1)

    iou = inter_area / (bx1_area + bx2_area - inter_area + 1e-6)
    
    return iou

def nms(boxes, num_classes, conf_thres=0.5, nms_thres=0.4):
    """
        nms steps:
            (1) convert boxes [batch_size, all_anchors, 5 + num_classes] to detections [batch_size, all_anchors, 4 + 1 + 2]
                boxes's [5 + num_classes] 5: x, y, w, h, pred
                detections's [4 + 1 + 2]  x1, y1, x2, y2, pred, class_conf, class_pred
            (2) 遍历每一张图片
            (3) 去除  pred 小于 conf_thres 的 boxes， 并且合成 detections [batch_size, 4 + 1 + 2]
            (4) 获取unique classes, 遍历每一个classes, 获取一定区域内 class_conf 最高的 box
            (5) 重合度越高的box被去除， 保留ious < nms_thres 的框 
    """
    # x y w h -> x1 y1 x2 y2
    shape_boxes = np.zeros_like(boxes[:,:,:4])
    shape_boxes[:, :, 0] = boxes[:, :, 0] - 0.5 * boxes[:, :, 2]
    shape_boxes[:, :, 1] = boxes[:, :, 1] - 0.5 * boxes[:, :, 3]
    shape_boxes[:, :, 2] = boxes[:, :, 0] + 0.5 * boxes[:, :, 2]
    shape_boxes[:, :, 3] = boxes[:, :, 1] + 0.5 * boxes[:, :, 3]
    boxes[:, :, :4] = shape_boxes
    bs = boxes.shape[0]
    output = []
    # (2) 遍历每一张图片
    for i in range(bs):
        # (2) 去除  pred 小于 conf_thres 的 boxes， 并且合成 detections [batch_size, 4 + 1 + 2]
        # [batch_size, all_anchors, 5 + num_classes] -> [all_anchors, 5 + num_classes]
        detections = boxes[i]
        mask = detections[4] > conf_thres
        detections = detections[mask]
        if detections.shape[0] == 0:
            continue

        # [all_anchors, 5 + num_classes] -> [all_anchors, 4 + 1 + 2]
        class_conf = np.expand_dims(np.max(detections[:, 4], axis=1), axis=-1)
        class_pred = np.expand_dims(np.argmax(detections[:, 4], axis=1), axis=-1)
        detections = np.concatenate([detections[:,:5], class_conf, class_pred], axis=-1)

        unique_classes = np.unique(detections[:,-1])
        if len(unique_classes) == 0:
            continue
        
        best_box = []
        # (4) 获取unique classes, 遍历每一个classes, 获取一定区域内 class_conf 最高的 box
        for cls in unique_classes:
            cls_mask = detections[:,-1] == cls
            detection = detections[cls_mask]
            # [anchors, x1, y1, x2, y2, pred, class_conf, class_pred]
            # sort by the pred score
            index = np.argsort(detection[:, 4])[::-1]
            detection = detection[index]

            while detection.shape[0] > 0:
                best_box.append(detection[0])
                if len(detection) == 1:
                    break
                ious = iou(best_box[-1], detection[1:])
                detection = detection[1:][ious < nms_thres]
        output.append(best_box)
    return np.array(output)
