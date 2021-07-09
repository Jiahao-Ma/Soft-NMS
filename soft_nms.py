import numpy as np
from numpy.lib.arraysetops import unique

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


def soft_nms(boxes, num_classes, conf_thres=0.5, sigma=0.5, nms_thres=0.001):
    """
        (1) [batch_size, all_anchors, 5 + num_classes] -> [batch_size, all_anchors, 4 + 1 + 2]
        (2) 遍历每一张图片
        (3) 去除 pred < conf_thres 的框, 合成 [batch_size, all_anchors, 4 + 1 + 2]
        (4) 遍历 unique classes, 对 class_conf 排序, 对 同类的框 求 ious, 去除重合度比较高的框
    """
    # x, y, w, h -> x1, y1, x2, y2
    shape_box = boxes[:, :, :4]
    shape_box[:, :, 0] = boxes[:, :, 0] - 0.5 * boxes[:, :, 2]
    shape_box[:, :, 1] = boxes[:, :, 1] - 0.5 * boxes[:, :, 3]
    shape_box[:, :, 2] = boxes[:, :, 0] + 0.5 * boxes[:, :, 2]
    shape_box[:, :, 3] = boxes[:, :, 1] + 0.5 * boxes[:, :, 3]
    boxes[:, :, :4] = shape_box

    output = []
    bs = boxes.shape[0]
    for i in range(bs):
        # [batch_size, all_anchors, 5 + num_classes] -> [all_anchors, 5 + num_classes]
        detections = boxes[i]
        # x1, y1, x2, y2, pred, num_classes
        mask = detections[:, 4] > conf_thres
        detections = detections[mask]
        
        if len(detections) == 0:
            continue
        
        # x1, y1, x2, y2, pred, num_classes -> x1, y1, x2, y2, pred, class_conf, class_pred
        class_conf = np.expand_dims(np.max(detections[:, 5:], axis=1), axis=-1)
        class_pred = np.expand_dims(np.argmax(detections[:, 5:], axis=1), axis=-1)
        detections = np.concatenate([detections[:, :5], class_conf, class_pred], axis=1)

        unique_class = np.unique(detections[:, -1])
        if len(unique_class) == 0:
            continue
        for cls in unique_class:
            cls_mask = detections[:, -1] == cls
            detection = detections[cls_mask]
            # sort the detection by pred
            conf_index = np.argsort(detection[:, 4])[::-1]
            detection = detection[conf_index]
            
            best_boxes = []
            while detection.shape[0] > 0:
                best_boxes.append(detection[0])
                if detection.shape[0] == 1:
                    break
                ious = ious(best_boxes[-1], detection[1:])
                detection[1:, 4] = np.exp(ious * ious / sigma) * detection[1:, 4]
                detection = detection[1:]
                conf_index = np.argsort(detections[:, 4])[::-1]
                detection = detection[conf_index]
            # [anchors_num, x1,y1,x2,y2,pred,class_conf,class_pred]
            best_boxes = np.array(best_boxes)
            keep = best_boxes[:, 4] > nms_thres
            best_boxes = best_boxes[keep]
            output.append(best_boxes)
    return output

def single_class_softnms(boxes, boxscores, sigma=0.5, thresh=0.001):
    N = boxes.shape[0]
    detection = np.concatenate([boxes, boxscores.reshape(-1, 1), np.arange(0, N).reshape(N,1)], axis=1)
    conf_indx = np.argsort(detection[:, 4])[::-1]
    detection = detection[conf_indx]

    best_boxes = []
    while detection.shape[0] > 0:
        best_boxes.append(detection[0])
        if detection.shape[0] == 1:
            break
        ious = iou(best_boxes[-1].reshape(1, -1), detection[1:])
        detection[1:, 4] = np.exp(ious * ious / sigma) * detection[1:, 4]
        detection = detection[1:]
        conf_indx = np.argsort(detection[:, 4])[::-1]
        detection = detection[conf_indx]
    best_boxes = np.array(best_boxes)
    keep = best_boxes[:, -1][best_boxes[:,4] > thresh]
    return keep



if __name__ == '__main__':
    
    def test():
            # boxes and boxscores
        boxes = np.array([[200, 200, 400, 400],
                            [220, 220, 420, 420],
                            [200, 240, 400, 440],
                            [240, 200, 440, 400],
                            [1, 1, 2, 2]], dtype=np.float)
        boxscores = np.array([0.8, 0.7, 0.6, 0.5, 0.9], dtype=np.float)

        print(single_class_softnms(boxes, boxscores))
    
    test()
            
