import os
import pickle
import cv2
import numpy as np
import torch
from tqdm import tqdm


from data_format_converter import convert_data
from utils import py_nms, combine_data_list, crop_landmark_image, pad, processed_image
from utils import save_hard_example, generate_bbox, read_annotation, convert_to_square, calibrate_box
from utils import get_landmark_from_neg


device = torch.device("cuda")
pnet = torch.jit.load(r'./model/PNet.pth')
pnet.to(device)
pnet.eval()
softmax_p = torch.nn.Softmax(dim=0)
rnet = torch.jit.load(r'./model/RNet.pth')
rnet.to(device)
rnet.eval()
softmax_r = torch.nn.Softmax(dim=-1)


def predict_pnet(infer_data):
    infer_data = torch.tensor(infer_data, dtype=torch.float32)
    infer_data = torch.unsqueeze(infer_data, dim=0)
    infer_data = infer_data.to(device)
    cls_prob, bbox_pred, _ = pnet(infer_data)
    cls_prob = torch.squeeze(cls_prob)
    cls_prob = softmax_p(cls_prob)
    bbox_pred = torch.squeeze(bbox_pred)
    return cls_prob.detach().cpu().numpy(), bbox_pred.detach().cpu().numpy()


def predict_rnet(infer_data):
    infer_data = torch.tensor(infer_data, dtype=torch.float32)
    infer_data = infer_data.to(device)
    cls_prob, bbox_pred, _ = rnet(infer_data)
    cls_prob = softmax_r(cls_prob)
    return cls_prob.detach().cpu().numpy(), bbox_pred.detach().cpu().numpy()


def detect_pnet(im, min_face_size, scale_factor, thresh):
    net_size = 12
    current_scale = float(net_size) / min_face_size
    im_resized = processed_image(im, current_scale)
    _, current_height, current_width = im_resized.shape
    all_boxes = list()
    while min(current_height, current_width) > net_size:
        cls_cls_map, reg = predict_pnet(im_resized)
        boxes = generate_bbox(cls_cls_map[1, :, :], reg, current_scale, thresh)
        current_scale *= scale_factor
        im_resized = processed_image(im, current_scale)
        _, current_height, current_width = im_resized.shape
        if boxes.size == 0:
            continue
            pass
        keep = py_nms(boxes[:, :5], 0.5, mode='Union')
        boxes = boxes[keep]
        all_boxes.append(boxes)
        pass
    if len(all_boxes) == 0:
        return None
    all_boxes = np.vstack(all_boxes)
    keep = py_nms(all_boxes[:, 0:5], 0.7, mode='Union')
    all_boxes = all_boxes[keep]
    bbw = all_boxes[:, 2] - all_boxes[:, 0] + 1
    bbh = all_boxes[:, 3] - all_boxes[:, 1] + 1
    boxes_c = np.vstack([all_boxes[:, 0] + all_boxes[:, 5] * bbw,
                         all_boxes[:, 1] + all_boxes[:, 6] * bbh,
                         all_boxes[:, 2] + all_boxes[:, 7] * bbw,
                         all_boxes[:, 3] + all_boxes[:, 8] * bbh,
                         all_boxes[:, 4]])
    boxes_c = boxes_c.T
    return boxes_c


def detect_rnet(im, dets, thresh):
    h, w, c = im.shape
    dets = convert_to_square(dets)
    dets[:, 0:4] = np.round(dets[:, 0:4])
    [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(dets, w, h)
    delete_size = np.ones_like(tmpw) * 20
    ones = np.ones_like(tmpw)
    zeros = np.zeros_like(tmpw)
    num_boxes = np.sum(np.where((np.minimum(tmpw, tmph) >= delete_size), ones, zeros))
    cropped_ims = np.zeros((num_boxes, 3, 24, 24), dtype=np.float32)
    if int(num_boxes) == 0:
        print('P模型检测结果为空！')
        return None, None
    for i in range(int(num_boxes)):
        if tmph[i] < 20 or tmpw[i] < 20:
            continue
            pass
        tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
        try:
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            img = cv2.resize(tmp, (24, 24))
            img = img.transpose((2, 0, 1))
            img = (img - 127.5) / 128
            cropped_ims[i, :, :, :] = img
            pass
        except:
            continue
            pass
        pass
    cls_scores, reg = predict_rnet(cropped_ims)
    cls_scores = cls_scores[:, 1]
    keep_inds = np.where(cls_scores > thresh)[0]
    if len(keep_inds) > 0:
        boxes = dets[keep_inds]
        boxes[:, 4] = cls_scores[keep_inds]
        reg = reg[keep_inds]
        pass
    else:
        return None, None
    keep = py_nms(boxes, 0.6, mode='Union')
    boxes = boxes[keep]
    boxes_c = calibrate_box(boxes, reg[keep])
    return boxes, boxes_c


def crop_48_box_image(min_face_size, scale_factor, p_thresh, r_thresh):
    pos_save_dir = r'./48/positive'
    part_save_dir = r'./48/part'
    neg_save_dir = r'./48/negative'
    save_dir = r'./48/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        pass
    if not os.path.exists(pos_save_dir):
        os.mkdir(pos_save_dir)
        pass
    if not os.path.exists(part_save_dir):
        os.mkdir(part_save_dir)
        pass
    if not os.path.exists(neg_save_dir):
        os.mkdir(neg_save_dir)
        pass
    data = read_annotation(r'./dataset/box.txt')
    all_boxes = []
    landmarks = []
    empty_array = np.array([])
    for image_path in tqdm(data['images']):
        assert os.path.exists(image_path), 'image not exists'
        image_path = '/'.join(image_path.split('\\'))
        im = cv2.imread(image_path)
        boxes_c = detect_pnet(im, min_face_size, scale_factor, p_thresh)
        if boxes_c is None:
            all_boxes.append(empty_array)
            landmarks.append(empty_array)
            continue
            pass
        boxes, boxes_c = detect_rnet(im, boxes_c, r_thresh)
        if boxes_c is None:
            all_boxes.append(empty_array)
            landmarks.append(empty_array)
            continue
            pass
        all_boxes.append(boxes_c)
        pass
    save_file = r'./48/detections.pkl'
    with open(save_file, 'wb') as f:
        pickle.dump(all_boxes, f, 1)
        pass
    save_hard_example(48)
    pass


if __name__ == '__main__':
    min_face_size = 20
    scale_factor = 0.79
    p_thresh = 0.6
    r_thresh = 0.7
    print('开始生成bbox图像数据')
    crop_48_box_image(min_face_size, scale_factor, p_thresh, r_thresh)
    print('开始生成landmark图像数据')
    data_list = get_landmark_from_neg()
    crop_landmark_image(r'./', data_list, 48, argument=True)
    print('开始合成数据列表')
    combine_data_list(r'./48')
    print('开始合成图像文件')
    convert_data(r'./48', r'./48/alldata')
    pass
