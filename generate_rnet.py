import os
import pickle
import cv2
import numpy as np
import torch
from tqdm import tqdm


from data_format_converter import convert_data
from utils import py_nms, combine_data_list, crop_landmark_image
from utils import save_hard_example, generate_bbox, read_annotation, processed_image
from utils import get_landmark_from_neg


device = torch.device("cuda")
pnet = torch.jit.load(r'./model/PNet.pth')
pnet.to(device)
pnet.eval()
softmax_p = torch.nn.Softmax(dim=0)


def predict(infer_data):
    infer_data = torch.tensor(infer_data, dtype=torch.float32)
    infer_data = torch.unsqueeze(infer_data, dim=0)
    infer_data = infer_data.to(device)
    cls_prob, bbox_pred, _ = pnet(infer_data)
    cls_prob = torch.squeeze(cls_prob)
    cls_prob = softmax_p(cls_prob)
    bbox_pred = torch.squeeze(bbox_pred)
    return cls_prob.detach().cpu().numpy(), bbox_pred.detach().cpu().numpy()


def detect_pnet(im, min_face_size, scale_factor, thresh):
    net_size = 12
    current_scale = float(net_size) / min_face_size
    im_resized = processed_image(im, current_scale)
    _, current_height, current_width = im_resized.shape
    all_boxes = list()
    while min(current_height, current_width) > net_size:
        cls_cls_map, reg = predict(im_resized)
        boxes = generate_bbox(cls_cls_map[1, :, :], reg, current_scale, thresh)
        current_scale *= scale_factor
        im_resized = processed_image(im, current_scale)
        _, current_height, current_width = im_resized.shape
        if boxes.size == 0:
            continue
            pass
        keep = py_nms(boxes[:, :5], 0.7, mode='Union')
        boxes = boxes[keep]
        all_boxes.append(boxes)
        pass
    if len(all_boxes) == 0:
        return None
    all_boxes = np.vstack(all_boxes)
    keep = py_nms(all_boxes[:, 0:5], 0.7)
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


def crop_24_box_image(min_face_size, scale_factor, thresh):
    pos_save_dir = r'./24/positive'
    part_save_dir = r'./24/part'
    neg_save_dir = r'./24/negative'
    save_dir =  r'./24/'
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
        boxes_c = detect_pnet(im, min_face_size, scale_factor, thresh)
        if boxes_c is None:
            all_boxes.append(empty_array)
            landmarks.append(empty_array)
            continue
            pass
        all_boxes.append(boxes_c)
        pass
    save_file = os.path.join(save_dir, 'detections.pkl')
    save_file = '/'.join(save_file.split('\\'))
    with open(save_file, 'wb') as f:
        pickle.dump(all_boxes, f, 1)
        pass
    save_hard_example(24)
    pass


if __name__ == '__main__':
    min_face_size = 20
    scale_factor = 0.79
    thresh = 0.6
    print('开始生成bbox图像数据')
    crop_24_box_image(min_face_size, scale_factor, thresh)
    print('开始生成landmark图像数据')
    data_list = get_landmark_from_neg()
    crop_landmark_image(r'./', data_list, 24, argument=True)
    print('开始合成数据列表')
    combine_data_list(r'./24')
    print('开始合成图像文件')
    convert_data(r'./24', r'./24/alldata')
    pass
