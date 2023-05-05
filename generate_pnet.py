import os

import cv2
import numpy as np
from tqdm import tqdm

from data_format_converter import convert_data
from utils import IOU, combine_data_list, crop_landmark_image
from utils import get_landmark_from_neg


def crop_12_box_image():
    box_file = r'./dataset/box.txt'
    npr = np.random
    pos_save_dir = r'./12/positive'
    part_save_dir = r'./12/part'
    neg_save_dir = r'./12/negative'
    save_dir = r'./12/'
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
    f1 = open(r'./12/positive.txt', 'w')
    f2 = open(r'./12/negative.txt', 'w')
    f3 = open(r'./12/part.txt', 'w')
    with open(box_file, 'r') as f:
        boxs = f.readlines()
        pass
    num = len(boxs)
    print('总共的图片数： %d' % num)
    p_idx = 0
    n_idx = 0
    d_idx = 0
    idx = 0
    for b in tqdm(boxs):
        b = '/'.join(b.split('\\'))
        b = b.strip().split(' ')
        box = list(map(float, b[1:5]))
        boxes = np.array(box, dtype=np.float32).reshape(-1, 4)
        img = cv2.imread(b[0])
        idx += 1
        height, width, channel = img.shape
        neg_num = 0
        while neg_num < 50:
            size = npr.randint(12, min(width, height) / 2)
            nx = npr.randint(0, width - size)
            ny = npr.randint(0, height - size)
            crop_box = np.array([nx, ny, nx + size, ny + size])
            Iou = IOU(crop_box, boxes)
            cropped_im = img[ny:ny + size, nx:nx + size, :]
            resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)
            if np.max(Iou) < 0.3:
                save_file = os.path.join(neg_save_dir, '%s.jpg' % n_idx)
                save_file = '/'.join(save_file.split('\\'))
                f2.write(neg_save_dir + '/%s.jpg' % n_idx + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1
                neg_num += 1
                pass
            pass
        for box in boxes:
            x1, y1, x2, y2 = box
            w = x2 - x1 + 1
            h = y2 - y1 + 1
            if max(w, h) < 20 or x1 < 0 or y1 < 0:
                continue
                pass
            for i in range(5):
                size = npr.randint(12, min(width, height) / 2)
                delta_x = npr.randint(max(-size, -x1), w)
                delta_y = npr.randint(max(-size, -y1), h)
                nx1 = int(max(0, x1 + delta_x))
                ny1 = int(max(0, y1 + delta_y))
                if nx1 + size > width or ny1 + size > height:
                    continue
                    pass
                crop_box = np.array([nx1, ny1, nx1 + size, ny1 + size])
                Iou = IOU(crop_box, boxes)
                cropped_im = img[ny1:ny1 + size, nx1:nx1 + size, :]
                resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)
                if np.max(Iou) < 0.3:
                    save_file = os.path.join(neg_save_dir, '%s.jpg' % n_idx)
                    save_file = '/'.join(save_file.split('\\'))
                    f2.write(neg_save_dir + '/%s.jpg' % n_idx + ' 0\n')
                    cv2.imwrite(save_file, resized_im)
                    n_idx += 1
                    pass
                pass
            for i in range(20):
                size = npr.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))
                if w < 5:
                    continue
                    pass
                delta_x = npr.randint(-w * 0.2, w * 0.2)
                delta_y = npr.randint(-h * 0.2, h * 0.2)
                nx1 = int(max(x1 + w / 2 + delta_x - size / 2, 0))
                ny1 = int(max(y1 + h / 2 + delta_y - size / 2, 0))
                nx2 = nx1 + size
                ny2 = ny1 + size
                if nx2 > width or ny2 > height:
                    continue
                    pass
                crop_box = np.array([nx1, ny1, nx2, ny2])
                offset_x1 = (x1 - nx1) / float(size)
                offset_y1 = (y1 - ny1) / float(size)
                offset_x2 = (x2 - nx2) / float(size)
                offset_y2 = (y2 - ny2) / float(size)
                cropped_im = img[ny1:ny2, nx1:nx2, :]
                resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)
                box_ = box.reshape(1, -1)
                iou = IOU(crop_box, box_)
                if iou >= 0.65:
                    save_file = os.path.join(pos_save_dir, '%s.jpg' % p_idx)
                    save_file = '/'.join(save_file.split('\\'))
                    f1.write(pos_save_dir + '/%s.jpg' % p_idx + ' 1 %.2f %.2f %.2f %.2f\n' % (offset_x1,
                                                                                              offset_y1, offset_x2,
                                                                                              offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    p_idx += 1
                    pass
                elif iou >= 0.4:
                    save_file = os.path.join(part_save_dir, '%s.jpg' % d_idx)
                    save_file = '/'.join(save_file.split('\\'))
                    f3.write(part_save_dir + '/%s.jpg' % d_idx + ' -1 %.2f %.2f %.2f %.2f\n' % (offset_x1,
                                                                                                offset_y1, offset_x2,
                                                                                                offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    d_idx += 1
                    pass
                pass
            pass
        pass
    print('%s 个图片已处理，pos：%s  part: %s neg:%s' % (idx, p_idx, d_idx, n_idx))
    f1.close()
    f2.close()
    f3.close()
    pass


if __name__ == '__main__':
    print('开始生成bbox图像数据')
    crop_12_box_image()
    print('开始生成landmark图像数据')
    data_list = get_landmark_from_neg()
    crop_landmark_image(r'./', data_list, 12, argument=True)
    print('开始合成数据列表')
    combine_data_list(r'./12')
    print('开始合成图像文件')
    convert_data(r'./12', r'./12/alldata')
    pass
