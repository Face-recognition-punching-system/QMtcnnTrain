import pickle
import numpy as np
import random
import os
import cv2
from tqdm import tqdm


class BBox:
    def __init__(self, box):
        self.left = box[0]
        self.top = box[1]
        self.right = box[2]
        self.bottom = box[3]
        self.x = box[0]
        self.y = box[1]
        self.w = box[2] - box[0]
        self.h = box[3] - box[1]
        pass

    def project(self, point):
        x = (point[0] - self.x) / self.w
        y = (point[1] - self.y) / self.h
        return np.asarray([x, y])

    def reproject(self, point):
        x = self.x + self.w * point[0]
        y = self.y + self.h * point[1]
        return np.asarray([x, y])

    def reprojectLandmark(self, landmark):
        p = np.zeros((len(landmark), 2))
        for i in range(len(landmark)):
            p[i] = self.reproject(landmark[i])
        return p

    def projectLandmark(self, landmark):
        p = np.zeros((len(landmark), 2))
        for i in range(len(landmark)):
            p[i] = self.project(landmark[i])
        return p
    pass


def processed_image(img, scale):
    height, width, channels = img.shape
    new_height = int(height * scale)
    new_width = int(width * scale)
    new_dim = (new_width, new_height)
    img_resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR)
    image = np.array(img_resized).astype(np.float32)
    image = image.transpose((2, 0, 1))
    image = (image - 127.5) / 128
    return image


def IOU(box, boxes):
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)
    inter = w * h
    return inter / (box_area + area - inter + 1e-10)


def get_landmark_from_neg(with_landmark=True):
    with open(r'./dataset/box.txt', 'r') as f:
        lines = f.readlines()
        pass
    result = []
    for line in lines:
        line = line.strip()
        components = line.split(' ')
        img_path = components[0]
        box = (components[1], components[3], components[2], components[4])
        box = [float(_) for _ in box]
        box = list(map(int, box))
        if not with_landmark:
            result.append((img_path, BBox(box)))
            continue
            pass
        landmark = np.zeros((5, 2))
        for index in range(5):
            rv = (float(components[5 + 2 * index]), float(components[5 + 2 * index + 1]))
            landmark[index] = rv
            pass
        result.append((img_path, BBox(box), landmark))
        pass
    return result


def combine_data_list(data_dir):
    npr = np.random
    pos_path = os.path.join(data_dir, 'positive.txt')
    part_path = os.path.join(data_dir, 'part.txt')
    neg_path = os.path.join(data_dir, 'negative.txt')
    landmark_path = os.path.join(data_dir, 'landmark.txt')
    all_data_list_path = os.path.join(data_dir, 'all_data_list.txt')
    tmp_path = os.path.join(data_dir, 'temp.txt')
    pos_path = "/".join(pos_path.split("\\"))
    part_path = "/".join(part_path.split("\\"))
    neg_path = "/".join(neg_path.split("\\"))
    landmark_path = "/".join(landmark_path.split("\\"))
    all_data_list_path = "/".join(all_data_list_path.split("\\"))
    tmp_path = "/".join(tmp_path.split("\\"))
    with open(pos_path, 'r') as f:
        pos = f.readlines()
        pass
    with open(part_path, 'r') as f:
        neg = f.readlines()
        pass
    with open(neg_path, 'r') as f:
        part = f.readlines()
        pass
    with open(landmark_path, 'r') as f:
        landmark = f.readlines()
        pass
    with open(all_data_list_path, 'w') as f:
        base_num = len(pos) // 1000 * 1000
        s1 = '整理前的数据：neg数量：{} pos数量：{} part数量:{} landmark: {} 基数:{}'.format(len(neg), len(pos), len(part),
                                                                                          len(landmark), base_num)
        print(s1)
        neg_keep = npr.choice(len(neg), size=base_num * 3, replace=base_num * 3 > len(neg))
        part_keep = npr.choice(len(part), size=base_num, replace=base_num > len(part))
        pos_keep = npr.choice(len(pos), size=base_num, replace=base_num > len(pos))
        landmark_keep = npr.choice(len(landmark), size=base_num * 2, replace=base_num * 2 > len(landmark))
        s2 = '整理后的数据：neg数量：{} pos数量：{} part数量:{} landmark数量：{}'.format(len(neg_keep), len(pos_keep),
                                                                                     len(part_keep), len(landmark_keep))
        print(s2)
        with open(tmp_path, 'a', encoding='utf-8') as f_temp:
            f_temp.write('%s\n' % s1)
            f_temp.write('%s\n' % s2)
            f_temp.flush()
            pass
        for i in pos_keep:
            f.write("/".join(pos[i].split("\\")))
            pass
        for i in neg_keep:
            f.write("/".join(neg[i].split("\\")))
            pass
        for i in part_keep:
            f.write("/".join(part[i].split("\\")))
            pass
        for i in landmark_keep:
            f.write("/".join(landmark[i].split("\\")))
            pass
        pass
    pass


def crop_landmark_image(data_dir, data_list, size, argument=True):
    npr = np.random
    image_id = 0
    output = os.path.join(data_dir, str(size))
    output = "/".join(output.split("\\"))
    if not os.path.exists(output):
        os.makedirs(output)
        pass
    dstdir = os.path.join(output, 'landmark')
    dstdir = "/".join(dstdir.split("\\"))
    if not os.path.exists(dstdir):
        os.mkdir(dstdir)
        pass
    lambdarkdir = os.path.join(output, 'landmark.txt')
    lambdarkdir = "/".join(lambdarkdir.split("\\"))
    f = open(lambdarkdir, 'w')
    idx = 0
    for (imgPath, box, landmarkGt) in tqdm(data_list):
        F_imgs = []
        F_landmarks = []
        imgPath = "/".join(imgPath.split("\\"))
        img = cv2.imread(imgPath)
        img_h, img_w, img_c = img.shape
        gt_box = np.array([box.left, box.top, box.right, box.bottom])
        f_face = img[box.top:box.bottom, box.left:box.right]
        try:
            f_face = cv2.resize(f_face, (size, size))
            pass
        except Exception as e:
            print(e)
            print('resize成网络输入大小，跳过')
            continue
            pass
        landmark = np.zeros((5, 2))
        for index, one in enumerate(landmarkGt):
            rv = ((one[0] - gt_box[0]) / (gt_box[2] - gt_box[0]), (one[1] - gt_box[1]) / (gt_box[3] - gt_box[1]))
            landmark[index] = rv
            pass
        F_imgs.append(f_face)
        F_landmarks.append(landmark.reshape(10))
        if argument:
            landmark = np.zeros((5, 2))
            idx = idx + 1
            x1, y1, x2, y2 = gt_box
            gt_w = x2 - x1 + 1
            gt_h = y2 - y1 + 1
            if max(gt_w, gt_h) < 40 or x1 < 0 or y1 < 0:
                continue
                pass
            for i in range(10):
                box_size = npr.randint(int(min(gt_w, gt_h) * 0.8), np.ceil(1.25 * max(gt_w, gt_h)))
                try:
                    delta_x = npr.randint(-gt_w * 0.2, gt_w * 0.2)
                    delta_y = npr.randint(-gt_h * 0.2, gt_h * 0.2)
                    pass
                except Exception as e:
                    print(e)
                    print('随机裁剪图像大小，跳过')
                    continue
                    pass
                nx1 = int(max(x1 + gt_w / 2 - box_size / 2 + delta_x, 0))
                ny1 = int(max(y1 + gt_h / 2 - box_size / 2 + delta_y, 0))
                nx2 = nx1 + box_size
                ny2 = ny1 + box_size
                if nx2 > img_w or ny2 > img_h:
                    continue
                    pass
                crop_box = np.array([nx1, ny1, nx2, ny2])
                cropped_im = img[ny1:ny2 + 1, nx1:nx2 + 1, :]
                resized_im = cv2.resize(cropped_im, (size, size))
                iou = IOU(crop_box, np.expand_dims(gt_box, 0))
                if iou > 0.65:
                    F_imgs.append(resized_im)
                    for index, one in enumerate(landmarkGt):
                        rv = ((one[0] - nx1) / box_size, (one[1] - ny1) / box_size)
                        landmark[index] = rv
                    F_landmarks.append(landmark.reshape(10))
                    landmark = np.zeros((5, 2))
                    landmark_ = F_landmarks[-1].reshape(-1, 2)
                    box = BBox([nx1, ny1, nx2, ny2])
                    if random.choice([0, 1]) > 0:
                        face_flipped, landmark_flipped = flip(resized_im, landmark_)
                        face_flipped = cv2.resize(face_flipped, (size, size))
                        F_imgs.append(face_flipped)
                        F_landmarks.append(landmark_flipped.reshape(10))
                        pass
                    if random.choice([0, 1]) > 0:
                        face_rotated_by_alpha, landmark_rorated = rotate(img, box, box.reprojectLandmark(landmark_), 5)
                        landmark_rorated = box.projectLandmark(landmark_rorated)
                        face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (size, size))
                        F_imgs.append(face_rotated_by_alpha)
                        F_landmarks.append(landmark_rorated.reshape(10))
                        face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rorated)
                        face_flipped = cv2.resize(face_flipped, (size, size))
                        F_imgs.append(face_flipped)
                        F_landmarks.append(landmark_flipped.reshape(10))
                        pass
                    if random.choice([0, 1]) > 0:
                        face_rotated_by_alpha, landmark_rorated = rotate(img, box, box.reprojectLandmark(landmark_), -5)
                        landmark_rorated = box.projectLandmark(landmark_rorated)
                        face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (size, size))
                        F_imgs.append(face_rotated_by_alpha)
                        F_landmarks.append(landmark_rorated.reshape(10))
                        face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rorated)
                        face_flipped = cv2.resize(face_flipped, (size, size))
                        F_imgs.append(face_flipped)
                        F_landmarks.append(landmark_flipped.reshape(10))
                        pass
                    pass
                pass
            pass
        F_imgs, F_landmarks = np.asarray(F_imgs), np.asarray(F_landmarks)
        for i in range(len(F_imgs)):
            if np.sum(np.where(F_landmarks[i] <= 0, 1, 0)) > 0:
                continue
                pass
            if np.sum(np.where(F_landmarks[i] >= 1, 1, 0)) > 0:
                continue
                pass
            cv2.imwrite(os.path.join(dstdir, '%d.jpg' % image_id), F_imgs[i])
            landmarks = list(map(str, list(F_landmarks[i])))
            f.write(os.path.join(dstdir, '%d.jpg' % image_id) + ' -2 ' + ' '.join(landmarks) + '\n')
            image_id += 1
            pass
        pass
    f.close()
    pass


def flip(face, landmark):
    face_flipped_by_x = cv2.flip(face, 1)
    landmark_ = np.asarray([(1 - x, y) for (x, y) in landmark])
    landmark_[[0, 1]] = landmark_[[1, 0]]
    landmark_[[3, 4]] = landmark_[[4, 3]]
    return face_flipped_by_x, landmark_


def rotate(img, box, landmark, alpha):
    center = ((box.left + box.right) / 2, (box.top + box.bottom) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, alpha, 1)
    img_rotated_by_alpha = cv2.warpAffine(img, rot_mat, (img.shape[1], img.shape[0]))
    landmark_ = np.asarray([(rot_mat[0][0] * x + rot_mat[0][1] * y + rot_mat[0][2],
                             rot_mat[1][0] * x + rot_mat[1][1] * y + rot_mat[1][2]) for (x, y) in landmark])
    face = img_rotated_by_alpha[box.top:box.bottom + 1, box.left:box.right + 1]
    return face, landmark_


def convert_to_square(box):
    square_box = box.copy()
    h = box[:, 3] - box[:, 1] + 1
    w = box[:, 2] - box[:, 0] + 1
    max_side = np.maximum(w, h)
    square_box[:, 0] = box[:, 0] + w * 0.5 - max_side * 0.5
    square_box[:, 1] = box[:, 1] + h * 0.5 - max_side * 0.5
    square_box[:, 2] = square_box[:, 0] + max_side - 1
    square_box[:, 3] = square_box[:, 1] + max_side - 1
    return square_box


def read_annotation(label_path):
    data = dict()
    images = []
    bboxes = []
    with open(label_path, 'r') as f:
        lines = f.readlines()
        pass
    for line in lines:
        labels = line.strip().split(' ')
        imagepath = labels[0]
        if not imagepath:
            break
            pass
        images.append(imagepath)
        one_image_bboxes = []
        xmin = float(labels[1])
        ymin = float(labels[2])
        xmax = float(labels[3])
        ymax = float(labels[4])
        one_image_bboxes.append([xmin, ymin, xmax, ymax])
        bboxes.append(one_image_bboxes)
        pass
    data['images'] = images
    data['bboxes'] = bboxes
    return data


def pad(bboxes, w, h):
    tmpw, tmph = bboxes[:, 2] - bboxes[:, 0] + 1, bboxes[:, 3] - bboxes[:, 1] + 1
    num_box = bboxes.shape[0]
    dx, dy = np.zeros((num_box,)), np.zeros((num_box,))
    edx, edy = tmpw.copy() - 1, tmph.copy() - 1
    x, y, ex, ey = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    tmp_index = np.where(ex > w - 1)
    edx[tmp_index] = tmpw[tmp_index] + w - 2 - ex[tmp_index]
    ex[tmp_index] = w - 1
    tmp_index = np.where(ey > h - 1)
    edy[tmp_index] = tmph[tmp_index] + h - 2 - ey[tmp_index]
    ey[tmp_index] = h - 1
    tmp_index = np.where(x < 0)
    dx[tmp_index] = 0 - x[tmp_index]
    x[tmp_index] = 0
    tmp_index = np.where(y < 0)
    dy[tmp_index] = 0 - y[tmp_index]
    y[tmp_index] = 0
    return_list = [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]
    return_list = [item.astype(np.int32) for item in return_list]
    return return_list


def calibrate_box(bbox, reg):
    bbox_c = bbox.copy()
    w = bbox[:, 2] - bbox[:, 0] + 1
    w = np.expand_dims(w, 1)
    h = bbox[:, 3] - bbox[:, 1] + 1
    h = np.expand_dims(h, 1)
    reg_m = np.hstack([w, h, w, h])
    aug = reg_m * reg
    bbox_c[:, 0:4] = bbox_c[:, 0:4] + aug
    return bbox_c


def py_nms(dets, thresh, mode="Union"):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if mode == "Union":
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            pass
        elif mode == "Minimum":
            ovr = inter / np.minimum(areas[i], areas[order[1:]])
            pass
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
        pass
    return keep


def generate_bbox(cls_map, reg, scale, threshold):
    stride = 2
    cellsize = 12
    t_index = np.where(cls_map > threshold)
    if t_index[0].size == 0:
        return np.array([])
    dx1, dy1, dx2, dy2 = [reg[i, t_index[0], t_index[1]] for i in range(4)]
    reg = np.array([dx1, dy1, dx2, dy2])
    score = cls_map[t_index[0], t_index[1]]
    boundingbox = np.vstack([np.round((stride * t_index[1]) / scale),
                             np.round((stride * t_index[0]) / scale),
                             np.round((stride * t_index[1] + cellsize) / scale),
                             np.round((stride * t_index[0] + cellsize) / scale),
                             score,
                             reg])
    return boundingbox.T


def save_hard_example(save_size):
    filename = r'./dataset/box.txt'
    data = read_annotation(filename)
    im_idx_list = data['images']
    gt_boxes_list = data['bboxes']
    pos_save_dir = r'./%d/positive' % save_size
    part_save_dir = r'./%d/part' % save_size
    neg_save_dir = './%d/negative' % save_size
    if not os.path.exists(pos_save_dir):
        os.mkdir(pos_save_dir)
        pass
    if not os.path.exists(part_save_dir):
        os.mkdir(part_save_dir)
        pass
    if not os.path.exists(neg_save_dir):
        os.mkdir(neg_save_dir)
        pass
    neg_file = open(r'./%d/negative.txt' % save_size, 'w')
    pos_file = open(r'./%d/positive.txt' % save_size, 'w')
    part_file = open(r'./%d/part.txt' % save_size, 'w')
    det_boxes = pickle.load(open('./%d/detections.pkl' % save_size, 'rb'))
    assert len(det_boxes) == len(im_idx_list), "预测结果和真实数据数量不一致"
    n_idx = 0
    p_idx = 0
    d_idx = 0
    pbar = tqdm(total=len(im_idx_list))
    for im_idx, dets, gts in zip(im_idx_list, det_boxes, gt_boxes_list):
        pbar.update(1)
        gts = np.array(gts, dtype=np.float32).reshape(-1, 4)
        if dets.shape[0] == 0:
            continue
            pass
        im_idx = "/".join(im_idx .split("\\"))
        img = cv2.imread(im_idx)
        dets = convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])
        neg_num = 0
        for box in dets:
            x_left, y_top, x_right, y_bottom, _ = box.astype(int)
            width = x_right - x_left + 1
            height = y_bottom - y_top + 1
            if width < 20 or x_left < 0 or y_top < 0 or x_right > img.shape[1] - 1 or y_bottom > img.shape[0] - 1:
                continue
                pass
            Iou = IOU(box, gts)
            cropped_im = img[y_top:y_bottom + 1, x_left:x_right + 1, :]
            resized_im = cv2.resize(cropped_im, (save_size, save_size), interpolation=cv2.INTER_LINEAR)
            if np.max(Iou) < 0.3 and neg_num < 60:
                save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
                save_file = "/".join(save_file.split("\\"))
                neg_file.write(save_file + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1
                neg_num += 1
                pass
            else:
                idx = np.argmax(Iou)
                assigned_gt = gts[idx]
                x1, y1, x2, y2 = assigned_gt
                offset_x1 = (x1 - x_left) / float(width)
                offset_y1 = (y1 - y_top) / float(height)
                offset_x2 = (x2 - x_right) / float(width)
                offset_y2 = (y2 - y_bottom) / float(height)
                if np.max(Iou) >= 0.65:
                    save_file = os.path.join(pos_save_dir, "%s.jpg" % p_idx)
                    save_file = "/".join(save_file.split("\\"))
                    pos_file.write(
                        save_file + ' 1 %.2f %.2f %.2f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    p_idx += 1
                    pass
                elif np.max(Iou) >= 0.4:
                    save_file = os.path.join(part_save_dir, "%s.jpg" % d_idx)
                    save_file = "/".join(save_file.split("\\"))
                    part_file.write(
                        save_file + ' -1 %.2f %.2f %.2f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    d_idx += 1
                    pass
                pass
            pass
        pass
    pbar.close()
    neg_file.close()
    part_file.close()
    pos_file.close()
    pass
