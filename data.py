import mmap
import cv2
import numpy as np
from torch.utils.data import Dataset


class ImageData(object):
    def __init__(self, data_path):
        self.offset_dict = {}
        for line in open(data_path + '.header', 'rb'):
            key, val_pos, val_len = line.split('\t'.encode('ascii'))
            self.offset_dict[key] = (int(val_pos), int(val_len))
        self.fp = open(data_path + '.data', 'rb')
        self.m = mmap.mmap(self.fp.fileno(), 0, access=mmap.ACCESS_READ)
        print('正在加载数据标签...')
        self.label = {}
        self.box = {}
        self.landmark = {}
        label_path = data_path + '.label'
        for line in open(label_path, 'rb'):
            key, bbox, landmark, label = line.split(b'\t')
            self.label[key] = int(label)
            self.box[key] = [float(x) for x in bbox.split()]
            self.landmark[key] = [float(x) for x in landmark.split()]
        print('数据加载完成，总数据量为：%d' % len(self.label))
        pass

    def get_img(self, key):
        p = self.offset_dict.get(key, None)
        if p is None:
            return None
        val_pos, val_len = p
        return self.m[val_pos:val_pos + val_len]

    def get_label(self, key):
        return self.label.get(key)

    def get_bbox(self, key):
        return self.box.get(key)

    def get_landmark(self, key):
        return self.landmark.get(key)

    def get_keys(self):
        return self.label.keys()
    pass


def process(image):
    image = np.fromstring(image, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    assert (image is not None), 'image is None'
    image = np.array(image).astype(np.float32)
    image = image.transpose((2, 0, 1))
    image = (image - 127.5) / 128
    return image


class CustomDataset(Dataset):
    def __init__(self, data_path):
        super(CustomDataset, self).__init__()
        self.imageData = ImageData(data_path)
        self.keys = self.imageData.get_keys()
        self.keys = list(self.keys)
        np.random.shuffle(self.keys)
        pass

    def __getitem__(self, idx):
        key = self.keys[idx]
        img = self.imageData.get_img(key)
        assert (img is not None)
        label = self.imageData.get_label(key)
        assert (label is not None)
        bbox = self.imageData.get_bbox(key)
        landmark = self.imageData.get_landmark(key)
        img = process(img)
        label = np.array([label], np.int64)
        bbox = np.array(bbox, np.float32)
        landmark = np.array(landmark, np.float32)
        return img, label, bbox, landmark

    def __len__(self):
        return len(self.keys)
    pass
