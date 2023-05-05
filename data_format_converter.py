import os
import struct
import uuid
from tqdm import tqdm
import cv2


class DataSetWriter(object):
    def __init__(self, prefix):
        self.data_file = open(prefix + '.data', 'wb')
        self.header_file = open(prefix + '.header', 'wb')
        self.label_file = open(prefix + '.label', 'wb')
        self.offset = 0
        self.header = ''
        pass

    def add_img(self, key, img):
        self.data_file.write(struct.pack('I', len(key)))
        self.data_file.write(key.encode('ascii'))
        self.data_file.write(struct.pack('I', len(img)))
        self.data_file.write(img)
        self.offset += 4 + len(key) + 4
        self.header = key + '\t' + str(self.offset) + '\t' + str(len(img)) + '\n'
        self.header_file.write(self.header.encode('ascii'))
        self.offset += len(img)
        pass

    def add_label(self, label):
        self.label_file.write(label.encode('ascii') + '\n'.encode('ascii'))
        pass
    pass


def convert_data(data_folder, output_prefix):
    data_list_path = os.path.join(data_folder, 'all_data_list.txt')
    train_list = open(data_list_path, "r").readlines()
    train_image_list = []
    for i, item in enumerate(train_list):
        sample = item.split(' ')
        image = sample[0]
        label = int(sample[1])
        bbox = [0, 0, 0, 0]
        landmark = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        if len(sample) == 6:
            bbox = [float(i) for i in sample[2:]]
            pass
        if len(sample) == 12:
            landmark = [float(i) for i in sample[2:]]
            pass
        train_image_list.append((image, label, bbox, landmark))
        pass
    print("训练数据大小：", len(train_image_list))
    writer = DataSetWriter(output_prefix)
    for image, label, bbox, landmark in tqdm(train_image_list):
        try:
            key = str(uuid.uuid1())
            img = cv2.imread(image)
            _, img = cv2.imencode('.bmp', img)
            writer.add_img(key, img.tostring())
            label_str = str(label)
            bbox_str = ' '.join([str(x) for x in bbox])
            landmark_str = ' '.join([str(x) for x in landmark])
            writer.add_label('\t'.join([key, bbox_str, landmark_str, label_str]))
            pass
        except:
            continue
            pass
        pass
    pass
