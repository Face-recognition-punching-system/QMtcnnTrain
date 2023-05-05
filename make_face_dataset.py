from mtcnn_cv2 import MTCNN
import cv2
import glob
from tqdm import tqdm


if __name__ == '__main__':
    detector = MTCNN()
    folder = glob.glob(r"F:/BaiduNetdiskDownload/64_CASIA-FaceV5/CASIA-FaceV5/*")
    box_file = open("./dataset/box.txt", 'w')
    """
    box_file.writelines('%s\n' % (str(length)))
    box_file.writelines('image_id x_1 y_1 width height\n')
    keypoints_file.writelines('%s\n' % (str(length)))
    keypoints_file.writelines(
        'lefteye_x lefteye_y righteye_x righteye_y nose_x nose_y leftmouth_x leftmouth_y rightmouth_x rightmouth_y\n')
    """
    for i in tqdm(folder):
        i = "/".join(i.split("\\"))
        img = cv2.imread(i)
        faces = detector.detect_faces(img)
        for face in faces:
            keypoints = face['keypoints']
            box = face["box"]
            box_file.writelines('%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s\n' % (i, box[0], box[1], box[0] + box[2], box[1] + box[3],keypoints['left_eye'][0], keypoints['left_eye'][1], keypoints['right_eye'][0], keypoints['right_eye'][1],
            keypoints['nose'][0], keypoints['nose'][1], keypoints['mouth_left'][0], keypoints['mouth_left'][1],
            keypoints['mouth_right'][0], keypoints['mouth_right'][1]))
            pass
        pass
    box_file.close()
    pass
