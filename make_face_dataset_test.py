from mtcnn_cv2 import MTCNN
import cv2

if __name__ == '__main__':
    detector = MTCNN()
    img = cv2.imread("E:/Users/Administrator/Pictures/lena512color.jpg")
    faces = detector.detect_faces(img)
    for face in faces:
        """
        {
            'box': [202, 182, 158, 210], 
            'confidence': 0.9974948167800903,
            'keypoints': {
                'left_eye': (267, 265), 
                'right_eye': (335, 267), 
                'nose': (316, 317), 
                'mouth_left': (265, 348),
                'mouth_right': (321, 352)}
            }
        """
        keypoints = face['keypoints']
        box = face["box"]
        cv2.rectangle(img,
                      (box[0], box[1]),
                      (box[0] + box[2], box[1] + box[3]),
                      (0, 155, 255),
                      2)
        cv2.circle(img, (keypoints['left_eye']), 2, (0, 155, 255), 2)
        cv2.circle(img, (keypoints['right_eye']), 2, (0, 155, 255), 2)
        cv2.circle(img, (keypoints['nose']), 2, (0, 155, 255), 2)
        cv2.circle(img, (keypoints['mouth_left']), 2, (0, 155, 255), 2)
        cv2.circle(img, (keypoints['mouth_right']), 2, (0, 155, 255), 2)

    cv2.imshow("test", img)
    cv2.waitKey(0)