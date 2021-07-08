# -*- coding: utf-8 -*-
# @Brief: 获取预测框的数据，并转mAP的txt检测格式
from predict import Yolov4Predict
from PIL import Image
import config.config as cfg
import os
from tqdm import tqdm


class YOLOmAP(Yolov4Predict):
    def __init__(self, model_path):
        super().__init__(model_path)
        self.score = 0.5

    def detect_single_image(self, image, image_id):
        f = open("./input/detection-results/" + image_id + ".txt", "w")

        # 读取预测结果
        out_boxes, out_scores, out_classes = self.predict(image)

        for i, c in list(enumerate(out_classes)):
            predicted_class = self.class_names[c]
            top, left, bottom, right = out_boxes[i]
            score = str(float(out_scores[i]))

            f.write("{} {:.6} {} {} {} {}\n".format(
                predicted_class, score, str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    yolo = YOLOmAP("../model/yolov4_15.1723.h5")

    image_infos = open("../config/test.txt").read().strip().split('\n')

    if not os.path.exists("./input"):
        os.makedirs("./input")
    if not os.path.exists("./input/detection-results"):
        os.makedirs("./input/detection-results")
    if not os.path.exists("./input/images-optional"):
        os.makedirs("./input/images-optional")

    process_bar = tqdm(range(len(image_infos)), ncols=100, unit="step")
    for i in process_bar:
        image_boxes = image_infos[i].split(' ')
        image_path = image_boxes[0]
        image_id = os.path.basename(image_path)[:-4]

        image = Image.open(image_path)
        # image = Image.open(image_path)
        # 开启后在之后计算mAP可以可视化
        # image.save("./input/images-optional/"+image_id+".jpg")
        yolo.detect_single_image(image, image_id)
        # print(image_id, " done!")

    print("Conversion completed!")
