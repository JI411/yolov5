#!/usr/bin/env python
# coding: utf-8

import os
import cv2
import time
import random
import argparse
import numpy as np
from pathlib import Path

SEED = 411
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-copy', '-c', type=int, default=3,
                        help='add no more than max-copy new boxes to train')
    parser.add_argument('--add-all', '-a', action='store_true',
                        help='add no less than max-copy new boxes to train')
    parser.add_argument('--max-size', '-s', type=int, default=3,
                        help='copy-paste box if image_size < max-size in every dimensions')
    parser.add_argument('--data', '-d', type=str, default='yolov5/data/dataset',
                        help='path to the folder for saving processed images')
    return parser.parse_args()

def copy_paste(opt):

    def get_pascal_voc_boxes(bbox, width, height):
        x, y, w, h = bbox
        x, w = x * width, w * width
        y, h = y * height, h * height
        return int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)  # x1, y1, x2, y2

    def save_labels_and_boxes(labels, boxes, image_name, image):
        cv2.imwrite(opt.images / image_name, image)
        description = ''
        for label, bbox in zip(labels, boxes):
            description += str(label) + ' ' + ' '.join(str(b) for b in bbox) + '\n'

        with open(opt.labels / image_name.replace('.jpg', '.txt'), 'w+') as txt_file:
            txt_file.write(description[:-1])

    def intercept_skip_crop(box_a, box_b):
        xa, ya, wa, ha = box_a
        xb, yb, wb, hb = box_b
        # w, h = max((wa, wb)), max((ha, hb))
        return abs(xa - xb) < wa / 2 and abs(ya - yb) < ha / 2

    num_new_boxes = 0
    for i, img_name in enumerate(opt.images.glob('*.jpg')):

        with open(opt.labels / img_name.replace('.jpg', '.txt'), 'r') as file:
            file_data = [[float(i) for i in line[:-1].split()] for line in file]

        target = {'boxes': [], 'labels': []}
        for line in file_data:
            target['labels'].append(int(line[0]))
            target['boxes'].append(line[1:])
        # {'boxes': [[0.35, 0.26, 0.40, 0.39]], 'labels': [15]}
        img = cv2.imread(opt.images / img_name)

        # cropped to train
        for j, box in enumerate(target['boxes']):
            labels_crop = target['labels'].copy()
            boxes_crop = target['boxes'].copy()
            labels_crop.pop(j)
            box = boxes_crop.pop(j)

            if any(intercept_skip_crop(box, box_crop) for box_crop in boxes_crop):
                continue
            num_new_boxes += 1

            cropped_img = img.copy()
            x1, y1, x2, y2 = get_pascal_voc_boxes(box)
            cv2.rectangle(cropped_img, (x1, y1), (x2, y2), average, -1)
            save_labels_and_boxes(labels=labels_crop, boxes=boxes_crop,
                                  image_name=f'cropped{j}_{img_name}', image=cropped_img)
    len_train = len(os.listdir(YOLO_IMAGES_PATH))
    print(f'train: {len_train}, empty: {n_empty}, cropped: {n_cropped}')


if __name__ == '__main__':
    options = parse_options()
    options.images = Path('options') / 'images' / 'train'
    options.labels = Path('options') / 'labels' / 'train'
    if not options.images.is_dir():
        print(f'your path to images: {options.images}')
        raise ValueError('yolo_data_path must be path to labels and images dir')
    elif not options.labels.is_dir():
        print(f'your path to labels: {options.labels}')
        raise ValueError('yolo_data_path must be path to labels and images dir')

    print()
    print('***** Start *****')

    start_time = time.time()
    copy_paste(opt=options)

    print(f'--- {(time.time() - start_time) // 60} минут ---')
    print('***** Finish *****')
    print()

