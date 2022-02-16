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
    parser.add_argument('--data', '-d', type=str, default='yolov5_dataset_01_augall',
                        help='path to the folder for saving processed images')
    return parser.parse_args()

class SmallObjectAugmentation(object):
    def __init__(self, thresh=64*64, prob=0.5, copy_times=3, epochs=30, all_objects=False, one_object=False):
        # sourcery skip: square-identity
        """
        sample = {'img':img, 'annot':annots}
        img = [height, width, 3]
        annot = [xmin, ymin, xmax, ymax, label]
        thresh： the detection threshold of the small object. If annot_h * annot_w < thresh, the object is small
        prob: the prob to do small object augmentation
        epochs: the epochs to do
        """
        self.thresh = thresh
        self.prob = prob
        self.copy_times = copy_times
        self.epochs = epochs
        self.all_objects = all_objects
        self.one_object = one_object
        if self.all_objects or self.one_object:
            self.copy_times = 1

    def issmallobject(self, h, w):
        return h * w <= self.thresh

    @staticmethod
    def compute_overlap(annot_a, annot_b):
        if annot_a is None:
            return False
        left_max = max(annot_a[0], annot_b[0])
        top_max = max(annot_a[1], annot_b[1])
        right_min = min(annot_a[2], annot_b[2])
        bottom_min = min(annot_a[3], annot_b[3])
        inter = max(0, (right_min-left_max)) * max(0, (bottom_min-top_max))
        return inter != 0

    def same_overlap(self, new_annot, annots):
        return any(self.compute_overlap(new_annot, annot) for annot in annots)

    def create_copy_annot(self, h, w, annot, annots):
        # sourcery skip: for-index-underscore
        annot = annot.astype(int)
        annot_h, annot_w = annot[3] - annot[1], annot[2] - annot[0]
        for epoch in range(self.epochs):
            random_x, random_y = np.random.randint(int(annot_w / 2), int(w - annot_w / 2)), \
                                 np.random.randint(int(annot_h / 2), int(h - annot_h / 2))
            xmin, ymin = random_x - annot_w / 2, random_y - annot_h / 2
            xmax, ymax = xmin + annot_w, ymin + annot_h
            if xmin < 0 or xmax > w or ymin < 0 or ymax > h:
                continue
            new_annot = np.array([xmin, ymin, xmax, ymax, annot[4]]).astype(int)

            if self.same_overlap(new_annot, annots):
                continue

            return new_annot
        return None

    @staticmethod
    def add_patch_in_img(annot, copy_annot, image):
        copy_annot = copy_annot.astype(int)
        image[annot[1]:annot[3], annot[0]:annot[2], :] = image[copy_annot[1]:copy_annot[3], copy_annot[0]:copy_annot[2], :]
        return image

    def __call__(self, sample):
        if self.all_objects and self.one_object:
            return sample
        if np.random.rand() > self.prob:
            return sample

        img, annots = sample['img'], sample['annot']
        h, w = img.shape[0], img.shape[1]

        small_object_list = []
        for idx in range(annots.shape[0]):
            annot = annots[idx]
            annot_h, annot_w = annot[3] - annot[1], annot[2] - annot[0]
            if self.issmallobject(annot_h, annot_w):
                small_object_list.append(idx)

        l = len(small_object_list)
        # No Small Object
        if l == 0:
            return sample

        # Refine the copy_object by the given policy
        # Policy 2:
        copy_object_num = np.random.randint(0, l)
        # Policy 3:
        if self.all_objects:
            copy_object_num = l
        # Policy 1:
        if self.one_object:
            copy_object_num = 1

        random_list = random.sample(range(l), copy_object_num)
        annot_idx_of_small_object = [small_object_list[idx] for idx in random_list]
        select_annots = annots[annot_idx_of_small_object, :]
        annots = annots.tolist()
        try:
            for idx in range(copy_object_num):
                annot = select_annots[idx]
                annot_h, annot_w = annot[3] - annot[1], annot[2] - annot[0]

                if self.issmallobject(annot_h, annot_w) is False: continue

                for _ in range(self.copy_times):
                    new_annot = self.create_copy_annot(h, w, annot, annots,)
                    if new_annot is not None:
                        img = self.add_patch_in_img(new_annot, annot, img)
                        annots.append(new_annot)
            return {'img': img, 'annot': np.array(annots)}
        except:
            return sample


def copy_paste(opt):

    def get_pascal_voc_boxes(bbox, width, height):
        x, y, w, h = bbox
        x, w = x * width, w * width
        y, h = y * height, h * height
        return [int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)]  # x1, y1, x2, y2

    def save_labels_and_boxes(labels, boxes, image_name, image):
        cv2.imwrite(opt.images / image_name, image)
        description = ''
        for label, bbox in zip(labels, boxes):
            description += str(label) + ' ' + ' '.join(str(b) for b in bbox) + '\n'

        with open(opt.labels / image_name.replace('.jpg', '.txt'), 'w+') as txt_file:
            txt_file.write(description[:-1])

    aug = SmallObjectAugmentation(thresh=128 * 128,
                                  prob=0.5,
                                  copy_times=3,
                                  epochs=30,
                                  all_objects=False,
                                  one_object=False)

    num_new_boxes = 0
    for i, img_name in enumerate(opt.images.glob('*.jpg')):

        with open(opt.labels / img_name.with_suffix('.txt').name, 'r') as file:
            file_data = [[float(i) for i in line.split()] for line in file]

        target = {'boxes': [], 'labels': []}
        for line in file_data:
            target['labels'].append(line[0])
            target['boxes'].append(line[1:])
        # {'boxes': [[0.35, 0.26, 0.40, 0.39]], 'labels': [15]}
        img = cv2.imread(str(img_name))
        height, width, _ = img.shape

        # copy to image
        annots = [get_pascal_voc_boxes(box, height=height, width=width) + [label]
                  for box, label in zip(target['boxes'], target['labels'])]
        transformed = aug({'img': img, 'annot': np.array(annots)})
        yolo_annot = []
        for annot in transformed['annot']:
            x1, y1, x2, y2, label = annot
            x1 = x1 / width
            x2 = x2 / width
            y1 = y1 / height
            y2 = y2 / height
            w = (max(x1, x2) - min(x1, x2))
            h = (max(y1, y2) - min(y1, y2))
            xc = min(x1, x2) + w / 2
            yc = min(y1, y2) + h / 2

            yolo_annot.append(f'0 {xc} {yc} {w} {h}')

        cv2.imwrite(str(img_name), transformed['img'])

        with open(opt.labels / img_name.with_suffix('.txt').name, 'w+') as txt_file:
            txt_file.write('\n'.join(yolo_annot))

    len_train = len(os.listdir(opt.labels))
    print(f'train: {len_train}, copy (new boxes): {num_new_boxes}')


if __name__ == '__main__':
    options = parse_options()
    options.images = Path(options.data) / 'images' / 'train'
    options.labels = Path(options.data) / 'labels' / 'train'
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

