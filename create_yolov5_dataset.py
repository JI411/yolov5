"""
Create yolov5 dataset to "TensorFlow - Help Protect the Great Barrier Reef" competition
"""
# pylint: disable=line-too-long

import ast
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from tqdm import tqdm


def get_bbox(annots: list[dict[str, int]]) -> list[list[int]]:
    """
    Transform [{'x': 525, 'y': 196, 'width': 63, 'height': 51}, ...] to [[525, 196, 63, 51], ...]

    :param annots: annotations in great barrier reef competitions format
    :return: list of bboxes from annotations
    """
    return [list(annot.values()) for annot in annots]


def rm_tree(path: Path) -> None:
    """
    Remove path; if is dir, remove child paths

    :param path: path to remove
    """
    for child in path.glob('*'):
        if child.is_file():
            child.unlink()
        else:
            rm_tree(child)
    path.rmdir()

def remove_and_recreate_dataset_dirs(root: Path, clear_dirs: bool = True) -> None:
    """
    Create yolov5 dirs structure

    :param root: path to yolov5 dataset dir
    :param clear_dirs: if true, remove existing dits and that they contain
    """
    images = root / 'images'
    labels = root / 'labels'
    images_train = images / 'train'
    labels_train = labels / 'train'
    images_val = images / 'val'
    labels_val = labels / 'val'

    if clear_dirs:
        rm_tree(root)

    for directory in (root, images, labels, images_train, images_val, labels_train, labels_val):
        if not directory.is_dir():
            directory.mkdir()

    for directory in (images_train, images_val, labels_train, labels_val):
        with open(directory / '.gitignore', 'w', encoding='utf-8') as gitignore:
            gitignore.write('')

    with open(labels_train / 'classes.txt', 'w', encoding='utf-8') as classes_labelimg:
        classes_labelimg.write('star')
    with open(labels_val / 'classes.txt', 'w', encoding='utf-8') as classes_labelimg:
        classes_labelimg.write('star')


def copy_image(path: str, new_path: str) -> None:
    """
    Create copy of image from path to new_path

    :param path: image source
    :param new_path: path to new image
    """
    img = cv2.imread(path)
    cv2.imwrite(new_path, img)

def coco2yolo(height: int,
              width: int,
              bboxes: NDArray[NDArray[np.float64]]) -> NDArray[NDArray[np.float64]]:
    """
    Transform coco bbox format to yolo bbox format.
    coco == [xmin, ymin, w, h]
    yolo == [x_center, y_center, w, h] (normalized to [0, 1])

    :param height: image height
    :param width: image width
    :param bboxes: coco bboxes
    :return: yolo bboxes
    """

    bboxes = bboxes.copy().astype(float)  # otherwise, all value will be 0 as voc_pascal dtype is np.int

    # normalizing
    bboxes[..., [0, 2]] = bboxes[..., [0, 2]] / width
    bboxes[..., [1, 3]] = bboxes[..., [1, 3]] / height

    # conversion (xmin, ymin) => (xc, yc)
    bboxes[..., [0, 1]] = bboxes[..., [0, 1]] + bboxes[..., [2, 3]] / 2
    return bboxes

def new_path_to_image_and_annotation(yolov5_dataset_dir: Path, image_path: Path, is_train: bool) -> Tuple[Path, Path]:
    """
    Create new paths for image and annotations in yolov5 dataset from old path in format ../folder_name/image_name.jpg

    :param yolov5_dataset_dir: root dir of yolov5 dataset
    :param image_path: old path to image
    :param is_train: if true, add this image to train, else to validation
    :return: annotations path and new path to image
    """
    train_or_val = 'train' if is_train else 'val'
    filename = image_path.name
    video_id = image_path.parent.name
    label_path = yolov5_dataset_dir / 'labels' / train_or_val / f'{video_id}_{filename}'.replace('.jpg', '.txt')
    new_image_path = yolov5_dataset_dir / 'images' / train_or_val / f'{video_id}_{filename}'
    return label_path, new_image_path

def create_dataset(dataframe, yolov5_dataset_dir) -> None:
    """
    Create yolov5 dataset (write annotations and copy images to yolov5_dataset_dir) from dataframe

    :param dataframe: pandas df with columns:
            * name - path to image
            * is_train - if true, add this image to train, else to validation
            * height - image height
            * width - image width
            * bboxes - bboxes in coco format
    :param yolov5_dataset_dir: root dir of yolov5 dataset
    """
    remove_and_recreate_dataset_dirs(root=yolov5_dataset_dir)
    all_bboxes = []
    for row in tqdm(dataframe.itertuples()):
        label_path, new_image_path = new_path_to_image_and_annotation(yolov5_dataset_dir=yolov5_dataset_dir,
                                                                      image_path=Path(row.image_path),
                                                                      is_train=row.is_train)

        copy_image(path=row.image_path, new_path=str(new_image_path))

        bboxes_coco = np.array(row.bboxes).astype(np.float32).copy()
        labels = (0,) * len(bboxes_coco)
        annot = ''

        # Create YOLO Annotation
        with open(label_path, 'w', encoding='utf-8') as annotations_file:
            bboxes_yolo = coco2yolo(row.height, row.width, bboxes_coco)
            bboxes_yolo = np.clip(bboxes_yolo, 0, 1)
            all_bboxes.extend(bboxes_yolo)
            for label, bbox in zip(labels, bboxes_yolo):
                bbox = bbox.astype(str)
                bbox = ' '.join(bbox)
                annot += f'{label} {bbox} \n'
            annotations_file.write(annot)


YOLOV5_DATASET_DIR = Path().cwd() / 'yolov5_dataset'

df = pd.read_csv(Path().cwd() /
                 'reef-cv-strategy-subsequences-dataframes' /
                 'train-validation-split' /
                 'train-0.1.csv')
df = df[df['has_annotations']]

df['image_path'] = df['image_path'].str.replace('../input', str(Path().cwd()), regex=False)
df['width'] = 1280
df['height'] = 720
df['annotations'] = df['annotations'].apply(ast.literal_eval)
df['bboxes'] = df['annotations'].apply(get_bbox)
create_dataset(dataframe=df, yolov5_dataset_dir=YOLOV5_DATASET_DIR)
print('Done!')
