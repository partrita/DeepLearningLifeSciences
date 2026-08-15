from __future__ import division
from __future__ import unicode_literals

import os
import logging
import deepchem
import numpy as np
import pandas as pd

"""
당뇨병성 망막병증(Diabetic Retinopathy) 영상 로더입니다.
"""

logger = logging.getLogger(__name__)


def load_images_DR(split="random", seed=None):
    """당뇨병성 망막병증(DR) 영상을 불러오는 로더입니다."""
    data_dir = deepchem.utils.get_data_dir()
    images_path = os.path.join(data_dir, "DR", "train")
    label_path = os.path.join(data_dir, "DR", "trainLabels.csv")
    if not os.path.exists(images_path) or not os.path.exists(label_path):
        logger.warn(
            "데이터를 찾을 수 없습니다. \n\
        모든 영상 파일(.png)은 다음 폴더에 저장되어야 합니다: $DEEPCHEM_DATA_DIR/DR/train/,\n\
        해당 라벨 파일은 $DEEPCHEM_DATA_DIR/DR/trainLabels.csv 경로에 저장되어야 합니다.\n\
        데이터 접근에 대해서는 https://www.kaggle.com/c/diabetic-retinopathy-detection를 참조하세요."
        )

    image_names = os.listdir(images_path)
    raw_images = []
    for im in image_names:
        if (
            im.endswith(".jpeg")
            and not im.startswith("cut_")
            and "cut_" + im not in image_names
        ):
            raw_images.append(im)
    if len(raw_images) > 0:
        cut_raw_images(raw_images, images_path)

    image_names = [
        p
        for p in os.listdir(images_path)
        if p.startswith("cut_") and p.endswith(".png")
    ]

    all_labels = dict(zip(*np.transpose(np.array(pd.read_csv(label_path)))))

    print("전체 영상 개수: %d" % len(image_names))
    labels = np.array(
        [all_labels[os.path.splitext(n)[0][4:]] for n in image_names]
    ).reshape((-1, 1))
    image_full_paths = [os.path.join(images_path, n) for n in image_names]

    classes, cts = np.unique(list(all_labels.values()), return_counts=True)
    weight_ratio = dict(zip(classes, np.max(cts) / cts.astype(float)))
    weights = np.array([weight_ratio[label[0]] for label in labels]).reshape((-1, 1))

    dat = deepchem.data.ImageDataset(image_full_paths, labels, weights)
    if split is None:
        return dat

    splitters = {
        "index": deepchem.splits.IndexSplitter(),
        "random": deepchem.splits.RandomSplitter(),
    }
    if seed is not None:
        np.random.seed(seed)
    splitter = splitters[split]
    train, valid, test = splitter.train_valid_test_split(dat)
    all_dataset = (train, valid, test)
    return all_dataset


def cut_raw_images(all_images, path):
    """영상 전처리를 수행합니다:
    (1) 망막을 포함한 중앙 영역을 정사각형으로 자릅니다.
    (2) 해상도를 512 * 512로 조정합니다.
    """
    print("처리할 영상 개수: %d" % len(all_images))
    try:
        import cv2
    except:  # noqa: E722
        logger.warn("영상 전처리를 위해 OpenCV가 필요합니다.")
        return

    for i, img_path in enumerate(all_images):
        if i % 100 == 0:
            print("%d번째 영상 처리 중..." % i)
        if os.path.exists(
            os.path.join(path, "cut_" + os.path.splitext(img_path)[0] + ".png")
        ):
            continue
        img = cv2.imread(os.path.join(path, img_path))
        edges = cv2.Canny(img, 10, 30)
        coords = list(zip(*np.where(edges > 0)))
        n_p = len(coords)

        coords.sort(key=lambda x: (x[0], x[1]))
        center_0 = int((coords[int(0.01 * n_p)][0] + coords[int(0.99 * n_p)][0]) / 2)
        coords.sort(key=lambda x: (x[1], x[0]))
        center_1 = int((coords[int(0.01 * n_p)][1] + coords[int(0.99 * n_p)][1]) / 2)

        edge_size = min(
            [center_0, img.shape[0] - center_0, center_1, img.shape[1] - center_1]
        )
        img_cut = img[
            (center_0 - edge_size) : (center_0 + edge_size),
            (center_1 - edge_size) : (center_1 + edge_size),
        ]
        img_cut = cv2.resize(img_cut, (512, 512))
        cv2.imwrite(
            os.path.join(path, "cut_" + os.path.splitext(img_path)[0] + ".png"), img_cut
        )
