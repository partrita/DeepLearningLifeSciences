#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
2018년 9월 10일 월요일 생성

@작성자: zqwu
"""

import deepchem as dc
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers

from sklearn.metrics import confusion_matrix, accuracy_score


class DRModel(dc.models.KerasModel):
    def __init__(
        self,
        n_tasks=1,
        image_size=512,
        n_downsample=6,
        n_init_kernel=16,
        n_fully_connected=[1024],
        n_classes=5,
        augment=False,
        batch_size=100,
        **kwargs,
    ):
        """
        매개변수(Parameters)
        ----------
        n_tasks: int
          작업(Task)의 수
        image_size: int
          입력 영상의 해상도 (정사각형)
        n_downsample: int
          2의 거듭제곱 비율로 다운샘플링 수행
        n_init_kernel: int
          첫 번째 합성곱 층(Convolutional layer)의 커널 크기
        n_fully_connected: list of int
          합성곱 연산 후 이어지는 완전 연결 층(FC layer)의 구조
        n_classes: int
          예측할 클래스 수 (분류 모드에서만 사용)
        augment: bool
          데이터 증강(Data augmentation) 사용 여부
        """
        self.n_tasks = n_tasks
        self.image_size = image_size
        self.n_downsample = n_downsample
        self.n_init_kernel = n_init_kernel
        self.n_fully_connected = n_fully_connected
        self.n_classes = n_classes
        self.augment = augment

        # 입력 플레이스홀더(Inputs placeholder)
        self.inputs = tf.keras.Input(
            shape=(self.image_size, self.image_size, 3), dtype=tf.float32
        )
        # 데이터 전처리 및 증강
        in_layer = DRAugment(
            self.augment, batch_size, size=(self.image_size, self.image_size)
        )(self.inputs)
        # 첫 번째 합성곱 층
        in_layer = layers.Conv2D(
            int(self.n_init_kernel), kernel_size=7, padding="same"
        )(in_layer)
        in_layer = layers.BatchNormalization()(in_layer)
        in_layer = layers.ReLU()(in_layer)

        # 최대 풀링(Max pooling)을 통한 다운샘플링
        res_in = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))(in_layer)

        for ct_module in range(self.n_downsample - 1):
            # 각 모듈은 잔차 합성곱 블록(Residual convolutional block)과
            # 이어지는 합성곱 다운샘플링 층으로 구성됩니다.
            in_layer = layers.Conv2D(
                int(self.n_init_kernel * 2 ** (ct_module - 1)),
                kernel_size=1,
                padding="same",
            )(res_in)
            in_layer = layers.BatchNormalization()(in_layer)
            in_layer = layers.ReLU()(in_layer)
            in_layer = layers.Conv2D(
                int(self.n_init_kernel * 2 ** (ct_module - 1)),
                kernel_size=3,
                padding="same",
            )(in_layer)
            in_layer = layers.BatchNormalization()(in_layer)
            in_layer = layers.ReLU()(in_layer)
            in_layer = layers.Conv2D(
                int(self.n_init_kernel * 2**ct_module), kernel_size=1, padding="same"
            )(in_layer)
            res_a = layers.BatchNormalization()(in_layer)

            res_out = res_in + res_a
            res_in = layers.Conv2D(
                int(self.n_init_kernel * 2 ** (ct_module + 1)),
                kernel_size=3,
                strides=2,
                activation=tf.nn.relu,
                padding="same",
            )(res_out)
            res_in = layers.BatchNormalization()(res_in)

        # 최종 결과에 대한 최대 풀링
        in_layer = layers.Lambda(lambda x: tf.reduce_max(x, axis=(1, 2)))(res_in)

        regularizer = tf.keras.regularizers.l2(0.1)
        for layer_size in self.n_fully_connected:
            # 완전 연결 층
            in_layer = layers.Dense(
                layer_size, activation=tf.nn.relu, kernel_regularizer=regularizer
            )(in_layer)
            # 은닉층을 위한 드롭아웃(Dropout)
            # in_layer = layers.Dropout(0.25)(in_layer)

        logit_pred = layers.Dense(self.n_tasks * self.n_classes)(in_layer)
        logit_pred = layers.Reshape((self.n_tasks, self.n_classes))(logit_pred)
        output = layers.Softmax()(logit_pred)

        keras_model = tf.keras.Model(inputs=self.inputs, outputs=[output, logit_pred])
        super(DRModel, self).__init__(
            keras_model,
            loss=dc.models.losses.SparseSoftmaxCrossEntropy(),
            output_types=["prediction", "loss"],
            batch_size=batch_size,
            **kwargs,
        )


def DRAccuracy(y, y_pred):
    y = np.argmax(y, 1)
    y_pred = np.argmax(y_pred, 1)
    return accuracy_score(y, y_pred)


def DRSpecificity(y, y_pred):
    y_pred = (np.argmax(y_pred, 1) > 0) * 1
    y = (y > 0) * 1
    TN = sum((1 - y_pred) * (1 - y))
    N = sum(1 - y)
    return float(TN) / N


def DRSensitivity(y, y_pred):
    y = np.argmax(y, 1)
    y_pred = (np.argmax(y_pred, 1) > 0) * 1
    y = (y > 0) * 1
    TP = sum(y_pred * y)
    P = sum(y)
    return float(TP) / P


def ConfusionMatrix(y, y_pred):
    y = np.argmax(y, 1)
    y_pred = np.argmax(y_pred, 1)
    return confusion_matrix(y, y_pred)


def QuadWeightedKappa(y, y_pred):
    y = np.argmax(y, 1)
    y_pred = np.argmax(y_pred, 1)
    cm = confusion_matrix(y, y_pred)
    classes_y, counts_y = np.unique(y, return_counts=True)
    classes_y_pred, counts_y_pred = np.unique(y_pred, return_counts=True)
    E = np.zeros((classes_y.shape[0], classes_y.shape[0]))
    for i, c1 in enumerate(classes_y):
        for j, c2 in enumerate(classes_y_pred):
            E[c1, c2] = counts_y[i] * counts_y_pred[j]
    E = E / np.sum(E) * np.sum(cm)
    w = np.zeros((classes_y.shape[0], classes_y.shape[0]))
    for i in range(classes_y.shape[0]):
        for j in range(classes_y.shape[0]):
            w[i, j] = float((i - j) ** 2) / (classes_y.shape[0] - 1) ** 2
    re = 1 - np.sum(w * cm) / np.sum(w * E)
    return re


class DRAugment(layers.Layer):
    def __init__(
        self,
        augment,
        batch_size,
        distort_color=True,
        central_crop=True,
        size=(512, 512),
        **kwargs,
    ):
        """
        매개변수(Parameters)
        ----------
        augment: bool
          데이터 증강 사용 여부
        batch_size: int
          배치 내 영상 개수
        distort_color: bool
          색상 왜곡 무작위 적용 여부
        central_crop: bool
          중앙 중심의 무작위 자르기 적용 여부
        size: int
          입력 영상의 해상도 (정사각형)
        """
        self.augment = augment
        self.batch_size = batch_size
        self.distort_color = distort_color
        self.central_crop = central_crop
        self.size = size
        super(DRAugment, self).__init__(**kwargs)

    def call(self, inputs, training=True):
        parent_tensor = inputs / 255.0
        if not self.augment or not training:
            return parent_tensor
        else:

            def preprocess(img):
                img = tf.image.random_flip_left_right(img)
                img = tf.image.random_flip_up_down(img)
                img = tf.image.rot90(img, k=np.random.randint(0, 4))
                if self.distort_color:
                    img = tf.image.random_brightness(img, max_delta=32.0 / 255.0)
                    img = tf.image.random_saturation(img, lower=0.5, upper=1.5)
                    img = tf.clip_by_value(img, 0.0, 1.0)
                if self.central_crop:
                    # 절단된 가우시안 분포로부터 샘플 비율을 추출합니다.
                    img = tf.image.central_crop(
                        img, np.clip(np.random.normal(1.0, 0.06), 0.8, 1.0)
                    )
                    img = tf.image.resize(
                        tf.expand_dims(img, 0), tf.convert_to_tensor(self.size)
                    )[0]
                return img

            return tf.map_fn(preprocess, parent_tensor)
