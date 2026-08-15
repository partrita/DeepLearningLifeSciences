import deepchem as dc
import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np
import os
import re

RETRAIN = False

# 데이터셋을 로드합니다.
image_dir = "BBBC005_v1_images"
files = []
labels = []
for f in os.listdir(image_dir):
    if f.endswith(".TIF"):
        files.append(os.path.join(image_dir, f))
        labels.append(int(re.findall("_C(.*?)_", f)[0]))
dataset = dc.data.ImageDataset(files, np.array(labels))
splitter = dc.splits.RandomSplitter()
train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
    dataset, seed=123
)

# 모델을 생성합니다.
features = tf.keras.Input(shape=(520, 696, 1))
prev_layer = features
for num_outputs in [16, 32, 64, 128, 256]:
    prev_layer = layers.Conv2D(
        num_outputs, kernel_size=5, strides=2, activation=tf.nn.relu
    )(prev_layer)
output = layers.Dense(1)(layers.Flatten()(prev_layer))
keras_model = tf.keras.Model(inputs=features, outputs=output)
learning_rate = dc.models.optimizers.ExponentialDecay(0.001, 0.9, 250)
model = dc.models.KerasModel(
    keras_model,
    loss=dc.models.losses.L2Loss(),
    learning_rate=learning_rate,
    model_dir="models/model",
)

if not os.path.exists("./models"):
    os.mkdir("models")
if not os.path.exists("./models/model"):
    os.mkdir("models/model")

if not RETRAIN:
    model.restore()

# 모델을 훈련하고 테스트 세트에서 성능을 평가합니다.
if RETRAIN:
    print("50 에포크 동안 모델 학습을 시작합니다.")
    model.fit(train_dataset, nb_epoch=50)

y_pred = model.predict(test_dataset).flatten()
rmse = np.sqrt(np.mean((y_pred - test_dataset.y) ** 2))
print(f"테스트 세트 RMSE(평균 제곱근 오차): {rmse}")
