import deepchem as dc
import tensorflow as tf
import tensorflow.keras.layers as layers
# import matplotlib.pyplot as plt

# 특정 서열이 RNA 간섭(RNA interference)에 얼마나 효과적인지 예측하는 모델을 훈련합니다.

# 모델을 구축합니다.
features = tf.keras.Input(shape=(21, 4))
prev = features
for i in range(2):
    prev = layers.Conv1D(
        filters=10, kernel_size=10, activation=tf.nn.relu, padding="same"
    )(prev)
    prev = layers.Dropout(rate=0.3)(prev)
output = layers.Dense(units=1, activation=tf.math.sigmoid)(layers.Flatten()(prev))
keras_model = tf.keras.Model(inputs=features, outputs=output)
model = dc.models.KerasModel(
    keras_model, loss=dc.models.losses.L2Loss(), batch_size=1000, model_dir="rnai"
)

# 데이터를 로드합니다.
train = dc.data.DiskDataset("train_siRNA")
valid = dc.data.DiskDataset("valid_siRNA")

# 모델을 훈련하며 훈련 및 검증 데이터셋에서의 성능 변화를 확인합니다.
metric = dc.metrics.Metric(dc.metrics.pearsonr, mode="regression")
for i in range(20):
    model.fit(train, nb_epoch=10)
    print(f"에포크 {i*10+10} - 훈련 데이터 피어슨 상관계수: {model.evaluate(train, [metric])['pearsonr']}")
    print(f"에포크 {i*10+10} - 검증 데이터 피어슨 상관계수: {model.evaluate(valid, [metric])['pearsonr']}")
