import deepchem as dc
import tensorflow as tf
import tensorflow.keras.layers as layers

# 전사 인자(Transcription Factor)인 JUND의 결합 부위를 예측하는 모델을 훈련합니다.

# 모델을 구축합니다.
features = tf.keras.Input(shape=(101, 4))
prev = features
for i in range(3):
    prev = layers.Conv1D(
        filters=15, kernel_size=10, activation=tf.nn.relu, padding="same"
    )(prev)
    prev = layers.Dropout(rate=0.5)(prev)
logits = layers.Dense(units=1)(layers.Flatten()(prev))
output = layers.Activation(tf.math.sigmoid)(logits)
keras_model = tf.keras.Model(inputs=features, outputs=[output, logits])
model = dc.models.KerasModel(
    keras_model,
    loss=dc.models.losses.SigmoidCrossEntropy(),
    output_types=["prediction", "loss"],
    batch_size=1000,
    model_dir="tf",
)

# 데이터를 로드합니다.
train = dc.data.DiskDataset("train_dataset")
valid = dc.data.DiskDataset("valid_dataset")

# 모델을 훈련하며 훈련 및 검증 데이터셋에서의 성능 변화를 확인합니다.
metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
for i in range(20):
    model.fit(train, nb_epoch=10)
    print(f"에포크 {i*10+10} - 훈련 데이터 ROC-AUC 점수: {model.evaluate(train, [metric])}")
    print(f"에포크 {i*10+10} - 검증 데이터 ROC-AUC 점수: {model.evaluate(valid, [metric])}")
