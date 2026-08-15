import deepchem as dc
import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np

# 학습된 전사 인자(TF) 결합 모델의 출력을 최대화하는 입력을 찾습니다.

# 먼저 모델을 구축합니다.
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
    model_dir="../tf",
)
model.restore()

# 무작위 서열로 시작합니다.
best_sequence = np.random.randint(4, size=101)
best_score = float(model.predict_on_batch([dc.metrics.to_one_hot(best_sequence, 4)]))

# 서열에 무작위 변화를 주고, 점수가 높아지는 경우에만 해당 변화를 유지합니다.
for step in range(1000):
    index = np.random.randint(101)
    base = np.random.randint(4)
    if best_sequence[index] != base:
        sequence = best_sequence.copy()
        sequence[index] = base
        score = float(model.predict_on_batch([dc.metrics.to_one_hot(sequence, 4)]))
        if score > best_score:
            best_sequence = sequence
            best_score = score


print("최적 서열(Best sequence):", "".join(["ACGT"[i] for i in best_sequence]))
print("최고 점수(Best score):", score)
