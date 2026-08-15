import deepchem as dc
import numpy as np
import matplotlib.pyplot as plt

# 모델 예측의 불확실성을 추정합니다.

# 먼저 모델을 생성하고 학습을 진행합니다.
tasks, datasets, transformers = dc.molnet.load_delaney(featurizer="GraphConv")
train_dataset, valid_dataset, test_dataset = datasets
model = dc.models.GraphConvModel(
    n_tasks=1, mode="regression", dropout=0.2, uncertainty=True
)
model.fit(train_dataset, nb_epoch=100)

# 테스트 세트에서 예측값과 그 불확실성을 예측합니다.
y_pred, y_std = model.predict_uncertainty(test_dataset)

# 예측된 불확실성(표준 편차)과 실제 절대 오차 사이의 관계를 시각화합니다.
plt.scatter(y_std, np.abs(y_pred - test_dataset.y))
plt.plot([0, 0.7], [0, 1.4], "k:")
plt.xlim([0.1, 0.7])
plt.xlabel("예측된 표준 편차(Predicted Standard Deviation)")
plt.ylabel("절대 오차(Absolute Error)")
plt.show()
