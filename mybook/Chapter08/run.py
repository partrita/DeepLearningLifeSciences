import deepchem as dc

# import numpy as np
# import pandas as pd
import os
import logging
from model import DRModel, DRAccuracy, ConfusionMatrix, QuadWeightedKappa
from data import load_images_DR

"""
2018년 9월 10일 월요일 생성

@작성자: zqwu
"""

RETRAIN = True
train, valid, test = load_images_DR(split="random", seed=123)

# 모델 정의 및 구축
model = DRModel(
    n_init_kernel=32,
    batch_size=32,
    learning_rate=1e-5,
    augment=True,
    model_dir="./test_model",
)
if not os.path.exists("./test_model"):
    os.mkdir("test_model")
if not RETRAIN:
    os.system("sh get_pretrained_model.sh")
    model.restore(checkpoint="./test_model/model-84384")
metrics = [
    dc.metrics.Metric(DRAccuracy, mode="classification"),
    dc.metrics.Metric(QuadWeightedKappa, mode="classification"),
]
cm = [dc.metrics.Metric(ConfusionMatrix, mode="classification")]

logger = logging.getLogger("deepchem.models.tensorgraph.tensor_graph")
logger.setLevel(logging.DEBUG)

if RETRAIN:
    print("10 에포크 동안 모델 학습을 시작합니다.")
    model.fit(train, nb_epoch=10, checkpoint_interval=1000)

print("훈련 데이터 지표 평가를 시작합니다.")
print(model.evaluate(train, metrics, n_classes=5))
print("검증 데이터 혼동 행렬(Confusion Matrix) 평가를 시작합니다.")
print(model.evaluate(valid, cm, n_classes=5))
print("테스트 데이터 혼동 행렬 평가를 시작합니다.")
print(model.evaluate(test, cm, n_classes=5))
