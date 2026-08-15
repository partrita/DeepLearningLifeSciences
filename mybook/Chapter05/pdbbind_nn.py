import deepchem as dc

# 신경망을 활용해 PDBBind 데이터셋의 결합 친화도를 예측합니다. 먼저 데이터를 로드합니다.

featurizer = dc.feat.RdkitGridFeaturizer(
    voxel_width=2.0,
    sanitize=True,
    flatten=True,
    feature_types=["hbond", "salt_bridge", "pi_stack", "cation_pi", "ecfp", "splif"],
)
pdbbind_tasks, pdbbind_datasets, transformers = dc.molnet.load_pdbbind(
    featurizer=featurizer, splitter="random", subset="core"
)
train_dataset, valid_dataset, test_dataset = pdbbind_datasets

# 모델을 생성하고 학습을 진행합니다.
n_features = train_dataset.X.shape[1]
model = dc.models.MultitaskRegressor(
    n_tasks=len(pdbbind_tasks),
    n_features=n_features,
    layer_sizes=[2000, 1000],
    dropouts=0.5,
    model_dir="pdbbind_nn",
    learning_rate=0.0003,
)
model.fit(train_dataset, nb_epoch=50)

# 모델의 성능을 평가합니다.
metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)
train_scores = model.evaluate(train_dataset, [metric], transformers)
test_scores = model.evaluate(test_dataset, [metric], transformers)

print("훈련 데이터 평가 점수(Train scores)")
print(train_scores)
print("테스트 데이터 평가 점수(Test scores)")
print(test_scores)
