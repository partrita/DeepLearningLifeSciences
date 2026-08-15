import deepchem as dc
from deepchem.models.optimizers import ExponentialDecay
from deepchem.models.seqtoseq import AspuruGuzikAutoEncoder
import numpy as np
from rdkit import Chem

# 새로운 분자를 생성하기 위해 변분 오토인코더(VAE)를 훈련합니다.
# 먼저 훈련 데이터를 로드합니다.
tasks, datasets, transformers = dc.molnet.load_muv()
train_dataset, valid_dataset, test_dataset = datasets
train_smiles = train_dataset.ids

# 사용된 토큰(문자) 집합과 SMILES 문자열의 최대 길이를 확인합니다.
tokens = set()
for s in train_smiles:
    tokens = tokens.union(set(s))
tokens = sorted(list(tokens))
max_length = max(len(s) for s in train_smiles)

# 모델을 구축합니다.
batch_size = 100
batches_per_epoch = len(train_smiles) / batch_size
learning_rate = ExponentialDecay(0.001, 0.95, batches_per_epoch)
model = AspuruGuzikAutoEncoder(
    tokens,
    max_length,
    model_dir="vae",
    batch_size=batch_size,
    learning_rate=learning_rate,
)


# 모델을 훈련합니다.
def generate_sequences(epochs):
    for i in range(epochs):
        for s in train_smiles:
            yield (s, s)


model.fit_sequences(generate_sequences(50))

# 새로운 분자를 생성합니다.
predictions = model.predict_from_embeddings(np.random.normal(size=(1000, 196)))
molecules = []
for p in predictions:
    smiles = "".join(p)
    if Chem.MolFromSmiles(smiles) is not None:
        molecules.append(smiles)

print("생성된 분자 목록:")
for m in molecules:
    print(m)
