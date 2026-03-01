import deepchem as dc
import numpy as np
import matplotlib.pyplot as plt
# Estimate the uncertainty in a model's predictions.

# Start by creating and training the model.
tasks, datasets, transformers = dc.molnet.load_delaney(featurizer="GraphConv")
train_dataset, valid_dataset, test_dataset = datasets
model = dc.models.GraphConvModel(
    n_tasks=1, mode="regression", dropout=0.2, uncertainty=True
)
model.fit(train_dataset, nb_epoch=100)

# Predict values and uncertainties on the test set.
y_pred, y_std = model.predict_uncertainty(test_dataset)

# Plot a graph of absolute error versus predicted uncertainty.
plt.scatter(y_std, np.abs(y_pred - test_dataset.y))
plt.plot([0, 0.7], [0, 1.4], "k:")
plt.xlim([0.1, 0.7])
plt.xlabel("Predicted Standard Deviation")
plt.ylabel("Absolute Error")
plt.show()
