import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import lazypredict
from lazypredict.Supervised import LazyClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

X, y = make_classification(n_samples=1000, n_features=3, n_informative=3, n_redundant=0, n_clusters_per_class=1, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

svm = SVC(kernel='rbf', random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=-1)
        return x

def train_policy_gradient(X_train, y_train, input_size, hidden_size, output_size, epochs=1000, lr=0.01):
    model = PolicyNetwork(input_size, hidden_size, output_size)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = loss_fn(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
    return model

def predict_policy_gradient(model, X_test):
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    outputs = model(X_test_tensor)
    _,predicted = torch.max(outputs, 1)
    return predicted.numpy()
input_size = X_train.shape[1]
hidden_size = 10
output_size = 2

policy_model = train_policy_gradient(X_train, y_train, input_size, hidden_size, output_size)
y_pred_policy = predict_policy_gradient(policy_model, X_test)
accuracy_policy = accuracy_score(y_test, y_pred_policy)

clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=y_test, marker='o', label='Actual')

ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=y_pred_svm, marker='^', label='Non-Linear SVM Predictions')
ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=y_pred_policy, marker='s', label='Policy Gradient Predictions')

ax.set_xlabel('X-Feature')
ax.set_ylabel('Y-Feature')
ax.set_zlabel('Z-Feature')
ax.set_title(f'Non-Linear SVM Accuracy: {accuracy_svm:.2f}, Policy Gradient Accuracy: {accuracy_policy:.2f}')
ax.legend()

plt.show()

print(models)
