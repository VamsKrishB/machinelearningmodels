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
from collections import defaultdict

X, y = make_classification(n_samples=1000, n_features=3, n_informative=3, n_redundant=0, n_clusters_per_class=1, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Linear SVM
svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)

# Q-learning implementation
class QLearningClassifier:
    def __init__(self, n_features, n_classes, alpha=0.1, gamma=0.9, epsilon=0.1, n_epochs=100):
            self.n_features = n_features
            self.n_classes = n_classes
            self.alpha = alpha
            self.gamma = gamma
            self.epsilon = epsilon
            self.n_epochs = n_epochs
            self.q_table = defaultdict(lambda: np.zeros(n_classes))

    def _discretize_state(self, state):
            return tuple(np.round(state, decimals=1))

    def fit(self, X, y):
        for epoch in range(self.n_epochs):
            for xi, yi in zip(X, y):
                  state = self._discretize_state(xi)
                  if np.random.rand() < self.epsilon:
                      action = np.random.randint(self.n_classes)
                  else:
                      action = np.argmax(self.q_table[state])

                  reward = 1 if action == yi else -1
                  next_state = self._discretize_state(xi)  
                  next_max = np.max(self.q_table[next_state])
                  self.q_table[state][action] += self.alpha * (reward + self.gamma * next_max - self.q_table[state][action])

    def predict(self, X):
        predictions = []
        for xi in X:
            state = self._discretize_state(xi)
            action = np.argmax(self.q_table[state])
            predictions.append(action)
        return np.array(predictions)
ql_classifier = QLearningClassifier(n_features=3, n_classes=2)
ql_classifier.fit(X_train, y_train)
y_pred_ql = ql_classifier.predict(X_test)
accuracy_ql = accuracy_score(y_test, y_pred_ql)

clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=y_test, marker='o', label='Actual')
ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=y_pred_svm, marker='^', label='Linear SVM Predictions')
ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=y_pred_ql, marker='s', label='Q-learning Predictions')

ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')
ax.set_title(f'Linear SVM Accuracy: {accuracy_svm:.2f}, Q-learning Accuracy: {accuracy_ql:.2f}')
ax.legend()
plt.show()
print(models)
