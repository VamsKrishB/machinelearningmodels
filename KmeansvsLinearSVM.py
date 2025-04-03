import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import lazypredict
from lazypredict.Supervised import LazyClassifier

X, y = make_classification(n_samples=1000, n_features=3, n_informative=3, n_redundant=0, n_clusters_per_class=1, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Linear SVM
svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)

# K-Means Clustering
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_train)
y_pred_kmeans = kmeans.predict(X_test)

y_pred_kmeans = np.where(y_pred_kmeans == 0, 1, 0)
accuracy_kmeans = accuracy_score(y_test, y_pred_kmeans)

clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=y_test, marker='o', label='Actual')

ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=y_pred_svm, marker='^', label='Linear SVM Predictions')
ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=y_pred_kmeans, marker='s', label='K-Means Predictions')

ax.set_xlabel('X-Feature')
ax.set_ylabel('Y-Feature')
ax.set_zlabel('Z-Feature')
ax.set_title(f'Linear SVM Accuracy: {accuracy_svm:.2f}, K-Means clustering Accuracy: {accuracy_kmeans:.2f}')
ax.legend()

plt.show()

print(models)

