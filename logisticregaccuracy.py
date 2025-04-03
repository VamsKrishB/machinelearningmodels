import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression

X, y = make_classification(n_samples=1000, n_features=3, n_informative=2, n_redundant=0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = LazyClassifier(predictions=True, random_state=42)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lin = lin_reg.predict(X_test)
lin_reg_score = lin_reg.score(X_test, y_test)
models.loc['LinearRegression'] = [lin_reg_score, None, None, None, None]

print(models)

logistic_regression_score = models.loc['LogisticRegression']['Accuracy']
linear_regression_score = models.loc['LinearRegression']['Accuracy']
model_names = ['LogisticRegression', 'LinearRegression']
scores = [logistic_regression_score, linear_regression_score]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=y_test, marker='o')

for i, model in enumerate(model_names):
    ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], label=f'{model} (Score: {scores[i]:.2f})')

    ax.set_xlabel('X-Feature')
    ax.set_ylabel('Y-Feature')
    ax.set_zlabel('Z-Feature')
    ax.set_title('Classification Data with Model Scores')
    ax.legend()

    plt.show()
