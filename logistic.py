!pip install lazypredict
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=10, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = LazyClassifier(predictions=True, custom_metric=None)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)

svm_result = models.loc['SVC']
logreg_result = models.loc['LogisticRegression']

print("SVM Result:")
print(svm_result)
print("\nLogistic Regression Result:")
print(logreg_result)

# Optional: Print predictions if needed
print(predictions['SVC'])
print(predictions['LogisticRegression'])
