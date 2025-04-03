import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = LazyClassifier(predictions=True, random_state=42)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)

print(models)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
lin_reg_predictions = lin_reg.predict(X_test)
lin_reg_predictions_binary = [1 if pred >= 0.5 else 0 for pred in lin_reg_predictions]
lin_reg_accuracy = accuracy_score(y_test, lin_reg_predictions_binary)
log_reg_result = models.loc['LogisticRegression']
comparison_df = {
    'Model': ['LinearRegression', 'LogisticRegression'],
        'Accuracy': [lin_reg_accuracy, log_reg_result['Accuracy']]
        }
comparison_df = pd.DataFrame(comparison_df)

plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='Accuracy', data=comparison_df)
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.xlabel('Model')
plt.ylim(0, 1)
plt.show()
