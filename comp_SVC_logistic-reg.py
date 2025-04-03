import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier

X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = LazyClassifier(predictions=True, random_state=42)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)
print(models)
svm_result = models.loc['LinearSVC']
log_reg_result = models.loc['LogisticRegression']
comparison_df = models.loc[['LinearSVC', 'LogisticRegression']]

plt.figure(figsize=(10, 6))
sns.barplot(x=comparison_df.index, y=comparison_df['Accuracy'])
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.xlabel('Model')
plt.ylim(0, 1)
plt.show()
