import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from lazypredict.Supervised import LazyClassifier

X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0,random_state=42, n_clusters_per_class=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = LazyClassifier(predictions=True, random_state=42)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)

linear_svm_acc = models.loc['LinearSVC']['Accuracy']
rbf_svm_acc = models.loc['SVC']['Accuracy']

print(f'Linear SVM Accuracy: {linear_svm_acc}')
print(f'Non-linear SVM (RBF Kernel) Accuracy: {rbf_svm_acc}')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

model_names = ['Linear SVM', 'Non-linear SVM']
accuracies = [linear_svm_acc, rbf_svm_acc]

                           
x_pos = np.arange(len(model_names))
y_pos = np.zeros(len(model_names)) 
z_pos = np.zeros(len(model_names))  

                          
dx = np.ones(len(model_names))  
dy = np.ones(len(model_names))  
dz = accuracies  

ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, color=['Red', 'Yellow'], alpha=0.6)

ax.set_xticks(x_pos)
ax.set_xticklabels(model_names)
ax.set_xlabel('Model')
ax.set_ylabel('Y Label')  
ax.set_zlabel('Accuracy')
ax.set_title('Comparison of Linear SVM and Non-linear SVM')
plt.show()
