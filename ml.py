import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

wine = load_wine()
X = wine.data
y = wine.target
gaussianNB = GaussianNB()
randomForest = RandomForestClassifier(random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.7, test_size = 0.3, random_state = 0, stratify = y)

gaussianNB.fit(X_train,y_train)
randomForest.fit(X_train,y_train)

resultGaussianNB = gaussianNB.predict(X_test)
resultRandomForest = randomForest.predict(X_test)

cmGaussianNB = confusion_matrix(y_test, resultGaussianNB)
cmRandomForest = confusion_matrix(y_test, resultRandomForest)

print(cmGaussianNB)
print(cmRandomForest)

plt.matshow(cmGaussianNB)
plt.title('Confusion matrix for Gaussian NB')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

plt.matshow(cmRandomForest)
plt.title('Confusion matrix for Random Forest')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


