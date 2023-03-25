import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC


randomState = 42
rng = np.random.default_rng(randomState)

bcdata = make_classification(5000, random_state=randomState)
X, y = bcdata[0], bcdata[1]
y[y==0] = -1

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=randomState)

n, p = X_train.shape
xi = 1.0e-4
C = 1.0

betas = rng.standard_normal(p+1)
alpha = 1.0e-3
batchsize = 10

## YOUR IMPLEMENTATION GOES HERE --->

# SGD

# SGD with Momentum & Adam (only for groups with four members)

## <--- YOUR IMPLEMENTATION GOES HERE

# Print out the accuracy score of your prediction

# Linear SVM
SVM = LinearSVC(max_iter=T, dual=False)
SVM.fit(X_train, y_train)
yhat = SVM.predict(X_test)
print('Linear SVM: \t\t\t {0:.4f}'.format(accuracy_score(yhat, y_test)))

# Polynomial ve RBF kernels
SVM = SVC(kernel='poly')
SVM.fit(X_train, y_train)
yhat = SVM.predict(X_test)
print('SVM (polynomial): \t\t {0:.4f}'.format(accuracy_score(yhat, y_test)))

SVM = SVC(kernel='rbf')
SVM.fit(X_train, y_train)
yhat = SVM.predict(X_test)
print('SVM (rbf): \t\t\t {0:.4f}'.format(accuracy_score(yhat, y_test)))