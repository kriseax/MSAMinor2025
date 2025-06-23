from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from LogisticRegression import LogisticRegression
from sklearn.preprocessing import StandardScaler

#import a dataset from sklearn
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

#scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=1234)

#create an instance of a logistic regression classifier
clf = LogisticRegression(lr=0.03)
clf.fit(X_train, y_train)
y_pred = clf.classify(X_test)

accuracy = clf.accuracy(y_pred, y_test)
print(accuracy)


