from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression

#import a dataset from sklearn
X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

fig = plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], y, color = "b", marker="o", s = 30)
plt.show()

#run linear regression
reg = LinearRegression(lr=0.01)
reg.fit(X_train, y_train)
predictions = reg.predict(X_test)
reg.print_stats(y_test, predictions)

#plot a scatter plot off traing data
training_predictions = reg.predict(X)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8,6))
plt.scatter(X_train, y_train, s=10)
plt.scatter(X_test, y_test, s=10)
plt.plot(X, training_predictions, linewidth=2)
plt.legend(["Training Data", "Test Data", "Pedictions"])
plt.show()

#plot the error vs iterations of GD
plt.plot(reg.cost)
plt.title("Model Training: Gradient Descent Cost")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.show()