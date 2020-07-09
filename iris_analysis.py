#Model to predict the species of an iris flower 
# based on length and width of its petals and sepals

#import the needed libraries from sklearn.datasets import load iris

from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.metrics import accuracy_score
import numpy as np

#loading the iris dataset
iris = load_iris()

x = iris.data #Array of data
#to display the dataset, uncomment the next line of code
# print(x) 

y = iris.target #arrays of labels(i.e answers) of each data entry
# print(y)

#getting label names i.e the three flower species
y_names = iris.target_names
#to print the name of the flowers, uncomment line 24-26
# print("These are the name of the iris flower species" , y_names)
# for flower_names in y_names:
#     print(flower_names)

#taking random indices to split the dataset into train and test
test_ids = np.random.permutation(len(x))
# print(test_ids)

#splitting data and labels into train and test
#keeping last 10 entries for testing, rest for training

x_train = x[test_ids[:-10]]
x_test = x[test_ids[-10:]]

y_train = y[test_ids[:-10]]
y_test = y[test_ids[-10:]]

#classifying using decision tree
clf = tree.DecisionTreeClassifier()
# print(clf)

#training(fitting) the classifier with the training set
clf.fit(x_train, y_train)

#predictions on the test dataset
pred = clf.predict(x_test)
print("prediction labels", pred) #prediction labels i.e flower species
print("Actual labels result" , y_test) #actual labels
accuracy = (accuracy_score(pred, y_test)) # prediction accuracy
accuracy_percent = accuracy * 100 #accuracy percent
print("accuracy percent :", accuracy_percent)



