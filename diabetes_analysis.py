import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


#load the diabetes dataset
diabetes = datasets.load_diabetes()

#use only one feature
diabetes_X = diabetes.data[:, np.newaxis, 2]

#Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

#Split the target into training/testing sets
diabetes_Y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

#create linear regression object
regr = linear_model.LinearRegression()

#train the model using the training sets
regr.fit(diabetes_X_train, diabetes_Y_train)

#make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)

#the coefficients
print('Coefficient : \n', regr.coef_)

#the mean squared error
print("Mean squared error: %.2f"
% mean_squared_error(diabetes_y_test, diabetes_y_pred))

#explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))

#plot outputs
#plot against the x_test against y_test
plt.scatter(diabetes_X_test, diabetes_y_test, color = 'black')

#draw the line
plt.plot(diabetes_X_test, diabetes_y_pred, color = 'blue', linewidth = 3)
plt.xticks(())
plt.yticks(())

#display the map
plt.show()

