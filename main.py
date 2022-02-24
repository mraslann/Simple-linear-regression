#Mohamed Raslan

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

path = "http://bit.ly/w-data"
Data = pd.read_csv(path)
print(Data.head())
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.title('Hours studied vs Score')
plt.scatter(Data.Hours, Data.Scores)
plt.show()

X = Data.iloc[:, :-1].values
Y = Data.iloc[:, 1].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0, test_size=0.2)
linreg = LinearRegression()
linreg.fit(X_train, Y_train)

Y0 = linreg.intercept_ + linreg.coef_ * X_train

plt.scatter(X_train, Y_train)
plt.plot(X_train, Y0)
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.title("Regression line(Train set)")
plt.show()

Y_pred = linreg.predict(X_test)
plt.plot(X_test, Y_pred)
plt.scatter(X_test, Y_test)
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.title("Regression line(Test set)")
plt.show()

Y_test1 = list(Y_test)
prediction = list(Y_pred)
df_compare = pd.DataFrame({'Actual': Y_test1, 'Result': prediction})
print(df_compare)

Prediction_score = linreg.predict([[9.25]])
print(f"The predicted score for a student studying 9.25 hours is {Prediction_score}")
