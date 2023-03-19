import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor


class DecisionTreeRegressionModel:
    def __init__(self, data_X, data_y):
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(data_X, data_y, test_size=0.2, random_state=0)

        self.regressor = DecisionTreeRegressor(random_state=0)
        self.regressor.fit(self.X_train, self.y_train)
        self.y_prediction = self.regressor.predict(self.X_test)

        r2 = r2_score(self.y_test, self.y_prediction)
        print(f"Decision Tree Regression R2 Score: {r2}")

    def show_prediction(self):
        np.set_printoptions(precision=2)
        print(np.concatenate((self.y_prediction.reshape(len(self.y_prediction),1), self.y_test.reshape(len(self.y_test),1)),1))