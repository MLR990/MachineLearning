import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


class PolynomialRegressionModel:
    def __init__(self, data_X, data_y):
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(data_X, data_y, test_size=0.2, random_state=0)

        self.poly_reg = PolynomialFeatures(degree=4)
        self.X_poly = self.poly_reg.fit_transform(self.X_train)
        self.regressor = LinearRegression()
        self.regressor.fit(self.X_poly, self.y_train)
        self.y_prediction = self.regressor.predict(self.poly_reg.transform(self.X_test))

        r2 = r2_score(self.y_test, self.y_prediction)
        print(f"Polynomial Regression R2 Score: {r2}")

    def show_prediction(self):
        np.set_printoptions(precision=2)
        print(np.concatenate((self.y_prediction.reshape(len(self.y_prediction), 1), self.y_test.reshape(len(self.y_test), 1)), 1))
