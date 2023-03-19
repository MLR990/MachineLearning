import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score
import warnings

class SVRegressionModel:
    def __init__(self, data_X, data_y):
        warnings.filterwarnings("ignore")

        self.y = data_y.reshape(len(data_y), 1)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data_X, self.y, test_size=0.2, random_state=0)

        self.sc_X = StandardScaler()
        self.sc_y = StandardScaler()
        self.X_train = self.sc_X.fit_transform(self.X_train)
        self.y_train = self.sc_y.fit_transform(self.y_train)

        self.regressor = SVR(kernel='rbf')

        self.regressor.fit(self.X_train, self.y_train)

        self.y_pred = self.sc_y.inverse_transform(self.regressor.predict(self.sc_X.transform(self.X_test)).reshape(-1, 1))
        print(f"Support Vector Regression R2 Score: {r2_score(self.y_test, self.y_pred)}")

    def show_prediction(self):
        np.set_printoptions(precision=2)
        print(np.concatenate((self.y_pred.reshape(len(self.y_pred), 1), self.y_test.reshape(len(self.y_test), 1)), 1))
