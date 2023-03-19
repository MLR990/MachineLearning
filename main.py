import pandas as pd
from Regression.LinearRegression import LinearRegressionModel
from Regression.PolynomialRegression import PolynomialRegressionModel
from Regression.SupportVector import SVRegressionModel
from Regression.DecisionTree import DecisionTreeRegressionModel
from Regression.RandomForest import RandomForestRegressionModel
from Classification.LogisticRegression import LogisticRegressionModel
from Classification.KNearestNeighbors import KNearestNeighborsModel
from Classification.SupportVectorMachine import SupportVectorMachineModel
from Classification.KernelSVM import KernelSVMModel
from Classification.NaiveBayes import NaiveBayesModel
from Classification.DecisionTree import DecisionTreeClassificationModel
from Classification.RandomForest import RandomForestClassificationModel


# Regressions
def check_regressions(X, y):
    linear = LinearRegressionModel(data_X=X, data_y=y)
    # linear.show_prediction()

    poly = PolynomialRegressionModel(data_X=X, data_y=y)
    # poly.show_prediction()

    svr = SVRegressionModel(data_X=X, data_y=y)
    # svr.show_prediction()

    tree = DecisionTreeRegressionModel(data_X=X, data_y=y)
    # tree.show_prediction()

    forest = RandomForestRegressionModel(data_X=X, data_y=y)
    # forest.show_prediction()


# Classifications
def check_classifications(X, y):
    logistic = LogisticRegressionModel(data_X=X, data_y=y)
    k_nearest = KNearestNeighborsModel(data_X=X, data_y=y)
    support_vector = SupportVectorMachineModel(data_X=X, data_y=y)
    kernel_svm = KernelSVMModel(data_X=X, data_y=y)
    naive_bayes = NaiveBayesModel(data_X=X, data_y=y)
    decision_tree = DecisionTreeClassificationModel(data_X=X, data_y=y)
    random_forest = RandomForestClassificationModel(data_X=X, data_y=y)


# regression_dataset = pd.read_csv('Regression/Data.csv')
# X = regression_dataset.iloc[:, :-1].values
# y = regression_dataset.iloc[:, -1].values
# check_regressions(X, y)
#

classification_dataset = pd.read_csv('Classification/Data.csv')
X = classification_dataset.iloc[:, :-1].values
y = classification_dataset.iloc[:, -1].values
check_classifications(X, y)
