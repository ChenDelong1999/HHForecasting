from sklearn import tree, neighbors, ensemble
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

import numpy as np
from dataset import CSVDataset


def get_metrics(model):
    TRAIN_DC = model.score(train_set.inputs, train_set.targets)
    TEST_DC = model.score(test_set.inputs, test_set.targets)

    Train_runoff_prediction = model.predict(train_set.inputs)
    TRAIN_MSE = np.average((train_set.targets - Train_runoff_prediction) ** 2)
    Test_runoff_prediction = model.predict(test_set.inputs)
    TEST_MSE = np.average((test_set.targets - Test_runoff_prediction) ** 2)

    return round(TRAIN_MSE, 2), round(TEST_MSE, 2), round(TRAIN_DC, 2), round(TEST_DC, 2)


if __name__ == '__main__':

    svr = SVR(kernel="rbf", C=1e2, gamma=0.1)
    decision_tree = tree.DecisionTreeRegressor(max_depth=4)
    linear_regression = LinearRegression()
    kNN = neighbors.KNeighborsRegressor()
    random_forest = ensemble.RandomForestRegressor(n_estimators=64)
    gradient_boosting = ensemble.GradientBoostingRegressor(n_estimators=64)
    bagging = ensemble.BaggingRegressor()

    models = [svr, decision_tree, linear_regression, kNN, random_forest, gradient_boosting, bagging]
    names = ['SVR', 'Decision Tree', 'Linear Regression', 'k-NN', 'RandomForest', 'Gradient Boosting', 'Bagging']

    for dataset_name in ['ChangHua', 'TunXi']:
        train_set = CSVDataset(forecast_range=6, dataset=dataset_name, mode='train',
                               train_test_split_ratio=0.7, sample_length=72)
        test_set = CSVDataset(forecast_range=6, dataset=dataset_name, mode='test',
                              train_test_split_ratio=0.7, sample_length=72)
        print('\n', dataset_name)
        print('Model, TRAIN_MSE, TEST_MSE, TRAIN_DC, TEST_DC')
        for i in range(len(models)):
            model = models[i]
            model.fit(train_set.inputs, train_set.targets)
            metrics = get_metrics(model)
            print(f'{names[i]}, {str(metrics)[1:-1]}')
