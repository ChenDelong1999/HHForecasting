import os

import sklearn.linear_model
from sklearn import tree, neighbors, ensemble
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

from dataset import CSVDataset
from utils import get_mean_squared_error, \
    get_deterministic_coefficient, \
    get_Nash_efficiency_coefficient,\
    get_Kling_Gupta_efficiency


def get_metrics(model):
    # TRAIN_DC = model.score(train_set.raindrop, train_set.runoff)
    # TEST_DC = model.score(test_set.raindrop, test_set.runoff)

    Train_runoff_prediction = model.predict(train_set.raindrop)
    TRAIN_MSE = get_mean_squared_error(Train_runoff_prediction,train_set.runoff)
    TRAIN_DC = get_deterministic_coefficient(Train_runoff_prediction,train_set.runoff)
    TRAIN_NSE = get_Nash_efficiency_coefficient(Train_runoff_prediction,train_set.runoff)
    TRAIN_KGE = get_Kling_Gupta_efficiency(Train_runoff_prediction,train_set.runoff)

    
    Test_runoff_prediction = model.predict(test_set.raindrop)
    TEST_MSE = get_mean_squared_error(Test_runoff_prediction, test_set.runoff)
    TEST_DC = get_deterministic_coefficient(Test_runoff_prediction, test_set.runoff)
    TEST_NSE = get_Nash_efficiency_coefficient(Test_runoff_prediction, test_set.runoff)
    TEST_KGE = get_Kling_Gupta_efficiency(Test_runoff_prediction, test_set.runoff)

    

    return round(TRAIN_MSE, 2), round(TEST_MSE, 2), \
           round(TRAIN_DC, 2), round(TEST_DC, 2),\
           round(TRAIN_NSE, 2), round(TEST_NSE, 2),\
           round(TRAIN_KGE, 2), round(TEST_KGE, 2)

            ############
            #   PAN    #
            ############
# def print_results(results):
#     file = 'results.csv'
#     with open(file, 'a+') as f:
#         print('\n--- printing results to file {}'.format(file))
#         print('time', file=f, end=',')
#         for key in results.keys():
#             print(key, end=',', file=f)
#         print(file=f)
#         print(datetime.datetime.now(), file=f, end=',')
#
#         for i in range(len(results[key])):
#             for key in results.keys():
#                 print(results[key][i], end=',', file=f)
#         print(file=f)

            ############
            #   PAN    #
            ############
# if __name__ == '__main__':
#     dataPath = "TunXi/"
#     svr = SVR(kernel="rbf", C=1e2, gamma=0.1)
#     decision_tree = tree.DecisionTreeRegressor(max_depth=4)
#     linear_regression = LinearRegression()
#     kNN = neighbors.KNeighborsRegressor()
#     random_forest = ensemble.RandomForestRegressor(n_estimators=64)
#     gradient_boosting = ensemble.GradientBoostingRegressor(n_estimators=64)
#     bagging = ensemble.BaggingRegressor()
#
#     models = [svr, decision_tree, linear_regression, kNN, random_forest, gradient_boosting, bagging]
#     names = ['SVR', 'Decision Tree', 'Linear Regression', 'k-NN', 'RandomForest', 'Gradient Boosting', 'Bagging']
#
#     for root, dirs, files in os.walk(dataPath):
#         for filename in files:
#             currPath = os.path.join(root, filename)
#             file = open(currPath)
#             train_set = CSVDataset(forecast_range=1, dataset=currPath, mode='train',
#                                train_test_split_ratio=0.7, sample_length=72)
#             test_set = CSVDataset(forecast_range=1, dataset=currPath, mode='test',
#                               train_test_split_ratio=0.7, sample_length=72)
#             print('\n', filename);
#             print('Model, TRAIN_MSE, TEST_MSE, TRAIN_DC, TEST_DC, TRAIN_NSE, TEST_NSE, TRAIN_KGE, TEST_KGE')
#             for i in range(len(models)):
#                 model = models[i]
#                 model.fit(train_set.inputs, train_set.targets)
#                 metrics = get_metrics(model)
#                 print_results(f'{names[i]}, {(metrics)[1:-1]}')


                                ############
                                #   YAN    #
                                ############

if __name__ == '__main__':

    svr = SVR(kernel="rbf", C=1e2, gamma=0.1)
    decision_tree = tree.DecisionTreeRegressor(max_depth=4)
    linear_regression = LinearRegression()
    kNN = neighbors.KNeighborsRegressor()
    random_forest = ensemble.RandomForestRegressor(n_estimators=64)
    gradient_boosting = ensemble.GradientBoostingRegressor(n_estimators=64)
    bagging = ensemble.BaggingRegressor()
    ridge = sklearn.linear_model.Ridge()

    models = [svr, decision_tree, linear_regression, kNN, random_forest, gradient_boosting, bagging, ridge]
    names = ['SVR', 'Decision Tree', 'Linear Regression', 'k-NN', 'RandomForest', 'Gradient Boosting','Bagging', 'Ridge']

    for dataset in ['ChangHua','TunXi']:#or['ChangHua','TunXi']
        l = os.listdir('dataset/' + dataset + '/')
        sensors = list(l)
        for dataset_name in sensors:
            if os.path.splitext(dataset_name)[1] == '.csv':
                dataset_name = dataset + '/' + dataset_name
                train_set = CSVDataset(forecast_range=6, dataset=dataset_name, mode='train',
                                       train_test_split_ratio=0.7, sample_length=72)
                test_set = CSVDataset(forecast_range=6, dataset=dataset_name, mode='test',
                                      train_test_split_ratio=0.7, sample_length=72)

                print('\n', dataset_name)
                print('Model, TRAIN_MSE, TEST_MSE, TRAIN_DC, TEST_DC','TRAIN_NSE', 'TEST_NSE', 'TRAIN_KGE', 'TEST_KGE')
                for i in range(len(models)):
                    model = models[i]
                    model.fit(train_set.raindrop, train_set.runoff)
                    metrics = get_metrics(model)
                    print(f'{names[i]}, {str(metrics)[1:-1]}')
