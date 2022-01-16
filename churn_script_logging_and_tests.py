'''
This module tests and logs the results of churn_library file

AUTHOR: Danilo Leira
DATE: January 2022
'''

import os
import logging
import churn_library as cls

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        data_pandas = import_data("./data/bank_data.csv")
    except FileNotFoundError as err:
        logging.error("Testing import_data: The file wasn't found")
        raise err

    try:
        assert data_pandas.shape[0] > 0
        assert data_pandas.shape[1] > 0
        logging.info("Testing import_data: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err
    return data_pandas


def test_eda(perform_eda, data_pandas):
    '''
    test perform eda function
    '''
    perform_eda(data_pandas)

    for file in [
        'churn_hist',
        'customer_age_hist',
        'marital_status_freq',
        'total_trans_distplot',
        'corr_heatmap'
    ]:

        try:
            with open(f'./images/eda/{file}.png', 'r'):
                logging.info("Testing perform_eda: SUCCESS")
        except FileNotFoundError as err:
            logging.error(
                "Testing perform_eda: %s imagem can not be found",
                file)
            raise err


def test_encoder_helper(encoder_helper, data_pandas):
    '''
    test encoder helper
    '''

    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    data_pandas = encoder_helper(data_pandas, cat_columns, 'Churn')

    try:
        for col in cat_columns:
            assert col in data_pandas.columns
        logging.info("Testing encoder_helper: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: The dataframe is missing some columns")
        raise err

    return data_pandas


def test_perform_feature_engineering(perform_feature_engineering, df_encoded):
    '''
    test perform_feature_engineering
    '''

    X_train, X_test, y_train, y_test = perform_feature_engineering(
        df_encoded, 'Churn')

    try:
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert len(y_train) > 0
        assert len(y_test) > 0
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.error("Testing perform_feature_engineering: "
                      "The four objects that should be returned were not.")
        raise err

    return X_train, X_test, y_train, y_test


def test_train_models(train_models, X_train, X_test, y_train, y_test):
    '''
    test train_models
    '''

    train_models(X_train, X_test, y_train, y_test)

    for file in [
        'rf_class_report',
        'lr_class_report',
        'random_forest_importances',
        'roc_curve'
    ]:

        try:
            with open(f'./images/results/{file}.png', 'r'):
                logging.info("Testing train_models: SUCCESS")
        except FileNotFoundError as err:
            logging.error(
                "Testing train_models: %s imagem can not be found",
                file)
            raise err

    for file in ['rfc_model', 'logistic_model']:

        try:
            with open(f'./models/{file}.pkl'):
                logging.info("Testing train_models: SUCCESS")
        except FileNotFoundError as err:
            logging.error(
                "Testing train_models: %s model can not be found",
                file)
            raise err


if __name__ == "__main__":

    DF = test_import(cls.import_data)
    test_eda(cls.perform_eda, DF)
    DF_ENCODED = test_encoder_helper(cls.encoder_helper, DF)
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = test_perform_feature_engineering(
        cls.perform_feature_engineering, DF)
    test_train_models(cls.train_models, X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)
