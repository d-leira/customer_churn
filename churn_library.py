'''
Module trains and evaluate models for churn prediction

AUTHOR: Danilo Leira
DATE: january 2022
'''


# import libraries
import os
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
        pth: a path to the csv
    output:
        data_pandas: pandas dataframe
    '''

    data_pandas = pd.read_csv(pth, index_col= 0)
    data_pandas['Churn'] = data_pandas['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    return data_pandas


def perform_eda(data_pandas):
    '''
    perform eda on data_pandas and save figures to images folder
    input:
            data_pandas: pandas dataframe

    output:
            None
    '''

    plt.figure(figsize=(20, 10))
    plt.ylabel('Frequency', fontsize = 20)
    plt.xlabel('Churn', fontsize = 20)
    plt.title('Churn distribution', fontsize = 24)
    data_pandas['Churn'].hist()
    plt.savefig('./images/eda/churn_hist.png')

    plt.figure(figsize=(20, 10))
    plt.ylabel('Frequency', fontsize = 20)
    plt.xlabel('Customer_Age', fontsize = 20)
    plt.title('Customer age distribution', fontsize = 24)
    data_pandas['Customer_Age'].hist()
    plt.savefig('./images/eda/customer_age_hist.png')

    plt.figure(figsize=(20, 10))
    plt.ylabel(r'% of total', fontsize = 20)
    plt.xlabel('Marital_Status', fontsize = 20)
    plt.title('Marital status distribution', fontsize = 24)
    data_pandas.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig('./images/eda/marital_status_freq.png')

    plt.figure(figsize=(20, 10))
    plt.ylabel('Density', fontsize = 20)
    plt.title('Total transaction count distribution', fontsize = 24)
    sns.distplot(data_pandas['Total_Trans_Ct'])
    plt.savefig('./images/eda/total_trans_distplot.png')

    plt.figure(figsize=(20, 10))
    plt.title('Correlation matrix', fontsize = 24)
    sns.heatmap(data_pandas.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig('./images/eda/corr_heatmap.png')


def encoder_helper(data_pandas, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            data_pandas: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name

    output:
            data_pandas: pandas dataframe with new columns for
    '''

    for cat_feat in category_lst:
        categories_list = []
        categories_groups = data_pandas.groupby(cat_feat).mean()[response]

        for val in data_pandas[cat_feat]:
            categories_list.append(categories_groups.loc[val])

        data_pandas[f'{cat_feat}_Churn'] = categories_list

    return data_pandas


def perform_feature_engineering(data_pandas, response):
    '''
    input:
              data_pandas: pandas dataframe
              response: string of response name

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    y_target = data_pandas[response]
    X_data = pd.DataFrame()
    X_data[keep_cols] = data_pandas[keep_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_target, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def roc_curve_image(lrc_model, cv_rfc_model, X_data, y_data):
    '''
    produces roc curve for a trained model and stores it as image in images folder
    input:
            lrc_model: logistic regression model object
            cv_rfc_model: random forest model object
            X_data: pandas dataframe of X values
            y_data: pandas Series of y values

    output:
             None
    '''

    # plots
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    plot_roc_curve(lrc_model, X_data, y_data, ax=ax, alpha=0.8)
    plot_roc_curve(
        cv_rfc_model,
        X_data,
        y_data,
        ax=ax,
        alpha=0.8)
    plt.savefig('./images/results/roc_curve.png')


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''

    # scores
    plt.figure(figsize=(6, 4.5), tight_layout = True)
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.4, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.5, str(classification_report(y_train, y_train_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.85, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig('./images/results/rf_class_report.png')

    plt.figure(figsize=(6, 4.5), tight_layout = True)
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.4, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.5, str(classification_report(y_train, y_train_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.85, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig('./images/results/lr_class_report.png')


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''

    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 9), tight_layout = True)

    # Create plot title
    plt.title("Feature Importance", fontsize = 24)
    plt.ylabel('Importance', fontsize = 20)

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=45)

    plt.savefig(f'{output_pth}/random_forest_importances.png')


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''

    rfc = RandomForestClassifier(random_state=42, n_jobs = -1)
    lrc = LogisticRegression(n_jobs = -1)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)
    
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)
    
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    roc_curve_image(lrc, cv_rfc.best_estimator_, X_test, y_test)
    
    feature_importance_plot(cv_rfc.best_estimator_, X_test, './images/results')
    
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')


if __name__ == '__main__':

    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    DF = import_data(r'./data/bank_data.csv')

    perform_eda(DF)

    df_encoded = encoder_helper(DF, cat_columns, 'Churn')

    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = perform_feature_engineering(
        df_encoded, 'Churn')

    train_models(X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)
