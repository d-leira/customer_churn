# Predict Customer Churn

Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Table of contents
* [Project Description](#project-description)
* [Files in the Repo](#files-in-the-repo)
* [Setup](#setup)

## Project Description
A simple pipeline to train a machine learning model to predict customer churn. The project performs the following tasks:

* Read Data
* Perform EDA
* Perform mean enconding for categorical features
* Preform feature engineering
* Train models
* Evaluate models

## Files in the Repo
📦root  
 ┣ 📂data  
 ┃ ┗ 📜bank_data.csv  
 ┣ 📂images  
 ┃ ┣ 📂eda  
 ┃ ┃ ┣ 📜churn_hist.png  
 ┃ ┃ ┣ 📜corr_heatmap.png  
 ┃ ┃ ┣ 📜customer_age_hist.png  
 ┃ ┃ ┣ 📜marital_status_freq.png  
 ┃ ┃ ┗ 📜total_trans_distplot.png  
 ┃ ┗ 📂results  
 ┃ ┃ ┣ 📜lr_class_report.png  
 ┃ ┃ ┣ 📜random_forest_importances.png  
 ┃ ┃ ┣ 📜rf_class_report.png  
 ┃ ┃ ┗ 📜roc_curve.png  
 ┣ 📂logs  
 ┃ ┗ 📜churn_library.log  
 ┣ 📂models  
 ┃ ┣ 📜logistic_model.pkl  
 ┃ ┗ 📜rfc_model.pkl  
 ┣ 📜.gitignore  
 ┣ 📜Guide.ipynb  
 ┣ 📜README.md  
 ┣ 📜__init__.py  
 ┣ 📜churn_library.py  
 ┣ 📜churn_notebook.ipynb  
 ┗ 📜churn_script_logging_and_tests.py  

## Running Files
Install dependencies to run the pipeline with the following command.

```bash
pip install -r requirements.txt
```

To run the pipeline module, after having all dependencies installed, use the following comand.

```bash
ipython churn_library.py
```

After excuted, the pipeline should save the output of EDA in [./images/eda](./images/eda), containing histograms, frequencies and a correlation matrix of the input features images. The models objects generated will be saved in [./models](./models), and the performance of the models will be saved in [./images/results](./images/results), you should have classification reports, feature importances and roc curve images.

If you notice some file missing or some error in the pipeline, you should consider test the code with the following command.

```bash
ipython churn_script_logging_and_tests.py
```

In the log file saved in [./logs](./logs), you can check of the pipeline is running as expected.

