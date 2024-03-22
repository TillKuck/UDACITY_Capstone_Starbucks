# Udacity Data Scientist Nanodegree Capstone Project
## Starbucks-Capstone-Project
This repository has all the code and report for my Udacity Data Scientist Nanodegree Capstone project.

### Project Overview :
This data set contains simulated data that mimics customer behavior on the Starbucks rewards mobile app. Once every few days, Starbucks sends out an offer to users of the mobile app. An offer can be merely an advertisement for a drink or an actual offer such as a discount or BOGO (buy one get one free). Some users might not receive any offer during certain weeks. Not all users receive the same offer, and that was the challenge to solve with this data set. Our task is to combine transaction, demographic and offer data to determine which demographic groups respond best to which offer type. This data set is a simplified version of the real Starbucks app because the underlying simulator only has one product whereas Starbucks actually sells dozens of products.

### Problem Statement :
1. Can a machine-learning classification model accurately predict whether a user will respond to a specific offer type?
2. What factors influence a user’s response to a specific offer type within the rewards program?

### Libraries used
This project was written in Python, using Jupyter Notebooks and python scripts. The relevant Python packages for this project are as follows:

pandas
numpy
sklearn.compose import ColumnTransformer
sklearn.pipeline import Pipeline
sklearn.preprocessing import StandardScaler, OneHotEncoder
sklearn.model_selection import GridSearchCV, train_test_split
xgboost.sklearn import XGBClassifier
sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score
matplotlib.pyplot
seaborn

### File Descriptions
This repo contains 3 folders.

**data:**
This folder contains all data files:  
- Three raw data files in json format from kaggle: https://www.kaggle.com/blacktile/starbucks-app-customer-reward-program-data
- Six pickle files that hold the processed data

**develop:**
This folder contains 4 jupyter notebooks used to facilitate the CRISP-DM data science process:  
1_Starbucks_Capstone_BusinessUnderstanding.ipynb  
2_EDA_DataUnderstanding_DataCleaning_DataViz.ipynb  
3_DataAnalysis_DataPreparation.ipynb  
4_ML_Modelling_Evaluation.ipynb

**source:**
- model.py: containing the XGBoostModel class
- utils.py: containing helper functions such as save and load pickle file

### Further information
**Medium Blog Post:**
https://medium.com/@till.kuckelkorn/improving-campaign-targeting-for-starbucks-reward-program-0c7847845542

**Licensing, Authors, Acknowledgements:**
Data for coding project was provided by Udacity and Starbucks.
