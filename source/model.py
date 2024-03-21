# import numpy as np
# import pandas as pd
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.model_selection import GridSearchCV, train_test_split
# from xgboost import XGBClassifier
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score
# import matplotlib.pyplot as plt
# from xgboost import plot_importance

# class XGBoostModel:
#     """
#     A class for building, training, and evaluating an XGBoost classifier model.
#     -------------
#     Attributes:
#     - pipeline: The pipeline used for preprocessing the data.
#     - model: The trained XGBoost classifier model.
#     - param_grid: The hyperparameter grid for tuning the model.
#     """
#     def __init__(self):
#         """
#         Initializes the XGBoostModel class.
#         """
#         self.pipeline = None
#         self.model = None

#     def split_train_test(self, data, target_col, test_size=0.25, random_state=42):
#         """
#         Splits data into training and testing sets.
#         -------------
#         Parameters:
#         - data: The DataFrame containing the data.
#         - target_col: The name of the target column.
#         -------------
#         Returns:
#         - X_train: The feature matrix of the training data.
#         - X_test: The feature matrix of the testing data.
#         - y_train: The target labels of the training data.
#         - y_test: The target labels of the testing data.
#         """
#         X = data.drop(target_col, axis=1)
#         y = data[target_col]
#         return train_test_split(X, y, test_size=test_size, random_state=random_state)

#     def build_pipeline(self, X):
#         """
#         Builds a preprocessing pipeline based on the data features.
#         -------------
#         Parameters:
#         - X: The feature matrix.
#         -------------
#         Returns:
#         - pipeline: The preprocessing pipeline.
#         """
#         # Identify categorical and numerical features
#         categorical_features = X.select_dtypes(include=['object']).columns.tolist()
#         numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

#         # Define preprocessing steps for categorical and numerical features
#         categorical_transformer = Pipeline(steps=[
#             ('onehot', OneHotEncoder(handle_unknown='ignore'))
#             ])

#         numerical_transformer = Pipeline(steps=[
#             ('scaler', StandardScaler())
#             ])

#         # Combine preprocessing steps
#         preprocessor = ColumnTransformer(
#             transformers=[
#                 ('num', numerical_transformer, numerical_features),
#                 ('cat', categorical_transformer, categorical_features)
#                 ])

#         # Define pipeline
#         self.pipeline = Pipeline(steps=[('preprocessor', preprocessor),
#                                         ('classifier', XGBClassifier())
#                                         ])
#         return self.pipeline

#     def fit_model(self, X_train, y_train):
#         """
#         Fits the XGBoost classifier model to the training data.
#         -------------
#         Parameters:
#         - X_train: The feature matrix of the training data.
#         - y_train: The target labels of the training data.
#         """
#         # Define parameter grid
#         param_grid = {
#             'classifier__n_estimators': [100, 200, 300], # [100, 200, 300]
#             'classifier__max_depth': [3, 5, 7], # [3, 5, 7]
#             'classifier__learning_rate': [0.1, 0.01, 0.001], # [0.1, 0.01, 0.001]
#             'classifier__objective': ['binary:logistic']
#         }
        
#         # Calculate scale_pos_weight
#         neg_count = np.sum(y_train == 0)
#         pos_count = np.sum(y_train == 1)
#         scale_pos_weight = round(neg_count / pos_count, 2)
        
#         # Update param_grid with scale_pos_weight
#         param_grid['classifier__scale_pos_weight'] = [scale_pos_weight]
        
#         # Perform grid search with cross-validation
#         self.model = GridSearchCV(self.pipeline, param_grid, cv=5, verbose=3)
#         self.model.fit(X_train, y_train)
#         print('Best model params:\n', self.model.best_params_)

#     def predict(self, X_test):
#         """
#         Makes predictions using the trained model on new data.
#         -------------
#         Parameters:
#         - X_test: The feature matrix of the test data.
#         -------------
#         Returns:
#         - y_pred: The predicted target labels.
#         """
#         y_pred = self.model.predict(X_test)
#         return y_pred

#     def evaluate(self, y_true, y_pred):
#         """
#         Evaluates the model's performance on test data.
#         -------------
#         Parameters:
#         - y_test: The target labels of the test data.
#         - y_pred: From the model predicted labels of the test data.
#         -------------
#         Returns:
#         - evaluation_metrics: Dictionary containing evaluation metrics.
#         """
#         # Calculate evaluation metrics
#         accuracy = accuracy_score(y_true, y_pred)
#         f1 = f1_score(y_true, y_pred)
#         roc_auc = roc_auc_score(y_true, y_pred)
#         conf_matrix = confusion_matrix(y_true, y_pred)
#         baseline_accuracy = max(y_true.mean(), 1 - y_true.mean())

#         # Print evaluation metrics
#         print("Baseline Accuracy (always predict majority class):", baseline_accuracy)
#         print("\nEvaluation Metrics:")
#         print("Accuracy:", accuracy)
#         print("F1 Score:", f1)
#         print("ROC AUC Score:", roc_auc)
#         print("\nConfusion Matrix:")
#         print(conf_matrix)
#         print("\nClassification Report:")
#         print(classification_report(y_true, y_pred))
        
#     def plot_feature_importance(self):
#         """
#         Plots feature importances.

#         Parameters:
#         - model: The trained model with a 'feature_importances_' attribute.
#         - feature_names: List of feature names corresponding to the importances.
#         """
#         feature_important = self.model.get_booster().get_score(importance_type='weight')
#         keys = list(feature_important.keys())
#         values = list(feature_important.values())

#         data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)
#         data.nlargest(40, columns="score").plot(kind='barh', figsize = (20,10)) ## plot top 40 features
        
        

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

class XGBoostModel:
    """
    A class for building, training, and evaluating an XGBoost classifier model.
    -------------
    Attributes:
    - pipeline: The pipeline used for preprocessing the data.
    - model: The trained XGBoost classifier model.
    """
    def __init__(self):
        """
        Initializes the XGBoostModel class.
        """
        self.pipeline = None
        self.model = None

    def split_train_test(self, data, target_col, test_size=0.25, random_state=42):
        """
        Splits data into training and testing sets.
        -------------
        Parameters:
        - data: The DataFrame containing the data.
        - target_col: The name of the target column.
        -------------
        Returns:
        - X_train: The feature matrix of the training data.
        - X_test: The feature matrix of the testing data.
        - y_train: The target labels of the training data.
        - y_test: The target labels of the testing data.
        """
        X = data.drop(target_col, axis=1)
        y = data[target_col]
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def build_pipeline(self, X):
        """
        Builds a preprocessing pipeline based on the data features.
        -------------
        Parameters:
        - X: The feature matrix.
        -------------
        Returns:
        - pipeline: The preprocessing pipeline.
        """
        # Identify categorical and numerical features
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

        # Define preprocessing steps for categorical and numerical features
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])

        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
            ])

        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
                ])

        # Define pipeline
        self.pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                        ('classifier', XGBClassifier())
                                        ])
        return self.pipeline

    def fit_model(self, X_train, y_train):
        """
        Fits the XGBoost classifier model to the training data.
        -------------
        Parameters:
        - X_train: The feature matrix of the training data.
        - y_train: The target labels of the training data.
        """
        # Define parameter grid
        param_grid = {
            'classifier__n_estimators': [100, 200, 300], # [100, 200, 300]
            'classifier__max_depth': [3, 5, 7], # [3, 5, 7]
            'classifier__learning_rate': [0.1, 0.01, 0.001], # [0.1, 0.01, 0.001]
            'classifier__objective': ['binary:logistic']
        }
        
        # Calculate scale_pos_weight
        neg_count = np.sum(y_train == 0)
        pos_count = np.sum(y_train == 1)
        scale_pos_weight = round(neg_count / pos_count, 2)
        
        # Update param_grid with scale_pos_weight
        param_grid['classifier__scale_pos_weight'] = [scale_pos_weight]
        
        # Perform grid search with cross-validation
        self.model = GridSearchCV(self.pipeline, param_grid, cv=5, verbose=3)
        self.model.fit(X_train, y_train)
        print('Best model params:\n', self.model.best_params_)

    def predict(self, X_test):
        """
        Makes predictions using the trained model on new data.
        -------------
        Parameters:
        - X_test: The feature matrix of the test data.
        -------------
        Returns:
        - y_pred: The predicted target labels.
        """
        y_pred = self.model.predict(X_test)
        return y_pred

    def evaluate(self, y_true, y_pred):
        """
        Evaluates the model's performance on test data.
        -------------
        Parameters:
        - y_test: The target labels of the test data.
        - y_pred: From the model predicted labels of the test data.
        -------------
        Returns:
        - evaluation_metrics: Dictionary containing evaluation metrics.
        """
        # Calculate evaluation metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_pred)
        conf_matrix = confusion_matrix(y_true, y_pred)
        baseline_accuracy = max(y_true.mean(), 1 - y_true.mean())

        # Print evaluation metrics
        print("Baseline Accuracy (always predict majority class):", baseline_accuracy)
        print("\nEvaluation Metrics:")
        print("Accuracy:", accuracy)
        print("F1 Score:", f1)
        print("ROC AUC Score:", roc_auc)
        print("\nConfusion Matrix:")
        print(conf_matrix)
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))
        
    def plot_feature_importance(self, feature_names):
        """
        Plots feature importances.

        Parameters:
        - feature_names: List of feature names corresponding to the importances.
        """
        importances = self.model.best_estimator_.named_steps['classifier'].feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances")
        plt.bar(range(len(feature_names)), importances[indices], align='center')
        plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=90)
        plt.xlabel('Feature')
        plt.ylabel('Importance')
        plt.show()