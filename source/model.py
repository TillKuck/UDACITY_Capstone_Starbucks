import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm

class XGBoostModel:
    def __init__(self):
        self.pipeline = None
        self.model = None
        self.param_grid = {}

    def split_train_test(self, data, target_col, test_size=0.25, random_state=42):
        X = data.drop(target_col, axis=1)
        y = data[target_col]
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def build_pipeline(self, X):
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
        # Calculate scale_pos_weight
        neg_count = np.sum(y_train == 0)
        pos_count = np.sum(y_train == 1)
        scale_pos_weight = round(neg_count / pos_count, 2)
        
        # Define parameters for hyperparameter tuning
        self.param_grid = {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [3, 5, 7],
            'classifier__learning_rate': [0.1, 0.01, 0.001],
            'classifier__min_samples_leaf': [1, 2, 4],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__objective': ['binary:logistic'],
            'classifier__scale_pos_weight': [scale_pos_weight]
        }
        
        # Perform grid search with cross-validation
        self.model = GridSearchCV(self.pipeline, self.param_grid, cv=5, verbose=1)
        with tqdm(total=len(self.param_grid['classifier__n_estimators']) *
                        len(self.param_grid['classifier__max_depth']) *
                        len(self.param_grid['classifier__learning_rate']) *
                        len(self.param_grid['classifier__min_samples_leaf']) *
                        len(self.param_grid['classifier__min_samples_split']),
                  desc="Hyperparameter Tuning") as pbar:
            self.model.fit(X_train, y_train)
            pbar.update()
            
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        y_pred = self.model.predict(X_test)
        return y_pred

    def evaluate(self, y_true, y_pred):
        # Calculate evaluation metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_pred)
        conf_matrix = confusion_matrix(y_true, y_pred)
        baseline_accuracy = max(y_true.mean(), 1 - y_true.mean())

        # Print evaluation metrics
        print("Evaluation Metrics:")
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        print("ROC AUC Score:", roc_auc)
        print("\nConfusion Matrix:")
        print(conf_matrix)
        print("\nBaseline Accuracy (always predict majority class):", baseline_accuracy)
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))
