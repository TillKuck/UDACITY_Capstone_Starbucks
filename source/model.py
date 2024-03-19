import pandas as pd

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class XGBoostModel(BaseEstimator, TransformerMixin):

    def load_data(data_path, label_col):
        df = pd.read_pickle(data_path)

        return df


    def create_train_test_data(df):
        X = df.drop(label_col, axis=1)
        y = df[label_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        return X_train, X_test, y_train, y_test
    

    def build_model(X_train, y_train):
        pipeline = Pipeline([
            ('categories', OneHotEncoder(dtype=int)),
            ('scaler', StandardScaler()),
            ('clf', GradientBoostingClassifier(random_state=42))
            ]),

        parameters = {
            'clf__n_estimators': [50, 100, 200],
            'clf__learning_rate': [0.5, 0.1, 0.01],
            'clf__max_depth': [3, 4, 5],
            'min_samples_leaf': [1, 2, 3],
            'clf__min_samples_split': [2, 3]
        }

        cv = GridSearchCV(pipeline, param_grid=parameters)

        return cv
    

    def fit(model, X_train, y_train):
        return model.fit(X_train, y_train)
    

    def predict(model, X_test):
       y_pred = model.predict(X_test)

       return y_pred


    def evaluate_model(model, y, y_test, y_pred):
        print('Base Accuracy:', y.mean())
        print('Model Accuracy:', (y_test == y_pred).mean())
        print('Classification Report:\n', classification_report(y_test, y_pred))
        print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))