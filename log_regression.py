# Very simple Regression Model
# Last edit: 06.10.2023 12:20

import pandas as pd
from sklearn.linear_model import LogisticRegression


class lr():
    
    def __init__(self, X_train, y_train) -> None:
        """Initialize Logistic Regression and fit model with training set

        Args:
            X_train (pd.DataFrame): Training set, x values_
            y_train (pd.DataFrame): Training set, y values
        """
        self.lrm = LogisticRegression(multi_class='auto', solver='lbfgs')
        self.lrm.fit(X_train, y_train)
        
    def make_prediction(self, X) -> pd.DataFrame:
        """Make prediction on X using fitted modell

        Returns:
            pd.DataFrame: y_predict on X
        """
        return self.lrm.predict(X)
    
