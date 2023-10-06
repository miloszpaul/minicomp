# Class for random forest including optimizer
# Strongly modified from original source: ChatGPT
# added model fit object to output
# created separate predict_f() for predictions on external dataset
# created get_model to return the model
# set optuna_n_trials as a parameter to optimization function 

import lightgbm as lgb
import optuna
import pandas as pd
from sklearn.metrics import f1_score, make_scorer


class LGBM():
    """Class for LightGBM Forest and Optuna optimizer
    """
    
    def __init__(self, X_train, X_valid, y_train, y_valid) -> None:
        """Init class, store training and validation sets. Modify data where needed

        Args:
            X_train (pd.DataFrame): Training set, x values
            X_valid (pd.DataFrame): Validation set, x values
            y_train (pd.DataFrame): Training set, y values
            y_valid (pd.DataFrame): Validation set, y values
        """
        # F1 score tracker. Use f1 micro for drivendata competition
        self.f1_micro_scorer = make_scorer(f1_score, average='micro')
        
        # Initialize the best_f1_score variable as a list
        self.best_f1_score = [0.0]
        
        # For optimization, classes need to go from [0,x]. Dataset starts at 1. Make adjustment
        self.y_train = y_train - 1
        self.y_valid = y_valid - 1
        
        # No change at X
        self.X_train = X_train
        self.X_valid = X_valid
        

    def _objective(self, trial) -> float:
        """Set parameters for optimization, keep track of f1"""
    
        params = {
            'objective': 'multiclass',
            'num_class': 3,  # Set the number of classes in your dataset
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'num_leaves': trial.suggest_int('num_leaves', 10, 100),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 20),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'lambda_l2': trial.suggest_float('lambda_l2', 0, 1),
            'verbose': -1  # Suppress LightGBM output,
        }

        # Create the LightGBM dataset
        train_data = lgb.Dataset(self.X_train, label=self.y_train)
        valid_data = lgb.Dataset(self.X_valid, label=self.y_valid, reference=train_data)

        # Train the multiclass classification model without early stopping
        model = lgb.train(
            params,
            train_data,
            valid_sets=[valid_data],
            num_boost_round=100
        )

        # Predict probabilities for each class on the validation set
        y_pred_prob = model.predict(self.X_valid, num_iteration=model.best_iteration)

        # Determine the predicted class by selecting the one with the highest probability
        y_pred = y_pred_prob.argmax(axis=1)

        # Adjust the predicted class labels back to the original range of 1 - 3
        self.y_pred_original_range = y_pred + 1

        # Calculate F1 score using micro averaging (Drivendata.com uses F1 micro)
        f1 = f1_score(self.y_valid, self.y_pred_original_range, average='micro')
        
        # Update the best F1 score if a better score is found
        if f1 > self.best_f1_score[0]:
            self.best_f1_score[0] = f1
        
        # Perform early stopping manually
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        return f1
    
    def optimization(self, X, y,optuna_n_trials):
        """Run optimization. Build trees, tweak hyperparameters and return prediction of test set

        Args:
            X (pd.DataFrame): Full dataset (training and validation), x values
            y (pd.DataFrame): Full dataset (training and validation), y values
            optuna_n_trials: optimization parameter for optuna: number of trials

        Returns:
            pd.DataFrame: Prediction of y values on testset
        """
            
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())  # Maximize F1 score
        study.optimize(self._objective, n_trials=optuna_n_trials)
        
        # Info on current run
        print('Number of finished trials:', len(study.trials))
        print('Best trial:')
        trial = study.best_trial

        #print('  Value: {:.3f}'.format(trial.value))
        #print('  Params: ')
        #for key, value in trial.params.items():
        #    print('    {}: {}'.format(key, value))
        
        # store best parameter set    
        best_params = trial.params
        
        # Lovely ASCII line, report best f1 score
        print("--------------------------------")
        print(f"Best F1 Score: {self.best_f1_score[0]}")
        print("--------------------------------")


        self.final_model = lgb.LGBMClassifier(
            objective='multiclass',
            **best_params
        )

        # fit model on full dataset
        self.fitted_model=self.final_model.fit(X, y)
        return None

    def get_model(self):
        return   self.fitted_model      
        
    def predict_f(self,X_test) -> pd.DataFrame:
        '''
        Args:
            X_test (pd.DataFrame): Testset, x values
        
        Returns:
            pd.DataFrame: Prediction of y values on testset
        '''
        return self.fitted_model.predict(X_test)
    
    def feature_importance_table(self) ->pd.DataFrame:
        '''
        returns feature importance of the fitted model
        '''
        return  pd.DataFrame(zip(self.fitted_model.feature_importances_,self.X_train.columns), columns=['Value','Feature'])

