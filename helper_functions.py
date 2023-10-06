### Helper functions
# Last edit: 06.10.2023 08:58

import pandas as pd

def imports() -> tuple:
    """Load data from csv files

    Returns:
        tuple: Tuple of 3 DataFrame: X, Y, X_test
    """
    X = pd.read_csv('data/train_values.csv')
    Y = pd.read_csv('data/train_labels.csv')
    X_test = pd.read_csv('data/test_values.csv')
    return X, Y, X_test
    
    
def write_output(X_test, y_pred) -> None:
    """Create output.csv following drivendata.com submission standards and write file to disk

    Args:
        X_test (pd.Dataframe): Test dataset, x
        y_pred (pd.Dataframe): Predictions, y
    """
    # Create Dataframe for output
    output_df = pd.DataFrame({'building_id': X_test['building_id'], 'damage_grade': y_pred})
    
    # Write to csv
    output_df.to_csv('output.csv', index=False)
    
    
    
def test_column_equality(a, b) -> None:
    """_summary_

    Args:
        a (pd.Dataframe): _description_
        b (pd.Dataframe): _description_
    """
    if a.columns.equals(b.columns):
        print("Both DataFrames have the same columns.")
    else:
        print("The columns of the DataFrames are different.")