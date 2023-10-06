### Helper functions
# Last edit: added viz_f

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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


def histograms_f(X,y,cols_ignore=('building_id'),bins=30):
    '''
    for each variable, creates histograms and 100% stacked histograms
    '''
    # Merge X and y for bivariate analysis
    data = X.copy()
    data['damage_grade'] = y

    # Define a color palette
    palette = sns.color_palette("tab10", n_colors=data['damage_grade'].nunique())
    
    # Loop through each column in X_train
    for col in X.columns:
        if col not in cols_ignore and pd.api.types.is_numeric_dtype(X[col]):
            plt.figure(figsize=(14,6))
            
            # Histogram with damage_grade distribution as color
            plt.subplot(1,2,1)
            sns.histplot(data=data, x=col, hue='damage_grade', element="step", common_norm=False, stat="probability", palette=palette)
            plt.title(f'Distribution of {col} by Damage Grade')
            
            # 100% stacked histogram
            plt.subplot(1,2,2)
            
            # Compute histograms for each damage_grade
            grades = sorted(data['damage_grade'].unique())
            histograms = [np.histogram(data[data['damage_grade'] == grade][col], bins=bins) for grade in grades]
            base = np.zeros_like(histograms[0][0], dtype=np.float64)
            
            for idx, (grade, histogram) in enumerate(zip(grades, histograms)):
                total_height = sum([hist[0] for hist in histograms])
                # Check for zero total height
                if np.all(total_height < 0.001):
                    plt.bar(histogram[1][:-1], np.ones_like(histogram[0]), width=np.diff(histogram[1]), color='gray', align='edge')
                    break
                
                heights = histogram[0] / total_height
                plt.bar(histogram[1][:-1], heights, width=np.diff(histogram[1]), bottom=base, label=f'Damage Grade {grade}', align='edge', color=palette[idx])
                base += heights
                
            plt.title(f'100% Stacked Distribution of {col} by Damage Grade')
            plt.legend()
                
            plt.tight_layout()
            plt.show()