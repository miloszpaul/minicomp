# visualisation functions
# Last edit: added viz_f

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def hist2x2_f(X,y,cols_include=['geo_level_1_id','age'],cols_ignore=[],bins=30):
    '''
    for each selected column of X, creates histograms and 100% stacked histograms, with categories of y color-encoded
    '''
    # copy, select columns
    data = X.copy()
    data=data[cols_include]
    for ic in cols_ignore:
        try:
             data=data.drop(columns = ic)
        except   KeyError:
            print(f"Column '{ic}' not found in data. Skipping.") 
    # select only numeric columns
    data=data.select_dtypes(include='number')
    # Merge X and y 
    data['damage_grade'] = y

    # Define a color palette
    palette = sns.color_palette("tab10", n_colors=data['damage_grade'].nunique())
    
    # Loop through each column in data
    for col in data.columns:
            if col=='damage_grade':
                 break

            plt.figure(figsize=(14,6))
            
            # Histogram with damage_grade distribution as color
            plt.subplot(1,2,1)
            sns.histplot(data=data, x=col, hue='damage_grade', element="step", common_norm=False,  palette=palette) #stat="probability",
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
             
                heights = histogram[0] / (total_height+0.001)
                plt.bar(histogram[1][:-1], heights, width=np.diff(histogram[1]), bottom=base, label=f'Damage Grade {grade}', align='edge', color=palette[idx])
                base += heights
                
            plt.title(f'100% Stacked Distribution of {col} by Damage Grade')
            plt.legend()
                
            plt.tight_layout()
            plt.show()



def plot_feature_imp(feature_imp_df):
    '''
    plots feature importance
    '''
    plt.figure(figsize=(20, 10))
    sns.barplot(x="Value", y="Feature", data=feature_imp_df.sort_values(by='Value',ascending=False))
    plt.title('Feature importance')
    plt.tight_layout()
    plt.show()


def plot_pairwise_cor(X_train,y):
    '''
    plots corelations of each column of X_train and y
    '''
    data = pd.concat([X_train, y], axis=1)

    # Calculate correlation matrix
    correlations = data.select_dtypes(include=[np.number]).drop('building_id', axis=1).corr()[y.name].drop(y.name)
    correlations=correlations.sort_values()
    # Plot bar chart
    correlations.plot(kind='bar', color='skyblue')

    plt.title('Pairwise Correlation of each feature with label')
    plt.show()