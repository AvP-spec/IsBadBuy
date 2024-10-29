import pandas as pd
import numpy as np


def unanimity(pred: list):
    """
    Computes a unanimity-based prediction from multiple model predictions.

    Parameters:
    -----------
    pred : list of pd.DataFrame
        A list of pandas DataFrames, each containing a single column of predictions 
        with values 0 or 1, and all DataFrames having the same index (each representing 
        predictions from different models for the same set of samples).

    Returns:
    --------
    pd.Series
        A pandas Series with the same index as the input DataFrames, where each value is:
        - 1 if all models unanimously predict 1 for that sample,
        - 0 if at least one model predicts 0 for that sample.
        
    Example:
    --------
    >>> unanimity([pd.DataFrame([1, 1, 1]), pd.DataFrame([0, 1, 1] , pd.DataFrame([0, 0, 1])])
    >>> [0, 0, 1]
    """
    
    n = len(pred)
    df_pred = pd.concat(pred, axis=1)
    df_pred['sum'] = df_pred.sum(axis=1)
    return (df_pred['sum'] == n).astype(int)


if __name__ == "__main__":
    
    print('\ntest of function \u001b[34m unanimity \u001b[0m')
    arr1 = pd.DataFrame([1, 1, 1, 0, 1])
    arr2 = pd.DataFrame([1, 1, 0, 0, 1])
    arr3 = pd.DataFrame([1, 0, 1, 0, 1])
    arr4 = pd.DataFrame([1, 1, 0, 1, 1])
    lst = [arr1, arr2, arr3, arr4]
    fun_output = pd.Series([1, 0, 0, 0, 1])
    result = (unanimity(lst) == fun_output).sum() == 5 
    if result:
        x = '\u001b[32m'
    else:
        x = '\u001b[31m'
    print( 'test passed:' + x, result,  '\u001b[0m' + '\n') 
    
