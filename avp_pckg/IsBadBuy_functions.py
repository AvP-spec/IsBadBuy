import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from avp_pckg.DataFrame import AvPdataFrame 
from avp_pckg.avp_model_selection import cross_validate_pipe
from avp_pckg.avp_model_selection import plot_scores, wheels_type_split
from avp_pckg.avp_model_selection import PrepareColsBase, PrepareColsTEncoder

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score # validation_curve, 
from sklearn.model_selection import cross_validate # for multiple score!!

from sklearn.metrics import classification_report, f1_score # accuracy_score, recall_scor

index_col='RefId'
cols_cat = [ 'Auction', 'VehicleAge',  'WheelType',
           'BYRNO', 'VNZIP1', # info byer
           'Make', 'Model', 'Trim', 'SubModel', # info model
           'Color',  'PRIMEUNIT', 'AUCGUART', 'Size', #  info model, low information
            ##
          #  'IsOnlineSale', 'Transmission', # low information
          #  'Nationality', 'TopThreeAmericanName', # redundant information
          # 'VNST', 'VehYear', 'WheelTypeID' # redundant information
            ] 

cols_num = [
            'MMRAcquisitionAuctionAveragePrice', 'MMRAcquisitionAuctionCleanPrice',
            'MMRAcquisitionRetailAveragePrice', 'MMRAcquisitonRetailCleanPrice',
            'MMRCurrentAuctionAveragePrice', 'MMRCurrentAuctionCleanPrice',
            'MMRCurrentRetailAveragePrice', 'MMRCurrentRetailCleanPrice', 
            'VehOdo', 'VehBCost', 'WarrantyCost']

# cols = cols_cat + cols_num
fname = 'data\\features_train.csv'
def load_features(fname=fname, cols_cat=cols_cat, cols_num=cols_num, index_col='RefId'):
    cols = cols_cat + cols_num + [index_col]
    features = pd.read_csv(fname, usecols=cols, index_col='RefId') # index_col=0,
    
    if 'WheelTypeID' in features.columns:
        features.loc[:, 'WheelTypeID'] = features['WheelTypeID'].astype(str)
        
    features.loc[:, cols_cat] = features[cols_cat].fillna(value='empty')
    features.loc[:, cols_num] = features[cols_num].fillna(value=0)
    
    return features


def calc_price_diff(df):
    
    df_ = df.copy()
    
    df_.loc[:, 'RetailClean'] = df_['MMRAcquisitonRetailCleanPrice'] - df_['MMRCurrentRetailCleanPrice']
    df_.loc[:, 'AcqClean'] = df_['MMRAcquisitonRetailCleanPrice'] - df_['MMRAcquisitionAuctionCleanPrice']
    df_.loc[:, 'AcqRetail'] = df_['MMRAcquisitonRetailCleanPrice'] - df_['MMRAcquisitionRetailAveragePrice'] 
    df_.loc[:, 'AcqAuc'] = df_['MMRAcquisitionAuctionCleanPrice'] - df_['MMRAcquisitionAuctionAveragePrice']
    
    cols = ['MMRAcquisitionAuctionAveragePrice', 'MMRAcquisitionAuctionCleanPrice',
            'MMRAcquisitionRetailAveragePrice', 'MMRAcquisitonRetailCleanPrice',
            'MMRCurrentAuctionAveragePrice', 'MMRCurrentAuctionCleanPrice',
            'MMRCurrentRetailAveragePrice', 'MMRCurrentRetailCleanPrice', ]
    
    df_ = df_.drop(columns=cols)
    return df_