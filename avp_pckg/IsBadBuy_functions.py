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


def norm_submodel(df:pd.DataFrame):
    '''split submodel string in columns'''
    df_ = df.copy()
    
    ## misprint correctoions
    df_['SubModel'] = df_['SubModel'].str.replace('HATCKBACK', 'HATCHBACK')
    df_['SubModel'] = df_['SubModel'].str.replace('HARTOP', 'HARDTOP')
    df_['SubModel'] = df_['SubModel'].str.replace('LIMTED', 'LIMITED')
    
    ## extract 2D an 4D from the sting \d = didgit, D=D, \s = space, ()? = if exist 
    df_['D'] = df_['SubModel'].str.extract(r'(\dD\s)?')
    df_['Remaining'] = df_['SubModel'].str.replace(r'(\dD\s*)', '', regex=True)
    ## extract engine volume (4.6L) with L
    df_[['L']] = df_['Remaining'].str.extract(r'(\d\.\dL)')
    df_['Remaining'] = df_['Remaining'].str.replace(r'(\d\.\dL)', '', regex=True)
    ## extract engine volume (4.6 ) whithout L
    df_[['L_tmp']] = df_['Remaining'].str.extract(r'(\d\.\d)')
    df_['Remaining'] = df_['Remaining'].str.replace(r'(\d\.\d)', '', regex=True)
    mask = df_['L_tmp'].notna()
    df_.loc[mask, 'L'] = df_['L_tmp']
    df_.drop(columns=['L_tmp'], inplace=True)
    df_['L'] = df_['L'].str.replace('L', '')
    ## TO DO remove L 
    
    
    ## extract spesial terms
    lst = ['CAB', 'CREW', 'EXT', 'SPORTBACK', 'SPORT', 'QUAD', 'HARDTOP', 'HYBRID', 
           'PREMIER', 'PREMIUM',  'LIMITED', 'POPULAR', 'COMFORT', 'CARGO', 'SPECIAL', 'DELUXE',
           'CLASSIC', 'VALUE', 'PLUS', 'PANEL', 'TRAC'
           'TURBO', 'TOURING', 'GRAND', 'CUSTOM', 'LUXURY', 'CONVENIENCE', 'SIGNATURE',
           'NAVIGATION', 'AUTO', 'DURATEC', 'HEMI', 'AWD', 'PACKAGE', 'HIGHLINE', 'PRERUNNER',
           '5SP', '6SP', 'FFV', 'ZX5', 'ZX4', 'ZX3', 'ZX2'
           ]
    
    for word in lst:
        df_[word] = df_['Remaining'].str.extract(f'({word})')
        df_['Remaining'] = df_['Remaining'].str.replace(word, '')
    
    
    ## extract first word 
    df_[['Type', 'sbTrim']] = df_['Remaining'].str.extract(r'(^\S+)\s*(.*)')
    df_['sbTrim'] = df_['sbTrim'].str.strip() 
    df_ = df_.drop(columns='Remaining')   
    
    
  
    return df_
    

