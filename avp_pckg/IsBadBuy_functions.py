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
    
    ## extract spesial terms to a binary columns
    ## the terms appear atleast 2 times in different combinations => 
    ## => information should be splited for normal form 
    lst = ['CAB', 'CREW', 'EXT', 'SPORTBACK', 'SPORT', 'QUAD', 'HARDTOP', 'HYBRID', 
           'PREMIER', 'PREMIUM',  'LIMITED', 'POPULAR', 'COMFORT', 'CARGO', 'SPECIAL', 'DELUXE',
           'CLASSIC', 'VALUE', 'PLUS', 'PANEL', 'TRAC',
           'TURBO', 'TOURING', 'GRAND', 'CUSTOM', 'LUXURY', 'CONVENIENCE', 'SIGNATURE',
           'NAVIGATION', 'AUTO', 'DURATEC', 'HEMI', 'AWD', 'PACKAGE', 'HIGHLINE', 'PRERUNNER',
           '5SP', '6SP', 'FFV', 'XUV', 'ZX5', 'ZX4', 'ZX3', 'ZX2', 'ZWD'
           ]
    ## ChatGPT: ZX2 = 2D, ZX3 = 3D, ..., ZX5 = 5D, ZWD = Wagon
    
    for word in lst:
        mask = df_['Remaining'].str.contains(f'{word}')
        df_.loc[mask, word] = 1 
        df_[word] = df_[word].fillna(0).astype(int)       
        df_['Remaining'] = df_['Remaining'].str.replace(word, '')
        # print(word, ':', df_[word].sum())
    
    ## extract first word 
    df_[['Type', 'sbTrim']] = df_['Remaining'].str.extract(r'(^\S+)\s*(.*)')
    df_['sbTrim'] = df_['sbTrim'].str.strip() 
    df_ = df_.drop(columns='Remaining')   
    mask = df_['sbTrim'] == ''
    # print(mask.sum())
    df_.loc[mask, 'sbTrim' ] = 'empty'
    df_.loc[:, 'sbTrim'] = df_.loc[:, 'sbTrim'].fillna('empty')
      
    return df_
 
def expand_truncated(df, col_mame:str,  trunc:list, replacement:str ):
    df_ = df.copy()
    pattern = r'(\s+(' + '|'.join(trunc) + r'))$'
    df_[col_mame] = df_[col_mame].str.replace(pattern, replacement, regex=True).str.strip()
    return df_

def mistprints_model(df:pd.DataFrame):
    
    df_ = df.copy()
    
    lst = ['Multipl', 'Multiple E', 'Multiple En', 'Multiple Eng', 
           'Multiple Engi', 'Multiple Engin', 'Multiple Engine'] 
    df_ = expand_truncated(df_, col_mame='Model', trunc=lst, replacement=' Multiple')
    
    df_['Model'] = df_['Model'].str.replace('6C 4.2L I-', '6C 4.2L') # 'ENVOY 4WD 6C 4.2L'
    
    lst = ['PICKU', 'PIC']
    df_ = expand_truncated(df_, col_mame='Model', trunc=lst, replacement=' PICKUP')
    
    df_['Model'] = df_['Model'].str.replace('CAMRY 4C EI I-4 2.2L', 
                                            'CAMRY 4C EFI I-4 2.2L')
    df_['Model'] = df_['Model'].str.replace('ECLIPSE EI V6 3.0L S', 
                                            'ECLIPSE EFI V6 3.0L SFI')
    
    df_['Model'] = df_['Model'].str.replace('ESCORT 4-FI-2.0L', 
                                            'ESCORT 4C MFI 2.0L')
    
    df_['Model'] = df_['Model'].str.replace('TAURUS 3.0L V6 EFI F', 
                                            'TAURUS 3.0L V6 EFI')
    
    df_['Model'] = df_['Model'].str.replace('2500 SILVERADO PICKUP', 
                                            '2500HD SILVERADO PICKUP')
    lst = ['SF', 'S']
    df_ = expand_truncated(df_, col_mame='Model',  trunc=lst, replacement=' SFI')
    
    lst = ['FI DO']
    df_ = expand_truncated(df_, col_mame='Model',  trunc=lst, replacement=' SFI DOHC')
    ## DOHC=DOH=DO=D -Dual Overhead Camshaft
    lst = ['DOH', 'DO', 'D']
    df_ = expand_truncated(df_, col_mame='Model', trunc=lst, replacement=' DOHC')
    # EFI = EF = EI = E - Electronic Fuel Injection
    lst = ['/EF', 'EF', 'E', 'EI']#  ,
    df_ = expand_truncated(df_, col_mame='Model', trunc=lst, replacement=' EFI')
    

    
    return df_
    
 
def norm_model(df:pd.DataFrame):
    print('norm_model() echo')
    df_ = df.copy()
    
    ## misprints
    df_['Model'] = df_['Model'].str.replace(' I 4', ' I-4')
    

    ## TO DO does not work ## does not exist?
    df_['Model'] = df_['Model'].str.replace('3.2 TL 3.2L V6 FI DOHC', 
                                            'TL 3.2L V6 SFI DOHC')
    
    df_['Model'] = df_['Model'].str.replace('DOCH', 'DOHC')
    
    # df_['Model'] = df_['Model'].str.replace('LE SABR', 'LE SABRE')
    # df_['Model'] = df_['Model'].str.replace('LE SABREE', 'LE SABRE')
    df_['Model'] = df_['Model'].str.replace('SILHOUETT', 'SILHOUETTE')
    df_['Model'] = df_['Model'].str.replace('PROTEG', 'PROTEGE')
    df_['Model'] = df_['Model'].str.replace('FIVE HUNDRE', 'FIVE HUNDRED')
    
    ## TO DO does not work
    # df_['Model'] = df_['Model'].str.replace('VIB EFI', 'VIBE EFI')
    # df_['Model'] = df_['Model'].str.replace(r'\bVIB\s+EFI\b', 'VIBE EFI', regex=True, case=False)
    # df_['Model'] = df_['Model'].str.replace(r'VIB[\W\s]*EFI', 'VIBE EFI', regex=True)
    print("Before replacement:", df_['Model'][df_['Model'].str.contains('VIB EFI', na=False)].head(2))
    df_['Model'] = df_['Model'].str.replace('VIB EFI', 'VIBE EFI', regex=False)
    print("After replacement:", df_['Model'][df_['Model'].str.contains('VIBE EFI', na=False)].head(2))
    




    
    
    ## HO => for esier removemal
    ## TO DO doesnot work
    df_['Model'] = df_['Model'].str.replace('MFI HO', 
                                            'MFI HIHGUOT')
    
    ## truncated entries
    # SPI - Single Point Injection (SPI != SFI)
    
    ## MPI = MFI = M = MP = MF - Multiport Fuel Injection
    lst = ['MP', 'MF', 'M'] 
    df_ = expand_truncated(df_, col_mame='Model', trunc=lst, replasment=' MFI')
    df_['Model'] = df_['Model'].str.replace(' MPI', ' MFI', regex=True).str.strip()
    ## EFI = EF = EI = E - Electronic Fuel Injection
    lst = ['/EF', 'EF', 'E', 'EI']
    df_ = expand_truncated(df_, col_mame='Model', trunc=lst, replasment=' EFI')
    ## SFI - Sequential Fuel Injection
    df_ = expand_truncated(df_, col_mame='Model', trunc=['SF'], replasment=' SFI')


    ## SO - Single Overhead Camshaft => transfer to SOHC, show similarity with DOHC
    df_ = expand_truncated(df_, col_mame='Model', trunc=['SO'], replasment=' SOHC')
    ## NA = N - Natural Aspiration => NatAsp, for esier removemal
    df_ = expand_truncated(df_, col_mame='Model', trunc=['NA', 'N'], replasment=' NatAsp')
    df_['Model'] = df_['Model'].str.replace(' NA ', ' NatAsp ')
    
    df_['Model'] = df_['Model'].str.replace(' FW ', ' FWD ')
    df_['Model'] = df_['Model'].str.replace(' AW ', ' AWD ')
    # Unspecifi Unsp Unspecified
    lst = ['Unspecifi', 'Unspecif', 'Unsp']
    df_ = expand_truncated(df_, col_mame='Model', trunc=lst, replasment='Unspecified')
    # PIC =  Pick-Up
    df_ = expand_truncated(df_, col_mame='Model', trunc=['PIC'], replasment=' PICKUP')
    
    ## extract  2WD, 4WD, FWD
    ## 2WD = Two-Wheel Drive, "2WD" typically implies a rear-wheel-drive (RWD)
    ## FWD = front-wheel drive, all-wheel drive (AWD) = 4WD
    df_['WD'] = df_['Model'].str.extract(r'(\s.WD)')  
    df_.loc[:,'Remaining'] = df_['Model'].copy()  
    df_['Remaining'] = df_['Remaining'].str.replace(r'(\s.WD)', '', regex=True)
    

    
    ## extract engine volume (4.6L) with L
    df_[['model_L']] = df_['Remaining'].str.extract(r'(\d\.\dL)')
    df_['Remaining'] = df_['Remaining'].str.replace(r'(\d\.\dL)', '', regex=True)
    ## extract engine volume (4.6 ) whithout L
    df_[['L_tmp']] = df_['Remaining'].str.extract(r'(\d\.\d)')
    df_['Remaining'] = df_['Remaining'].str.replace(r'(\d\.\d)', '', regex=True)
    mask = df_['L_tmp'].notna()
    df_.loc[mask, 'model_L'] = df_['L_tmp']
    df_.drop(columns=['L_tmp'], inplace=True)
    df_['model_L'] = df_['model_L'].str.replace('L', '')
    
    df_['Remaining'] = df_['Remaining'].str.replace('/', '')

    
    ## extrat terms
    lst = [
           ' EXT', 'GRAND', ' PICKUP', 'Multiple Eng', 'SOLARA', 
           'Unspecified', 'SPORT', 'HIHGUOT',
           ' V6', ' V8', ' 4C', ' 6C', ' V', ' I4', ' I6', ' 2V' ' 4V',
           'I-4', ' I-6', ' 4B',
           '1500', '2500HD',
           ' EFI', ' MFI',' MPI', ' SFI', ' DOHC', ' SOHC', ' DI', 
           # ' SF', ' EF',  ' DOH',
           ' SPI', ' XL',
           #' DO', ' DI',  
           ' 24V', ' 16V', 'NatAsp', 'MR2',
           # ' E', ' M', ' PI', ' C', ' P', ' D',
           ]
    ## 6C = 6 cylinders, V=V6 = 6 cylinders in V-configuration
    ## I4, I-4:  an inline 4-cylinder engine
    ## M - manual transmission?, MPI - Multiport Fuel Injection = MFI
    
    ## DOHC=DOH=DO=D -Dual Overhead Camshaft
    
    for word in lst:
        mask = df_['Remaining'].str.contains(f'{word}')
        df_.loc[mask, word] = 1 
        df_[word] = df_[word].fillna(0).astype(int)       
        df_['Remaining'] = df_['Remaining'].str.replace(word, '')
        df_['Remaining'] = df_['Remaining'].str.strip() 

        
    ## extract Trim specifikations at the end of the model name:    
    lst_trim = [
        '4.', '2.', '5.', '6.',
        'S', 'I',
    ]
    pattern = r'(\s+' + '|'.join(lst_trim) + r')$'
    df_['Remaining'] = df_['Remaining'].str.strip()
    df_['mdTrim'] = df_['Remaining'].str.extract(pattern)
    df_['Remaining'] = df_['Remaining'].str.replace(pattern, '', regex=True).str.strip()
    
    df_.loc[:, 'ModelShort'] = df_['Remaining'].copy()
   # df_.drop(columns=['Remaining'], inplace=True)
    return df_

