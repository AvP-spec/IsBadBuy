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

 
 
def expand_truncated(df, col_mame:str,  trunc:list, replacement:str ):
    df_ = df.copy()
    pattern = r'(\s+(' + '|'.join(trunc) + r'))$'
    df_[col_mame] = df_[col_mame].str.replace(pattern, replacement, regex=True).str.strip()
    return df_


def mistprints_model(df:pd.DataFrame):
    
    df_ = df.copy()
    
    lst = ['Multipl', 'Multiple E', 'Multiple En', 'Multiple Eng', 
           'Multiple Engi', 'Multiple Engin', 'Multiple Engine', 'Mult'] 
    df_ = expand_truncated(df_, col_mame='Model', trunc=lst, replacement=' Multiple')
    
    df_['Model'] = df_['Model'].str.replace('6C 4.2L I-', '6C 4.2L') # 'ENVOY 4WD 6C 4.2L'
    
    lst = ['PICKU', 'PIC']
    df_ = expand_truncated(df_, col_mame='Model', trunc=lst, replacement=' PICKUP')
    
    df_['Model'] = df_['Model'].str.replace('300 3.5L / 6.0L V12', 
                                            '300 3.5L V6')
    
    df_['Model'] = df_['Model'].str.replace('GRAND AM V6 3.4L V 6', 
                                            'GRAND AM V6 3.4L')
    
    df_['Model'] = df_['Model'].str.replace('3.2 CL 3.2L V 6', 
                                            'CL 3.2L V6')
    
    df_['Model'] = df_['Model'].str.replace('3.4L V 6', 
                                            '3.4L V6')
    
    df_['Model'] = df_['Model'].str.replace('CAMRY 4C EI I-4 2.2L', 
                                            'CAMRY 4C EFI I-4 2.2L')
    df_['Model'] = df_['Model'].str.replace('ECLIPSE EI V6 3.0L S', 
                                            'ECLIPSE EFI V6 3.0L SFI')
    
    df_['Model'] = df_['Model'].str.replace('ESCORT 4-FI-2.0L', 
                                            'ESCORT 4C MFI 2.0L')
    
    df_['Model'] = df_['Model'].str.replace('TAURUS 3.0L V6 EFI F', 
                                            'TAURUS 3.0L V6 EFI')
    
    df_['Model'] = df_['Model'].str.replace('2500HD SILVERADO PICKUP', 
                                            '2500 SILVERADO PICKUP')
    
    df_['Model'] = df_['Model'].str.replace(' AWD 6C 4', 
                                            ' AWD 6C')
    df_['Model'] = df_['Model'].str.replace('TRAILBLAZER 2WD 6C 4', 
                                            'TRAILBLAZER 2WD 6C')
    lst = ['SF', 'S']
    df_ = expand_truncated(df_, col_mame='Model',  trunc=lst, replacement=' SFI')
    lst = ['SM', 'SMP']
    df_ = expand_truncated(df_, col_mame='Model',  trunc=lst, replacement=' SMPI')
    
    lst = ['FI DO']
    df_ = expand_truncated(df_, col_mame='Model',  trunc=lst, replacement=' SFI DOHC')
    ## DOHC=DOH=DO=D -Dual Overhead Camshaft
    lst = ['DOH', 'DO', 'D']
    df_ = expand_truncated(df_, col_mame='Model', trunc=lst, replacement=' DOHC')
    ## SO - Single Overhead Camshaft => transfer to SOHC, show similarity with DOHC
    df_ = expand_truncated(df_, col_mame='Model', trunc=['SO'], replacement=' SOHC')
    ## MPI = MFI = M = MP = MF - Multiport Fuel Injection
    lst = ['MP', 'MF', 'M'] 
    df_ = expand_truncated(df_, col_mame='Model', trunc=lst, replacement=' MFI')
    df_['Model'] = df_['Model'].str.replace(' MPI', ' MFI', regex=True).str.strip()
    ## EFI = EF = EI = E - Electronic Fuel Injection
    lst = ['/EF', 'EF', 'E', 'EI', '/E']
    df_ = expand_truncated(df_, col_mame='Model', trunc=lst, replacement=' EFI')
    ## SFI - Sequential Fuel Injection / # SPI - Single Point Injection (SPI != SFI)
    df_ = expand_truncated(df_, col_mame='Model', trunc=['SF'], replacement=' SFI')
    ## Unspecifi Unsp Unspecified
    lst = ['Unspecifi', 'Unspecif', 'Unsp']
    df_ = expand_truncated(df_, col_mame='Model', trunc=lst, replacement='Unspecified')
    ## for easier remooval  
    df_ = expand_truncated(df_, col_mame='Model', trunc=['HO'], replacement='HighOutput')
    ## NA = N - Natural Aspiration => NatAsp, for esier removemal
    df_ = expand_truncated(df_, col_mame='Model', trunc=['NA', 'N'], replacement=' NatAsp')
    df_['Model'] = df_['Model'].str.replace(' NA ', ' NatAsp ')
    ## 4WD -> AWD 
    df_['Model'] = df_['Model'].str.replace(' 4WD', ' AWD')
    df_ = expand_truncated(df_, col_mame='Model', trunc=['PICKUP 2'], replacement=' PICKUP 2WD')
    df_ = expand_truncated(df_, col_mame='Model', trunc=['PICKUP 4'], replacement=' PICKUP AWD')
    ##
    df_['Model'] = df_['Model'].str.replace(' I 4', ' I4')
    df_['Model'] = df_['Model'].str.replace(' I-4', ' I4')
    df_['Model'] = df_['Model'].str.replace(' I-6', ' I6')
    df_['Model'] = df_['Model'].str.replace(' I-6', ' I6')
    
    ## drop insufficient information:
    lst = ['2.', '3.', '4.', '5.', '6.', 'I', '6-230/250-1V', '4'] 
    df_ = expand_truncated(df_, col_mame='Model', trunc=lst, replacement='')

    return df_
    
 
def norm_model(df:pd.DataFrame):
    print('norm_model() echo')
    df_ = df.copy()
    
    ## extract  2WD, 4WD, FWD, AWD
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
           ' EXT', 'GRAND', ' PICKUP', 'Multiple', 'SOLARA', 
           'Unspecified', ' SPORT ', 'HighOutput',
           ' V6', ' V8', ' 4C', ' 6C', ' 5C', ' V', ' I4', ' I5', ' I6', ' 2V', ' 4V', 
           ' 2B', ' 4B',
           '1500', '2500', # car load
           ' EFI', ' MFI',' MPI', ' SFI', ' DOHC', ' SOHC', ' DI', ' SMPI',
           ' SPI', ' XL',
           ' 24V', ' 16V', 'NatAsp', 'MR2',
           ]
    ## 6C = 6 cylinders, V=V6 = 6 cylinders in V-configuration
    ## I4, I-4:  an inline 4-cylinder engine
    ## M - manual transmission?, MPI - Multiport Fuel Injection = MFI
    ## Sequential Multi-Point Injection (SMPI)
    
    for word in lst:
        mask = df_['Remaining'].str.contains(f'{word}')
        df_.loc[mask, word.strip()] = 1 
        df_[word.strip()] = df_[word.strip()].fillna(0).astype(int)       
        df_['Remaining'] = df_['Remaining'].str.replace(word, '')
        df_['Remaining'] = df_['Remaining'].str.strip() 
    
    #df_.loc[:, 'ModelShort'] = df_['Remaining'].copy()
    df_ = df_.rename(columns={'Remaining': 'ModelShort'})

    return df_

   
def clean_transmission(df):
    if 'Transmission' not in df.columns:
        print('no Transmission coulumn')
        return df  
    else:
        print(' clean_transmission echo')
        df.loc[:, 'Transmission'] = df['Transmission'].str.upper()
        mask = df['Transmission'].isna() # == 'empty'
        df.loc[mask, 'Transmission'] = 'AUTO'           
        return df

def mistprints_submodel(df:pd.DataFrame):
    df_ = df.copy()
    df_['SubModel'] = df_['SubModel'].str.replace('HATCKBACK', 'HATCHBACK')
    df_['SubModel'] = df_['SubModel'].str.replace('HARTOP', 'HARDTOP')
    df_['SubModel'] = df_['SubModel'].str.replace('LIMTED', 'LIMITED')
    # dublicated model name
    df_['SubModel'] = df_['SubModel'].str.replace('MAZDA3 ', '')
    df_['SubModel'] = df_['SubModel'].str.replace('MAZDA5 ', '')
    return df_ 
           
def norm_submodel(df:pd.DataFrame):
    '''split submodel string in columns'''
    df_ = df.copy()
    
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
    lst = ['CAB', 'CREW',  'SPORTBACK', 'SPORT', 'QUAD', 'HARDTOP', 'HYBRID', 
           'PREMIER', 'LIMITED', 'TOURING', 'EDGE', # Trim level
           'CLASSIC', # Trim level
           'PREMIUM', 'POPULAR', 'COMFORT', 'CARGO', 'SPECIAL', 'DELUXE',
            'VALUE', 'PLUS', 'PANEL', 'TRAC',
           'TURBO',  'GRAND', 'CUSTOM', 'LUXURY', 'CONVENIENCE', 'SIGNATURE',
           'NAVIGATION', 'AUTO', 'DURATEC', 'HEMI', 'PACKAGE', 'HIGHLINE', 'PRERUNNER',
           '5SP', '6SP', 'FFV', 'XUV', 'ZX5', 'ZX4', 'ZX3', 'ZX2', 'ZWD',
           'AWD', 'EXT',
           ]
    ## ChatGPT: ZX2 = 2D, ZX3 = 3D, ..., ZX5 = 5D, ZWD = Wagon
    
    for word in lst:
        mask = df_['Remaining'].str.contains(f'{word}')
        df_.loc[mask, 'sb_'+ word] = 1 
        df_['sb_'+ word] = df_['sb_' + word].fillna(0).astype(int)       
        df_['Remaining'] = df_['Remaining'].str.replace(word, '')
        # print(word, ':', df_[word].sum())
    
    ## extract first word 
    df_['Remaining'] = df_['Remaining'].str.strip() 
    df_[['Type', 'sb_Trim']] = df_['Remaining'].str.extract(r'(^\S+)\s*(.*)')
    df_['sb_Trim'] = df_['sb_Trim'].str.strip() 
    #df_ = df_.drop(columns='Remaining')   
    mask = df_['sb_Trim'] == ''
    # print(mask.sum())
    df_.loc[mask, 'sb_Trim' ] = 'empty'
    df_.loc[:, 'sb_Trim'] = df_.loc[:, 'sb_Trim'].fillna('empty')
    
    ## insert identified Trim keys
    mask = df_['sb_PREMIER'] == 1
    df_.loc[mask, 'sb_Trim'] = 'Pre'
    mask = df_['sb_LIMITED'] == 1
    df_.loc[mask, 'sb_Trim'] = 'Lim'
    mask = df_['sb_TOURING'] == 1
    df_.loc[mask, 'sb_Trim'] = 'Tou'
    mask = df_['sb_Trim'] == 'ADVENTURER'
    df_.loc[mask, 'sb_Trim'] = 'Adv'
    mask = df_['sb_EDGE'] == 1
    df_.loc[mask, 'sb_Trim'] = 'Edg'
    mask = df_['sb_CLASSIC'] == 1
    df_.loc[mask, 'sb_Trim'] = 'Cla'
    
    cols = ['sb_PREMIER', 'sb_LIMITED',   'sb_EDGE', 'sb_CLASSIC'] # 'sb_TOURING',
    df_ = df_.drop(columns=cols)
    
      
    return df_        
   
        
def clean_df(df):
    df = clean_transmission(df)
    df = mistprints_model(df)
    df = norm_model(df)
    df = mistprints_submodel(df)
    
    return df
        
