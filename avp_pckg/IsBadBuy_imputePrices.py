import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator

from avp_pckg.DataFrame import AvPdataFrame 
from avp_pckg.TransformerCols import LinearImputer, TreeRegresor


class IsBadBuy_prepare_cols(TransformerMixin, BaseEstimator,):
    
    ## columns to set as string data type:
    cols_dtcut = ['VehicleAge','VehYear', 'BYRNO', 'VNST', 'VNZIP1', 'IsOnlineSale']
    cols_drop = ['PurchDate', 'VehYear', 'WheelTypeID',  'TopThreeAmericanName', 
                 'VNST',  'Nationality', 'Trim'
                ]
    
    def __init__(self) -> None:
        pass 
    
    def set_dt_str(self, X:pd.DataFrame):
        df = X.copy()
        for col in self.cols_dtcut:
            df.loc[:, col] = df.loc[:, col].astype('str')
        return df
            
    def drop_cols(self, X:pd.DataFrame):
        df = X.copy()
        df = df.drop(columns=self.cols_drop)    
        return df
    
    def calc_price_diff(self, X):   
        df_ = X.copy()
    
        df_.loc[:, 'RetailClean'] = df_['MMRAcquisitonRetailCleanPrice'] - df_['MMRCurrentRetailCleanPrice']
        df_.loc[:, 'AcqClean'] = df_['MMRAcquisitonRetailCleanPrice'] - df_['MMRAcquisitionAuctionCleanPrice']
        df_.loc[:, 'AcqRetail'] = df_['MMRAcquisitonRetailCleanPrice'] - df_['MMRAcquisitionRetailAveragePrice'] 
        df_.loc[:, 'AcqAuc'] = df_['MMRAcquisitionAuctionCleanPrice'] - df_['MMRAcquisitionAuctionAveragePrice']
        
        # cols = ['MMRAcquisitionAuctionAveragePrice', 'MMRAcquisitionAuctionCleanPrice',
        #         'MMRAcquisitionRetailAveragePrice', 'MMRAcquisitonRetailCleanPrice',
        #         'MMRCurrentAuctionAveragePrice', 'MMRCurrentAuctionCleanPrice',
        #         'MMRCurrentRetailAveragePrice', 'MMRCurrentRetailCleanPrice', ]
        
        # df_ = df_.drop(columns=cols)
        return df_
    
    def fit(self, X=None, y=None):
        return self
    
    def transform(self, X:AvPdataFrame, y=None):
        df = X.copy()
        # print('transform echo')
        df = self.set_dt_str(df)
        df = self.drop_cols(df)
        df = self.calc_price_diff(df)
        # print('no price diff')
        return df
           
        
class Cleaner(TransformerMixin, BaseEstimator,):
    def __init__(self) -> None:
        pass
    
    def fill_na(self, X:pd.DataFrame):
        df = X.copy()
        cols_cat = df.select_dtypes('object').columns
        cols_num = df.select_dtypes('number').columns
        df.loc[:, cols_cat] = df.loc[:, cols_cat].fillna('empty')
        df.loc[:, cols_num] = df.loc[:, cols_num].fillna(0)
        return df
    
    def clean_transmission(self, df):
        if 'Transmission' not in df.columns:
            print('no Transmission coulumn')
            return df  
        else:
            print(' clean_transmission echo')
            df.loc[:, 'Transmission'] = df['Transmission'].str.upper()
            mask = (df['Transmission'] == 'empty'.upper()) | (df['Transmission'].isna())
            df.loc[mask, 'Transmission'] = 'AUTO'           
            return df
        
    def expand_truncated(self, X, col_mame:str,  trunc:list, replacement:str ):
        df = X.copy()
        pattern = r'(\s+(' + '|'.join(trunc) + r'))$'
        df[col_mame] = df[col_mame].str.replace(pattern, replacement, regex=True).str.strip()
        return df
    
    def mistprints_model(self, X:pd.DataFrame):
        df_ = X.copy()
        
        lst = ['Multipl', 'Multiple E', 'Multiple En', 'Multiple Eng', 
            'Multiple Engi', 'Multiple Engin', 'Multiple Engine', 'Mult'] 
        df_ = self.expand_truncated(df_, col_mame='Model', trunc=lst, replacement=' Multiple')
        
        df_['Model'] = df_['Model'].str.replace('6C 4.2L I-', '6C 4.2L') # 'ENVOY 4WD 6C 4.2L'
        
        lst = ['PICKU', 'PIC']
        df_ = self.expand_truncated(df_, col_mame='Model', trunc=lst, replacement=' PICKUP')
        
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
        df_ = self.expand_truncated(df_, col_mame='Model',  trunc=lst, replacement=' SFI')
        lst = ['SM', 'SMP']
        df_ = self.expand_truncated(df_, col_mame='Model',  trunc=lst, replacement=' SMPI')
        
        lst = ['FI DO']
        df_ = self.expand_truncated(df_, col_mame='Model',  trunc=lst, replacement=' SFI DOHC')
        ## DOHC=DOH=DO=D -Dual Overhead Camshaft
        lst = ['DOH', 'DO', 'D']
        df_ = self.expand_truncated(df_, col_mame='Model', trunc=lst, replacement=' DOHC')
        ## SO - Single Overhead Camshaft => transfer to SOHC, show similarity with DOHC
        df_ = self.expand_truncated(df_, col_mame='Model', trunc=['SO'], replacement=' SOHC')
        ## MPI = MFI = M = MP = MF - Multiport Fuel Injection
        lst = ['MP', 'MF', 'M'] 
        df_ = self.expand_truncated(df_, col_mame='Model', trunc=lst, replacement=' MFI')
        df_['Model'] = df_['Model'].str.replace(' MPI', ' MFI', regex=True).str.strip()
        ## EFI = EF = EI = E - Electronic Fuel Injection
        lst = ['/EF', 'EF', 'E', 'EI', '/E']
        df_ = self.expand_truncated(df_, col_mame='Model', trunc=lst, replacement=' EFI')
        ## SFI - Sequential Fuel Injection / # SPI - Single Point Injection (SPI != SFI)
        df_ = self.expand_truncated(df_, col_mame='Model', trunc=['SF'], replacement=' SFI')
        ## Unspecifi Unsp Unspecified
        lst = ['Unspecifi', 'Unspecif', 'Unsp']
        df_ = self.expand_truncated(df_, col_mame='Model', trunc=lst, replacement='Unspecified')
        ## for easier remooval  
        df_ = self.expand_truncated(df_, col_mame='Model', trunc=['HO'], replacement='HighOutput')
        ## NA = N - Natural Aspiration => NatAsp, for esier removemal
        df_ = self.expand_truncated(df_, col_mame='Model', trunc=['NA', 'N'], replacement=' NatAsp')
        df_['Model'] = df_['Model'].str.replace(' NA ', ' NatAsp ')
        ## 4WD -> AWD 
        df_['Model'] = df_['Model'].str.replace(' 4WD', ' AWD')
        df_ = self.expand_truncated(df_, col_mame='Model', trunc=['PICKUP 2'], replacement=' PICKUP 2WD')
        df_ = self.expand_truncated(df_, col_mame='Model', trunc=['PICKUP 4'], replacement=' PICKUP AWD')
        ##
        df_['Model'] = df_['Model'].str.replace(' I 4', ' I4')
        df_['Model'] = df_['Model'].str.replace(' I-4', ' I4')
        df_['Model'] = df_['Model'].str.replace(' I-6', ' I6')
        df_['Model'] = df_['Model'].str.replace(' I-6', ' I6')
        
        ## drop insufficient information:
        lst = ['2.', '3.', '4.', '5.', '6.', 'I', '6-230/250-1V', '4'] 
        df_ = self.expand_truncated(df_, col_mame='Model', trunc=lst, replacement='')

        return df_
        
    def mistprints_submodel(self, X:pd.DataFrame):
        df_ = X.copy()
        df_['SubModel'] = df_['SubModel'].str.replace('HATCKBACK', 'HATCHBACK')
        df_['SubModel'] = df_['SubModel'].str.replace('HARTOP', 'HARDTOP')
        df_['SubModel'] = df_['SubModel'].str.replace('LIMTED', 'LIMITED')
        # dublicated model name
        df_['SubModel'] = df_['SubModel'].str.replace('MAZDA3 ', '')
        df_['SubModel'] = df_['SubModel'].str.replace('MAZDA5 ', '')
        return df_ 
    
    def norm_model(self, X:pd.DataFrame):
        print('norm_model() echo')
        df_ = X.copy()
        
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
            ' EXT', 'GRAND', 'Multiple', 'SOLARA', 
            'Unspecified', ' SPORT ', 'HighOutput',
            ' V6', ' V8', ' 4C', ' 6C', ' 5C', ' V', ' I4', ' I5', ' I6', # cylinders
            ' 2V', ' 4V', ' 24V', ' 16V', # valves in cylinder or in total
            ' 2B', ' 4B',
            '1500', '2500',  ' PICKUP', # car load
            ' EFI', ' MFI',' MPI', ' SFI', ' DOHC', ' SOHC', ' DI', ' SMPI',
            ' SPI', ' XL',
                'NatAsp', 'MR2',
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
        
        # form 'md_Size' column
        mask = (df_.loc[:, '1500'] == 1) | (df_.loc[:, '2500'] == 1)
        # print('norm_model mask.sum = ', mask.sum())
        df_.loc[mask, 'md_Size'] = 'LARGE TRUCK'
        mask = (df_.loc[:, 'PICKUP'] == 1 ) & ( df_.loc[:, 'md_Size'] != 'LARGE TRUCK')
        # print('norm_model mask.sum = ', mask.sum())
        df_.loc[mask, 'md_Size'] = 'SMALL TRUCK'
        mask = (df_.loc[:, 'Size'] == 'empty') & df_.loc[:, 'md_Size'].notnull()
        # print('norm_model mask.sum = ', mask.sum())
        df_.loc[mask, 'Size'] = df_.loc[:, 'md_Size']
        cols = ['1500', '2500', 'PICKUP', 'md_Size']
        df_ = df_.drop(columns=cols)

        return df_

    def norm_submodel(self, X:pd.DataFrame):
        '''split submodel string in columns'''
        df_ = X.copy()
        df_.loc[df_['SubModel'].isna(), 'SubModel'] = 'empty'
        
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
            'PREMIER', 'LIMITED', # Trim level
            'GRAND TOURING', 'TOURING', 'EDGE', # Trim level  'GRAND',
            'CLASSIC', 'CUSTOM', 'LUXURY', 'SIGNATURE', # Trim level
            'PREMIUM', 'POPULAR', 'COMFORT', 'CARGO', 'SPECIAL', 'DELUXE',
                'VALUE', 'PLUS', 'PANEL', 'TRAC',
            'TURBO',    'CONVENIENCE', 
            'NAVIGATION', 'AUTO', 'DURATEC', 'HEMI', 'PACKAGE', 'HIGHLINE', 'PRERUNNER',
            '5SP', '6SP', 'FFV', 'XUV', 'ZX5', 'ZX4', 'ZX3', 'ZX2', 'ZWD',
            'AWD', 'EXT',
            ]
        ## ChatGPT: ZX2 = 2D, ZX3 = 3D, ..., ZX5 = 5D, ZWD = Wagon
        
        for word in lst:
            # print(word)
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
        mask = df_['sb_GRAND TOURING'] == 1
        df_.loc[mask, 'sb_Trim'] = 'GT'
        mask = df_['sb_TOURING'] == 1
        df_.loc[mask, 'sb_Trim'] = 'Tou'
        mask = df_['sb_Trim'] == 'ADVENTURER'
        df_.loc[mask, 'sb_Trim'] = 'Adv'
        mask = df_['sb_EDGE'] == 1
        df_.loc[mask, 'sb_Trim'] = 'Edg'
        mask = df_['sb_CLASSIC'] == 1
        df_.loc[mask, 'sb_Trim'] = 'Cla'
        mask = df_['sb_CUSTOM'] == 1
        df_.loc[mask, 'sb_Trim'] = 'Cus'
        mask = df_['sb_LUXURY'] == 1
        df_.loc[mask, 'sb_Trim'] = 'Lux'  
        mask = df_['sb_SIGNATURE'] == 1
        df_.loc[mask, 'sb_Trim'] = 'Sig'  
        
        mask = df_['sb_SPECIAL'] == 1
        df_.loc[mask, 'sb_Trim'] = 'Spe'
        df_.loc[df_['Type'] == 'LS', 'sb_Trim'] = 'LS'
        df_.loc[df_['Type'] == 'LS', 'Type'] = 'empty'
        
        cols = ['sb_PREMIER', 'sb_LIMITED', 'sb_GRAND TOURING',
                'sb_TOURING','sb_EDGE', 'sb_CLASSIC', 'sb_CUSTOM', 
                'sb_LUXURY', 'sb_SIGNATURE', 'sb_SPECIAL', 
                ] # 'sb_TOURING',
        df_.loc[df_['sb_Trim']=='LAREDO', 'sb_Trim'] = 'Lar'
        df_ = df_.drop(columns=cols)
        
        return df_        
    
    
    def clean_df(self, df):
        print('clean_df() echo')
        df = self.fill_na(df)
        df = self.clean_transmission(df)
        df = self.mistprints_model(df)
        df = self.mistprints_submodel(df)
        df = self.norm_model(df)
        df = self.norm_submodel(df)
        
        mask = df['model_L'].isna() &  df['L'].notnull()
        df.loc[mask, 'model_L' ] = df['L']
        df = df.drop(columns=['L'])
        
        df.loc[:, 'EXT'] = df['sb_EXT']
        df = df.drop(columns=['sb_EXT'])
        
        mask = df['sb_AWD'] == 1
        df.loc[mask, 'WD'] == ' AWD'
        df = df.drop(columns=['sb_AWD'])
        
        cols = ['Remaining',
        'sb_CAB', 'sb_CREW', 'sb_SPORTBACK', 'sb_SPORT', 'sb_QUAD',
        'sb_HARDTOP', 'sb_HYBRID', 'sb_PREMIUM', 'sb_POPULAR', 'sb_COMFORT',
        'sb_CARGO', 'sb_DELUXE', 'sb_VALUE', 'sb_PLUS', 'sb_PANEL', 'sb_TRAC',
        'sb_TURBO', 'sb_CONVENIENCE', 'sb_NAVIGATION', 'sb_AUTO', 'sb_DURATEC',
        'sb_HEMI', 'sb_PACKAGE', 'sb_HIGHLINE', 'sb_PRERUNNER', 'sb_5SP',
        'sb_6SP', 'sb_FFV', 'sb_XUV', 'sb_ZX5', 'sb_ZX4', 'sb_ZX3', 'sb_ZX2',
        'sb_ZWD']
        df = df.drop(columns=cols)
        
        return df

        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        df = X.copy()
        df = self.clean_df(df)
        return df
    
    
class IsBadBuyImputer(TransformerMixin, BaseEstimator):
    ''' attempt to involve imputing of numerical collumns'''
    cols_tree_reg = [
            'Auction',
            'VehicleAge', 
            'ModelShort', # 'Model', 
            'SubModel',  #'Type',
            'VehOdo', 
            'Size',   
            'Trim', #'sb_Trim',
      #      'VehBCost',
            'model_L',
            'WD',
            'V6', 'V8', '4C', '6C', '5C', 'V', 'I4', 'I5', 'I6', 
             'XL', 'NatAsp', 'MR2', 'D',
           'MMRAcquisitionAuctionAveragePrice', 'MMRAcquisitionAuctionCleanPrice',
           'MMRAcquisitionRetailAveragePrice', 'MMRAcquisitonRetailCleanPrice',
            ] 
    def __init__(self, percent=1, verbose=0) -> None:

        self.verbose = verbose
        self.percent = percent
        self.lin_imp1 = LinearImputer(col_x='VehBCost', 
                                    col_y='MMRAcquisitionAuctionAveragePrice')
        self.lin_imp2 = LinearImputer(col_x='VehBCost', 
                                    col_y='MMRAcquisitionAuctionCleanPrice')
        self.lin_imp3 = LinearImputer(col_x='VehBCost', 
                                    col_y='MMRAcquisitionRetailAveragePrice')
        self.lin_imp4 = LinearImputer(col_x='VehBCost', 
                                    col_y='MMRAcquisitonRetailCleanPrice')
        self.tree_regr1 = TreeRegresor(col_target='MMRCurrentAuctionAveragePrice',
                                      cols_reg=self.cols_tree_reg,
                                      cols_impurity_detection=['MMRCurrentAuctionAveragePrice', 'VehBCost' ],
                                      nan_condition=('<', 100),
                                      percent = percent,
                                      verbose=verbose,
                                      )
        self.tree_regr2 = TreeRegresor(col_target='MMRCurrentAuctionCleanPrice',
                                      cols_reg=self.cols_tree_reg,
                                      cols_impurity_detection=['MMRCurrentAuctionCleanPrice', 'VehBCost' ],
                                      nan_condition=('<', 100),
                                      percent = percent,
                                      verbose=verbose,
                                      )
        self.tree_regr3 = TreeRegresor(col_target='MMRCurrentRetailAveragePrice',
                                      cols_reg=self.cols_tree_reg,
                                      cols_impurity_detection=['MMRCurrentRetailAveragePrice', 'VehBCost' ],
                                      nan_condition=('<', 100),
                                      percent = percent,
                                      verbose=verbose,
                                      )
        self.tree_regr4 = TreeRegresor(col_target='MMRCurrentRetailCleanPrice',
                                      cols_reg=self.cols_tree_reg,
                                      cols_impurity_detection=['MMRCurrentRetailCleanPrice', 'VehBCost' ],
                                      nan_condition=('<', 100),
                                      percent = percent,
                                      verbose=verbose,
                                      )
        
        
    def fit(self, X, y=None):
        df = X.copy()
        
        ## imputed Aqc cols are used to impute Current cols
        ## => the transform is nessesary 
        df = self.lin_imp1.fit_transform(df)
        df = self.lin_imp2.fit_transform(df)
        df = self.lin_imp3.fit_transform(df)
        df = self.lin_imp4.fit_transform(df)
        print('fit: Aqc cols impute is ready')
        ## transform left for a  simmetry
        df = self.tree_regr1.fit_transform(df)
        df = self.tree_regr2.fit_transform(df)
        df = self.tree_regr3.fit_transform(df)
        df = self.tree_regr4.fit_transform(df)
        print('fit: Current cols impute is ready')
        
        return self
    
    def transform(self, X, y=None):
        df = X.copy()
        
        ## imputed Aqc cols are used to impute Current cols
        ## => the transform is nessesary 
        df = self.lin_imp1.transform(df)
        df = self.lin_imp2.transform(df)
        df = self.lin_imp3.transform(df)
        df = self.lin_imp4.transform(df)
        print('transform: Aqc cols impute is ready')
        print(f'{type(df)=}')
        ## transform left for a  simmetry
        df = self.tree_regr1.transform(df)
        df = self.tree_regr2.transform(df)
        df = self.tree_regr3.transform(df)
        df = self.tree_regr4.transform(df)
        print('transform: Current cols impute is ready')
        print(f'{type(df)=}')

        return df    