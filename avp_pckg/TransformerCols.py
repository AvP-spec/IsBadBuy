from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import RANSACRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.cluster import DBSCAN

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import operator

from avp_pckg.DataFrame import AvPdataFrame
# from avp_pckg.avp_model_selection import PrepareColsBase


class PrepareColsBasePipe:
    def __init__(self, cols_cat=[], cols_num=[], 
                 max_cat=25, cols_binary = []) -> None:
        self.cols_cut = cols_cat
        self.cols_num = cols_num
        self.max_cat = max_cat
        self.cols_binary = cols_binary
        print('class PrepareColsBasePipe echo')
        
    def define_cols(self, X:pd.DataFrame):
        print('define_cols() echo')
        df = X.copy()
        self.cols_cut = df.select_dtypes('object').columns
        self.cols_num = df.select_dtypes('number').columns
        ## TO DO define binary columns. 
        # self.cols_binary = [name for name in self.cols_num if df[name].unique() == [0, 1]]
        
        
    def make_pipe(self):
        pipe_cat = Pipeline(steps=[
            ('impute_empty', SimpleImputer(strategy='constant', fill_value='empty')),
            ('ohe', OneHotEncoder(handle_unknown='infrequent_if_exist', max_categories=self.max_cat)) 
            ])

        pipe_num = Pipeline(steps=[
                ('impute_null', SimpleImputer(strategy='constant', fill_value=0)),
                ("scaler", StandardScaler()) # not neseccary for tree estimator
            ])
        
        pipe_binary = Pipeline(steps=[
            ('impute_null', SimpleImputer(strategy='constant', fill_value=0)),
        ])

        prepare_cols = ColumnTransformer(
                transformers=[
                    ('transform_categorical', pipe_cat, self.cols_cut),
                    ('transform_numerical', pipe_num, self.cols_num),
                    ('keep bynary cols', pipe_binary, self.cols_binary)
                ],
                remainder='drop',
            )
        return prepare_cols   
                   
# TransformerMixin, BaseEstimator, AvPdataFrame
class CatColReducer(TransformerMixin, BaseEstimator,):
    '''categories reduction in categorical column'''
    freq_range = {'02': {'maxf': 2, 'minf': 0},
                '05': { 'maxf': 5, 'minf': 2},
                '10': {'maxf': 10, 'minf': 5},
                '15': {'maxf': 15, 'minf': 10},
                '20': {'maxf': 20, 'minf': 15},
                '30': {'maxf': 30, 'minf': 20},
                '40': {'maxf': 40, 'minf': 30},
                '50': {'maxf': 50, 'minf': 40},
                '60': {'maxf': 60, 'minf': 50},
                '70': {'maxf': 70, 'minf': 60},
                '80': {'maxf': 80, 'minf': 70},
                '90': {'maxf': 90, 'minf': 80},
                '100': {'maxf': 100, 'minf': 90},
                 }
    
    def __init__(self, cols_fit:list) -> None:
        '''
        Ags:
        X:   pd.DataFrame featureas
        y:   pd.DataFrame target (sould have column name => y.columns[0])
        cols: list of columns names (should be in X) for transformation
        '''
        # print('echo class CatColReducer')
        self.cols_fit = cols_fit
        self.cols_dict = {}             # result of fit function, for each column dict of new categories
        # print('echo class CatColReducer: init ends')  
        
    def combine_X_y(self, X, y):
        df = X.copy()
        df.insert(0, y.columns[0], y.copy())
        return AvPdataFrame(df)
        
        
    def fit_col(self, X, y, col):
        '''get dickt for one column
           {new_category1: [old_1, old_2],
           new_category2: [old_3, old_4, old_5],
           ....}
        ''' 
        df_ = self.combine_X_y(X, y)
        count_range = {f'{col}_E': {'max_counts': X.shape[0], 'min_counts': 10_000},
			f'{col}_D': {'max_counts': 10_000, 'min_counts': 1_000},
			f'{col}_C': {'max_counts': 1_000, 'min_counts': 100},
			f'{col}_B': {'max_counts': 100, 'min_counts': 10},
			f'{col}_A': {'max_counts': 10, 'min_counts': 0},
			}
        
        df = df_.calc_frequency(col, target=y.columns[0])
        col_dict = {}
        for new_cat in count_range.keys():
            ### select counts category 
            max_counts = count_range[new_cat]['max_counts']
            min_counts = count_range[new_cat]['min_counts']
            dct = {} 

            for el in self.freq_range.keys():
                ### add subcategory by frequency
                max_freq = self.freq_range[el]['maxf']
                min_freq = self.freq_range[el]['minf']
                
                counts_ = df.columns[1]
                freq_ = df.columns[2]
                df_tmp = df[(df[counts_] < max_counts) & ( df[counts_] >= min_counts )]
                df_tmp = df_tmp[(df_tmp[freq_] < max_freq) & ( df_tmp[freq_] >= min_freq )]
                value = list(df_tmp.index) # categories names within counts and categories ranges
                if value:
                    dct[new_cat + el] = value
                    
            col_dict.update(dct)
        return col_dict
        
    def fit(self, X, y):
        '''create column_dicts for each column 
        should return self - for sklearn pipeline
        '''
        # print('fit of CatColReduction echo')
        for col in self.cols_fit:
            # print('col'.center(50, '-'))
            dct = self.fit_col(X=X, y=y, col=col) 
            self.cols_dict.update({col: dct})
        return self
            
    def transform_col(self, df, col, dct):
        '''transform single column'''
        df_ = df.copy()
        ### dct = {new_category1: [old_1, old_2], }
        for key_ in dct:
            ### group categories to new_category = key_
            mask = df_[col].isin(dct[key_])
            df_.loc[mask, col+'_avp'] = key_
            # df_.loc[mask, col] = key_
            
        df_.drop(columns=[col], inplace=True)
        return df_
        
    def transform(self, X, y=None):
        '''transform columns in self.cols_dict.keys()
        should return X.copy() - for sklearn pipline
        '''
        #print('trandform function CatColReduction echo')
        df = X.copy()
        # print(self.cols_dict)  
        for col in self.cols_dict.keys():
            #print(col) 
            if col not in X.columns:
                print(f'{col} not in CatColReduser dict. Make a fit') 
                return None        
            assert col in X.columns
            df = self.transform_col(df, col, self.cols_dict[col])
        return df
             
             
             
class NumToCatTransformer(TransformerMixin, BaseEstimator,):

    def __init__(self, cols_dict) -> None:
        self.cols_dict = cols_dict
        self.original_names = []
        self.new_cols_names = []
    
    def fit(self, X=None, y=None):
        return self
    
    
    def num_col_transformer(self, df, col='VehOdo', split=[50_000, 75_000, 100_000]):
        ''' assign categories to numerical vaues in df['col'] with thresholds in 'split'
        '''
        df_ = df.copy()
        s_min = df[col].min()
        s_max = df[col].max()
        splt = [s_min] + split + [s_max]
        # print(splt)
        i = 0
        for value in splt[1:]:
            # print(splt[i]) # low split value
            # print(s) # high split value          
            l_name = col + str(value)
            # print(l_name)
            mask = ( df_[col] <= value) & (df_[col] > splt[i]) 
            df_.loc[mask, col+'_avp'] = l_name
           # df_.loc[mask, col] = l_name
            i += 1
        df_.drop(columns=[col], inplace=True) 
        #df_ = df_.rename(columns={col+'_avp': col})
        return df_
    
    def transform(self, X=None, y=None):
 
        df_ = X.copy()
        for key in self.cols_dict.keys():             
            if key in df_.columns:
                # print(key) 
                df_ = self.num_col_transformer(df_, col=key, split=self.cols_dict[key])
                self.original_names.append(key)
                self.new_cols_names.append(key + '_avp')
    
        return df_


class LinearImputer(TransformerMixin, BaseEstimator):
    '''col_y - to be imputed'''
    def __init__(self, 
                 col_x:str, 
                 col_y:str, 
                 random_state=42, 
                 verbose=False,
                 threshold=0) -> None:
        self.col_x = col_x
        self.col_y = col_y
        self.random_state = random_state
        self.verbose = verbose
        self.threshold = threshold  # value=0 if value < threshold
        self.model = None
        #label_encoder = None
        
    def fit(self, X=None, y=None):
        df_ = X.copy()
        # impute NaN values ​​as 0, model does not accept NaN
        df_.loc[df_[self.col_x].isnull(),  self.col_x] = 0
        df_.loc[df_[self.col_y].isnull(),  self.col_y] = 0
        
        self.model = RANSACRegressor(random_state=self.random_state)
        self.model.fit(df_.loc[:, [self.col_x]], df_.loc[:, self.col_y])
        
        return self
    
    def transform(self, X=None, y=None):
        df_ = X.copy()
        # impute NaN values ​​as 0, model does not accept NaN
        df_.loc[df_[self.col_x].isnull(),  self.col_x] = 0
        df_.loc[df_[self.col_y].isnull(),  self.col_y] = 0
        
        #print('echo LinearImputer2')
        mask_y0 = df_[self.col_y] <= self.threshold
        #print(mask_y0)
        x0 = df_.loc[mask_y0, [self.col_x]]
        if x0.shape[0]:
            y0 = self.model.predict(x0)
            df_.loc[mask_y0, self.col_y] = y0

            if self.verbose:
                sns.regplot(
                        data= df_,
                        x=self.col_x, 
                        y=self.col_y,
                        scatter_kws={'s': 3, 'color': 'r'},
                        line_kws={'color': 'r'}
                        )
                sns.regplot(
                        x=x0, 
                        y=y0,
                        scatter_kws={'s': 3, 'color': 'b'},
                        line_kws={'color': 'b'}
                        )  
                        
                        
        return df_

#################################################################
#################################################################
class IsBadBuyTransformer(TransformerMixin, BaseEstimator):
    ''' does not bring advantage in comparison with base model'''
    def __init__(self, 
                 cols_cat=[], # categorical column names list
                 cols_num=[], # numerical column name list
                 num_to_cat_dict={}, # cols_num for transfer into categorical
                 cols_MCA_list=[], # multi-categorical annalisis columns
                 max_cat=15, # maximum categories
                 impute=False,
                 diff = False,
                 target_encoder = False
                 ) -> None:
        
        #print('__init__ hallo_cut self _ str')
        self.cols_cat = cols_cat
        self.cols_num = cols_num
        self.cols_num_ = cols_num # reserve copy to return after fit
        self.cols_cat_ = cols_cat # reserve copy 
        self.num_to_cat_dict = num_to_cat_dict
        self.cols_MCA_list = cols_MCA_list
        self.max_cat = max_cat
        self.impute = impute
        self.diff = diff
        self.target_encoder = target_encoder
        
        self.cols_catL = []
        self.cols_catS = []
        self.to_category = None
        self.categories_reduser = None
        
        self.ohe = None
        self.scaler = None
        self.cols_tranformer= None 
        self.imputer_list = []
        
    def calc_price_diff(self, X):
       # print('ehco calc_price_diff')
        df_ = X.copy()
    
        df_.loc[:, 'RetailClean'] = df_['MMRAcquisitonRetailCleanPrice'] - df_['MMRCurrentRetailCleanPrice']
        df_.loc[:, 'AcqClean'] = df_['MMRAcquisitonRetailCleanPrice'] - df_['MMRAcquisitionAuctionCleanPrice']
        df_.loc[:, 'AcqRetail'] = df_['MMRAcquisitonRetailCleanPrice'] - df_['MMRAcquisitionRetailAveragePrice'] 
        df_.loc[:, 'AcqAuc'] = df_['MMRAcquisitionAuctionCleanPrice'] - df_['MMRAcquisitionAuctionAveragePrice']
        
        cols = ['MMRAcquisitionAuctionAveragePrice', 'MMRAcquisitionAuctionCleanPrice',
                'MMRAcquisitionRetailAveragePrice', 'MMRAcquisitonRetailCleanPrice',
                'MMRCurrentAuctionAveragePrice', 'MMRCurrentAuctionCleanPrice',
                'MMRCurrentRetailAveragePrice', 'MMRCurrentRetailCleanPrice', ]
        
        self.cols_num = list(set(self.cols_num) - set(cols)) + ['RetailClean', 'AcqClean', 'AcqRetail', 'AcqAuc']
        df_ = df_.drop(columns=cols)
        return df_

        
    def merge_cat_cols(self, df, cols):
        '''All columns in cols merged in one column.
        All entries in columns transformed as:
        new_str = str1*str2*..
        '''
        df_ = df.copy()
        #print(df_.columns)
        new_col = '*'.join(cols)
        #print(new_col)
        self.cols_cat.append(new_col)
        df_.loc[:, new_col] = ''
        for col in cols:
            df_.loc[:, new_col] += df_[col].astype(str) + '*'
        # print(df_.columns)    
        df_.drop(columns=list(cols), inplace=True)
        self.cols_cat = list(set(self.cols_cat) - set(cols))
        return df_ 
    

    def split_cat_cols_names(self, X, y):
        df = X[self.cols_cat].copy()
        df.insert(0, y.columns[0], y.copy())
        df = AvPdataFrame(df)
        df_imp = df.cols_importance(cols=self.cols_cat, target=y.columns[0])
        ### 'No_categories' - number of categories calculated by cols_importance()
        df_catS = df_imp[df_imp['No_categories'] <= self.max_cat] 
        cols_catS = list(df_catS.index)
        df_catL = df_imp[df_imp['No_categories'] > self.max_cat]
        cols_catL = list(df_catL.index)
        return cols_catS, cols_catL
    
    
    def imput_transform(self, X):
        X_= X.copy()
        for transformer in self.imputer_list:
            X_ = transformer.transform(X_)
        return X_
   
    
    def fit(self, X=None, y=None):
        X_ = X.copy()
        y_ = y.copy()
        
        ############### impute #########################
        if self.impute:
            impute_VehBcost = LinearImputer(
                                            col_x='MMRAcquisitionAuctionAveragePrice', 
                                            col_y='VehBCost', 
                                            random_state=42, 
                                            verbose=True,
                                            threshold=500 )
            impute_VehBcost.fit(X_)
            self.imputer_list.append(impute_VehBcost)
            # print('echo impute_VehBcost.fit(X_)')
            
            cols_price = ['MMRAcquisitionAuctionAveragePrice', 'MMRAcquisitionAuctionCleanPrice',
                        'MMRAcquisitionRetailAveragePrice', 'MMRAcquisitonRetailCleanPrice',
                        'MMRCurrentAuctionAveragePrice', 'MMRCurrentAuctionCleanPrice',
                        'MMRCurrentRetailAveragePrice', 'MMRCurrentRetailCleanPrice',] 
            for col in cols_price:
                impute_price = LinearImputer(
                                            col_x='VehBCost', 
                                            col_y=col, 
                                            random_state=42, 
                                            verbose=False,
                                            threshold=100 )
                impute_price.fit(X_)
                self.imputer_list.append(impute_VehBcost)
                
            X_ = self.imput_transform(X_)
        
        ############### price difference #########################
        if self.diff:
            X_ = self.calc_price_diff(X_)                  
        # print(X_.head(3))    
        
        ############### numerical to categorical #########################
        ### transformation make sence only if the columns wil be merge after
        ### DecisionTree make categorical split better   
       # print('to categorical initiating')
       # print(self.num_to_cat_dict.keys())
       # print(X_.columns)
        self.to_category = NumToCatTransformer(self.num_to_cat_dict)
        X_ = self.to_category.transform(X_)
        
        self.cols_num = list(set(self.cols_num) - set(self.to_category.original_names))
        self.cols_cat = self.cols_cat + self.to_category.new_cols_names 
       # print('to categorical columns renamed')
       # print(self.cols_cat)
       # print(X_.columns)
        
       # print(self.cols_MCA_list)
       # print('fit df.columns', df.columns)
       
       ############### merge columns #########################
        for cols in self.cols_MCA_list:
            print('echo transform MCA list itteration', cols)
            #print(X_.columns)
            X_ = self.merge_cat_cols(X_, cols)
        #print(X_.columns)
        # print(self.cols_cat)

        ############### reduce categories #########################
        if self.target_encoder:
          #  print('split names')
            self.cols_catS, self.cols_catL = self.split_cat_cols_names(X_, y_)
          #  print('split names passed')
          #  print(self.cols_catS, self.cols_catL)
          #  print(self.cols_cat)
            self.categories_reduser = CatColReducer(cols_fit=self.cols_catL)       
            X_ = self.categories_reduser.fit_transform(X_, y_)
            ### indicate changed columns names
            i = 0
            for i in range(len(self.cols_catL)):
                self.cols_catL[i] += '_avp'
            #print(self.cols_catL)
            self.cols_cat = self.cols_catS + self.cols_catL
          #  print(self.cols_cat)
        #print(df.head())
        
        ############### final pipe #########################
        self.ohe = OneHotEncoder(handle_unknown='infrequent_if_exist',
                                 max_categories=self.max_cat,
                               #  drop='first' # , None,
                                 #sparse_output=False,
                                 )
        self.scaler = StandardScaler()
        
        self.cols_tranformer = ColumnTransformer(
            transformers=[
                ('ohe', self.ohe, self.cols_cat),
                ('scaler', self.scaler, self.cols_num)
            ],
            remainder='drop',
            # sparse_threshold=0.3,
            n_jobs=-1,
        ) # .set_output(transform="pandas") # do not support sparse output

        self.cols_tranformer.fit(X_)
        
        ##### price difference function change cols_num, return to original
        if self.diff:
            self.cols_num = self.cols_num_
       
        return self
    
    
    def transform(self, X=None, y=None):
        print('transformator echo')
        X_ = X.copy()
        
        ############### impute #########################
        if self.impute:
            X_ = self.imput_transform(X_)
            
        ############### price difference #########################
        if self.diff:
            X_ = self.calc_price_diff(X_)  
            
        ############### numerical to categorical #########################
        ### transformation make sence only if the columns wil be merge after 
        X_ = self.to_category.transform(X_)
        
        ############### merge columns #########################
        for cols in self.cols_MCA_list:
            print('echo transform MCA list itteration', cols)
            X_ = self.merge_cat_cols(X_, cols)
            
        ############### reduce categories #########################
        if self.target_encoder:
            X_ = self.categories_reduser.transform(X_)
            # print('echo transform after categories_reduser.transform(X_)')
        
        ############### final pipe #########################
        X_ = self.cols_tranformer.transform(X_)
        # print('echo transform after cols_tranformer.transform(X_) new')
        # X_.set_output(transform="pandas")
        # print('after set_output(transform="pandas")')
        
        return X_ 
    
    
    def __str__(self):
        return 'I am IsBadBuyTransformer'
    

class TreeRegresor(TransformerMixin, BaseEstimator): # 
    def __init__(self, 
                 col_target:'str',
                 cols_reg:'list',
                 cols_impurity_detection:'list',
                 nan_condition = ('==', 0),
                 percent = 1, # percent of outliers in data
                 verbose = 0,
                 ) -> None:
        self.col_target = col_target 
        self.cols_reg = cols_reg
        self.cols_impurity_detection = cols_impurity_detection
        self.tree_regressor = None
        self.nan_condition = nan_condition
        self.ops = {'<': operator.lt,
               '>': operator.gt,
               '==': operator.eq,
               '>=': operator.ge,
               '<=': operator.le
               }
        self.verbose = verbose
        self.percent = percent

        
    def remove_rows(self, df):

        df_ = df.copy()
       # mask = df[self.col_target] == rows_value # rows_value=0
        mask = self.ops[self.nan_condition[0]](df[self.col_target],  
                                          self.nan_condition[1])
        df_.drop(df_[mask].index, inplace=True)
        print(f'number of rows with value {self.nan_condition[0]} {self.nan_condition[1]} removed = {mask.sum()}')
        mask = df_[self.col_target].isna()
        df_.drop(df_[mask].index, inplace=True)
        print(f'number of rows with value isna() removed = {mask.sum()}')
        return df_
    
           
    def remove_outliers(self, X: pd.DataFrame, percent = 1):
        percent /= 100
        # reduce df to colums of interest 
        df = X.loc[:, self.cols_impurity_detection].copy()
        
        prepare_cols_imp = PrepareColsBasePipe()
        prepare_cols_imp.define_cols(df) 
        isoforest_pipe = Pipeline(steps=[
        ('preprocessing', prepare_cols_imp.make_pipe()),
        ('model', IsolationForest(n_jobs=-1, random_state=42, contamination=percent)
        )])
        isoforest_pipe.fit(df)
        isoresult = isoforest_pipe.predict(df)
        
        df.loc[:, 'Outliers'] = isoresult
        mask = df['Outliers'] == -1
        print('removed outliers rows:', mask.sum())
        # to return full df 
        df = X.copy()
        df.drop(df[mask].index, inplace=True)
        return df
   
    
    def dbscan_price_dif(self, X):
        df = X[self.cols_impurity_detection].copy()
        df = self.remove_rows(df)
        
        prepare_cols = PrepareColsBasePipe()
        prepare_cols.define_cols(df)
        
        db_pipe = Pipeline(steps=[
        ('preprocessing', prepare_cols.make_pipe()),
        ('db', DBSCAN(eps=0.34)
        )])
        
        db_pipe.fit(df)
        labels = db_pipe['db'].labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        print("Estimated number of clusters: %d" % n_clusters_)
        print("Estimated number of noise points: %d" % n_noise_)
        
        cols = df.columns
        sns.scatterplot(data=df, x=cols[1], y=cols[0], hue=db_pipe['db'].labels_, s=3)
        plt.show()
    
    
    def fit(self, X: pd.DataFrame, y=None ):
        ## for plotting outliers we need that cols_impurity_detection
        ##  will be in df, so we could not use df = X.loc[:, self.cols_reg].copy()
        
        df = X.copy()
        df = self.remove_rows(df)
        df = self.remove_outliers(df, percent=self.percent)
        if self.verbose == 2:
            ## plot Outliers detected 
            fig, ax = plt.subplots(figsize=(5,4))
            ax.set(title="Outliers" )
            sns.scatterplot(data=X, x=self.cols_impurity_detection[1], 
                    y=self.cols_impurity_detection[0],
                    ax=ax,
                    s=2
                    )
            sns.scatterplot(data=df, x=self.cols_impurity_detection[1], 
                    y=self.cols_impurity_detection[0],
                    ax=ax,
                    s=1
                    )
            plt.show()
        
        ## define target after outliers removement but before df truncation
        target = df[self.col_target].copy()
        df = df.loc[:, self.cols_reg]   
        
      #  print('TreeRegressor echo: before BasePipe')
        prepare_cols_reg = PrepareColsBasePipe()
        prepare_cols_reg.define_cols(df)
       # print('TreeRegressor echo: BasePipe')
        self.tree_regressor = Pipeline(steps=[
        ('preprocessing', prepare_cols_reg.make_pipe()),
        ('model', DecisionTreeRegressor(max_depth=100, 
                                        min_samples_leaf=3, 
                                        random_state=42,)
        )])
       # print('TreeRegressor echo: full Pipe inicialized')
        print(target.isna().sum())
        self.tree_regressor.fit(df, target)
       # print('TreeRegressor echo: full Pipe fit')

        if self.verbose > 0:
            pred_reg = self.tree_regressor.predict(df.loc[:, self.cols_reg])
            fig, ax = plt.subplots(figsize=(5,4))
            ax.set(title="Quality of prediction",
                         xlabel="target",
                         ylabel="predicted",)
        
            ax.scatter(x=target , y=pred_reg, s=1)
            ax.plot(target, target, color='r')
            plt.show() 
            print(target.shape)
            err = np.sqrt(sum(((target-pred_reg)**2))/target.shape[0])
            print('fit error', err)

        return self
    
    def transform(self, X, y=None):
        df = X.copy()
        
        mask = (df[self.col_target].isna() | 
                self.ops[self.nan_condition[0]](df[self.col_target],  
                                               self.nan_condition[1]) )
       # print(mask.sum())
        pred_imput = self.tree_regressor.predict(df.loc[mask, self.cols_reg])
        df.loc[mask, self.col_target] = pred_imput
        
     #   print(mask.sum())
     #   print(pred_imput.shape)
     #   print('TreeRegressor transform function echo')
        return df
    

class StandardOHETransformer(TransformerMixin, BaseEstimator):
    '''standard scaler and one-hot-encoder with auto recognision 
    of categorical and numerical columns
    auto fill NaN as 'empty' or '0'
    '''
    def __init__(self, max_cat=25) -> None:
        self.max_cat = max_cat
        self.cols_cut = None
        self.cols_num = None
        self.cols_binary = []
        
        self.pipe_cat = Pipeline(steps=[
            ('impute_empty', SimpleImputer(strategy='constant', fill_value='empty')),
            ('ohe', OneHotEncoder(handle_unknown='infrequent_if_exist', max_categories=self.max_cat)) 
            ])
        
        self.pipe_num = Pipeline(steps=[
                ('impute_null', SimpleImputer(strategy='constant', fill_value=0)),
                ("scaler", StandardScaler()) # not neseccary for tree estimator
            ])
        
        self.pipe_binary = Pipeline(steps=[
            ('impute_null', SimpleImputer(strategy='constant', fill_value=0)),
        ])
        
        self.column_transformer = None

        
    def define_cols(self, X:pd.DataFrame):
        self.cols_cut = X.select_dtypes('object').columns
        # print(f'{self.cols_cut=}')
        ## list() is used to anable list.remove('val) command  
        ## in determination of binary columns
        self.cols_num = list(X.select_dtypes('number').columns)   
        col_dt = X.select_dtypes('datetime').columns
        if not col_dt.empty:
            print('StandardOHETransformer columns will be dropped:')
            print('date-time columns:', col_dt) 
            
        for col in self.cols_num:
            if list(X[col].unique()) == [0, 1]:
                # print(col)
                self.cols_binary.append(col)
                self.cols_num.remove(col)
        
        
    def fit(self, X:pd.DataFrame, y=None):
        df = X.copy()
        self.define_cols(df)
        self.column_transformer = ColumnTransformer(
                transformers=[
                    ('transform_categorical', self.pipe_cat, self.cols_cut),
                    ('transform_numerical', self.pipe_num, self.cols_num),
                    ('keep bynary cols', self.pipe_binary, self.cols_binary)
                ],
                remainder='drop',  # 'passthrough',
            )
        self.column_transformer.fit(df)
       # print('StandardOHETransformer fit finished')

        return self
    
    def transform(self, X:pd.DataFrame, y=None):
        df = X.copy()
        df = self.column_transformer.transform(df)
        return df
    
    def get_feature_names_out(self):
        return self.column_transformer.get_feature_names_out()