import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from avp_pckg.DataFrame import AvPdataFrame 
from avp_pckg.TransformerCols import CatColReducer
from avp_pckg.TransformerCols import NumToCatTransformer
from avp_pckg.TransformerCols import IsBadBuyTransformer

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import TargetEncoder
from sklearn.compose import ColumnTransformer

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import validation_curve, cross_val_score
from sklearn.model_selection import cross_validate # for multiple score!!

from avp_pckg.TransformerCols import StandardOHETransformer


from sklearn.metrics import classification_report, accuracy_score, recall_score, f1_score, precision_score

class PrepareColsBase:
    def __init__(self, cols_cat:list, cols_num:list, 
                 max_cat:int, cols_binary = []) -> None:
        self.cols_cut = cols_cat
        self.cols_num = cols_num
        self.max_cat = max_cat
        self.cols_binary = cols_binary
        print('class PrepareColsBase echo')
        
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
    
    
class PrepareColsTEncoder:
    def __init__(self, 
                 cols_catS:list, # categorical columns with few categories < max_cat (25)
                 cols_catL:list, # categorical columns with categories > max_cat (25)
                 cols_num:list,  # numerical columns 
                 max_cat:int) -> None:
        self.cols_cutS = cols_catS
        self.cols_cutL = cols_catL
        self.cols_num = cols_num
        self.max_cat = max_cat
       # print('class repareColsTEncoder echo')
        
    def make_pipe(self):
      #  print('make_pipe echo, class PrepareColsTEncoder')
        
        pipe_catS = Pipeline(steps=[
            ('impute_empty', SimpleImputer(strategy='constant', fill_value='empty')),
          #  ('target_encoder', TargetEncoder(random_state=42, shuffle=False) ),
            ('ohe', OneHotEncoder(handle_unknown='infrequent_if_exist', max_categories=self.max_cat)) 
            ])
        
        pipe_catL = Pipeline(steps=[
            ('impute_empty', SimpleImputer(strategy='constant', fill_value='empty')),
            ('target_encoder', TargetEncoder(shuffle=False) ),
          #  ('ohe', OneHotEncoder(handle_unknown='infrequent_if_exist', max_categories=self.max_cat)) 
            ])

        pipe_num = Pipeline(steps=[
                ('impute_null', SimpleImputer(strategy='constant', fill_value=0)),
              #  ('target_encoder', TargetEncoder(random_state=42) ),
                ("scaler", StandardScaler()) # not neseccary for tree estimator
            ])

        prepare_cols = ColumnTransformer(
                transformers=[
                    ('transform_categorical_S', pipe_catS, self.cols_cutS),
                    ('transform_categorical_L', pipe_catL, self.cols_cutL),
                    ('transform_numerical', pipe_num, self.cols_num)
                ],
                remainder='drop',
            )
        return prepare_cols  
    
    
class PrepareColsAvP:
    def __init__(self, 
               #  X,
              #   y,
                 cols_catS:list, # categorical columns with few categories < 25
                 cols_catL:list, # categorical columns with categories > 25
                 cols_num:list,  # numerical columns 
                 num_to_cat_dict= None,
                 max_cat=25) -> None:
        
      #  self.X = X.copy()
      #  self.y = y.copy()
        self.cols_catS = cols_catS
        self.cols_catL = cols_catL
        self.cols_num = cols_num
        self.max_cat = max_cat
        self.num_to_cat_dict = num_to_cat_dict
                
     #   print('class repareColsAvP echo')
        
    def make_pipe(self):
       # print('make_pipe echo, PrepareColsAvP')
        # num_tranformer = NumToCatTransformer(self.num_to_cat_dict)
        # for col in self.num_to_cat_dict.keys():
        #     self.
        pipe_catS = Pipeline(steps=[
          #  ('impute_empty', SimpleImputer(strategy='constant', fill_value='empty')),
          #  ('target_encoder', TargetEncoder(random_state=42, shuffle=False) ),
            ('ohe', OneHotEncoder(handle_unknown='infrequent_if_exist', max_categories=self.max_cat)) 
            ])
        
        pipe_catL = Pipeline(steps=[
            ('target_encoder_AvP', CatColReducer(cols_fit=self.cols_catL) ),
            ('impute_empty', SimpleImputer(strategy='constant', fill_value='empty')),
          #  ('target_encoder_AvP', CatColReducer(cols_fit=self.cols_catL) ),
            ('ohe', OneHotEncoder(handle_unknown='infrequent_if_exist', max_categories=self.max_cat)) 
            ])

        pipe_num = Pipeline(steps=[
                ('impute_null', SimpleImputer(strategy='constant', fill_value=0)),
              #  ('target_encoder', TargetEncoder(random_state=42) ),
                ("scaler", StandardScaler()) # not neseccary for tree estimator
            ])

        prepare_cols = ColumnTransformer(
                transformers=[
                 #   ('trans_num_to_cat', NumToCatTransformer(self.cols_dict)),
                    ('transform_categorical_S', pipe_catS, self.cols_catS),
                    ('transform_categorical_L', pipe_catL, self.cols_catL),
                    ('transform_numerical', pipe_num, self.cols_num)
                ],
                remainder='drop',
            )
        return prepare_cols  
    
    
    
def split_cat_cols_names(X, y, cols_cat, max_cat=25):
    ''' seporate names of categorical collumns which have
    number of categories more then max_cat: 
    catL => large amount of categories
    from those which have number of categories less then max_cat
    catS => small amount of categories
    '''
  #  print('split_cat_cols_names')
  #  print(cols_cat)
    df = X[cols_cat].copy()
    y_ = pd.DataFrame(y.copy())
  #  print('pass')
    df.insert(0, y_.columns[0], y_)
    df = AvPdataFrame(df)
    df_imp = df.cols_importance(cols=cols_cat, target=y_.columns[0])
    # 'No_categories' - number of categories calculated by cols_importance()
    df_catS = df_imp[df_imp['No_categories'] <= max_cat] 
    cols_catS = list(df_catS.index)
  #  print('cols_catS:', cols_catS)
    df_catL = df_imp[df_imp['No_categories'] > max_cat]
    cols_catL = list(df_catL.index)
   # print('cols_catL:', cols_catL)
    return cols_catS, cols_catL

    
       
def cross_validate_pipe(X,          # features
                        y,          # target
                        cols_cat,   # list of categorical columns names
                        cols_num,   # list of numerical columns names
                        num_to_cat_dict={}, # columns names and parameters for transformation of cols_num  to cols_cat
                        param_name='', # name of parameter for cross-validation curve  
                        param_range=[],# list of the parameter values
                        cv=5,        # number of cross validation spits
                        max_cat=50,    # int, maximum categories for one-hot-encoder and target encoder
                        estimator_name='tree', # 'forest', 'logistic', 
                        pipe_name = 'base', # 'TargetEncoder', 'TE_AvP'
                        n_jobs=-1,
                        kwards_fixed = {
                            'class_weight':'balanced', 
                            'random_state':42
                            }, # key words for the model
                        cols_binary = [],
                        ):
    '''Calculate Cross-Validation curves for 
    f1, precision and recall metrics for selection of a model parameters. 
    
    Parameters:
    -----------
    esctimator_name:
        tree = DecisionTreeClassifier
        forest = RandomForestClassifier
        logistic = LogisticRegression classifier
    
    pipe_name: name of a pipline for data preprocessing 
    
    Returns:
    --------
    cross validation score dictionry 
    
    '''
    
    ############################################
    ## make copy of values which can be modefied 
    cols_num_ = cols_num.copy()
    cols_cat_ = cols_cat.copy()
    X_ = X.copy()
    y_ = y.copy()
    
    ## transfer numreical cols to categorical
    if num_to_cat_dict: 
        for col in num_to_cat_dict.keys():
          #  print(col + '_avp')
            cols_num_.remove(col)
            cols_cat_.append(col + '_avp')
        num_transformer = NumToCatTransformer(num_to_cat_dict)
        X_ = num_transformer.transform(X_)
       # print(X_.head(2))
        
    # print(cols_cat_)
    # print(cols_num_)
    
    ## seporate names of categorical collumns which have
    ## catL => large amount of categories
    ## catS => small amount of categories
    cols_catS, cols_catL = split_cat_cols_names(X_,
                                                y_,
                                                cols_cat=cols_cat_,
                                                max_cat=max_cat)

    estimator_dickt = {'tree':DecisionTreeClassifier,
                       'forest': RandomForestClassifier,
                       'logistic': LogisticRegression,
                       }
    
    pipe_dickt = {'base': PrepareColsBase(cols_cat=cols_cat, 
                                   cols_num=cols_num_,
                                   max_cat=max_cat,
                                   cols_binary=cols_binary).make_pipe(),
                  
                  'TargetEncoder': PrepareColsTEncoder(cols_catS=cols_catS,
                                     cols_catL=cols_catL, 
                                     cols_num=cols_num_, 
                                     max_cat=max_cat).make_pipe(),
                  
                  'TE_AvP': PrepareColsAvP(
                                     cols_catS=cols_catS,
                                     cols_catL=cols_catL, 
                                     num_to_cat_dict=num_to_cat_dict,
                                     cols_num=cols_num_, 
                                     max_cat=max_cat).make_pipe(),
                  }

    prepare_cols = pipe_dickt[pipe_name]
    
    ### calculate cross validation score 
    score_dickt = {}
    for value in param_range:
        print('cross_validate_pipe echo: parameter value = ', value)
        kwargs_par = {param_name: value}
     #   print('call pipe_tree')
        pipe_tree = Pipeline(steps=[
        ('preprocessing', prepare_cols),
        ('model', estimator_dickt[estimator_name](**kwards_fixed, **kwargs_par))
     #   ('model', estimator_dickt[estimator_name](class_weight='balanced', random_state=42, **kwargs_par))
     #   ('model', DecisionTreeClassifier(max_depth=depth, class_weight='balanced', random_state=42))
        ])  
         
        score = cross_validate(estimator=pipe_tree,  # Corrected estimator
                              X=X_,
                              y=y_,
                              scoring=['f1', 'precision', 'recall'],
                              return_train_score=True,
                              cv=cv,
                              n_jobs=n_jobs)

        score_dickt[value] =  score
        
    return score_dickt

 
def plot_single_score(df, ax=None, color='k', xlabel='max_depth', label='f1', title='test', xlog=False):
    
    sns.lineplot(x=df.index, y=df.mean(axis=1), legend='auto', linewidth=5, color=color, ax=ax, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('score')
    ax.set_title(title.upper()+ ' score ')
    for i in range(df.shape[1]):
        if xlog:
            ax.set(xscale="log")
        sns.lineplot(df, x=df.index, y=df[i], color=color, ax=ax)        
    ax.legend()
    
    

def plot_scores(score_dckt:dict, param_name='max_depth', xlog=False):
    
    df_score = pd.DataFrame.from_dict(score_dckt, orient='index')
    
    df_test_f1 = pd.DataFrame(df_score['test_f1'].tolist(), index=df_score.index)
    df_train_f1 = pd.DataFrame(df_score['train_f1'].tolist(), index=df_score.index)
    
    df_test_precision = pd.DataFrame(df_score['test_precision'].tolist(), index=df_score.index)
    df_train_precision = pd.DataFrame(df_score['train_precision'].tolist(), index=df_score.index)
    
    df_test_recall = pd.DataFrame(df_score['test_recall'].tolist(), index=df_score.index)
    df_train_recall = pd.DataFrame(df_score['train_recall'].tolist(), index=df_score.index)

    fig, axs = plt.subplots(ncols=2, figsize=(10, 5))
    
    plot_single_score(df_test_f1, axs[0], color='k', xlabel=param_name, label='f1', title='test', xlog=xlog)
    plot_single_score(df_test_precision, axs[0], color='g', xlabel=param_name, label='precision', title='test', xlog=xlog)
    plot_single_score(df_test_recall, axs[0], color='b',  xlabel=param_name, label='recall', title='test', xlog=xlog)

    plot_single_score(df_train_f1, axs[1], color='k', xlabel=param_name, label='f1', title='train', xlog=xlog)
    plot_single_score(df_train_precision, axs[1], color='g', xlabel=param_name, label='precision', title='train', xlog=xlog)
    plot_single_score(df_train_recall, axs[1], color='b',  xlabel=param_name, label='recall', title='train', xlog=xlog)


def print_scores(score_dict:dict):
    df_score = pd.DataFrame.from_dict(score_dict, orient='index')
    print('| {:11} | {:7} | {:6} | {:10} |'.format('par. value', 
                                                  'f1_mean', 
                                                  'f1_std', 
                                                  'precission'))
    for i in df_score.index:
        index = i
        f1_mean = np.round(np.mean(df_score.test_f1[i]), decimals=3)
        f1_std = np.round(np.std(df_score.test_f1[i]), decimals=4)
        precis_mean = np.round(np.mean(df_score.test_precision[i]), decimals=3)
        
        print('| {:11} | {:7} | {:6} | {:10} |'.format(index, 
                                                      f1_mean, 
                                                      f1_std, 
                                                      precis_mean))
      
    

def wheels_type_split(X, y, noWheel_sign=None):
    '''function for IsBadBuy dataset only'''
    df_X = X.copy() 
    df_y = y.copy()
    
    if noWheel_sign:
        print('noWheel_sign_c:', noWheel_sign, type(noWheel_sign) )
        X_noWheels = df_X[df_X['WheelType'] == noWheel_sign]
        
        y_noWheels = df_y.loc[X_noWheels.index]
        X_wheels = df_X[df_X['WheelType'] != noWheel_sign]
        y_wheels = df_y.loc[X_wheels.index]
        
    else:
        X_noWheels = df_X[df_X['WheelType'].isna()]
        y_noWheels = df_y.loc[X_noWheels.index]
            
        X_wheels = df_X[df_X['WheelType'].notna()]
        y_wheels = df_y.loc[X_wheels.index]
        
    return X_wheels, y_wheels,  X_noWheels, y_noWheels



def cross_validate_IsBadBuy(X,          # features
                        y,          # target
                        cols_cat,   # list of categorical columns names
                        cols_num,   # list of numerical columns names
                        num_to_cat_dict={}, # columns names and parameters for transformation of cols_num  to cols_cat
                        cols_MCA_list={}, 
                        param_name='', # name of parameter for cross-validation curve  
                        param_range=[],# list of the parameter values
                        cv=5,        # number of cross validation spits
                        max_cat=15,    # int, maximum categories for one-hot-encoder and target encoder
                        estimator_name='tree',
                        impute=True,
                        diff=True,
                        target_encoder=True,
                        n_jobs=-1,
                        kwards_fixed = {'class_weight':'balanced', 'random_state':42}
                        ):
    
    X_ = X.copy()
    y_ = y.copy()
    
    estimator_dickt = {'tree':DecisionTreeClassifier,
                    'forest': RandomForestClassifier,
                    'logistic': LogisticRegression,
                       }
    
    prepare_cols = IsBadBuyTransformer(
                                        cols_cat=cols_cat,
                                        cols_num=cols_num,
                                        max_cat=max_cat,
                                        num_to_cat_dict=num_to_cat_dict,
                                        cols_MCA_list=cols_MCA_list,  
                                        impute=impute,
                                        diff=diff,
                                        target_encoder=target_encoder                  
                                        )
    
    score_dickt = {}
    for value in param_range:
        print('cross_validate echo: parameter value = ', value)
        kwargs_par = {param_name: value}
     #   print('call pipe_tree')
        pipe_IsBadBuy = Pipeline(steps=[
        ('isbadbuy_transformer', prepare_cols),
        ('model', estimator_dickt[estimator_name](**kwards_fixed, **kwargs_par))
     #   ('model', estimator_dickt[estimator_name](class_weight='balanced', random_state=42, **kwargs_par))
     #   ('model', DecisionTreeClassifier(max_depth=depth, class_weight='balanced', random_state=42))
        ])  
         
        score = cross_validate(estimator=pipe_IsBadBuy,  # Corrected estimator
                              X=X_,
                              y=y_,
                              scoring=['f1', 'precision', 'recall'],
                              return_train_score=True,
                              cv=cv,
                              n_jobs=n_jobs)

        score_dickt[value] =  score
        
    return score_dickt


def cross_validate_transformer(X,          # features
                        y,          # target
                        param_name='', # name of parameter for cross-validation curve  
                        param_range=[],# list of the parameter values
                        cv=5,        # number of cross validation spits
                        max_cat=25,    # int, maximum categories for one-hot-encoder and target encoder
                        estimator_name='tree',
                      #  diff=True,
                        n_jobs=-1,
                        kwards_fixed = {'class_weight':'balanced', 'random_state':42}
                        ):
    
    X_ = X.copy()
    y_ = y.copy()
    
    estimator_dickt = {'tree':DecisionTreeClassifier,
                    'forest': RandomForestClassifier,
                    'logistic': LogisticRegression,
                    'svc' : SVC,
                    'kn' : KNeighborsClassifier,
                       }
    
    prepare_cols = StandardOHETransformer(
                                        max_cat=max_cat,          
                                        )
    
    score_dickt = {}
    for value in param_range:
        print('cross_validate echo: parameter value = ', value)
        kwargs_par = {param_name: value}
     #   print('call pipe_tree')
        pipe_IsBadBuy = Pipeline(steps=[
        ('st-ohe_transformer', prepare_cols),
        ('model', estimator_dickt[estimator_name](**kwards_fixed, **kwargs_par))
        ])  
         
        score = cross_validate(estimator=pipe_IsBadBuy,  # Corrected estimator
                              X=X_,
                              y=y_,
                              scoring=['f1', 'precision', 'recall'],
                              return_train_score=True,
                              cv=cv,
                              n_jobs=n_jobs)

        score_dickt[value] =  score
        
    return score_dickt