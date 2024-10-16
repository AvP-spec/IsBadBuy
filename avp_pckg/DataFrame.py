import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class AvPdataFrame(pd.DataFrame):
    
    ###############################################################
    ### methods for categorical features 
    #################################################################    
    def calc_frequency(self, col:str, target:str):
        '''Calculate frequency of the target for the column
        Args:
           col: str, column name
           target: str, name of the target column with values 0 or 1
        Returns:
           pd.DataFrame with columns: 
           index        - categories of col 
           {target}_sum - target summ fo category 
           count        - number of entries 
           {target}_%   - frequency in % {target}_sum / count * 100 
        '''
        
        if col not in self.columns:
            print(f'Error: no {col} column in the data frame')
            return None
        if target not in self.columns:
            print('Error: no target column in the data frame')
            return None       
                
        df_ = pd.DataFrame({f'{target}_sum': self.groupby(col)[target].sum().sort_index(),
                            'count': self.groupby(col)[target].count().sort_index()}).copy()
        
        df_.loc[:, f'{target}_%'] = df_[f'{target}_sum'] / df_['count'] * 100
        # print(df_.sort_values('count', ascending=False))  
              
        return df_.sort_values('count', ascending=False)
    
    
    counts_list = [10_000, 1_000, 100, 10]
    def cols_importance(self, cols: list, target: str, counts_list=counts_list, verbose=0):
        '''Calculate importance for each column in <cols> 
        as difference between max - min frequency of the <target>. Calculated in %
        for categories with count > number for each number in <counts_list>
        
        Args:
            cols: list of column names (str)
            target: str, name of the target column with values 0 or 1 
            counts_list: list of int 
            
        Return:
            pd.DataFrame with columns
            index         - column names
            No_categories - number_of_cutegories in the column
            delta<number> - imortance for each number in <counts_list>
        '''
        res_dickt = {}
        for col in cols:
            col_dickt = {}
            # print(col.center(50, '-'), '\n') ### for debaging 
            df_ = self.calc_frequency(col=col, target=target)
            number_of_cutegories = df_.shape[0]
            col_dickt["No_categories"] = number_of_cutegories
            
            for num in counts_list:
                ### select data with counts more then <num>
                df_num = df_[df_['count'] > num]
                ### calculate difference of max and min frequency of the target
                freq_diff = df_num[f'{target}_%'].max() - df_num[f'{target}_%'].min()
                ### save results for each <num>
                col_dickt[f"delta{num}"] = freq_diff
                if verbose:
                    print(f" delta > {num} = ", freq_diff)
                    print('number_of_cutegories =', number_of_cutegories)
                
            
            ### save results for each <col> 
            res_dickt[col] = col_dickt   
        
        # print(res_dickt) ### for debagging
        res_df = pd.DataFrame.from_dict(res_dickt, orient='index') # 'columns'
        return res_df
    
    ###############################################################
    ### methods for numerical features 
    ################################################################# 
    def calc_frequency_num(self, col:str, target:str, bin=100, ascending=True):
        ''' sort 'df' on 'col' values, and calculate frequncy of target for each bin''' 
        df_tmp = self[[target, col]].copy().sort_values(col, ignore_index=True, ascending=ascending)
        df_tmp.reset_index(inplace=True)
        # print('bin= ', bin)
        i = 0
        i_max = df_tmp.shape[0] - bin
        ii = 0
        dct_freq = {}
        for i in range(i_max):
            target_freq = df_tmp.loc[i:i+(bin - 1), target].sum() / bin
            av_col = df_tmp.loc[i:i+bin, col].mean()
            dct_freq.update({ii:[av_col, target_freq]})
            # print(aav_col, target_freq)
            i += bin
            ii += 1
            
        df_freq = pd.DataFrame.from_dict(dct_freq, orient='index', columns=[col+'_mean', target + '_freq'] )
        return df_freq
    
    
    def plot_frequency_num(df,
                        x_vlines=[3500, 7000, 12000], 
                        lines_color='r',
                        zoomX1=[800, 6000], 
                        zoomX2=[5000, 22000], 
                        xHist=[800, 15000],
                        hist_bins=100                       
                        ):
        
        x_name = df.columns[0]
        y_name = df.columns[1]
        ymax = df[y_name].max()
        
        fig, axs = plt.subplots(ncols=4, figsize=(20, 5))
        fig.suptitle(x_name, fontsize=16)
        
        i=0
        for i in range(3):
           #  print(i)
            sns.scatterplot(data=df,
                x=x_name,
                y=y_name,
                ax=axs[i],
                )
            axs[i].vlines(x_vlines, ymin=0, ymax=ymax, colors=lines_color)
                   
        axs[1].set_xlim(zoomX1)
        axs[2].set_xlim(zoomX2)
       
        df[x_name].plot(kind='hist', bins=hist_bins , ax=axs[3])
        axs[3].set_xlabel(x_name)
        axs[3].set_xlim(xHist) # 
