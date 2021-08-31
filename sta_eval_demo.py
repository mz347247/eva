# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 13:22:32 2020

@author: work14
"""

import os
import pandas as pd
pd.set_option('display.max_columns',200)
pd.options.mode.chained_assignment = None
import matplotlib.pyplot as plt
import numpy as np
perc = [.01, .05, .1, .25, .5, .75, .9, .95, .99]
import yaml


from DB_ANDY  import DB
db_config = {
    'user': 'user',
    'password': 'password',
    'host': '192.168.10.178',
    'port': '27017',
    'db': 'com_md_eq_cn'
} 
db = DB(db_config['host'], db_config['db'], db_config['user'], db_config['password'])   

def render_mpl_table(data, col_width=2.6, row_height=0.8, font_size=15,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0, seperate_col = 6,
                     ax=None, **kwargs):
    import six
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, cellLoc='center', **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in  six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] in [0,seperate_col] or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors)])
    return ax
###############################################################################

class sta_alpha_eval():
    def __init__(self, sta_input):
        f = open(sta_input, 'r', encoding='utf-8')
        cfg = f.read()
        dict_yaml = yaml.full_load(cfg)
        f.close()

        self.bench = dict_yaml['bench']
        self.startDate = str(dict_yaml['startDate'])
        self.endDate = str(dict_yaml['endDate'])
        self.baseAlpha = dict_yaml['baseAlpha']
        self.evalAlpha = dict_yaml['evalAlpha']
        self.eval_alpha = (self.baseAlpha + self.evalAlpha)[-1]
        self.actualRetLs = dict_yaml['actualRetLs']
        self.targetRet = dict_yaml['targetRet']
        self.targetCut = dict_yaml['targetCut']
        self.dataLoadPath = ''
        for ele in dict_yaml['dataLoadPath']:
            self.dataLoadPath = os.path.join(self.dataLoadPath, ele)
        self.resultSavePath = ''
        for ele in dict_yaml['resultSavePath']:
            self.resultSavePath = os.path.join(self.resultSavePath, ele)  
            
        self.MDPath = os.path.join(self.dataLoadPath,'simMarketData',self.bench)
        self.RETPath = os.path.join(self.dataLoadPath,'simSTA',self.bench)
        self.STAPath = os.path.join(self.dataLoadPath,'simSTA',self.bench)
        self.EVALPath = os.path.join(self.resultSavePath, self.bench, self.eval_alpha)
        self.CUTOFFPath = os.path.join(self.EVALPath, 'sta_' + self.targetCut)
        os.makedirs(self.CUTOFFPath, exist_ok = True)
        
    def alpha_eval(self):  
        targetAlpha = sorted(list(set(self.baseAlpha + self.evalAlpha)))
        df_stock = []
        for file in os.listdir(os.path.join(self.CUTOFFPath, 'daily')):
            df_file = pd.read_pickle(os.path.join(self.CUTOFFPath, 'daily', file))
            df_stock += [df_file]
        df_stock = pd.concat(df_stock).reset_index(drop=True)
        df_stock['datetime'] = pd.to_datetime(df_stock['date'].astype('str'))
                   
        print('\n', 'stats - new alpla: ', '\n', df_stock[df_stock.cat.isin(targetAlpha)].groupby(['cat','side'])[[
              'mean','std','skew','kurtosis','minimum','maximum']].mean())
        df_stock['month'] = df_stock['date'].map(lambda x:str(x)[:6])
        df_stock['year'] = df_stock['date'].map(lambda x:str(x)[:4])
    #    aggInfo_even = {'countStock':'sum','yHatStockCut':'mean','yHatStockAvg':'mean','actualRet90s':'mean','actualRet300s':'mean'}
    #    df_stock.groupby(['date','exchange','side','cat']).agg(aggInfo_even).tail(20)
        aggInfo = {'countStock':'sum'}
        aggCols = ['yHatStockCut','yHatStockAvg'] + [self.targetRet, 'vwapRet' + self.targetRet[9:], 'availNtl']
        for aggCol in aggCols: 
            df_stock[aggCol + '_w'] = df_stock[aggCol] * df_stock['countStock'] 
            aggInfo[aggCol + '_w'] = 'sum'
    
        stockList = sorted(list(df_stock.secid.unique()))
        stockList = [int(str(x)) for x in stockList]
        db = DB(db_config['host'], db_config['db'], db_config['user'], db_config['password'])        
        df_daily = db.read_daily('mdbar1d_tr',start_date=int(df_stock.date.min()),end_date=int(df_stock.date.max()),skey=stockList)
        df_daily.rename(columns = {'skey':'secid'}, inplace = True)
    #    df_daily['price_group'] = np.where(df_daily['close'] < 10, 'low', np.where(df_daily['close'] < 50, 'middle', 'high'))
        df_daily['price_group'] = np.where(df_daily['close'] < 5, '0-5', np.where(df_daily['close'] < 10, '5-10', 
                                  np.where(df_daily['close'] < 20, '10-20', np.where(df_daily['close'] < 50, '20-50', 
                                  np.where(df_daily['close'] < 100, '50-100', '100-')))))
        df_stock = df_stock.merge(df_daily[['secid','date','price_group']], on = ['secid','date'], how = 'left', validate = 'many_to_one')   
        
        ## print out stats for all trades
        group = ['side']
        df_total = df_stock.groupby(group + ['cat']).agg(aggInfo).reset_index()
        for aggCol in aggCols: 
            df_total[aggCol + '_w'] = df_total[aggCol + '_w'] / df_total['countStock']
        df_total = df_total[df_total.cat.isin(targetAlpha)].reset_index(drop = True)
        df_total['base_ret'] = np.where(df_total['cat'] == self.baseAlpha[0], df_total[self.targetRet+'_w'], np.nan)
        df_total['base_ret'] = df_total.groupby(group)['base_ret'].ffill()
        df_total['base_ret'] = df_total.groupby(group)['base_ret'].bfill()
        df_total['improvement_%'] = (df_total[self.targetRet+'_w']/df_total['base_ret'] - 1) * 100
        print('\n', df_total.groupby(group + ['cat'])[[self.targetRet+'_w', 'improvement_%', 'availNtl_w']].mean())
    
        from matplotlib.backends.backend_pdf import PdfPages
        if len(self.evalAlpha) == 0: 
            fileName = '_'.join([self.bench, targetAlpha[-1], self.targetCut, self.startDate, self.endDate]) + '.pdf'
        else: fileName = '_'.join([self.bench, self.evalAlpha[-1], self.targetCut, self.startDate, self.endDate]) + '.pdf'
        with PdfPages(os.path.join(self.EVALPath, fileName)) as pdf:              
            ## figure 1: summary of new alpha
            df_sum1 = df_stock.groupby(['cat','side'])[['mean','std','skew','kurtosis']].mean().reset_index()
            for col in ['mean','std','skew','kurtosis']:
                df_sum1[col] = df_sum1[col].map(lambda x: "{:.2f}".format(x))
            df_sum1 = df_sum1[df_sum1.cat.isin(targetAlpha)].reset_index(drop = True)
            df_sum2 = df_stock.groupby(['exchange','side','cat']).agg(aggInfo).reset_index()
            for aggCol in aggCols: 
                df_sum2[aggCol + '_w'] = df_sum2[aggCol + '_w'] / df_sum2['countStock']
            df_sum2 = df_sum2[df_sum2.cat.isin(targetAlpha)].reset_index(drop = True)
            df_sum2['base_ret'] = np.where(df_sum2['cat'] == self.baseAlpha[0], df_sum2[self.targetRet+'_w'], np.nan)
            df_sum2['base_ret'] = df_sum2.groupby(['exchange','side'])['base_ret'].ffill()
            df_sum2['base_ret'] = df_sum2.groupby(['exchange','side'])['base_ret'].bfill()
            df_sum2['improvement_%'] = (df_sum2[self.targetRet+'_w']/df_sum2['base_ret'] - 1) * 100
            df_sum2 = df_sum2.groupby(['exchange','side','cat'])[[self.targetRet+'_w', 'improvement_%', 'availNtl_w']].mean().reset_index()
            for col in [self.targetRet+'_w','improvement_%','availNtl_w']:
                df_sum2[col] = df_sum2[col].map(lambda x: "{:.2f}".format(x))
            df_sum1 = df_sum1.append(pd.DataFrame([['-']*len(df_sum1.columns)], columns = df_sum1.columns))
            df_sum1 = df_sum1.append(pd.DataFrame([list(df_sum2.columns)], columns = df_sum1.columns))
            df_sum2.columns = list(df_sum1.columns)
            df_sum = df_sum1.append(df_sum2).reset_index(drop = True)
            render_mpl_table(df_sum, seperate_col = len(df_sum1))
            plt.title('stats of new alpha & overall performance outlook \n',fontsize = 25)
            pdf.savefig()  
            plt.close()
    
            ## figure 2: overall performance for different price group           
            fig = plt.figure(figsize = (16, 20))
            plt.suptitle('performance for different price group', fontsize = 25)
            for side in ['all','buy','sell']:
                if side == 'all': i = 1
                elif side == 'buy': i = 2
                else : i = 3
                if side in ['buy','sell']:
                    group = ['side','price_group']
                    df_total = df_stock.groupby(group + ['cat']).agg(aggInfo).reset_index()
                    for aggCol in aggCols: 
                        df_total[aggCol + '_w'] = df_total[aggCol + '_w'] / df_total['countStock']
                    df_total = df_total[df_total.cat.isin(targetAlpha)].reset_index(drop = True)
                    df_total['base_ret'] = np.where(df_total['cat'] == self.baseAlpha[0], df_total[self.targetRet+'_w'], np.nan)
                    df_total['base_ret'] = df_total.groupby(group)['base_ret'].ffill()
                    df_total['base_ret'] = df_total.groupby(group)['base_ret'].bfill()
                    df_total['improvement_%'] = (df_total[self.targetRet+'_w']/df_total['base_ret'] - 1) * 100
                    df_side = df_total[df_total.side == side].groupby(group + ['cat'])[[self.targetRet+'_w', 'improvement_%', 'availNtl_w']].mean().reset_index()
                    df_group_number = df_daily.groupby(['date','price_group'])['secid'].nunique().reset_index().groupby(['price_group'])['secid'].mean()
                    df_side = df_side.merge(df_group_number.reset_index(), on = 'price_group')
                    df_side['secid'] = df_side['secid'].map(lambda x:'('+str(int(x))+' stocks)')
                    df_side['price_group'] = df_side['price_group'] + df_side['secid']
                    x_axis = 'price_group'
                else :
                    df_side = df_stock.groupby(['side','cat']).agg(aggInfo).reset_index()
                    for aggCol in aggCols: 
                        df_side[aggCol + '_w'] = df_side[aggCol + '_w'] / df_side['countStock']
                    df_side = df_side[df_side.cat.isin(targetAlpha)].reset_index(drop = True)  
                    x_axis = 'side'
                df_side = df_side.pivot(index = x_axis, columns = 'cat', values = self.targetRet+'_w').reset_index() 
                if side in ['buy','sell']:   
                    df_side['price_start'] = df_side['price_group'].map(lambda x:int(x.split('-')[0]))
                    df_side = df_side.sort_values('price_start').reset_index(drop = True)
                    df_side.drop(columns = ['price_start'], inplace = True)
                ax = fig.add_subplot(3,1,i)
                ind = np.arange(len(df_side))  # the x locations for the groups
                width = 0.1  # the width of the bars
                for i in range(len(targetAlpha)):
                    alpha = targetAlpha[i]
                    ax.bar(ind + width*(i-(len(targetAlpha)-1)/2), df_side[alpha], width, label=alpha)
                ax.set_xticks(ind)
                ax.set_xticklabels(df_side[x_axis])
                ax.set_title(side + ' side - ' + self.targetRet)
                ax.set_ylabel('return(bps)')
                plt.legend(loc = 'best') 
            pdf.savefig()  
            plt.close()
                        
            ## figure 3: historical top 3% yHat
            fig = plt.figure(figsize = (16, 20))
            plt.suptitle('mean of top 3% yHat', fontsize = 25)
            for side in ['buy','sell']:
                if side == 'buy': k = 1
                else : k = 2
                for ex in ['SH','SZ']:
                    if ex == 'SH': j = 1
                    else : j = 2
                    df_top5 = df_stock[(df_stock.side == side) & (df_stock.exchange == ex)].groupby(['cat','datetime'])['yHatStockAvg'].mean().reset_index()
                    df_hist = df_top5.pivot(index = 'datetime', columns = 'cat', values = 'yHatStockAvg').reset_index().sort_values('datetime')
                    ax = fig.add_subplot(4,1,2*(k-1)+j)
                    for alpha in targetAlpha:
                        ax.plot(df_hist['datetime'], df_hist[alpha], label = alpha + ' mean: ' + str(round(df_hist[alpha].mean(),2)), linestyle = '-')
                    ax.set_title(side + ' side - ' + ex + ' - yHatStockAvg')
                    ax.set_ylabel('yHat(bps)')
                    plt.legend(loc = 'best')
            pdf.savefig()  
            plt.close()
    
            ## figure 4: moments of new alpha
            df_moments = df_stock.groupby(['datetime','cat','side'])[['mean','std','skew','kurtosis']].mean().reset_index()
            fig = plt.figure(figsize = (16, 20))
            plt.suptitle('moments of new alpha across time', fontsize = 25)
            for moment in ['mean','std','skew','kurtosis']:
                if moment == 'mean': k = 1
                elif moment == 'std': k = 2 
                elif moment == 'skew': k = 3
                else : k = 4
                for side in ['buy','sell']:
                    if side == 'buy': j = 1
                    else : j = 2
                    df_moment = df_moments.loc[df_moments.side == side, ['datetime','cat',moment]].reset_index(drop = True)
                    df_hist = df_moment.pivot(index = 'datetime', columns = 'cat', values = moment).reset_index().sort_values('datetime')
                    ax = fig.add_subplot(4,2,2*(k-1)+j)
                    for alpha in targetAlpha:
                        ax.plot(df_hist['datetime'], df_hist[alpha], label = alpha + ' mean: ' + str(round(df_hist[alpha].mean(),2)), linestyle = '-')
                    ax.set_title(side + ' side - ' + moment)
                    ax.set_ylabel('yHat(bps)')
                    plt.legend(loc = 'best')
            pdf.savefig()  
            plt.close()
    
            ## figure 5: quantiles of new alpha
            df_quantiles = df_stock.groupby(['datetime','cat','side'])[['75p','90p','95p','99p']].mean().reset_index()
            fig = plt.figure(figsize = (16, 20))
            plt.suptitle('quantiles of new alpha across time', fontsize = 25)
            for quantile in ['75p','90p','95p','99p']:
                if quantile == '75p': k = 1
                elif quantile == '90p': k = 2 
                elif quantile == '95p': k = 3
                else : k = 4
                for side in ['buy','sell']:
                    if side == 'buy': j = 1
                    else : j = 2
                    df_quantile = df_quantiles.loc[df_quantiles.side == side, ['datetime','cat',quantile]].reset_index(drop = True)
                    df_hist = df_quantile.pivot(index = 'datetime', columns = 'cat', values = quantile).reset_index().sort_values('datetime')
                    ax = fig.add_subplot(4,2,2*(k-1)+j)
                    for alpha in targetAlpha:
                        ax.plot(df_hist['datetime'], df_hist[alpha], label = alpha + ' mean: ' + str(round(df_hist[alpha].mean(),2)), linestyle = '-')
                    ax.set_title(side + ' side - ' + quantile)
                    ax.set_ylabel('yHat(bps)')
                    plt.legend(loc = 'best')
            pdf.savefig()  
            plt.close()

            ## figure 6: daily volatility of new alpha
            df_quantiles = df_stock.groupby(['datetime','cat','side'])[['mean','std','95p','99p']].std().reset_index()
            fig = plt.figure(figsize = (16, 20))
            plt.suptitle('daily volatility across stocks across time', fontsize = 25)
            for quantile in ['mean','std','95p','99p']:
                if quantile == 'mean': k = 1
                elif quantile == 'std': k = 2 
                elif quantile == '95p': k = 3
                else : k = 4
                for side in ['buy','sell']:
                    if side == 'buy': j = 1
                    else : j = 2
                    df_quantile = df_quantiles.loc[df_quantiles.side == side, ['datetime','cat',quantile]].reset_index(drop = True)
                    df_hist = df_quantile.pivot(index = 'datetime', columns = 'cat', values = quantile).reset_index().sort_values('datetime')
                    ax = fig.add_subplot(4,2,2*(k-1)+j)
                    for alpha in targetAlpha:
                        ax.plot(df_hist['datetime'], df_hist[alpha], label = alpha + ' mean: ' + str(round(df_hist[alpha].mean(),2)), linestyle = '-')
                    ax.set_title(side + ' side - ' + quantile)
                    ax.set_ylabel('yHat(bps)')
                    plt.legend(loc = 'best')
            pdf.savefig()  
            plt.close()
            
            ## figure 7: historical monthly realized return
            aggInfo['datetime'] = 'first'   
            stats_horizon = 'month'
            df_stats = df_stock.groupby([stats_horizon,'exchange','side','cat']).agg(aggInfo).reset_index()
            for aggCol in aggCols: 
                df_stats[aggCol + '_w'] = df_stats[aggCol + '_w'] / df_stats['countStock']
            fig = plt.figure(figsize = (16, 20))
            plt.suptitle('monthly realized return', fontsize = 25)
            for side in ['buy','sell']:
                if side == 'buy': k = 1
                else : k = 2
                for ex in ['SH','SZ']:
                    if ex == 'SH': j = 1
                    else : j = 2
                    df_hist = df_stats.loc[(df_stats.side == side) & (df_stats.exchange == ex), [stats_horizon,'cat'] + [self.targetRet + '_w']
                                  ].pivot(index = stats_horizon, columns = 'cat', values = self.targetRet + '_w').reset_index()
                    ax = fig.add_subplot(4,1,2*(k-1)+j)
                    ind = np.arange(len(df_hist))  # the x locations for the groups
                    width = 0.15  # the width of the bars
                    for i in range(len(targetAlpha)):
                        alpha = targetAlpha[i]
                        ax.bar(ind + width*(i-(len(targetAlpha)-1)/2), df_hist[alpha], width, label=alpha)
                    ax.set_xticks(ind)
                    ax.set_xticklabels(df_hist[stats_horizon])
                    ax.set_title(side + ' side - ' + ex + ' - ' + self.targetRet)
                    plt.legend(loc = 'best')
            pdf.savefig()  
            plt.close()
                               
            ## figure 8: historical daily realized return
            aggInfo['datetime'] = 'first'         
            df_stats = df_stock.groupby(['date','exchange','side','cat']).agg(aggInfo).reset_index()
            for aggCol in aggCols: 
                df_stats[aggCol + '_w'] = df_stats[aggCol + '_w'] / df_stats['countStock']
            fig = plt.figure(figsize = (16, 20))
            plt.suptitle('daily realized return', fontsize = 25)
            for side in ['buy','sell']:
                if side == 'buy': k = 1
                else : k = 2
                for ex in ['SH','SZ']:
                    if ex == 'SH': j = 1
                    else : j = 2
                    df_hist = df_stats.loc[(df_stats.side == side) & (df_stats.exchange == ex), ['datetime','cat'] + [self.targetRet + '_w']
                                          ].pivot(index = 'datetime', columns = 'cat', values = self.targetRet + '_w').reset_index().sort_values('datetime')
                    ax = fig.add_subplot(4,1,2*(k-1)+j)          
                    for alpha in targetAlpha:
                        ax.plot(df_hist['datetime'], df_hist[alpha], label = alpha + ' mean: ' + str(round(df_hist[alpha].mean(),2)), linestyle = '-')
                    ax.set_title(side + ' side - ' + ex + ' - ' + self.targetRet)
                    ax.set_ylabel('return(bps)')
                    plt.legend(loc = 'best')
            pdf.savefig()  
            plt.close()
                    
            ## figure 9: intraday realized return and opportunities (top5% & 95p)
            df_stock_intra = []
            for file in os.listdir(os.path.join(self.CUTOFFPath, 'intraday')):
                df_file = pd.read_pickle(os.path.join(self.CUTOFFPath, 'intraday', file))
                df_file['hurdle_type'] = self.targetCut
                df_stock_intra += [df_file]
            df_stock_intra = pd.concat(df_stock_intra).reset_index(drop=True)    
            df_stock_intra = df_stock_intra.groupby(['hurdle_type','exchange','side','cat','mins_since_open'])[['secid', self.targetRet]].mean().reset_index()
            df_stock_intra[self.targetRet] = round((df_stock_intra[self.targetRet]*10000),2)
            for value in ['secid', self.targetRet]:
                if value == 'secid': 
                    suptitle = 'daily average oppo. across minutes since open'
                    ylabel = 'num of oppo.'
                else : 
                    suptitle = 'daily average return across minutes since open'
                    ylabel = 'return(bps)'
                for hurdle in [self.targetCut]:
                    fig = plt.figure(figsize = (16, 20))
                    plt.suptitle(suptitle, fontsize = 25)    
                    for side in ['buy','sell']:
                        if side == 'buy': k = 1
                        else : k = 2
                        for ex in ['SH','SZ']:
                            if ex == 'SH': j = 1
                            else : j = 2
                            df_intra = df_stock_intra.loc[(df_stock_intra.hurdle_type == hurdle) & (df_stock_intra.side == side) & (df_stock_intra.exchange == ex), 
                                                         ['mins_since_open','cat'] + [value]].pivot(index = 'mins_since_open', 
                                                        columns = 'cat', values = value).reset_index().sort_values('mins_since_open')
                            ax = fig.add_subplot(4,1,2*(k-1)+j)          
                            for alpha in targetAlpha:
                                ax.plot(df_intra['mins_since_open'], df_intra[alpha], label = alpha + ' mean: ' + str(round(df_hist[alpha].mean(),2)), linestyle = '-')
                            ax.set_title(side + ' side - ' + ex + ' - ' + ' - ' + hurdle)
                            ax.set_ylabel(ylabel)
                            plt.legend(loc = 'best')   
                    pdf.savefig()  
                    plt.close()      
        return None

def sta_cutoff_generation(simDate, c_eval):
    # db = DB(db_config['host'], db_config['db'], db_config['user'], db_config['password'])        
    df_daily = db.read_daily('mdbar1d_tr',start_date=int(simDate),
                                  end_date=int(simDate),index_name=[c_eval.bench])  
    staCols, retCols = c_eval.baseAlpha + c_eval.evalAlpha, c_eval.actualRetLs
    print('...... Now Processing Cutoff on ', simDate)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OMP_THREAD_LIMIT'] = '1'
    df_md = pd.read_parquet(os.path.join(c_eval.MDPath, 'md' + simDate + '.parquet'))
    df_md = df_md[['skey','date','ordering','datetime','ask1p','ask1q','bid1p','bid1q']].reset_index(drop = True)
    df_ret = pd.read_parquet(os.path.join(c_eval.RETPath, 'sta_ps_ret', 'ret' + simDate + '.parquet'))
    df_ret.columns = ['skey','date','ordering','passFilter','buyRet90s','sellRet90s']
    buyRetCols = []
    sellRetCols = []
    for horizon in [x[9:] for x in c_eval.actualRetLs]:
        buyRetCols.append('buyRet' + horizon)
        sellRetCols.append('sellRet' + horizon)
    df_ret = df_ret[['skey','date','ordering','passFilter'] + buyRetCols + sellRetCols].reset_index(drop = True)
    df_sta = pd.read_parquet(os.path.join(c_eval.STAPath, c_eval.baseAlpha[0], 'sta' + simDate + '.parquet'))
    buySTACols = ['yHatBuy']
    sellSTACols = ['yHatSell']
    for sta_cat in c_eval.evalAlpha:         
        df_sta_tgt = pd.read_parquet(os.path.join(c_eval.STAPath, sta_cat, 'sta' + simDate + '.parquet'))
        df_sta_tgt.rename(columns = {'yHatBuy':'yHatBuy_' + sta_cat[4:], 'yHatSell':'yHatSell_' + sta_cat[4:]}, inplace = True)
        df_sta = df_sta.merge(df_sta_tgt, on = ['skey','date','ordering'], how = 'outer', validate = 'one_to_one')
        buySTACols.append('yHatBuy_' + sta_cat[4:])
        sellSTACols.append('yHatSell_' + sta_cat[4:])

    rawData = df_md.merge(df_ret, on = ['skey','date','ordering'], how = 'left', validate = 'one_to_one')
    rawData = rawData.merge(df_sta, on = ['skey','date','ordering'], how = 'left', validate = 'one_to_one')
    rawData = rawData.sort_values(['skey','date','ordering']).reset_index(drop=True)
    rawData['passFilter'] = rawData['passFilter'].astype(bool)
    df_alpha = rawData.loc[(rawData.passFilter) & (rawData.skey.isin(df_daily.skey.unique())), :].reset_index(drop = True)
    df_alpha['exchange'] = df_alpha['skey'].map(lambda x:str(x)[:1])
    df_alpha['exchange'] = np.where(df_alpha['exchange'] == '1', 'SH', 'SZ')
    df_alpha['minute'] = df_alpha['datetime'].dt.hour * 60 + df_alpha['datetime'].dt.minute
    df_alpha['mins_since_open'] = np.where(df_alpha['minute'] <= 690, df_alpha['minute'] - 570, df_alpha['minute'] - 660)
    
    df_daily, df_intra = [], []  
    for ex in ['SH','SZ']:
        df_exchange = df_alpha[df_alpha.exchange == ex].reset_index(drop=True)
        for side in ['buy','sell']:
            # print(ex, side)
            if side == 'buy': df_side = df_exchange[['skey','mins_since_open','ask1p','ask1q'] + buySTACols + buyRetCols].reset_index(drop=True)
            else: df_side = df_exchange[['skey','mins_since_open','bid1p','bid1q'] + sellSTACols + sellRetCols].reset_index(drop=True)
            df_side.columns = ['secid','mins_since_open','price','quantity'] + staCols + retCols
            df_side = df_side.dropna(subset = staCols + retCols).reset_index(drop = True)
            if len(df_side) == 0: continue
            df_side['availNtl'] = (df_side['price'] * df_side['quantity'].clip(0, 500000)).clip(0, 500000)
            for col in retCols:
                df_side['availNtl_F%s'%(col[9:])] = df_side['availNtl'] * (1 + df_side[col])                
            for stock, df_stock in df_side.groupby('secid'): 
#                assert 1 == 2
                statsData = {}
                for col in ['date','exchange','side','secid','cat','mean','std',
                            'skew','kurtosis','75p','90p','95p','99p','minimum','maximum',
                            'countStock','availNtl','yHatStockCut','yHatStockAvg'] + retCols + ['vwapRet' + x[9:] for x in retCols]:
                    statsData[col] = []
                statsData['date'] = simDate
                statsData['exchange'] = ex
                statsData['side'] = side
                statsData['secid'] = stock
                for cat in staCols: 
                    statsData['mean'].append(df_stock[cat].mean())
                    statsData['std'].append(df_stock[cat].std())
                    statsData['skew'].append(df_stock[cat].skew())
                    statsData['kurtosis'].append(df_stock[cat].kurtosis())
                    statsData['75p'].append(df_stock[cat].quantile(.75))
                    statsData['90p'].append(df_stock[cat].quantile(.9))
                    statsData['95p'].append(df_stock[cat].quantile(.95))
                    statsData['99p'].append(df_stock[cat].quantile(.99))
                    statsData['minimum'].append(df_stock[cat].min())
                    statsData['maximum'].append(df_stock[cat].max())   
                    
                    df_cutoff = df_stock[(df_stock[cat] >= df_stock[cat].quantile(0.95))].reset_index(drop=True)
                    statsData['yHatStockCut'].append(df_stock[cat].quantile(0.95))  

                    # if len(df_cutoff) == 0: continue 
                    # assert len(df_cutoff) > 0
                    statsData['cat'].append(cat)
                    statsData['countStock'].append(df_cutoff.shape[0])
                    statsData['availNtl'].append(df_cutoff['availNtl'].mean())
                    statsData['yHatStockAvg'].append(df_cutoff[cat].mean())
                    for col in retCols:
                        statsData[col].append(df_cutoff[col].mean())
                        if len(df_cutoff) == 0: statsData['vwapRet' + col[9:]].append(np.nan)
                        else :statsData['vwapRet' + col[9:]].append(df_cutoff['availNtl_F%s'%(col[9:])].sum()/df_cutoff['availNtl'].sum() - 1)
                     
                    ## ADD BEGIN
                    df_cutoff = df_cutoff[['secid','mins_since_open'] + retCols].reset_index(drop=True)
                    df_cutoff['date'] = simDate
                    df_cutoff['exchange'] = ex
                    df_cutoff['side'] = side    
                    df_cutoff['cat'] = cat
                    df_intra += [df_cutoff]
                    ## ADD END
                    
                statsData = pd.DataFrame(statsData)
                statsData = statsData.dropna()
                for col in ['mean','std','75p','90p','95p','99p','minimum','maximum','yHatStockCut','yHatStockAvg'] + retCols + ['vwapRet' + x[9:] for x in retCols]:
                    statsData[col] = statsData[col].apply(lambda x: '%.2f'%(x*10000))
                    
                df_daily += [statsData]

    aggInfo = {'secid':'count'}
    for ret in retCols:
        aggInfo[ret] = 'mean'
    df_intra = pd.concat(df_intra).reset_index(drop=True)
    for col in ['date','secid','mins_since_open']:
        df_intra[col] = df_intra[col].astype('int32')
    for col in retCols:
        df_intra[col] = df_intra[col].astype('float32')
    df_intra = df_intra.groupby(['date','exchange','side','cat','mins_since_open']).agg(aggInfo).reset_index()
    os.makedirs(os.path.join(c_eval.CUTOFFPath, 'intraday'), exist_ok = True)
    df_intra.to_pickle(os.path.join(c_eval.CUTOFFPath, 'intraday', 'df_' + c_eval.targetCut + '_' + simDate + '.pkl'), protocol = 4)
                       
    df_daily = pd.concat(df_daily).reset_index(drop=True)
    for col in ['date','secid','countStock']:
        df_daily[col] = df_daily[col].astype('int32')
    for col in ['mean','std','skew','kurtosis','75p','90p','95p','99p','minimum','maximum','availNtl','yHatStockCut','yHatStockAvg'] + retCols + ['vwapRet' + x[9:] for x in retCols]:
        df_daily[col] = df_daily[col].astype('float32')
    os.makedirs(os.path.join(c_eval.CUTOFFPath, 'daily'), exist_ok = True)
    df_daily.to_pickle(os.path.join(c_eval.CUTOFFPath, 'daily', 'df_' + c_eval.targetCut + '_' + simDate + '.pkl'), protocol = 4)
    
    return df_daily
    
 

###############################################################################
import multiprocessing
from functools import partial  
def proc_multiDates(startDate, endDate, c_eval):
    dateList = []
    for simFile in os.listdir(MDPath):
        if (simFile[2:10] > endDate) | (simFile[2:10]  < startDate): continue   
        dateList.append(simFile[2:10])
        
    pool = multiprocessing.Pool(32)    
    
    pool.map(partial(sta_cutoff_generation, c_eval = c_eval), sorted(dateList))
    pool.close()
    pool.join()
    
if __name__ == '__main__':
    
    ###############################################################################   
    sta_input = os.path.join('/data','home','andy','shareWithBenny','sta_input_demo.yaml')
    f = open(sta_input, 'r', encoding='utf-8')
    cfg = f.read()
    dict_yaml = yaml.full_load(cfg)
    f.close()
    
    startDate = str(dict_yaml['startDate'])
    endDate = str(dict_yaml['endDate'])
    dataLoadPath = ''
    for ele in dict_yaml['dataLoadPath']:
        dataLoadPath = os.path.join(dataLoadPath, ele)
    MDPath = os.path.join(dataLoadPath, 'simMarketData', dict_yaml['bench'])
    
    sta_eval_run = sta_alpha_eval(sta_input)
    ###############################################################################
    
    proc_multiDates(startDate, endDate, sta_eval_run)

    sta_eval_run.alpha_eval() 

    