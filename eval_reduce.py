import os
import re
import sys
from collections import defaultdict
import pandas as pd
import numpy as np
from utils import *
from AShareReader import AShareReader
from CephClient import CephClient
import multiprocessing
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
from glob import glob
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import plotly.io as io
import plotly.express as px
import plotly.graph_objects as go
from eval import StaAlphaEval

import yaml

perc = [.01, .05, .1, .25, .5, .75, .9, .95, .99]

def read_pkl(path):
    return pd.read_pickle(path)

class StaAlphaEvalReduce(StaAlphaEval):
    def __init__(self, sta_input):
        super().__init__(sta_input)

        self.pdf_width = 30
        self.html_width = 1400
        self.template = 'seaborn'
        self.to_html = partial(io.to_html, include_plotlyjs='cdn', full_html=False)
        self.color = ['lightslategrey', 'steelblue', 'lightskyblue', 'paleturquoise', 'azure']

    def alpha_eval(self):
        with Pool(16) as p:
            self.daily_stat = pd.concat(p.map(read_pkl, glob(f"{self.cutoff_path}/daily/*.pkl")), ignore_index=True)

        for col in ['yHatHurdle','yHatAvg', 'actualRetAvg', 'vwActualRetAvg']:
            self.daily_stat[col] = self.daily_stat[col] * 10000
        self.daily_stat['datetime'] = pd.to_datetime(self.daily_stat['date'].astype('str'))
        self.daily_stat['month'] = self.daily_stat['date'].map(lambda x:str(x)[:6])
        self.daily_stat['year'] = self.daily_stat['date'].map(lambda x:str(x)[:4])

        stock_list = self.daily_stat.skey.unique()
        self.df_daily = self.stock_reader.Read_Stock_Daily('com_md_eq_cn', 'mdbar1d_jq', start_date=self.daily_stat.date.min(),
                                                            end_date=self.daily_stat.date.max(), stock_list=stock_list, 
                                                            cols_list=['skey','date','close'])
        self.df_daily['price_group'] = np.where(self.df_daily['close'] < 5, '0-5', np.where(self.df_daily['close'] < 10, '5-10', 
                                       np.where(self.df_daily['close'] < 20, '10-20', np.where(self.df_daily['close'] < 50, '20-50', 
                                       np.where(self.df_daily['close'] < 100, '50-100', '100-')))))

        self.daily_stat = self.daily_stat.merge(self.df_daily[['skey','date','price_group']], on = ['skey','date'], 
                                                how = 'left', validate = 'many_to_one')

        with Pool(16) as p:
            df_intraday = pd.concat(p.map(read_pkl, glob(f"{self.cutoff_path}/intraday/*.pkl")), ignore_index=True)

        self.intraday_stat = df_intraday.groupby(['exchange','side','sta_cat','mins_since_open'], observed=True)[['countOppo']].mean()
        self.intraday_stat['countOppo'] = self.intraday_stat['countOppo'] / len(stock_list)
        self.intraday_stat['vwActualRetAvg'] = (df_intraday.groupby(['exchange','side','sta_cat','mins_since_open'])
                                                             .apply(lambda x: weighted_average(x['vwActualRetAvg'], weights=x['availNtlSum'])))
        self.intraday_stat['vwActualRetAvg'] = self.intraday_stat['vwActualRetAvg'] * 10000
        self.intraday_stat = self.intraday_stat.reset_index()
        
        # self.pdf_report()
        self.html_report()

    def html_report(self):
        px.defaults.template = 'seaborn'
        report_name = '_'.join([self.universe, self.eval_alpha[-1]['name'], self.target_cut, self.eval_focus, 
                                self.start_date, self.end_date]) + '.html'
        with open(os.path.join(self.eval_path, report_name), 'w') as f:
            f.write('''<html>\n<head><meta charset="utf-8" /></head>\n<body>\n''')

            # sta all summary
            table1 = self.sta_all_summary_html()
            f.write(table1 + '\n')

            # distribution
            f.write("<h2>distribution of yHat and yTrue</h2>" + "\n")
            figs = self.sta_all_dist_html()
            f.write(figs + '\n')

            # drop actualRet
            self.daily_stat = self.daily_stat[~self.daily_stat['sta_cat'].str.startswith(self.target_ret)].copy()

            # sta cutoff summary
            table2 = self.sta_cutoff_summary_html()
            f.write(table2 + '\n')
            
            if self.eval_focus in ['ret', 'mixed']:
                # performance by side
                f.write("<h2>performance for different side</h2>" + "\n")
                figs = self.performance_by_side_html()
                f.write(figs + '\n')
                # performance by price group
                f.write("<h2>performance for different price group</h2>" + "\n")
                figs = self.performance_by_price_group_html()
                f.write(figs + '\n')
            if self.eval_focus in ['oppo', 'mixed']:
                # opportunities by side
                f.write("<h2>opportunities for different side</h2>" + "\n")
                figs = self.opportunities_by_side_html()
                f.write(figs + '\n')
                # performance by price group
                f.write("<h2>opportunities for different price group</h2>" + "\n")
                figs = self.opportunities_by_price_group_html()
                f.write(figs + '\n')

            # daily yHatAvg
            f.write(f"<h2>daily average yHatHurdle</h2>" + "\n")
            figs = self.daily_yHatHurdle_html()
            f.write(figs + '\n')

            if self.eval_focus in ['ret', 'mixed']:
                # monthly realized return
                f.write("<h2>monthly realized return</h2>" + "\n")
                figs = self.monthly_realized_return_html()
                f.write(figs + '\n')
            if self.eval_focus in ['oppo', 'mixed']:
                # monthly realized return
                f.write("<h2>monthly number of opportunities</h2>" + "\n")
                figs = self.monthly_opportunities_html()
                f.write(figs + '\n')

            if self.eval_focus in ['ret', 'mixed']:
                # daily realized return
                f.write("<h2>daily realized return</h2>" + "\n")
                figs = self.daily_realized_return_html()
                f.write(figs + '\n')
            if self.eval_focus in ['oppo', 'mixed']:
                # daily realized return
                f.write("<h2>daily number of opportunities</h2>" + "\n")
                figs = self.daily_opportunities_html()
                f.write(figs + '\n')

            # daily realized return
            f.write("<h2>intraday average return across minutes since open</h2>" + "\n")
            figs = self.intraday_realized_return_html()
            f.write(figs + '\n')

            # intraday number of opportunities
            f.write("<h2>intraday average oppo. across minutes since open</h2>" + "\n")
            figs = self.intraday_opportunities_html()
            f.write(figs + '\n')
            
            f.write('''\n</body>\n</html>''')

    def pdf_report(self):
        report_name = '_'.join([self.universe, self.eval_alpha[-1]['name'], self.target_cut, self.start_date, self.end_date]) + '.pdf'
        with PdfPages(os.path.join(self.eval_path, report_name)) as pdf:
            # sta all summary
            table1 = self.sta_all_summary_pdf()
            df_to_plt_table(table1)
            plt.title('general stats of yHat',fontsize = 25)
            plt.tight_layout()
            pdf.savefig()
            plt.close

            # sta cutoff summary
            table2 = self.sta_cutoff_summary_pdf()
            df_to_plt_table(table2)
            plt.title('overall performance outlook', fontsize = 26)
            plt.tight_layout()
            pdf.savefig()
            plt.close

            # performance by price group
            fig = self.performance_by_price_group_pdf()
            pdf.savefig(fig)
            plt.close()

            # daily yHatAvg
            fig = self.daily_yHatAvg_pdf()
            pdf.savefig(fig)
            plt.close()

            # monthly realized return
            fig = self.monthly_realized_return_pdf()
            pdf.savefig(fig)
            plt.close()

            # daily realized return
            fig = self.daily_realized_return_pdf()
            pdf.savefig(fig)
            plt.close()

            # intraday number of opportunities
            fig = self.intraday_opportunities_pdf()
            pdf.savefig(fig)
            plt.close()

            # daily realized return
            fig = self.intraday_realized_return_pdf()
            pdf.savefig(fig)
            plt.close()

    def sta_all_summary_pdf(self):
        df_total = self.daily_stat.groupby(['exchange','side','sta_cat'])[['mean','std','skew','kurtosis']].mean().reset_index()
        for col in ['mean','std','skew','kurtosis']:
            df_total[col] = df_total[col].map(lambda x: "{:.2f}".format(x))
        return df_total.set_index(['sta_cat'])
    
    def sta_cutoff_summary_pdf(self):
        df_total = self.daily_stat.groupby(['exchange', 'side', 'sta_cat'])[['countOppo']].mean()
        
        for col in ['yHatHurdle', 'yHatAvg', 'vwActualRetAvg']:
            df_total[col] = self.daily_stat.groupby(['exchange', 'side', 'sta_cat']).apply(lambda x: weighted_average(x[col], weights=x['availNtlSum']))
        df_total['availNtl'] = self.daily_stat.groupby(['exchange', 'side', 'sta_cat']).apply(lambda x: weighted_average(x['availNtlAvg'], weights=x['countOppo']))

        df_total.reset_index(inplace=True)

        df_total['base_ret'] = np.where(df_total['sta_cat'] == self.eval_alpha[0], df_total['vwActualRetAvg'], np.nan)
        df_total['base_ret'] = df_total.groupby(['exchange', 'side'])['base_ret'].ffill()
        df_total['base_ret'] = df_total.groupby(['exchange', 'side'])['base_ret'].bfill()

        df_total['improvement(%)'] = (df_total['vwActualRetAvg']/df_total['base_ret'] - 1) * 100

        for col in ['yHatAvg', 'yHatHurdle','vwActualRetAvg','improvement(%)','availNtl', 'countOppo']:
            df_total[col] = df_total[col].map(lambda x: "{:.2f}".format(x))

        df_total = df_total.drop("base_ret", axis=1)
        return df_total.set_index(['sta_cat'])

    def performance_by_price_group_pdf(self):
        fig = plt.figure(figsize = (16, 20))
        plt.suptitle('performance for different price group', fontsize = 25)
        for side in ['all','buy','sell']:
            if side == 'all': i = 1
            elif side == 'buy': i = 2
            else: i = 3
            if side in ['buy','sell']:
                df_total = (self.daily_stat.groupby(['side','price_group', 'sta_cat'])
                                .apply(lambda x: weighted_average(x['vwActualRetAvg'], weights=x['availNtlSum'])))
                df_total.name = 'vwActualRetAvg'
                df_total = df_total.reset_index()
                
                df_side = df_total[df_total.side == side]
                df_group_number = self.df_daily.groupby(['date','price_group'])['skey'].nunique().reset_index().groupby(['price_group'])['skey'].mean()
                df_side = df_side.merge(df_group_number.reset_index(), on = 'price_group')
                df_side['skey'] = df_side['skey'].map(lambda x:'('+str(int(x))+' stocks)')
                df_side['price_group'] = df_side['price_group'] + df_side['skey']
                x_axis = 'price_group'
            else :
                df_side = self.daily_stat.groupby(['side','sta_cat']).apply(lambda x: weighted_average(x['vwActualRetAvg'], weights=x['availNtlSum']))
                df_side.name = 'vwActualRetAvg'
                df_side = df_side.reset_index()
                x_axis = 'side'
            df_side = df_side.pivot(index = x_axis, columns = 'sta_cat', values = 'vwActualRetAvg').reset_index() 
            if side in ['buy','sell']:   
                df_side['price_start'] = df_side['price_group'].map(lambda x:int(x.split('-')[0]))
                df_side = df_side.sort_values('price_start').reset_index(drop = True)
                df_side.drop(columns = ['price_start'], inplace = True)
            ax = fig.add_subplot(3,1,i)
            ind = np.arange(len(df_side))  # the x locations for the groups
            width = 0.1  # the width of the bars
            for i, alpha in enumerate(self.eval_alpha):
                ax.bar(ind + width*(i-(len(self.eval_alpha)-1)/2), df_side[alpha], width, label=alpha, color=self.color[i])
            ax.set_xticks(ind)
            ax.set_xticklabels(df_side[x_axis])
            ax.set_title(side + ' side - ' + self.target_ret)
            ax.set_ylabel('return(bps)')
            plt.legend(loc = 'best')
        plt.tight_layout()
        return fig

    def daily_yHatAvg_pdf(self):
        fig = plt.figure(figsize = (16, 20))
        plt.suptitle(f'mean of {self.target_cut} yHat', fontsize = 25)
        for side in ['buy','sell']:
            if side == 'buy': k = 1
            else: k = 2
            for ex in ['SH','SZ']:
                if ex == 'SH': j = 1
                else : j = 2
                df_top = (self.daily_stat[(self.daily_stat.side == side) & (self.daily_stat.exchange == ex)]
                                           .groupby(['sta_cat','datetime'])['yHatAvg'].mean().reset_index())
                df_hist = df_top.pivot(index = 'datetime', columns = 'sta_cat', values = 'yHatAvg').reset_index().sort_values('datetime')
                if len(df_hist) == 0:
                    continue
                ax = fig.add_subplot(4,1,2*(k-1)+j)
                for i, alpha in enumerate(self.eval_alpha):
                    ax.plot(df_hist['datetime'], df_hist[alpha], label = alpha + ' mean: ' + str(round(df_hist[alpha].mean(),2)), 
                            linestyle = '-', color=self.color[i])
                ax.set_title(side + ' side - ' + ex + ' - yHatAvg')
                ax.set_ylabel('yHat(bps)')
                plt.legend(loc = 'best')
        plt.tight_layout()
        plt.show()
        return fig

    def monthly_realized_return_pdf(self):
        stats_horizon = 'month'
        df_stats = self.daily_stat.groupby([stats_horizon,'exchange','side','sta_cat'])[['datetime']].first()
        df_stats['vwActualRetAvg'] = (self.daily_stat.groupby([stats_horizon,'exchange','side','sta_cat'])
                                                     .apply(lambda x: weighted_average(x['vwActualRetAvg'], weights=x['availNtlSum'])))
        df_stats = df_stats.reset_index()
        fig = plt.figure(figsize=(16, 20))
        plt.suptitle('monthly realized return', fontsize = 25)
        for side in ['buy','sell']:
            if side == 'buy': k = 1
            else: k = 2
            for ex in ['SH','SZ']:
                if ex == 'SH': j = 1
                else : j = 2
                df_hist = (df_stats.loc[(df_stats.side == side) & (df_stats.exchange == ex), [stats_horizon,'sta_cat'] + ['vwActualRetAvg']]
                                   .pivot(index = stats_horizon, columns = 'sta_cat', values = 'vwActualRetAvg').reset_index())
                if len(df_hist) == 0:
                    continue
                ax = fig.add_subplot(4,1,2*(k-1)+j)
                ind = np.arange(len(df_hist))  # the x locations for the groups
                width = 0.15  # the width of the bars
                for i, alpha in enumerate(self.eval_alpha):
                    ax.bar(ind + width*(i-(len(self.eval_alpha)-1)/2), df_hist[alpha], width, label=alpha,color=self.color[i])
                ax.set_xticks(ind)
                ax.set_xticklabels(df_hist[stats_horizon])
                ax.set_title(side + ' side - ' + ex + ' - ' + self.target_ret)
                plt.legend(loc = 'best')
        plt.tight_layout()
        return fig

    def daily_realized_return_pdf(self):
        df_stats = self.daily_stat.groupby(['date','exchange','side','sta_cat'])[['datetime']].first()
        df_stats['vwActualRetAvg'] = (self.daily_stat.groupby(['date','exchange','side','sta_cat'])
                                                     .apply(lambda x: weighted_average(x['vwActualRetAvg'], weights=x['availNtlSum'])))
        df_stats = df_stats.reset_index()
        fig = plt.figure(figsize=(16, 20))
        plt.suptitle('daily realized return', fontsize = 25)
        for side in ['buy','sell']:
            if side == 'buy': k = 1
            else: k = 2
            for ex in ['SH','SZ']:
                if ex == 'SH': j = 1
                else : j = 2
                df_hist = (df_stats.loc[(df_stats.side == side) & (df_stats.exchange == ex), ['datetime','sta_cat'] + ['vwActualRetAvg']]
                                   .pivot(index = 'datetime', columns = 'sta_cat', values = 'vwActualRetAvg').reset_index())
                if len(df_hist) == 0:
                    continue
                ax = fig.add_subplot(4,1,2*(k-1)+j)
                for i, alpha in enumerate(self.eval_alpha):
                    ax.plot(df_hist['datetime'], df_hist[alpha], label = alpha + ' mean: ' + str(round(df_hist[alpha].mean(),2)), 
                            linestyle = '-', color=self.color[i])
                ax.set_title(side + ' side - ' + ex + ' - ' + self.target_ret)
                ax.set_ylabel('return(bps)')
                plt.legend(loc = 'best')
        plt.tight_layout()
        return fig

    def intraday_opportunities_pdf(self):
        suptitle = 'daily average oppo. across minutes since open'
        ylabel = 'num of oppo.'
        value = 'countOppo'
        fig = plt.figure(figsize = (16, 20))
        plt.suptitle(suptitle, fontsize = 25)    
        for side in ['buy','sell']:
            if side == 'buy': k = 1
            else: k = 2
            for ex in ['SH','SZ']:
                if ex == 'SH': j = 1
                else: j = 2
                df_intra = (self.intraday_stat.loc[(self.intraday_stat.side == side) & (self.intraday_stat.exchange == ex), 
                                                    ['mins_since_open','sta_cat'] + [value]]
                                                .pivot(index='mins_since_open', columns='sta_cat', values=value)
                                                .reset_index().sort_values('mins_since_open'))
                if len(df_intra) == 0:
                    continue
                ax = fig.add_subplot(4,1,2*(k-1)+j)          
                for i, alpha in enumerate(self.eval_alpha):
                    ax.plot(df_intra['mins_since_open'], df_intra[alpha], label = alpha, linestyle = '-', color=self.color[i])
                ax.set_title(side + ' side - ' + ex + ' - ' + ' - ' + self.target_cut)
                ax.set_ylabel(ylabel)
                plt.legend(loc = 'best')
        plt.tight_layout()
        return fig

    def intraday_realized_return_pdf(self):
        suptitle = 'daily average return across minutes since open'
        ylabel = 'return(bps)'
        value = 'vwActualRetAvg'
        fig = plt.figure(figsize = (16, 20))
        plt.suptitle(suptitle, fontsize = 25)    
        for side in ['buy','sell']:
            if side == 'buy': k = 1
            else: k = 2
            for ex in ['SH','SZ']:
                if ex == 'SH': j = 1
                else: j = 2
                df_intra = (self.intraday_stat.loc[(self.intraday_stat.side == side) & (self.intraday_stat.exchange == ex), 
                                                    ['mins_since_open','sta_cat'] + [value]]
                                                .pivot(index='mins_since_open', columns='sta_cat', values=value)
                                                .reset_index().sort_values('mins_since_open'))
                if len(df_intra) == 0:
                    continue
                ax = fig.add_subplot(4,1,2*(k-1)+j)          
                for i, alpha in enumerate(self.eval_alpha):
                    ax.plot(df_intra['mins_since_open'], df_intra[alpha], label = alpha, linestyle = '-', color=self.color[i])
                ax.set_title(side + ' side - ' + ex + ' - ' + ' - ' + self.target_cut)
                ax.set_ylabel(ylabel)
                plt.legend(loc = 'best')
        plt.tight_layout()
        return fig

    def sta_all_summary_html(self):
        tmp_cols = ['count_x','sum_x','sum_x2','sum_x3','sum_x4']
        df_total = self.daily_stat.groupby(['sta_cat','exchange','side'])[tmp_cols].sum().reset_index()

        df_total['mean'] = df_total['sum_x'] / df_total['count_x']
        ex2_tmp = df_total['sum_x2'] / df_total['count_x']
        ex3_tmp = df_total['sum_x3'] / df_total['count_x']
        ex4_tmp = df_total['sum_x4'] / df_total['count_x']
        
        df_total['std'] = np.sqrt(ex2_tmp - df_total['mean']**2)
        df_total['skew'] = (ex3_tmp - 3 * df_total['mean'] * df_total['std']**2 - df_total['mean']**3) / df_total['std']**3
        df_total['kurtosis'] = (ex4_tmp - 4 * df_total['mean'] * ex3_tmp - 3 * df_total['mean']**4 + 6 * df_total['mean']**2 * ex2_tmp) / df_total['std']**4 - 3

        for col in ['mean', 'std']:
            df_total[col] = df_total[col] * 10000
        for col in ['mean','std','skew','kurtosis']:
            df_total[col] = df_total[col].map(lambda x: "{:.2f}".format(x))
        df_total = df_total.drop(tmp_cols, axis=1)
        fig = go.Figure(data=[go.Table(header=dict(values=[f'<b>{col}</b>' for col in df_total.columns],align=['center', 'center'],font_size=14,height=30),
                      cells=dict(values=[df_total[col] for col in df_total.columns], align=['center', 'center'],
                                font_size=14,height=30))])
        fig.update_layout(font_family='sans-serif', title_text='yHat General Stats', title_x=0.5, title_yanchor='top', 
                          width=self.html_width, height=(1+len(df_total)) * 30 + 60, autosize=False, 
                          margin=dict(t=40, b=10, l=10, r=10))

        return io.to_html(fig, full_html=False, include_plotlyjs='cdn')
    
    def sta_all_dist_html(self):
        df_hist = pd.DataFrame(self.daily_stat.hist_x.tolist()).sort_index(axis=1)
        df_hist = pd.concat([self.daily_stat[['sta_cat']], df_hist], axis=1).groupby('sta_cat').sum().unstack().reset_index()
        df_hist.columns = ['index', 'sta_cat', 'value']
        df_hist['value'] = df_hist['value'] / df_hist.groupby('sta_cat')['value'].transform('sum')
        df_hist['group'] = np.where(df_hist['sta_cat'].str.startswith(self.target_ret), 'target', 'alpha')
        df_hist = df_hist.sort_values(['sta_cat', 'index'])

        fig = px.line(df_hist, x='index', y='value',line_dash='group', color='sta_cat', height=500, width=self.html_width,
                      labels={'index': 'return (bps)', 'value': 'probability density'},
                      hover_data={'group':False, 'sta_cat': False, 'value': ':.3f'},
                      category_orders={'sta_cat_prefix':[self.target_ret]})
        fig.update_layout(font_family='sans-serif', legend=dict(title_text="", bgcolor="LightSteelBlue"))

        return self.to_html(fig)

    def sta_cutoff_summary_html(self):
        df_total = self.daily_stat.groupby(['sta_cat', 'exchange', 'side'], observed=True)[['countOppo', 'topPercent']].mean()
        
        for col in ['yHatHurdle', 'yHatAvg', 'vwActualRetAvg']:
            df_total[col] = self.daily_stat.groupby(['sta_cat', 'exchange', 'side'], 
                                                    observed=True).apply(lambda x: weighted_average(x[col], weights=x['availNtlSum']))
        df_total['availNtl'] = self.daily_stat.groupby(['sta_cat', 'exchange', 'side'], 
                                                       observed=True).apply(lambda x: weighted_average(x['availNtlAvg'], weights=x['countOppo']))

        df_total.reset_index(inplace=True)

        if self.eval_focus == 'oppo':
            underlying_ls = ['countOppo']
        elif self.eval_focus == 'ret':
            underlying_ls = ['vwActualRetAvg']
        else:
            underlying_ls = ['countOppo', 'vwActualRetAvg']
        for underlying in underlying_ls:
            df_total['base'] = np.where(df_total['sta_cat'] == df_total['sta_cat'].min(), df_total[underlying], np.nan)
            df_total['base'] = df_total.groupby(['exchange', 'side'])['base'].ffill()
            df_total['base'] = df_total.groupby(['exchange', 'side'])['base'].bfill()
            
            df_total[f'{underlying}<br>improvement(%)'] = (df_total[underlying]/df_total['base'] - 1) * 100

        df_total['topPercent'] = df_total['topPercent'] * 100
        df_total.rename(columns={'topPercent':'topPercent(%)'}, inplace=True)
        
        for col in ['yHatAvg', 'yHatHurdle', 'vwActualRetAvg','availNtl', 'countOppo', 'topPercent(%)'] + \
                   [f'{underlying}<br>improvement(%)' for underlying in underlying_ls]:
            df_total[col] = df_total[col].map(lambda x: "{:.2f}".format(x))

        # add bold font
        for underlying in underlying_ls:
            df_total[f'{underlying}<br>improvement(%)'] = "<b>" + df_total[f'{underlying}<br>improvement(%)'].astype(str) + "</b>"

        df_total = df_total.drop("base", axis=1)

        fig = go.Figure(data=[go.Table(header=dict(values=[f'<b>{col}</b>' for col in df_total.columns],align=['center', 'center'],font_size=14,height=30),
                      cells=dict(values=[df_total[col] for col in df_total.columns], align=['center', 'center'],
                                font_size=14,height=30))])
        fig.update_layout(font_family='sans-serif', title_text='overall performance outlook', title_x=0.5, title_yanchor='top', 
                          width=self.html_width, height=(1+len(df_total)) * 30 + 120, autosize=False, 
                          margin=dict(t=40, b=10, l=10, r=10))

        return io.to_html(fig, full_html=False, include_plotlyjs='cdn')

    def performance_by_side_html(self):
        df_side = self.daily_stat.groupby(['side','sta_cat'], observed=True).apply(lambda x: weighted_average(x['vwActualRetAvg'], weights=x['availNtlSum']))
        df_side.name = 'vwActualRetAvg'
        df_side = df_side.reset_index()
        fig = px.bar(df_side, x='side', y='vwActualRetAvg', color='sta_cat', barmode='group', height=500, width=self.html_width,
                        labels={'vwActualRetAvg': 'return (bps)'},
                        hover_data={'sta_cat':False, 'vwActualRetAvg': ':.2f'})
        fig.update_layout(font_family='sans-serif',
                          title=dict(text='all side - '+self.target_ret, x=0.5, yanchor='top',font_size=18),
                          legend=dict(title_text="", bgcolor="LightSteelBlue"))

        return self.to_html(fig)

    def opportunities_by_side_html(self):
        df_side = self.daily_stat.groupby(['side','sta_cat'], observed=True)['countOppo'].mean()
        df_side = df_side.reset_index()
        fig = px.bar(df_side, x='side', y='countOppo', color='sta_cat', barmode='group', height=500, width=self.html_width,
                        labels={'countOppo': 'num of oppo'},
                        hover_data={'sta_cat':False, 'countOppo': ':.2f'})
        fig.update_layout(font_family='sans-serif',
                          title=dict(text='all side - '+'countOppo', x=0.5, yanchor='top',font_size=18),
                          legend=dict(title_text="", bgcolor="LightSteelBlue"))

        return self.to_html(fig)
    
    def performance_by_price_group_html(self):
        reports = []
        for side in ['buy','sell']:
            df_total = (self.daily_stat.groupby(['side','sta_cat', 'price_group'], observed=True)
                            .apply(lambda x: weighted_average(x['vwActualRetAvg'], weights=x['availNtlSum'])))
            df_total.name = 'vwActualRetAvg'
            df_total = df_total.reset_index()

            df_side = df_total[df_total.side == side]
            df_group_number = self.df_daily.groupby(['date','price_group'])['skey'].nunique().reset_index().groupby(['price_group'])['skey'].mean()
            df_side = df_side.merge(df_group_number.reset_index(), on = 'price_group')
            df_side['skey'] = df_side['skey'].map(lambda x:'('+str(int(x))+' stocks)')
            df_side['price_group'] = df_side['price_group'] + df_side['skey']
            
            df_side['price_start'] = df_side['price_group'].map(lambda x:int(x.split('-')[0]))
            df_side = df_side.sort_values(['price_start', 'sta_cat']).reset_index(drop = True)
            df_side.drop(columns = ['price_start'], inplace = True)
            fig = px.bar(df_side, x='price_group', y='vwActualRetAvg', color='sta_cat', barmode='group', height=500, width=self.html_width,
                         labels={'price_group': 'price group', 'vwActualRetAvg': 'return (bps)'},
                         hover_data={'sta_cat':False, 'vwActualRetAvg': ':.2f'})
            fig.update_layout(font_family='sans-serif',
                              title=dict(text=side+' side - '+self.target_ret, x=0.5, yanchor='top',font_size=18),
                              legend=dict(title_text="", bgcolor="LightSteelBlue"))
            reports.append(self.to_html(fig))

        return '\n'.join(reports)

    def opportunities_by_price_group_html(self):
        reports = []
        for side in ['buy','sell']:
            df_total = (self.daily_stat.groupby(['side','price_group', 'sta_cat'], observed=True)['countOppo'].mean())
            df_total = df_total.reset_index()

            df_side = df_total[df_total.side == side]
            df_group_number = self.df_daily.groupby(['date','price_group'])['skey'].nunique().reset_index().groupby(['price_group'])['skey'].mean()
            df_side = df_side.merge(df_group_number.reset_index(), on = 'price_group')
            df_side['skey'] = df_side['skey'].map(lambda x:'('+str(int(x))+' stocks)')
            df_side['price_group'] = df_side['price_group'] + df_side['skey']
            
            df_side['price_start'] = df_side['price_group'].map(lambda x:int(x.split('-')[0]))
            df_side = df_side.sort_values(['price_start', 'sta_cat']).reset_index(drop = True)
            df_side.drop(columns = ['price_start'], inplace = True)
            fig = px.bar(df_side, x='price_group', y='countOppo', color='sta_cat', barmode='group', height=500, width=self.html_width,
                         labels={'price_group': 'price group', 'countOppo': 'num of oppo'},
                         hover_data={'sta_cat':False, 'countOppo': ':.2f'})
            fig.update_layout(font_family='sans-serif',
                              title=dict(text=side+' side - '+'countOppo', x=0.5, yanchor='top',font_size=18),
                              legend=dict(title_text="", bgcolor="LightSteelBlue"))
            reports.append(self.to_html(fig))

        return '\n'.join(reports)

    def daily_yHatHurdle_html(self):
        reports = []
        for side in ['buy','sell']:
            for ex in ['SH','SZ']:
                df_top = (self.daily_stat[(self.daily_stat.side == side) & (self.daily_stat.exchange == ex)]
                                        .groupby(['sta_cat','datetime'], observed=True)['yHatHurdle'].mean().reset_index())
                if len(df_top) == 0:
                    continue
                fig = px.line(df_top, x='datetime', y='yHatHurdle', color='sta_cat', height=600, width=self.html_width,
                              labels={'datetime': '', 'yHatHurdle': 'yHatHurdle (bps)'},
                              hover_data={'sta_cat':False, 'yHatHurdle': ':.2f'})
                fig.update_layout(font_family='sans-serif',
                                  title=dict(text=side+' side - '+ex+' - yHatHurdle', x=0.5, yanchor='top',font_size=18),
                                  legend=dict(title_text="", bgcolor="LightSteelBlue"))
                fig.update_xaxes(range=(df_top['datetime'].min() - pd.offsets.Day(5), df_top['datetime'].max() + pd.offsets.Day(5)),
                                 rangeslider_visible=True)
                reports.append(self.to_html(fig))

        return '\n'.join(reports)

    def monthly_realized_return_html(self):
        reports = []
        stats_horizon = 'month'
        df_stats = self.daily_stat.groupby([stats_horizon,'exchange','side','sta_cat'], observed=True)[['datetime']].first()
        df_stats['vwActualRetAvg'] = (self.daily_stat.groupby([stats_horizon,'exchange','side','sta_cat'], observed=True)
                                                     .apply(lambda x: weighted_average(x['vwActualRetAvg'], weights=x['availNtlSum'])))
        df_stats = df_stats.reset_index()

        for side in ['buy','sell']:
            for ex in ['SH','SZ']:
                df_hist = df_stats.loc[(df_stats.side == side) & 
                                    (df_stats.exchange == ex), [stats_horizon,'sta_cat'] + ['vwActualRetAvg']].reset_index()
                if len(df_hist) == 0:
                    continue
                fig = px.bar(df_hist, x=stats_horizon, y='vwActualRetAvg', color='sta_cat', barmode='group', 
                             height=500, width=self.html_width,labels={'vwActualRetAvg': 'return (bps)'},
                             hover_data={'sta_cat':False, 'vwActualRetAvg': ':.2f'})
                fig.update_layout(font_family='sans-serif',
                                  title=dict(text=side+' side - '+ex+' - '+self.target_ret, x=0.5, yanchor='top',font_size=18),
                                  legend=dict(title_text="", bgcolor="LightSteelBlue"))
                reports.append(self.to_html(fig))

        return '\n'.join(reports)

    def monthly_opportunities_html(self):
        reports = []
        stats_horizon = 'month'
        df_stats = self.daily_stat.groupby([stats_horizon,'exchange','side','sta_cat'], observed=True)[['datetime']].first()
        df_stats['countOppo'] = (self.daily_stat.groupby([stats_horizon,'exchange','side','sta_cat'], observed=True)['countOppo'].mean())
        df_stats = df_stats.reset_index()

        for side in ['buy','sell']:
            for ex in ['SH','SZ']:
                df_hist = df_stats.loc[(df_stats.side == side) & 
                                    (df_stats.exchange == ex), [stats_horizon,'sta_cat'] + ['countOppo']].reset_index()
                if len(df_hist) == 0:
                    continue
                fig = px.bar(df_hist, x=stats_horizon, y='countOppo', color='sta_cat', barmode='group', 
                             height=500, width=self.html_width,labels={'countOppo': 'num of oppo'},
                             hover_data={'sta_cat':False, 'countOppo': ':.2f'})
                fig.update_layout(font_family='sans-serif',
                                  title=dict(text=side+' side - '+ex+' - '+'countOppo', x=0.5, yanchor='top',font_size=18),
                                  legend=dict(title_text="", bgcolor="LightSteelBlue"))
                reports.append(self.to_html(fig))

        return '\n'.join(reports)

    def daily_realized_return_html(self):
        reports = []
        df_stats = self.daily_stat.groupby(['date','exchange','side','sta_cat'], observed=True)[['datetime']].first()
        df_stats['vwActualRetAvg'] = (self.daily_stat.groupby(['date','exchange','side','sta_cat'], observed=True)
                                                    .apply(lambda x: weighted_average(x['vwActualRetAvg'], weights=x['availNtlSum'])))
        df_stats = df_stats.reset_index()
        for side in ['buy','sell']:
            for ex in ['SH','SZ']:
                df_hist = (df_stats.loc[(df_stats.side == side) & (df_stats.exchange == ex), ['datetime','sta_cat'] + ['vwActualRetAvg']].reset_index())
                if len(df_hist) == 0:
                    continue
                
                fig = px.line(df_hist, x='datetime', y='vwActualRetAvg', color='sta_cat', height=600, width=self.html_width,
                              labels={'datetime': 'date', 'vwActualRetAvg': 'return (bps)'},
                              hover_data={'sta_cat':False, 'vwActualRetAvg': ':.2f'})
                fig.update_layout(font_family='sans-serif',
                                  title=dict(text=side+' side - '+ex+' - '+self.target_ret, x=0.5, yanchor='top',font_size=18),
                                  legend=dict(title_text="", bgcolor="LightSteelBlue"))
                fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor='grey')
                fig.update_xaxes(range=(df_hist['datetime'].min() - pd.offsets.Day(5), df_hist['datetime'].max() + pd.offsets.Day(5)),
                                 rangeslider_visible=True)
                reports.append(self.to_html(fig))
        
        return '\n'.join(reports)

    def daily_opportunities_html(self):
        reports = []
        df_stats = self.daily_stat.groupby(['date','exchange','side','sta_cat'], observed=True)[['datetime']].first()
        df_stats['countOppo'] = (self.daily_stat.groupby(['date','exchange','side','sta_cat'], observed=True)['countOppo'].mean())
        df_stats = df_stats.reset_index()
        for side in ['buy','sell']:
            for ex in ['SH','SZ']:
                df_hist = (df_stats.loc[(df_stats.side == side) & (df_stats.exchange == ex), ['datetime','sta_cat'] + ['countOppo']].reset_index())
                if len(df_hist) == 0:
                    continue
                
                fig = px.line(df_hist, x='datetime', y='countOppo', color='sta_cat', height=600, width=self.html_width,
                              labels={'datetime': 'date', 'countOppo': 'num of oppo'},
                              hover_data={'sta_cat':False, 'countOppo': ':.2f'})
                fig.update_layout(font_family='sans-serif',
                                  title=dict(text=side+' side - '+ex+' - '+'countOppo', x=0.5, yanchor='top',font_size=18),
                                  legend=dict(title_text="", bgcolor="LightSteelBlue"))
                fig.update_xaxes(range=(df_hist['datetime'].min() - pd.offsets.Day(5), df_hist['datetime'].max() + pd.offsets.Day(5)),
                                 rangeslider_visible=True)
                reports.append(self.to_html(fig))
        
        return '\n'.join(reports)

    def intraday_opportunities_html(self):
        ylabel = 'num of oppo'
        value = 'countOppo'

        reports = []
        for side in ['buy','sell']:
            for ex in ['SH','SZ']:
                df_intra = (self.intraday_stat.loc[(self.intraday_stat.side == side) & (self.intraday_stat.exchange == ex), 
                                                    ['mins_since_open','sta_cat'] + [value]]
                                              .sort_values(['mins_since_open','sta_cat']).reset_index())
                if len(df_intra) == 0:
                    continue
                fig = px.line(df_intra, x='mins_since_open', y=value, color='sta_cat', height=600, width=self.html_width,
                             labels={'mins_since_open': '', value: ylabel},
                             hover_data={'sta_cat':False, 'mins_since_open':False, value: ':.2f'})
                fig.update_layout(font_family='sans-serif',
                                  title=dict(text=side+' side - '+ex+' - '+self.target_cut, x=0.5, yanchor='top',font_size=18),
                                  legend=dict(title_text="", bgcolor="LightSteelBlue"))
                fig.update_xaxes(tickmode='array', tickvals=list(range(0,240,30)),
                                 ticktext=['09:30', '10:00', '10:30', '11:00', '11:30 / 13:00', "13:30", "14:00", "14:30"],
                                 range=(-5, 240),
                                 rangeslider_visible=True)
                reports.append(self.to_html(fig))
        
        return '\n'.join(reports)

    def intraday_realized_return_html(self):
        ylabel = 'return (bps)'
        value = 'vwActualRetAvg'

        reports = []
        for side in ['buy','sell']:
            for ex in ['SH','SZ']:
                df_intra = (self.intraday_stat.loc[(self.intraday_stat.side == side) & (self.intraday_stat.exchange == ex), 
                                                    ['mins_since_open','sta_cat'] + [value]]
                                              .sort_values(['mins_since_open', 'sta_cat']).reset_index())
                if len(df_intra) == 0:
                    continue
                fig = px.line(df_intra, x='mins_since_open', y=value, color='sta_cat', height=600, width=self.html_width,
                              labels={'mins_since_open': '', value: ylabel},
                              hover_data={'sta_cat':False, 'mins_since_open':False, value: ':.2f'})
                fig.update_layout(font_family='sans-serif',
                                  title=dict(text=side+' side - '+ex+' - '+self.target_cut, x=0.5, yanchor='top',font_size=18),
                                  legend=dict(title_text="", bgcolor="LightSteelBlue"))
                fig.update_xaxes(tickmode='array', tickvals=list(range(0,240,30)),
                                 ticktext=['09:30', '10:00', '10:30', '11:00', '11:30 / 13:00', "13:30", "14:00", "14:30"],
                                 range=(-5,240),
                                 rangeslider_visible=True)
                fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor='grey')
                reports.append(self.to_html(fig))
        
        return '\n'.join(reports)

def main(sta_input):
    # sta_input = '/home/marlowe/Marlowe/eva/sta_input_ps.yaml'
    sta_eval_run = StaAlphaEvalReduce(sta_input)
    
    sta_eval_run.alpha_eval()

if __name__ == "__main__":
    main(sys.argv[1])