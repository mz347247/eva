import os
import re
from collections import defaultdict
import pandas as pd
import numpy as np
from utils import *
from AShareReader import AShareReader
from CephClient import CephClient
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
from glob import glob
from matplotlib.backends.backend_pdf import PdfPages
import plotly.io as io
import plotly.express as px
import plotly.graph_objects as go

import yaml

perc = [.01, .05, .1, .25, .5, .75, .9, .95, .99]

class sta_alpha_eval():
    def __init__(self, sta_input):
        f = open(sta_input, 'r', encoding='utf-8')
        cfg = f.read()
        dict_yaml = yaml.full_load(cfg)
        f.close()

        self.machine = dict_yaml['machine']
        self.bench = dict_yaml['bench']
        self.start_date = str(dict_yaml['start_date'])
        self.end_date = str(dict_yaml['end_date'])
        self.eval_alpha = dict_yaml['eval_alpha']
        self.target_ret = dict_yaml['target_return']
        self.target_cut = dict_yaml['target_cut']
        self.lookback_window = dict_yaml['lookback_window']
        self.eval_path = os.path.join(dict_yaml['save_path'], self.bench, self.eval_alpha[-1])
        self.cutoff_path = os.path.join(self.eval_path, f'sta_{self.target_cut}')
        try:
            self.target_horizon = int(re.search(r"\d+", self.target_ret).group(0))
        except (AttributeError, ValueError):
            raise ValueError(f"Invalid Input target_ret: {self.target_ret}") 

        try:
            self.target_number = int(4800 * int(re.search(r"\d+", self.target_cut).group(0)) / 100)
        except (AttributeError, ValueError):
            raise ValueError(f"Invalid Input target_cut: {self.target_cut}") 
        
        self.eval_alpha_dict = defaultdict(list)
        for alpha in self.eval_alpha:
            self.eval_alpha_dict[alpha.split("_")[-1]].append(alpha)

        self.pdf_width = 30
        self.html_width = 1400
        self.template = 'seaborn'
        self.to_html = partial(io.to_html, include_plotlyjs='cdn', full_html=False)
        self.color = ['lightslategrey', 'steelblue', 'lightskyblue', 'paleturquoise', 'azure']


    def alpha_eval(self):
        a = AShareReader(dll_path = '{0}/ceph_client/ceph-client.so'.format(os.environ['HOME']), 
                     config_path='{0}/dfs/ceph.conf'.format(os.environ['HOME']),
                     KEYRING_LOC = '{0}/dfs/ceph.key.keyring'.format(os.environ['HOME']))

        self.daily_stat = pd.concat([pd.read_pickle(path) for path in glob(f"{self.cutoff_path}/daily/*.pkl")], 
                                  ignore_index=True)
        self.daily_stat['datetime'] = pd.to_datetime(self.daily_stat['date'].astype('str'))
        self.daily_stat['month'] = self.daily_stat['date'].map(lambda x:str(x)[:6])
        self.daily_stat['year'] = self.daily_stat['date'].map(lambda x:str(x)[:4])

        stock_list = self.daily_stat.skey.unique()
        self.df_daily = a.Read_Stock_Daily('com_md_eq_cn', 'mdbar1d_jq', start_date=self.daily_stat.date.min(),
                                           end_date=self.daily_stat.date.max(), stock_list=stock_list)
        self.df_daily['price_group'] = np.where(self.df_daily['close'] < 5, '0-5', np.where(self.df_daily['close'] < 10, '5-10', 
                                       np.where(self.df_daily['close'] < 20, '10-20', np.where(self.df_daily['close'] < 50, '20-50', 
                                       np.where(self.df_daily['close'] < 100, '50-100', '100-')))))

        self.daily_stat = self.daily_stat.merge(self.df_daily[['skey','date','price_group']], on = ['skey','date'], 
                                                how = 'left', validate = 'many_to_one')

        df_intraday = pd.concat([pd.read_pickle(path) for path in glob(f"{self.cutoff_path}/intraday/*.pkl")], 
                                   ignore_index=True)

        self.intraday_stat = df_intraday.groupby(['exchange','side','sta_cat','mins_since_open'])[['countOppo']].mean()
        self.intraday_stat['vwActualRetAvg'] = (df_intraday.groupby(['exchange','side','sta_cat','mins_since_open'])
                                                             .apply(lambda x: weighted_average(x['vwActualRetAvg'], weights=x['availNtlSum'])))
        self.intraday_stat['vwActualRetAvg'] = round(self.intraday_stat['vwActualRetAvg'] * 10000, 2)
        self.intraday_stat = self.intraday_stat.reset_index()
        
        # self.pdf_report()
        self.html_report()

    def html_report(self):
        px.defaults.template = 'seaborn'
        report_name = '_'.join([self.bench, self.eval_alpha[-1], self.target_cut, self.start_date, self.end_date]) + '.html'
        with open(os.path.join(self.eval_path, report_name), 'w') as f:
            f.write('''<html>\n<head><meta charset="utf-8" /></head>\n<body>\n''')

            # sta all summary
            table1 = self.sta_all_summary_html()
            f.write(table1 + '\n')

            # sta cutoff summary
            table2 = self.sta_cutoff_summary_html()
            f.write(table2 + '\n')
            
            # performance by side
            f.write("<h2>performance for different side</h2>" + "\n")
            figs = self.performance_by_side_html()
            f.write(figs + '\n')

            # performance by price group
            f.write("<h2>performance for different price group</h2>" + "\n")
            figs = self.performance_by_price_group_html()
            f.write(figs + '\n')

            # daily yHatAvg
            f.write(f"<h2>{self.target_cut} yHatHurdle</h2>" + "\n")
            figs = self.daily_yHatHurdle_html()
            f.write(figs + '\n')

            # monthly realized return
            f.write("<h2>monthly realized return</h2>" + "\n")
            figs = self.monthly_realized_return_html()
            f.write(figs + '\n')

            # daily realized return
            f.write("<h2>daily realized return</h2>" + "\n")
            figs = self.daily_realized_return_html()
            f.write(figs + '\n')

            # intraday number of opportunities
            f.write("<h2>daily average oppo. across minutes since open</h2>" + "\n")
            figs = self.intraday_opportunities_html()
            f.write(figs + '\n')

            # daily realized return
            f.write("<h2>daily average return across minutes since open</h2>" + "\n")
            figs = self.intraday_realized_return_html()
            f.write(figs + '\n')
            
            f.write('''\n</body>\n</html>''')

    def pdf_report(self):
        report_name = '_'.join([self.bench, self.eval_alpha[-1], self.target_cut, self.start_date, self.end_date]) + '.pdf'
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
        df_total = self.daily_stat.groupby(['sta_cat','exchange','side'])[['mean','std','skew','kurtosis']].mean().reset_index()
        for col in ['mean','std','skew','kurtosis']:
            df_total[col] = df_total[col].map(lambda x: "{:.2f}".format(x))
        fig = go.Figure(data=[go.Table(header=dict(values=[f'<b>{col}</b>' for col in df_total.columns],align=['center', 'center'],font_size=14,height=30),
                      cells=dict(values=[df_total[col] for col in df_total.columns], align=['center', 'center'],
                                font_size=14,height=30))])
        fig.update_layout(font_family='sans-serif', title_text='yHat General Stats', title_x=0.5, title_yanchor='top', 
                          width=self.html_width, height=220, autosize=False, margin=dict(t=40, b=10, l=10, r=10))

        return io.to_html(fig, full_html=False, include_plotlyjs='cdn')
    
    def sta_cutoff_summary_html(self):
        df_total = self.daily_stat.groupby(['sta_cat', 'exchange', 'side'])[['countOppo']].mean()
        
        for col in ['yHatHurdle', 'yHatAvg', 'vwActualRetAvg']:
            df_total[col] = self.daily_stat.groupby(['sta_cat', 'exchange', 'side']).apply(lambda x: weighted_average(x[col], weights=x['availNtlSum']))
        df_total['availNtl'] = self.daily_stat.groupby(['sta_cat', 'exchange', 'side']).apply(lambda x: weighted_average(x['availNtlAvg'], weights=x['countOppo']))

        df_total.reset_index(inplace=True)

        df_total['base_ret'] = np.where(df_total['sta_cat'] == self.eval_alpha[0], df_total['vwActualRetAvg'], np.nan)
        df_total['base_ret'] = df_total.groupby(['exchange', 'side'])['base_ret'].ffill()
        df_total['base_ret'] = df_total.groupby(['exchange', 'side'])['base_ret'].bfill()

        df_total['improvement(%)'] = (df_total['vwActualRetAvg']/df_total['base_ret'] - 1) * 100

        for col in ['yHatAvg', 'yHatHurdle','vwActualRetAvg','improvement(%)','availNtl', 'countOppo']:
            df_total[col] = df_total[col].map(lambda x: "{:.2f}".format(x))

        df_total = df_total.drop("base_ret", axis=1)

        fig = go.Figure(data=[go.Table(header=dict(values=[f'<b>{col}</b>' for col in df_total.columns],align=['center', 'center'],font_size=14,height=30),
                      cells=dict(values=[df_total[col] for col in df_total.columns], align=['center', 'center'],
                                font_size=14,height=30))])
        fig.update_layout(font_family='sans-serif', title_text='overall performance outlook', title_x=0.5, title_yanchor='top', 
                          width=self.html_width, height=220, autosize=False, margin=dict(t=40, b=10, l=10, r=10))

        return io.to_html(fig, full_html=False, include_plotlyjs='cdn')

    def performance_by_side_html(self):
        df_side = self.daily_stat.groupby(['side','sta_cat']).apply(lambda x: weighted_average(x['vwActualRetAvg'], weights=x['availNtlSum']))
        df_side.name = 'vwActualRetAvg'
        df_side = df_side.reset_index()
        fig = px.bar(df_side, x='side', y='vwActualRetAvg', color='sta_cat', barmode='group', height=500, width=self.html_width,
                        labels={'vwActualRetAvg': 'return (bps)'},
                        hover_data={'sta_cat':False, 'vwActualRetAvg': ':.2f'})
        fig.update_layout(font_family='sans-serif',
                          title=dict(text='all side - '+'actualRet90s', x=0.5, yanchor='top',font_size=18),
                          legend=dict(title_text="", bgcolor="LightSteelBlue"))

        return self.to_html(fig)
    
    def performance_by_price_group_html(self):
        reports = []
        for side in ['buy','sell']:
            df_total = (self.daily_stat.groupby(['side','price_group', 'sta_cat'])
                            .apply(lambda x: weighted_average(x['vwActualRetAvg'], weights=x['availNtlSum'])))
            df_total.name = 'vwActualRetAvg'
            df_total = df_total.reset_index()

            df_side = df_total[df_total.side == side]
            df_group_number = self.df_daily.groupby(['date','price_group'])['skey'].nunique().reset_index().groupby(['price_group'])['skey'].mean()
            df_side = df_side.merge(df_group_number.reset_index(), on = 'price_group')
            df_side['skey'] = df_side['skey'].map(lambda x:'('+str(int(x))+' stocks)')
            df_side['price_group'] = df_side['price_group'] + df_side['skey']
            
            df_side['price_start'] = df_side['price_group'].map(lambda x:int(x.split('-')[0]))
            df_side = df_side.sort_values('price_start').reset_index(drop = True)
            df_side.drop(columns = ['price_start'], inplace = True)
            fig = px.bar(df_side, x='price_group', y='vwActualRetAvg', color='sta_cat', barmode='group', height=500, width=self.html_width,
                         labels={'price_group': 'price group', 'vwActualRetAvg': 'return (bps)'},
                         hover_data={'sta_cat':False, 'vwActualRetAvg': ':.2f'})
            fig.update_layout(font_family='sans-serif',
                              title=dict(text=side+' side - '+'actualRet90s', x=0.5, yanchor='top',font_size=18),
                              legend=dict(title_text="", bgcolor="LightSteelBlue"))
            reports.append(self.to_html(fig))

        return '\n'.join(reports)

    def daily_yHatHurdle_html(self):
        reports = []
        for side in ['buy','sell']:
            for ex in ['SH','SZ']:
                df_top = (self.daily_stat[(self.daily_stat.side == side) & (self.daily_stat.exchange == ex)]
                                        .groupby(['sta_cat','datetime'])['yHatHurdle'].mean().reset_index())
                if len(df_top) == 0:
                    continue
                fig = px.line(df_top, x='datetime', y='yHatHurdle', color='sta_cat', height=500, width=self.html_width,
                              labels={'datetime': '', 'yHatHurdle': 'yHatHurdle (bps)'},
                              hover_data={'sta_cat':False, 'yHatHurdle': ':.2f'})
                fig.update_layout(font_family='sans-serif',
                                  title=dict(text=side+' side - '+ex+' - yHatHurdle', x=0.5, yanchor='top',font_size=18),
                                  legend=dict(title_text="", bgcolor="LightSteelBlue"))
                fig.update_xaxes(range=(df_top['datetime'].min() - pd.offsets.Day(5), df_top['datetime'].max() + pd.offsets.Day(5)))
                reports.append(self.to_html(fig))

        return '\n'.join(reports)

    def monthly_realized_return_html(self):
        reports = []
        stats_horizon = 'month'
        df_stats = self.daily_stat.groupby([stats_horizon,'exchange','side','sta_cat'])[['datetime']].first()
        df_stats['vwActualRetAvg'] = (self.daily_stat.groupby([stats_horizon,'exchange','side','sta_cat'])
                                                     .apply(lambda x: weighted_average(x['vwActualRetAvg'], weights=x['availNtlSum'])))
        df_stats = df_stats.reset_index()
        fig = plt.figure(figsize=(16, 20))
        plt.suptitle('monthly realized return', fontsize = 25)
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

    def daily_realized_return_html(self):
        reports = []
        df_stats = self.daily_stat.groupby(['date','exchange','side','sta_cat'])[['datetime']].first()
        df_stats['vwActualRetAvg'] = (self.daily_stat.groupby(['date','exchange','side','sta_cat'])
                                                    .apply(lambda x: weighted_average(x['vwActualRetAvg'], weights=x['availNtlSum'])))
        df_stats = df_stats.reset_index()
        for side in ['buy','sell']:
            for ex in ['SH','SZ']:
                df_hist = (df_stats.loc[(df_stats.side == side) & (df_stats.exchange == ex), ['datetime','sta_cat'] + ['vwActualRetAvg']].reset_index())
                if len(df_hist) == 0:
                    continue
                
                fig = px.line(df_hist, x='datetime', y='vwActualRetAvg', color='sta_cat', height=500, width=self.html_width,
                              labels={'datetime': '', 'vwActualRetAvg': 'return (bps)'},
                              hover_data={'sta_cat':False, 'vwActualRetAvg': ':.2f'})
                fig.update_layout(font_family='sans-serif',
                                  title=dict(text=side+' side - '+ex+' - '+self.target_ret, x=0.5, yanchor='top',font_size=18),
                                  legend=dict(title_text="", bgcolor="LightSteelBlue"))
                fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor='grey')
                fig.update_xaxes(range=(df_hist['datetime'].min() - pd.offsets.Day(5), df_hist['datetime'].max() + pd.offsets.Day(5)))
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
                                              .reset_index().sort_values('mins_since_open'))
                if len(df_intra) == 0:
                    continue
                fig = px.line(df_intra, x='mins_since_open', y=value, color='sta_cat', height=500, width=self.html_width,
                            labels={'mins_since_open': '', value: ylabel},
                            hover_data={'sta_cat':False, value: ':.2f'})
                fig.update_layout(font_family='sans-serif',
                                  title=dict(text=side+' side - '+ex+' - '+self.target_cut, x=0.5, yanchor='top',font_size=18),
                                  legend=dict(title_text="", bgcolor="LightSteelBlue"))
                fig.update_xaxes(tickmode='array', tickvals=list(range(0,240,30)),
                                 ticktext=['09:30', '10:00', '10:30', '11:00', '11:30 / 13:00', "13:30", "14:00", "14:30"],
                                 range=(-5, 240))
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
                                              .reset_index().sort_values('mins_since_open'))
                if len(df_intra) == 0:
                    continue
                fig = px.line(df_intra, x='mins_since_open', y=value, color='sta_cat', height=500, width=self.html_width,
                              labels={'mins_since_open': '', value: ylabel},
                              hover_data={'sta_cat':False, value: ':.2f'})
                fig.update_layout(font_family='sans-serif',
                                  title=dict(text=side+' side - '+ex+' - '+self.target_cut, x=0.5, yanchor='top',font_size=18),
                                  legend=dict(title_text="", bgcolor="LightSteelBlue"))
                fig.update_xaxes(tickmode='array', tickvals=list(range(0,240,30)),
                                 ticktext=['09:30', '10:00', '10:30', '11:00', '11:30 / 13:00', "13:30", "14:00", "14:30"],
                                 range=(-5,240))
                fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor='grey')
                reports.append(self.to_html(fig))
        
        return '\n'.join(reports)

if __name__ == "__main__":
    sta_input = '/home/marlowe/Marlowe/eva/sta_input_demo.yaml'
    sta_eval_run = sta_alpha_eval(sta_input)
    
    sta_eval_run.alpha_eval()