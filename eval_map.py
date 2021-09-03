import os
import re
import sys
from collections import defaultdict
import pandas as pd
import numpy as np
from utils import *
from AShareReader import AShareReader
from CephClient import CephClient
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
from HPCutils import get_job_list
from eval import StaAlphaEval

class StaAlphaEvalMap(StaAlphaEval):

    def __init__(self, sta_input):
        super().__init__(sta_input)

        try:
            self.target_horizon = int(re.search(r"\d+", self.target_ret).group(0))
        except (AttributeError, ValueError):
            raise ValueError(f"Invalid Input target_ret: {self.target_ret}") 

        # top5p or top240
        self._target_number = self._target_ratio = self._target_return = None
        if self.target_cut.endswith("p"):
            try:
                self._target_ratio = int(re.search(r"\d+", self.target_cut).group(0)) / 100
                assert self._target_ratio < 1
            except (AttributeError, ValueError):
                raise ValueError(f"Invalid Input target_cut: {self.target_cut}")
        elif self.target_cut.endswith("bps"):
            try:
                self._target_return = int(re.search(r"\d+", self.target_cut).group(0))
            except (AttributeError, ValueError):
                raise ValueError(f"Invalid Input target_cut: {self.target_cut}")
        else:
            try:
                self._target_number = int(re.search(r"\d+", self.target_cut).group(0))
            except (AttributeError, ValueError):
                raise ValueError(f"Invalid Input target_cut: {self.target_cut}")
        
        self.eval_alpha_dict = defaultdict(list)
        for alpha in self.eval_alpha:
            self.eval_alpha_dict[alpha.split("_")[-1]].append(alpha)

    def generate_daily_sta_cutoff(self, date):
        df_daily = []
        df_intra = []

        # loop over alphas from mbd and lv2
        for sta_type, sta_ls in self.eval_alpha_dict.items():
            buy_sta_cols = []
            sell_sta_cols = []
            for ix, sta in enumerate(sta_ls):
                tmp_sta = self.sta_reader.read_file(f"/sta_alpha_eq_cn/{sta}/IC/sta{date}.parquet", "sta_alpha_eq_cn", sta)
                tmp_sta = tmp_sta.rename(columns={"yHatBuy": f"yHatBuy_{sta[4:]}", "yHatSell": f"yHatSell_{sta[4:]}"})
                buy_sta_cols.append(f"yHatBuy_{sta[4:]}")
                sell_sta_cols.append(f"yHatSell_{sta[4:]}")
                if ix == 0:
                    df_sta = tmp_sta
                else:
                    df_sta = pd.merge(df_sta, tmp_sta, on=['skey', 'date', 'ordering'], how='outer', validate='one_to_one')
            del tmp_sta

            universe = self.stock_reader.Read_Stock_Daily('com_md_eq_cn', f"chnuniv_{self.bench.lower()}", date, date).skey.unique()
            df_sta = df_sta[df_sta.skey.isin(universe)].reset_index(drop=True)

            # TODO: no need to read ask5q and bid5q if we have near limit status in DFS
            # TODO: tempararily modify this
            df_md = self.stock_reader.Read_Stock_Tick('com_md_eq_cn', f'md_snapshot_{sta_type}', start_date=date, end_date=date, 
                                       stock_list=universe, cols_list=['skey', 'date', 'time', 'clockAtArrival', 'ordering', 
                                                                       'ask1p', 'ask1q', 'bid1p', 'bid1q', 'ask5q', 'bid5q',
                                                                       'cum_volume', 'cum_amount', 'ApplSeqNum'])
            # if sta_type == 'mbd':
            #     tmp = self.stock_reader.Read_Stock_Tick('com_md_eq_cn', f'md_snapshot_l2', start_date=date, end_date=date, 
            #                                             stock_list=universe, cols_list=['skey', 'date','ApplSeqNum'])
            #     df_md = df_md[df_md.ApplSeqNum >= 0]
            #     tmp = tmp[tmp.ApplSeqNum >= 0]
            #     df_md = pd.merge(tmp, df_md, how='left', on=['skey', 'date','ApplSeqNum'], validate='many_to_one')
            #     df_md = df_md.drop_duplicates(subset=['skey', 'date', 'ordering']).sort_values(['skey', 'time'])
            #     del tmp
            
            df_md = df_md[(df_md.bid1p != 0) | (df_md.ask1p != 0)].reset_index(drop=True)
            
            # TODO: modify this when true return and near limit status is ready in DFS
            df_md['datetime'] = (df_md.date.astype(str).apply(lambda x: x[:4] + "-" + x[4:6] + "-" + x[6:]) + " " + 
                                 df_md.time.astype(int).astype(str).str.zfill(12).apply(lambda x: x[:2] + ":" + x[2:4] + ":" + x[4:6]))
            df_md['datetime'] = pd.to_datetime(df_md['datetime'])
            df_md = _get_eva_md(df_md, self.target_horizon, self.lookback_window)

            df_md = df_md[((df_md['time'] >= 9.3 * 1e10) & (df_md['time'] <= 11.3 * 1e10)) |
                          ((df_md['time'] >= 13 * 1e10) & (df_md['time'] < 14.5655 * 1e10))].reset_index(drop=True)

            df_alpha = df_md.merge(df_sta, on = ['skey','date','ordering'], how = 'left', validate = 'one_to_one')
            df_alpha = df_alpha.sort_values(['skey','date','ordering']).reset_index(drop=True)
            del df_md, df_sta

            df_alpha['exchange'] = df_alpha['skey'].map(lambda x:str(x)[:1])
            df_alpha['exchange'] = np.where(df_alpha['exchange'] == '1', 'SH', 'SZ')

            # TODO: modify this when we have SH mbd data
            # only compare SZ data if we compare mbd
            if "mbd" in self.eval_alpha_dict:
                df_alpha = df_alpha[df_alpha.exchange=="SZ"].reset_index(drop=True)

            df_alpha['minute'] = df_alpha['datetime'].dt.hour * 60 + df_alpha['datetime'].dt.minute
            df_alpha['mins_since_open'] = np.where(df_alpha['minute'] <= 690, df_alpha['minute'] - 570, df_alpha['minute'] - 660)
            df_alpha['buyAvailNtl'] = df_alpha['ask1p'] * df_alpha['ask1q']
            df_alpha['sellAvailNtl'] = df_alpha['bid1p'] * df_alpha['bid1q']
            basic_cols = ['skey','date','time','exchange','mins_since_open',
                          'nearLimit','cum_volume','cum_amount']
            for side in ['buy', 'sell']:
                if side == 'buy': 
                    df_side = df_alpha[basic_cols + buy_sta_cols + ['buyAvailNtl', f"buyRet{self.target_horizon}s"]].copy()
                else: 
                    df_side = df_alpha[basic_cols + sell_sta_cols + ['sellAvailNtl', f"sellRet{self.target_horizon}s"]].copy()

                df_side.columns = basic_cols + sta_ls + ['availNtl', self.target_ret]
                
                stat_data_all = (df_side.groupby(['skey', 'exchange', 'date'])[sta_ls + [self.target_ret]]
                                        .agg([('sum_x', 'sum'), ('count_x', 'count'),
                                              ('sum_x2', lambda x: (x**2).sum()),
                                              ('sum_x3', lambda x: (x**3).sum()),
                                              ('sum_x4', lambda x: (x**4).sum()),
                                              ('hist_x', lambda x: pd.cut(x, bins=np.arange(-0.003,0.00305, 0.0001), 
                                                                          labels=np.arange(-29.5, 30, 1)).value_counts().to_dict())]))
                stat_data_all.columns.names = ["sta_cat", ""]
                stat_data_all = stat_data_all.stack("sta_cat").reset_index()
                stat_data_all['sta_cat'] = np.where(stat_data_all['sta_cat']==self.target_ret, self.target_ret + "_" + sta_type,
                                                    stat_data_all['sta_cat'])
                stat_data_all['side'] = side

                df_side = df_side.melt(id_vars=basic_cols + ['availNtl', self.target_ret], value_vars=sta_ls, 
                                       var_name="sta_cat", value_name='sta')

                total_number = 4800 if sta_type=="l2" else 14400
                tolerance = 0.05 if (self._target_return is not None) else 0.01

                df_cutoff = (df_side.groupby(['skey', 'exchange', 'date', 'sta_cat'])
                                    .apply(lambda x: find_top_percent(x, col="sta",
                                                                      target_number=self._target_number, 
                                                                      target_ratio=self._target_ratio,
                                                                      target_return=self._target_return,
                                                                      ytrue_col=self.target_ret,
                                                                      total_number=total_number,
                                                                      filter_first=self.filter_first,
                                                                      tolerance=tolerance))
                                    .reset_index(drop=True))

                df_cutoff['side'] = side
                
                stat_data_cutoff = (df_cutoff.groupby(['skey', 'exchange', 
                                                       'date', 'sta_cat'])[['sta','availNtl',self.target_ret,'top_percent']]
                                             .agg(yHatAvg=('sta', 'mean'), 
                                                  countOppo=('sta', 'count'),
                                                  topPercent=('top_percent', 'mean'),
                                                  availNtlAvg=('availNtl', 'mean'),
                                                  availNtlSum=('availNtl', 'sum'),
                                                  yHatHurdle=('sta', 'min'),
                                                  actualRetAvg=(self.target_ret, 'mean')))
                stat_data_cutoff['vwActualRetAvg'] = (df_cutoff.groupby(['skey', 'exchange', 'date', 'sta_cat'])
                                                              .apply(lambda x: weighted_average(x[self.target_ret],
                                                                                                weights=x['availNtl'])))
                stat_data = pd.merge(stat_data_all, stat_data_cutoff, on=['skey', 'exchange', 'date', 'sta_cat'], how='left', validate='one_to_one')
                del stat_data_all, stat_data_cutoff

                df_daily.append(stat_data)
                df_intra.append(df_cutoff)

        # save df_daily
        df_daily = pd.concat(df_daily).reset_index(drop=True)
        os.makedirs(os.path.join(self.cutoff_path, 'daily'), exist_ok = True)
        df_daily.to_pickle(os.path.join(self.cutoff_path, 'daily', f'df_{self.target_cut}_{date}.pkl'), protocol = 4)

        # save df_intra
        df_intra = pd.concat(df_intra, ignore_index=True)
        df_intra = df_intra.dropna(subset=[self.target_ret]).reset_index(drop=True)
        for col in ['date','skey','mins_since_open']:
            df_intra[col] = df_intra[col].astype('int32')
        
        df_intra_stat = (df_intra.groupby(['exchange','date','side','sta_cat','mins_since_open'])[['skey', 'availNtl']]
                                 .agg(countOppo=('skey', 'count'),
                                      countStock=('skey', 'nunique'),
                                      availNtlSum=('availNtl', 'sum')))
        df_intra_stat[f'vwActualRetAvg'] = (df_intra.groupby(['exchange','date','side','sta_cat','mins_since_open'])
                                                    .apply(lambda x: weighted_average(x[self.target_ret],
                                                                                      weights=x['availNtl'])))
        df_intra_stat = df_intra_stat.reset_index()
        os.makedirs(os.path.join(self.cutoff_path, 'intraday'), exist_ok = True)
        df_intra_stat.to_pickle(os.path.join(self.cutoff_path, 'intraday', f'df_{self.target_cut}_{date}.pkl'), protocol = 4)

    def generate_sta_cutoff(self, dates):
        if self.machine=='personal-server':
            with NestablePool(32) as p:
                p.map(self.generate_daily_sta_cutoff, dates)
        elif self.machine=='HPC':
            jobs = get_job_list(dates)
            for job in jobs:
                self.generate_daily_sta_cutoff(job)

def _get_eva_md(tstock, forward_period, backward_period):
    tstock['index'] = tstock.index.values
    tstock['session'] = np.where(tstock['time'] < 13 * 1e10, 0, 1)

    assert not ((tstock.bid1p == 0) & (tstock.ask1p == 0)).any()
    tstock['safeBid1p'] = np.where(tstock.bid1p == 0, tstock.ask1p, tstock.bid1p)
    tstock['safeAsk1p'] = np.where(tstock.ask1p == 0, tstock.bid1p, tstock.ask1p)
    tstock['mid'] = (tstock.safeBid1p + tstock.safeAsk1p)/2
    assert (tstock['mid'] != 0).all()
    tstock['adjMid'] = (tstock.bid1q * tstock.safeAsk1p + tstock.ask1q * tstock.safeBid1p)/(tstock.bid1q + tstock.ask1q)

    groupAllData = tstock.groupby(['date', 'skey', 'session'])
    tstock['sessionStartIx'] = groupAllData['index'].transform('min')

    # FIXME: change to clockAtExchange in the future version of DFS
    for tm in [backward_period]:
        tmCol = 'L{}s_ix'.format(tm)
        tstock[tmCol] = groupAllData['clockAtArrival'].transform(lambda x: findTmValue(x, tm * 1e6, 'L')).astype(int)
    
    for tm in [forward_period]:
        tmCol = 'F{}s_ix'.format(tm)
        tstock[tmCol] = groupAllData['clockAtArrival'].transform(lambda x: findTmValue(x, tm * 1e6, 'F')).astype(int)

    for tm in [forward_period]:
        tmIx = tstock['F{}s_ix'.format(tm)].values + tstock['sessionStartIx'].values
        nanMask = tstock['F{}s_ix'.format(tm)].values == -1
        for col in ['adjMid']:
            targetCol = tstock[col].values[tmIx]
            targetCol[nanMask] = np.nan
            tstock['{}_F{}s'.format(col, tm)] = targetCol
    
    tstock['curNearLimit'] = np.where((tstock.ask5q == 0) | (tstock.bid5q == 0), 1.0, 0.0)
    
    tstock[f'buyRet{forward_period}s'] = (np.where(tstock.curNearLimit == 1, np.NaN, tstock[f'adjMid_F{forward_period}s'] / tstock.ask1p - 1)).clip(-.03,.03)
    tstock[f'sellRet{forward_period}s'] = (np.where(tstock.curNearLimit == 1, np.NaN, tstock.bid1p / tstock[f'adjMid_F{forward_period}s'] - 1)).clip(-.03,.03)

    for tm in [backward_period]:
        tmIx = tstock['L{}s_ix'.format(tm)].values + tstock['sessionStartIx'].values
        nanMask = tstock['L{}s_ix'.format(tm)].values == -1
        for col in ['curNearLimit']:
            targetCol = tstock[col].values[tmIx]
            targetCol[nanMask] = np.nan
            tstock['{}_L{}s'.format(col, tm)] = targetCol  

     # nearLimit == 1 if any nearLimit in past
    tstock.set_index(pd.DatetimeIndex(tstock['datetime']), inplace = True)

    tstock[f'nearLimit_L{backward_period}s_L0'] = (groupAllData['curNearLimit'].apply(lambda x:x.rolling(window = f'{backward_period}s', closed='both').max())) 
    tstock['nearLimit'] = (tstock[f'nearLimit_L{backward_period}s_L0'] == 1) | (tstock[f'curNearLimit_L{backward_period}s'] == 1)
    tstock.reset_index(drop=True, inplace=True)
    tstock.loc[tstock['nearLimit'], [f'buyRet{forward_period}s', f'sellRet{forward_period}s']] = np.nan

    col_ls = ['skey', 'date', 'time', 'datetime', 'ordering', 
              'ask1p', 'ask1q', 'bid1p', 'bid1q',
              f'buyRet{forward_period}s', f'sellRet{forward_period}s', 'nearLimit',
              'cum_volume', 'cum_amount']

    return tstock[col_ls]
            

if __name__ == "__main__":
    

    sta_input = '/home/marlowe/Marlowe/eva/sta_input_demo.yaml'
    sta_eval_run = StaAlphaEvalMap(sta_input)
    sta_eval_run.generate_daily_sta_cutoff(20200102)

    # sta_input = sys.argv[1]
    # sta_eval_run = StaAlphaEvalMap(sta_input)

    # paths = sta_eval_run.stock_reader.list_dir('/com_md_eq_cn/mdbar1d_jq', 'com_md_eq_cn', 'mdbar1d_jq')
    # dates = []
    # for path in paths:
    #     date = path.split("/")[-1].split(".")[0]
    #     if (date >= sta_eval_run.start_date) and (date <= sta_eval_run.end_date):
    #         dates.append(int(date))

    # # dates = [20200102, 20200103, 20200106, 20200107]
    # print(len(dates))
    # sta_eval_run.generate_sta_cutoff(dates)

    # partial_func = partial(_get_daily_sta_cutoff, sta_input=sta_input)
    # with Pool(4) as p:
    #     p.map(partial_func, dates)

    # for date in tqdm(dates[2:]):
    #     sta_eval_run.generate_daily_sta_cutoff(date)



    