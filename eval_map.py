import os
import re
import time
import sys
import pandas as pd
import numpy as np
from utils import *
from multiprocessing import Pool
from functools import partial
from HPCutils import get_job_list
from eval import StaAlphaEval
from glob import glob
from pandas.api.types import CategoricalDtype

class StaAlphaEvalMap(StaAlphaEval):

    def __init__(self, sta_input):
        super().__init__(sta_input)

        try:
            self.target_horizon = int(re.search(r"\d+", self.target_ret).group(0))
        except (AttributeError, ValueError):
            raise ValueError(f"Invalid Input target_ret: {self.target_ret}") 

        # top5p or top240
        self._target_number = self._target_ratio = None
        if self.target_cut.endswith("p"):
            try:
                self._target_ratio = int(re.search(r"\d+", self.target_cut).group(0)) / 100
                assert self._target_ratio < 1
            except (AttributeError, ValueError):
                raise ValueError(f"Invalid Input target_cut: {self.target_cut}")
        else:
            try:
                self._target_number = int(re.search(r"\d+", self.target_cut).group(0))
            except (AttributeError, ValueError):
                raise ValueError(f"Invalid Input target_cut: {self.target_cut}")

        self._filter_first = False

        if self.eval_focus == 'mixed':
            self._filter_first = True

    def generate_daily_sta_cutoff(self, date):
        df_daily = []
        df_intra = []
        sta_ls_all = []

        if self.universe != 'custom':
            universe = self.stock_reader.Read_Stock_Daily('com_md_eq_cn', f"chnuniv_{self.universe.lower()}", date, date).skey.unique()

        # loop over alphas from mbd and lv2
        for sta_type, sta_info_ls in self.eval_alpha_dict.items():
            buy_sta_cols = []
            sell_sta_cols = []
            sta_ls = []
            for ix, sta_info in enumerate(sta_info_ls):
                if sta_info['data_source'] == 'DFS':
                    tmp_sta = self.sta_reader.read_file(sta_info['data_path'].format(date=date), 
                                                        sta_info['pool_name'], sta_info['namespace'])
                else:
                    if '{skey}' in sta_info['data_path']:
                        tmp_sta = pd.concat([pd.read_parquet(path) for path in glob(sta_info['data_path'].replace("{skey}", "*")
                                                                                                         .format(date=date))],
                                             ignore_index=True)
                    else:
                        tmp_sta = pd.read_parquet(sta_info['data_path'].format(date=date))

                tmp_buy_cols = [col for col in tmp_sta.columns if re.match(sta_info['alpha_name']['buy'], col)]
                tmp_sell_cols = [col for col in tmp_sta.columns if re.match(sta_info['alpha_name']['sell'], col)]
                if len(tmp_buy_cols) == 1:
                    sta_ls.append(sta_info['name'])
                else:
                    sta_ls += [sta_info['name'] + "_" + re.search(sta_info['alpha_name']['buy'], col).group('label') 
                               + "_" + sta_type for col in tmp_buy_cols]
                tmp_rename = {col : (sta_info['name'] + "_" + col) for col in tmp_buy_cols + tmp_sell_cols}
                tmp_sta = tmp_sta.rename(columns=tmp_rename)
                buy_sta_cols += [(sta_info['name'] + "_" + col) for col in tmp_buy_cols]
                sell_sta_cols += [(sta_info['name'] + "_" + col) for col in tmp_sell_cols]
                if ix == 0:
                    df_sta = tmp_sta
                else:
                    df_sta = pd.merge(df_sta, tmp_sta, on=['skey', 'date', 'ordering'], how='outer', validate='one_to_one')
            del tmp_sta

            sta_ls_all += sta_ls

            if self.universe == 'custom':
                universe = df_sta.skey.unique()
            else:
                df_sta = df_sta[df_sta.skey.isin(universe)].reset_index(drop=True)

            if self.compute_ret:
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
            else:
                df_ret = self.sta_reader.read_file(f'/sta_md_eq_cn/sta_ret_{sta_type}/actual_return/{self.bench}/{date}.parquet', 
                                                    'sta_md_eq_cn', f'sta_ret_{sta_type}')
                df_md = self.sta_reader.read_file(f'/sta_md_eq_cn/sta_md_{sta_type}/{self.bench}/{date}.parquet', 
                                                    'sta_md_eq_cn', f'sta_md_{sta_type}')
                df_md = pd.merge(df_md[['skey', 'date', 'time','datetime', 'ordering', 
                                        'ask1p', 'ask1q', 'bid1p', 'bid1q',
                                        'cum_volume', 'cum_amount', f'nearLimit{self.lookback_window}s']],
                                 df_ret[['skey', 'date', 'ordering', f'buyRet{self.target_horizon}s', f'sellRet{self.target_horizon}s']],
                                 on = ['skey','date','ordering'], how = 'left', validate = 'one_to_one')
                del df_ret

            df_md = df_md[(df_md.skey.isin(universe)) & 
                          (((df_md['time'] >= 9.3 * 1e10) & (df_md['time'] <= 11.3 * 1e10)) |
                          ((df_md['time'] >= 13 * 1e10) & (df_md['time'] < 14.5655 * 1e10)))].reset_index(drop=True)

            df_alpha = df_md.merge(df_sta, on = ['skey','date','ordering'], how = 'left', validate = 'one_to_one')
            df_alpha = df_alpha.sort_values(['skey','date','ordering']).reset_index(drop=True)
            del df_md, df_sta

            df_alpha['exchange'] = df_alpha['skey'].astype(str).str[:1]
            df_alpha['exchange'] = np.where(df_alpha['exchange'] == '1', 'SH', 'SZ')

            # TODO: modify this when we have SH mbd data
            # only compare SZ data if we compare mbd
            if "mbd" in self.eval_alpha_dict:
                df_alpha = df_alpha[df_alpha.exchange=="SZ"].reset_index(drop=True)

            df_alpha['minute'] = df_alpha['datetime'].dt.hour * 60 + df_alpha['datetime'].dt.minute
            df_alpha['mins_since_open'] = np.where(df_alpha['minute'] <= 690, df_alpha['minute'] - 570, df_alpha['minute'] - 660)
            df_alpha['buyAvailNtl'] = (df_alpha['ask1p'] * df_alpha['ask1q']).clip(upper=100000)
            df_alpha['sellAvailNtl'] = (df_alpha['bid1p'] * df_alpha['bid1q']).clip(upper=100000)

            if self.eval_focus == "oppo":
                df_target = self.sta_reader.read_file(f'/sta_md_eq_cn/sta_ret_l2/target_return/IC/top240/{date}.parquet',
                                                       'sta_md_eq_cn', 'sta_ret_l2')

            basic_cols = ['skey','date','time','exchange','mins_since_open',
                          'cum_volume','cum_amount']
            
            for side in ['buy', 'sell']:
                if side == 'buy': 
                    df_side = df_alpha[basic_cols + buy_sta_cols + 
                                      ['buyAvailNtl', f"buyRet{self.target_horizon}s", f'nearLimit{self.lookback_window}s']].copy()
                else:
                    df_side = df_alpha[basic_cols + sell_sta_cols + 
                                      ['sellAvailNtl', f"sellRet{self.target_horizon}s", f'nearLimit{self.lookback_window}s']].copy()

                df_side.columns = basic_cols + sta_ls + ['availNtl', self.target_ret, 'nearLimit']
                
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

                df_side = df_side.melt(id_vars=basic_cols + ['availNtl', self.target_ret, 'nearLimit'], value_vars=sta_ls, 
                                       var_name="sta_cat", value_name='sta')

                total_number = 4800 if sta_type=="l2" else 14400

                if self.eval_focus == 'oppo':
                    target_return_col = f'{side}Ret{self.target_horizon}s'
                    df_side = df_side.merge(df_target[['skey', 'exchange', 'date', f'{side}Ret{self.target_horizon}s']], how='left', 
                                            on=['skey', 'exchange', 'date'], validate="many_to_one")
                else:
                    target_return_col = None

                df_cutoff = (df_side.groupby(['skey', 'exchange', 'date', 'sta_cat'])
                                    .apply(lambda x: find_top_percent(x, col="sta",
                                                                      target_number=self._target_number, 
                                                                      target_ratio=self._target_ratio,
                                                                      target_return_col=target_return_col,
                                                                      ytrue_col=self.target_ret,
                                                                      total_number=total_number,
                                                                      filter_first=self._filter_first))
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
                stat_data_cutoff = stat_data_cutoff.reset_index()

                # # record the target return
                # if (self.eval_focus == 'oppo') and (key=='base'):
                #     base_return[side] = stat_data_cutoff[['skey', 'exchange', 'date', 'vwActualRetAvg']]
                #     base_return[side].columns = ['skey', 'exchange', 'date', 'target_ret']

                stat_data = pd.merge(stat_data_all, stat_data_cutoff, on=['skey', 'exchange', 'date', 'sta_cat'], how='left', validate='one_to_one')
                del stat_data_all, stat_data_cutoff

                df_daily.append(stat_data)
                df_intra.append(df_cutoff)

        sta_ls_all += [(self.target_ret + "_" + sta_type) for sta_type in self.eval_alpha_dict.keys()]

        ## save the target return
        # tmp_buy = base_return['buy'].rename(columns={'target_ret': 'buyRet90s'})
        # tmp_sell = base_return['sell'].rename(columns={'target_ret': 'sellRet90s'})
        # tmp_df = pd.merge(tmp_buy, tmp_sell, how='outer', on=['skey', 'exchange', 'date'], validate='one_to_one')
        # self.sta_reader.write_df_to_ceph(tmp_df, f'/sta_md_eq_cn/sta_ret_l2/target_return/IC/top240/{date}.parquet', 'sta_md_eq_cn', 'sta_ret_l2')
        # del tmp_buy, tmp_sell

        sta_cat_type = CategoricalDtype(categories=sta_ls_all, ordered=True)
        # save df_daily
        df_daily = pd.concat(df_daily, ignore_index=True)
        df_daily['sta_cat'] = df_daily['sta_cat'].astype(sta_cat_type)
        os.makedirs(os.path.join(self.cutoff_path, 'daily'), exist_ok = True)
        df_daily.to_pickle(os.path.join(self.cutoff_path, 'daily', f'df_{self.target_cut}_{self.eval_focus}_{date}.pkl'), protocol = 4)

        # save df_intra
        df_intra = pd.concat(df_intra, ignore_index=True)
        df_intra = df_intra.dropna(subset=[self.target_ret]).reset_index(drop=True)
        df_intra['sta_cat'] = df_intra['sta_cat'].astype(sta_cat_type)
        for col in ['date','skey','mins_since_open']:
            df_intra[col] = df_intra[col].astype('int32')
        
        df_intra_stat = (df_intra.groupby(['exchange','date','side','sta_cat','mins_since_open'], observed=True)[['skey', 'availNtl']]
                                 .agg(countOppo=('skey', 'count'),
                                      availNtlSum=('availNtl', 'sum')))
        df_intra_stat[f'vwActualRetAvg'] = (df_intra.groupby(['exchange','date','side','sta_cat','mins_since_open'], observed=True)
                                                    .apply(lambda x: weighted_average(x[self.target_ret],
                                                                                      weights=x['availNtl'])))
        df_intra_stat = df_intra_stat.reset_index()
        os.makedirs(os.path.join(self.cutoff_path, 'intraday'), exist_ok = True)
        df_intra_stat.to_pickle(os.path.join(self.cutoff_path, 'intraday', f'df_{self.target_cut}_{self.eval_focus}_{date}.pkl'), protocol = 4)

    def generate_sta_cutoff(self, dates):
        if self.machine=='personal-server':
            with NestablePool(self.njobs) as p:
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


def main(sta_input):
    sta_eval_run = StaAlphaEvalMap(sta_input)

    paths = sta_eval_run.stock_reader.list_dir('/com_md_eq_cn/mdbar1d_jq', 'com_md_eq_cn', 'mdbar1d_jq')
    dates = []
    for path in paths:
        date = path.split("/")[-1].split(".")[0]
        if (date >= sta_eval_run.start_date) and (date <= sta_eval_run.end_date):
            dates.append(int(date))

    # # dates = [20200102, 20200103, 20200106, 20200107]
    # print(len(dates))
    sta_eval_run.generate_sta_cutoff(dates)

def test():
    start_time = time.time()
    sta_input = '/home/marlowe/Marlowe/eva/sta_input_ps.yaml'
    sta_eval_run = StaAlphaEvalMap(sta_input)
    sta_eval_run.generate_daily_sta_cutoff(20200701)
    end_time = time.time()
    print(end_time - start_time)

if __name__ == "__main__":
    main(sys.argv[1])
    # test()