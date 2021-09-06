import os
import re
from collections import defaultdict
import pandas as pd
import numpy as np
from AShareReader import AShareReader
from CephClient import CephClient
from multiprocessing import Pool
from utils import *
from functools import partial
from tqdm import tqdm
from glob import glob

a = AShareReader(dll_path = '{0}/ceph_client/ceph-client.so'.format(os.environ['HOME']), 
                     config_path='{0}/dfs/ceph.conf'.format(os.environ['HOME']),
                     KEYRING_LOC = '{0}/dfs/ceph.key.keyring'.format(os.environ['HOME']))
c = CephClient(dll_path = '{0}/ceph_client/ceph-client.so'.format(os.environ['HOME']), 
               config_path='{0}/dfs/ceph.conf'.format(os.environ['HOME']),
               KEYRING_LOC = '{0}/dfs/ceph.key.keyring'.format(os.environ['HOME']))

bench = "IC"
start_date = 20200102
end_date = 20201231

def get_meta(date):
    universe = a.Read_Stock_Daily('com_md_eq_cn', f"chnuniv_{bench.lower()}", date, date).skey.unique()
    results = []
    for stock in universe:
        md = a.Read_Stock_Tick('com_md_eq_cn', 'md_snapshot_l2', date, date, stock_list=[stock])
        results.append([md.skey.iloc[0], md.date.iloc[0], len(md), md.cum_volume.max(), md.cum_amount.max()])
    return pd.DataFrame(results, columns=['skey', 'date', 'nticks', 'volume', 'amount'])


paths = a.list_dir('/com_md_eq_cn/mdbar1d_jq', 'com_md_eq_cn', 'mdbar1d_jq')
dates = []
for path in paths:
    date = path.split("/")[-1].split(".")[0]
    if (date >= str(start_date)) and (date <= str(end_date)):
        dates.append(int(date))

with NestablePool(32) as p:
    results = pd.concat(p.map(get_meta, dates), ignore_index=True)

c.write_df_to_ceph(results, "/sta_md_eq_cn/sta_md_l2/meta.parquet", 'sta_md_eq_cn', 'sta_md_l2')