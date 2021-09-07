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
from HPCutils import get_job_list, submit
import yaml

class StaAlphaEval():
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
        self.eval_focus = dict_yaml['eval_focus']
        self.lookback_window = dict_yaml['lookback_window']
        self.eval_path = os.path.join(dict_yaml['save_path'], self.bench, self.eval_alpha[-1])
        self.cutoff_path = os.path.join(self.eval_path, f'sta_{self.target_cut}_{self.eval_focus}')
        
        if self.machine == "personal-server":
            self.stock_reader = AShareReader(dll_path = '{0}/ceph_client/ceph-client.so'.format(os.environ['HOME']), 
                                          config_path='{0}/dfs/ceph.conf'.format(os.environ['HOME']),
                                          KEYRING_LOC = '{0}/dfs/ceph.key.keyring'.format(os.environ['HOME']))
            self.sta_reader = CephClient(dll_path = '{0}/ceph_client/ceph-client.so'.format(os.environ['HOME']), 
                                        config_path='{0}/dfs/ceph.conf'.format(os.environ['HOME']),
                                        KEYRING_LOC = '{0}/dfs/ceph.key.keyring'.format(os.environ['HOME']))
        elif self.machine == "HPC":
            self.stock_reader = AShareReader()
            self.sta_reader = CephClient()
        else:
            raise ValueError(f"Invalid Input machine: {self.machine}")


if __name__ == "__main__":
    sta_input = '/home/marlowe_zhong/eva/sta_input_HPC.yaml'
    cwd = os.getcwd()
    
    map_sh = f'''#!/bin/sh
#SBATCH --output=/home/marlowe_zhong/eva/logs/%A-%a-%x.out
#SBATCH --error=/home/marlowe_zhong/eva/logs/%A-%a-%x.error
#SBATCH --mem-per-cpu=4G --ntasks=1
#SBATCH --time=30:00
#SBATCH --cpus-per-task=4
#SBATCH --array=0-250
srun -l python3 {cwd}/eval_map.py {sta_input}'''

    map_job_id = submit(map_sh, dryrun=False)

    reduce_sh = f'''#!/bin/sh
#SBATCH --output=/home/marlowe_zhong/eva/logs/%A-%a-%x.out
#SBATCH --error=/home/marlowe_zhong/eva/logs/%A-%a-%x.error
#SBATCH --dependency=afterok:{map_job_id}
#SBATCH --mem-per-cpu=4G --ntasks=1
#SBATCH --time=5:00
#SBATCH --cpus-per-task=2
srun -l python3 {cwd}/eval_reduce.py {sta_input}'''

    reduce_sh_id = submit(reduce_sh, dryrun=False)