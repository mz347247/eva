import os
import re
from collections import defaultdict
import pandas as pd
import numpy as np
from utils import *
from AShareReader import AShareReader
from CephClient import CephClient
import yaml

class StaAlphaEval():
    def __init__(self, sta_input):
        f = open(sta_input, 'r', encoding='utf-8')
        cfg = f.read()
        dict_yaml = yaml.full_load(cfg)
        f.close()

        self.machine = dict_yaml['machine']
        self.bench = dict_yaml['bench']
        self.universe = dict_yaml['universe']
        self.njobs = dict_yaml['njobs']
        self.start_date = str(dict_yaml['start_date'])
        self.end_date = str(dict_yaml['end_date'])
        self.eval_alpha = dict_yaml['eval_alpha']
        self.target_ret = dict_yaml['target_return']
        self.target_cut = dict_yaml['target_cut']
        self.eval_focus = dict_yaml['eval_focus']
        self.lookback_window = dict_yaml['lookback_window']
        self.compute_ret = dict_yaml['compute_ret']    

        self.eval_path = os.path.join(dict_yaml['save_path'], self.universe, self.eval_alpha[-1]['name'])
        self.cutoff_path = os.path.join(self.eval_path, f'sta_{self.target_cut}_{self.eval_focus}')
        self.log_path = dict_yaml['log_path']

        if ((self.machine == 'personal-server') and (self.njobs > 70)) or \
           ((self.machine == 'HPC') and (self.njobs > 999)):
           raise ValueError(f"Too many jobs {self.njobs} for the {self.machine}!")
        
        # setup the dfs reader
        self.stock_reader = AShareReader()
        self.sta_reader = CephClient()
        
        self.eval_alpha_dict = defaultdict(list)
        for alpha_info in self.eval_alpha:
            self.eval_alpha_dict[alpha_info['data_type']].append(alpha_info)