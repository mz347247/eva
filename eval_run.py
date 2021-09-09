import os
import sys
from HPCutils import submit
from eval import StaAlphaEval
import eval_map
import eval_reduce

# sta_input = '/home/marlowe/Marlowe/eva/sta_input_ps.yaml'
sta_input = sys.argv[1]
sta_eval_run = StaAlphaEval(sta_input)

cwd = os.getcwd()

if sta_eval_run.machine == "HPC":
    if 'mbd' in sta_eval_run.eval_alpha_dict:
        mem = '6G'
    else:
        mem = '4G'
    
    map_sh = f'''#!/bin/sh
#SBATCH --output={sta_eval_run.log_path}/%A-%a-%x.out
#SBATCH --error={sta_eval_run.log_path}/%A-%a-%x.error
#SBATCH --mem-per-cpu={mem} --ntasks=1
#SBATCH --time=30:00
#SBATCH --cpus-per-task=4
#SBATCH --array=0-{sta_eval_run.hpc_njobs}
srun -l python3 {cwd}/eval_map.py {sta_input}'''

    map_job_id = submit(map_sh, dryrun=False)

    reduce_sh = f'''#!/bin/sh
#SBATCH --output={sta_eval_run.log_path}/%A-%a-%x.out
#SBATCH --error={sta_eval_run.log_path}/%A-%a-%x.error
#SBATCH --dependency=afterok:{map_job_id}
#SBATCH --mem-per-cpu=4G --ntasks=1
#SBATCH --time=5:00
#SBATCH --cpus-per-task=2
srun -l python3 {cwd}/eval_reduce.py {sta_input}'''

    reduce_sh_id = submit(reduce_sh, dryrun=False)

else:
    eval_map.main(sta_input)

    eval_reduce.main(sta_input)