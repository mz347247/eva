import os
import re
from subprocess import Popen, PIPE, STDOUT

def get_slurm_env(name): 
    value = os.getenv(name) 
    assert value is not None 
    return value

def get_job_list(Ls):
    array_id = int(get_slurm_env("SLURM_ARRAY_TASK_ID"))
    array_size = int(get_slurm_env("SLURM_ARRAY_TASK_COUNT")) 
    # proc_id = int(get_slurm_env("SLURM_PROCID")) 
    # task_size = int(get_slurm_env("SLURM_NTASKS"))

    job_size = len(Ls) // array_size + 1
    jobLs = Ls[array_id * job_size: (array_id + 1) * job_size]

    return jobLs

def submit(batch, dryrun):
    """Submit an batch file
    Parameters
    ----------
    batch : str
        the batch string to submig, it could be str or utf8 bytes
    dryrun : bool
        When True, print the batches to stdout instead of submit them
    """
    if dryrun:
        print(batch)
    else:
        p = Popen(['sbatch'], stdin=PIPE, stdout=PIPE, stderr=STDOUT)
        if type(batch) is str:
            batch = batch.encode()
        out, _ = p.communicate(input=batch)
        print(out.decode(), end='')
        return re.search("\d+", out.decode()).group(0)

if __name__ == "__main__":
    batch1 = """#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=100M
#SBATCH --ntasks=1
#SBATCH --time=5:00

srun python script.py
"""
    job_id1 = submit(batch1, False)
    print(job_id1)

    batch2 = f"""#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=100M
#SBATCH --ntasks=1
#SBATCH --time=5:00
#SBATCH --dependency=afterok:{job_id1}

srun python script.py
"""
    submit(batch2, False)