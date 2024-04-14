import numpy as np
import os
import sys
os.chdir('/work/jdy/davin/proj/')
sys.path.append('/scratch/hill.davi/ukbiobank_spirometry/Quality/')
from SimCLR.utils import *

#from GPEC.utils import *

def mkdir_p(dir):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir):
        os.mkdir(dir)
    

job_directory = '/work/jdy/davin/proj/Tests/Models/blackbox_model_training/slurm_scripts'
mkdir_p(job_directory)


script_list = [
    '/work/jdy/davin/proj/Tests/Models/blackbox_model_training/census/train_census.py',
    '/work/jdy/davin/proj/Tests/Models/blackbox_model_training/german_credit/germancredit.py',
    '/work/jdy/davin/proj/Tests/Models/blackbox_model_training/synthetic_cosinv/cosinv.py',
    '/work/jdy/davin/proj/Tests/Models/blackbox_model_training/online_shoppers/onlineshoppers.py',
]

gpu = False

for script in script_list:
    job_name = "%s" %(script[-8:])
    job_file = os.path.join(job_directory,"%s.job" %(script[-8:]))
    python_script = os.path.basename(script)

    cmd = script
    if gpu:
        submit_slurm(cmd, job_file, conda_env = 'a100', partition = 'ai-jumpstart', exclude_nodes = ['d3146'], mem = 32, time_hrs = 24, job_name = job_name)
    else:
        percent_aijump = 1
        coin_flip = np.random.binomial(n=1,p=percent_aijump)
        if coin_flip == 1:
            submit_slurm(
                cmd,
                job_file,
                conda_env = 'a100',
                partition = 'ai-jumpstart',
                mem = 16,
                time_hrs = 24,
                job_name = job_name,
                n_gpu = 0,
                prioritize_cpu_nodes = True
            )
        else:
            submit_slurm(
                cmd,
                job_file,
                conda_env = 'a100',
                partition = 'short',
                mem = 16,
                time_hrs = 24,
                job_name = job_name,
                n_gpu = 0,
                prioritize_cpu_nodes = True 
            )

