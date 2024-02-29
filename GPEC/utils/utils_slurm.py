import os
import warnings
import sys
sys.path.append('./')
from GPEC.utils.utils_io import make_dir

def submit_slurm(python_script, job_file,job_out_dir = '', conda_env='a100', partition='gpu',mem=32, time_hrs = -1, n_gpu = 1, n_cpu = 1, exclude_nodes = None, job_name = 'script', prioritize_cpu_nodes = True, extra_line='', nodelist = None, exclusive = False):
    '''
    submit batch job to slurm

    args:
        exclude_nodes: list of specific nodes to exclude
    '''
    dname = os.path.dirname(python_script)
    if job_out_dir == '':
        job_out = os.path.join(dname, 'job_out')
    else:
        job_out = os.path.join(job_out_dir, 'job_out')
    make_dir(job_out)  # create job_out folder

    if partition not in ['gpu', 'short', 'ai-jumpstart']:
        raise ValueError('invalid partition specified')

    # default time limits
    time_default = {
        'gpu': 8,
        'short':24,
        'ai-jumpstart':24
    }
    # max time limits
    time_max = {
        'gpu': 8,
        'short':24,
        'ai-jumpstart':720
    }
    if time_hrs == -1:
        # set to default time limit
        time_hrs = time_default[partition]
    elif time_hrs > time_max[partition]:
        # set to maximum time limit if exceeded
        time_hrs = time_max[partition]
        warnings.warn('time limit set to maximum for %s partiton: %s hours' % (partition, str(time_hrs)))
    elif time_hrs < 0:
        raise ValueError('invalid (negative) time specified')

    with open(job_file, 'w') as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines("\n")
        fh.writelines("#SBATCH --job-name=%s\n" % (job_name))
        fh.writelines("#SBATCH --nodes=1\n")
        fh.writelines("#SBATCH --tasks-per-node=1\n")
        fh.writelines("#SBATCH --cpus-per-task=%s\n" % str(n_cpu))
        fh.writelines("#SBATCH --mem=%sGb \n" % str(mem))
        fh.writelines("#SBATCH --output=" + job_out + "/%j.out\n")
        fh.writelines("#SBATCH --error=" + job_out + "/%j.err\n")
        fh.writelines("#SBATCH --partition=%s\n" % (partition))
        fh.writelines("#SBATCH --time=%s:00:00\n" % (str(time_hrs)))
        if nodelist is not None:
            fh.writelines("#SBATCH --nodelist=%s\n" % (nodelist))
        if exclusive: fh.writelines("#SBATCH --exclusive\n")

        # exclude specific nodes
        if exclude_nodes is not None:
            exclude_str = ','.join(exclude_nodes)
            fh.writelines("#SBATCH --exclude=%s\n" % (exclude_str))

        # specify gpu
        if partition == 'gpu':
            fh.writelines("#SBATCH --gres=gpu:v100-sxm2:1\n")
        elif partition == 'ai-jumpstart':
            if n_gpu>0:
                fh.writelines("#SBATCH --gres=gpu:a100:%s\n" % (str(n_gpu)))
            elif prioritize_cpu_nodes:
                # exclude gpu nodes
                fh.writelines("#SBATCH --exclude=d[3146-3150]\n")

        fh.writelines("\n")
        # fh.writelines("module load anaconda3/2022.05\n")
        # fh.writelines("source activate %s\n" % conda_env)
        fh.writelines("CONDA_BASE=$(conda info --base) ; source $CONDA_BASE/etc/profile.d/conda.sh \n")
        fh.writelines("conda activate %s\n" % conda_env)
        fh.writelines("%s\n" % extra_line)
        fh.writelines("python -u %s" % python_script)
    os.system("sbatch %s" %job_file)

