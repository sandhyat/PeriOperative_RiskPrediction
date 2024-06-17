import sys
import os
from itertools import product
from scipy.stats import qmc
import numpy as np
import math

# This is used when the hyperparameter tuning is performed on a high performance cluster
LSF_vars = "LSF_DOCKER_VOLUMES='/<INPUT DATA DIRECTORY>/:/input/ /<OUTPUT RESULT SAVING DIRECTORY>/:/output/ /<SCRIPT DIRECTORY>/:/codes/'"

poweroftwo_samples = 5

## bounds are inclusive for integers
## third element is 1 if is an integer, 2 binary (so no value is passed)
## fourth element 1 if on log scale
bounds = { 'medid_embed_dim':[5, 25, 1, 0 ] ,
           'repr_dims_f':[100, 200, 1, 0 ] ,
           # 'repr_dims_m':[100, 400, 1, 0 ] ,  # turning off these two because we are currently wanting them all to the same pprojection for cross modal comparison
           # 'repr_dims_a':[50, 100, 1, 0] , # the original dimension is currently 2 so the representation dim is smaller.
           # 'preops_rep_dim':[80, 150, 1, 0 ],
           'preops_rep_dim_o': [50, 100, 1, 0],
           'preops_rep_dim_l': [50, 100, 1, 0],
            'cbow_rep_dim':[80, 120, 1, 0 ] ,
           'homemeds_rep_dim':[200, 400, 1, 0],
           'pmh_rep_dim':[50, 100, 1, 0],
           'prob_list_rep_dim':[50, 100, 1, 0],
           'outcome_rep_dim':[30, 65, 1, 0],
           'proj_dim':[40,80,1,0],
           'proj_head_depth':[2,5,1,0],
           'weight_preops':[0.1, 0.8, 0, 0],
           'weight_ts_preops':[0.1, 0.8, 0, 0],
           'weight_outcomes':[0.1,0.8,0,0],
           'weight_std': [0.1,0.8,0,0],
           'weight_cov': [0.1,0.8,0,0],
            'weight_mse':[0.0,0.0,0,0],
           'weight_ts_cross':[0.1,0.8,0,0],
           'lmd': [0.1, 0.8, 0, 0],  # weight between reconstruction and hierarchical
           'temp': [0.4,1.0, 0, 0],  # tau in the contrastive loss
           'segment_num': [0.1, 0.8, 1, 0],  ## 'number of time interval segment to mask, default: 3 time intervals'
           'mask_ratio_per_seg': [0.05, 0.3, 0, 0],  # fraction of the sequence length to mask for each time interval
           'batch-size':[16, 128, 1, 1 ],
           'epochs':[2, 15, 1, 0 ],
           'iters':[200, 8000,1,1],
           'lr':[.0001, .01, 0 , 1]
  }

sampler = qmc.Sobol(d =len(bounds.keys()))
sample = sampler.random_base2(m=poweroftwo_samples)

def makeargstring(sample, bounds):
  out = "" ## if I was clever I would do this with a list comprehension
  for samplei, thisvar in zip(sample, bounds.keys() ):
    if(bounds[thisvar][3]==1):
      target = bounds[thisvar][0] * math.exp(samplei* (math.log(bounds[thisvar][1])-math.log(bounds[thisvar][0] + bounds[thisvar][2]) ) )
    else:
      target = bounds[thisvar][0] + samplei* (bounds[thisvar][1]-bounds[thisvar][0] + bounds[thisvar][2]) ## plus one to include outer limit
    if(bounds[thisvar][2]==1):
      target = " --" + thisvar+ "=" + str(math.floor(target))
    elif (bounds[thisvar][2]==2):
      if(target > 0.5):
        target = " --" + thisvar
      else:
        target = ''
    else:
      target = " --" + thisvar + "=" + str(round(target, 4)) ## the rounding is just to make it prettier
    out = out + target
  return out

def makefilestring(sample, bounds):
  out = "" ## if I was clever I would do this with a list comprehension
  for samplei, thisvar in zip(sample, bounds.keys() ):
    if(bounds[thisvar][3]==1):
      target = bounds[thisvar][0] * math.exp(samplei* (math.log(bounds[thisvar][1])-math.log(bounds[thisvar][0] + bounds[thisvar][2]) ) )
    else:
      target = bounds[thisvar][0] + samplei* (bounds[thisvar][1]-bounds[thisvar][0] + bounds[thisvar][2]) ## plus one to include outer limit
    if(bounds[thisvar][2]==1):
      target = str(math.floor(target))
    else:
      target = str(round(target,4))
    out = out + "_" + target
  return out


task = "icu" # options from {'icu', 'mortality', 'aki1', 'aki2', 'aki3', 'pulm', 'PE', 'severe_present_1', 'postop_del','DVT','low_sbp_time','n_glu_high','cardiac'}
bsub_filename = 'bsub_command_TS_contra_'+task+'_multiView_tUrl.txt'



file_to_run = 'train_modular_tUrl.py'
output_file_name = '/output/logs/' +task +'-multiView_tUrl_HP_'


np.random.seed(100)
initial_seed_list = np.random.randint(10000, size=2)

with open(bsub_filename, 'w') as f:
    for i in range(np.power(2, poweroftwo_samples)):
        for seed in initial_seed_list:
            python_command = 'python /codes/' + \
              file_to_run + \
              " Flowsheets_meds_timesurl FM_timesurl_output --eval --preops --meds --alerts --pmh --problist --homemeds --postopcomp --outputcsv=HPTuningFullModel_"+ task+ "_MultiView_tUrl.csv  --outcome="+task+ \
              makeargstring(sample[i,:], bounds) + " --number_runs=1  --seed=" + str(seed) +' > ' \
             + output_file_name + makefilestring(sample[i,:], bounds) + '_' + str(seed) +'.out'

            comTemp = LSF_vars+ " bsub -G '<COMPUTE GROUP NAME>' -g '<NAME OF THE USER's JOB GROUP>' -n 8 -q general -R 'gpuhost' -gpu 'num=1:gmodel=TeslaV100_SXM2_32GB:gmem=16G' -M 256GB -R 'rusage[mem=256GB] span[hosts=1]' -a 'docker(docker121720/pytorch-for-ts:0.5)' " + "'" +str( python_command)+ "'"
            f.write(f"{comTemp}\n")
f.close()
