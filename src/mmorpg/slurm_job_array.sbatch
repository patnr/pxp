#!/usr/bin/env bash

# Why not merge batch_runner.py and slurm script, as per
# #the-slurm-script-does-not-have-to-be-written-in-bash of
# https://documentation.sigma2.no/jobs/job_scripts.html.
# - Modularity: may want to support other HPC systems than SLURM
# - Must insert shebang into batch_runner.py after python runtime
# - Must fix sys.path[0] since SLURM copies script to /var/spool/slurmd/job1297113

"$@"/$SLURM_ARRAY_TASK_ID
