#!/bin/bash

#SBATCH --job-name=agent-testing
#SBATCH --account=my@account
#SBATCH --array=0-8256
#SBATCH --partition=test_partition
#SBATCH --time=00:05:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=my@email.domain

export PROJECT_HOME=$WORK/projects/gym-saturation
source $PROJECT_HOME/venv/bin/activate
python $PROJECT_HOME/gym_saturation/agent_testing.py --step_limit 20 \
       --output_folder results \
       --problem_filename $(python $PROJECT_HOME/slurm-jobs/filename_getter.py)
