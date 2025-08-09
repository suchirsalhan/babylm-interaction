#!/bin/bash

#SBATCH --job-name="switchboard_complexity"
#SBATCH --output="switchboard_complexity_%A_%a.out"
#SBATCH --error="switchboard_complexity_%A_%a.err"
#SBATCH --time=5:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --array=1-10%5

##########################################
echo "Starting switchboard complexity analysis job (array job ${SLURM_ARRAY_TASK_ID})"
echo "$(date) start ${SLURM_JOB_ID}"
##########################################

# Load required modules
module load stack/2024-06
module load gcc/12.2.0
module load python/3.10.13

# Activate the virtual environment
VENV_DIR="$SCRATCH/complexity_project/venv"
source $VENV_DIR/bin/activate

# Set NLTK data path
export NLTK_DATA="$VENV_DIR/nltk_data"
echo "NLTK_DATA is set to: $NLTK_DATA"

# Create output directory for switchboard results
mkdir -p complexity_results_switchboard

# --- Execution ---
echo "Running compleximeter_chunked_switchboard.py for chunk ${SLURM_ARRAY_TASK_ID}..."
python compleximeter_chunked_switchboard.py

##############################################
echo "Job completed at $(date)"
############################################## 