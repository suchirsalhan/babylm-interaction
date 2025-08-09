#!/bin/bash

#SBATCH --job-name="complexity_analysis"
#SBATCH --output="complexity_analysis_%A_%a.out"
#SBATCH --error="complexity_analysis_%A_%a.err"
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --array=1-10%5

##########################################
echo "Starting complexity analysis job (array job ${SLURM_ARRAY_TASK_ID})"
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

# Create main output directories
mkdir -p complexity_results_with_levels
mkdir -p complexity_results_without_levels

# --- Execution ---
echo "Running compleximeter_chunked.py for chunk ${SLURM_ARRAY_TASK_ID}..."
python compleximeter_chunked.py --chunk-id ${SLURM_ARRAY_TASK_ID} --total-chunks 10

##############################################
echo "Job completed at $(date)"
############################################## 