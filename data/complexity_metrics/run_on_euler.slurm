#!/bin/bash

#SBATCH --job-name="complexity_analysis"
#SBATCH --output="complexity_analysis_%j.out"
#SBATCH --error="complexity_analysis_%j.err"
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G

##########################################
echo "Starting complexity analysis job"
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

# Verify Python environment and NLTK data
echo "Python version and location:"
which python
python --version

echo "Verifying NLTK data..."
python -c "
import nltk
import os
print('NLTK data path:', nltk.data.path)
print('NLTK_DATA environment variable:', os.getenv('NLTK_DATA'))
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
    print('NLTK data verified successfully')
except LookupError as e:
    print('Error: NLTK data not found:', e)
    exit(1)
"

# --- Execution ---
echo "Running compleximeter.py..."
python compleximeter.py

##############################################
echo "Job completed at $(date)"
############################################## 