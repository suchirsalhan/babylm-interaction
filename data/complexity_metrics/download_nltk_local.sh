#!/bin/bash

# Create a directory for NLTK data
mkdir -p nltk_data_download

# Download NLTK data using Python
python3 - << 'EOF'
import nltk
import os
import ssl

# Disable SSL verification (only for downloading NLTK data)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Set download directory
nltk.data.path.append("nltk_data_download")

# Download required data
datasets = [
    'punkt',
    'punkt_tab',
    'wordnet',
    'omw-1.4',
    'averaged_perceptron_tagger',
    'wordnet_ic',
    'brown',
    'stopwords',
    'tagsets',
    'universal_tagset',
    'maxent_ne_chunker',
    'words',
    'maxent_treebank_pos_tagger',
    'treebank'
]

for dataset in datasets:
    print(f"Downloading {dataset}...")
    try:
        nltk.download(dataset, download_dir='nltk_data_download', quiet=False)
        print(f"Successfully downloaded {dataset}")
    except Exception as e:
        print(f"Error downloading {dataset}: {e}")
        # Continue even if one dataset fails
        continue

print("All downloads completed")
EOF

# Create a tarball of the downloaded data
tar -czf nltk_data.tar.gz nltk_data_download/

echo "NLTK data downloaded and packaged in nltk_data.tar.gz"
echo "Please transfer this file to Euler using scp" 