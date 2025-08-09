#!/bin/bash

# Create a directory for NLTK data
mkdir -p nltk_data

# Download NLTK data using Python
python3 - << 'EOF'
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required NLTK data
nltk.download('punkt', download_dir='nltk_data')
nltk.download('wordnet', download_dir='nltk_data')
nltk.download('omw-1.4', download_dir='nltk_data')
nltk.download('averaged_perceptron_tagger', download_dir='nltk_data')
nltk.download('wordnet_ic', download_dir='nltk_data')
nltk.download('brown', download_dir='nltk_data')
nltk.download('stopwords', download_dir='nltk_data')
EOF

echo "NLTK data downloaded successfully to nltk_data directory" 