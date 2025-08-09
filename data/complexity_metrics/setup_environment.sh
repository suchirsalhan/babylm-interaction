#!/bin/bash

# Set up Python environment
echo "Setting up Python environment..."
module load stack/2024-06
module load gcc/12.2.0
module load python/3.10.13

# Create and activate virtual environment
VENV_DIR="$SCRATCH/complexity_project/venv"
echo "Creating virtual environment at $VENV_DIR"
python -m venv $VENV_DIR
source $VENV_DIR/bin/activate

# Install base dependencies first
echo "Installing base dependencies..."
python -m pip install --upgrade pip
python -m pip install nltk pandas numpy spacy textstat taaled cefrpy networkx requests
python -m pip install scikit-learn scipy matplotlib seaborn
python -m pip install inflect truecase
python -m pip install pyarrow fastparquet
python -m pip install tqdm  # For progress bars in batch processing

# Set up NLTK data directory in the virtual environment
echo "Setting up NLTK data..."
NLTK_DATA_DIR="$VENV_DIR/nltk_data"
mkdir -p "$NLTK_DATA_DIR"

# Extract NLTK data from the tar file
if [ -f "nltk_data.tar.gz" ]; then
    echo "Extracting pre-downloaded NLTK data..."
    tar -xzf nltk_data.tar.gz
    
    # Create the correct directory structure
    mkdir -p "$NLTK_DATA_DIR/corpora"
    mkdir -p "$NLTK_DATA_DIR/tokenizers"
    mkdir -p "$NLTK_DATA_DIR/taggers"
    
    echo "Moving files to correct locations..."
    
    # Function to safely move NLTK data
    move_nltk_data() {
        local src="$1"
        local dest="$2"
        if [ -d "$dest" ]; then
            echo "Removing existing directory: $dest"
            rm -rf "$dest"
        fi
        echo "Moving $src to $dest"
        mv "$src" "$dest"
    }
    
    # Extract and move corpora
    for corpus in brown omw-1.4 stopwords treebank wordnet_ic wordnet words; do
        if [ -f "nltk_data_download/corpora/$corpus.zip" ]; then
            echo "Extracting $corpus.zip..."
            (cd nltk_data_download/corpora && unzip -q "$corpus.zip")
            move_nltk_data "nltk_data_download/corpora/$corpus" "$NLTK_DATA_DIR/corpora/$corpus"
        elif [ -d "nltk_data_download/corpora/$corpus" ]; then
            move_nltk_data "nltk_data_download/corpora/$corpus" "$NLTK_DATA_DIR/corpora/$corpus"
        fi
    done
    
    # Extract and move tokenizers
    for tokenizer in punkt_tab punkt; do
        if [ -f "nltk_data_download/tokenizers/$tokenizer.zip" ]; then
            echo "Extracting $tokenizer.zip..."
            (cd nltk_data_download/tokenizers && unzip -q "$tokenizer.zip")
            move_nltk_data "nltk_data_download/tokenizers/$tokenizer" "$NLTK_DATA_DIR/tokenizers/$tokenizer"
        elif [ -d "nltk_data_download/tokenizers/$tokenizer" ]; then
            move_nltk_data "nltk_data_download/tokenizers/$tokenizer" "$NLTK_DATA_DIR/tokenizers/$tokenizer"
        fi
    done
    
    # Extract and move taggers
    for tagger in averaged_perceptron_tagger averaged_perceptron_tagger_eng maxent_treebank_pos_tagger universal_tagset; do
        if [ -f "nltk_data_download/taggers/$tagger.zip" ]; then
            echo "Extracting $tagger.zip..."
            (cd nltk_data_download/taggers && unzip -q "$tagger.zip")
            move_nltk_data "nltk_data_download/taggers/$tagger" "$NLTK_DATA_DIR/taggers/$tagger"
        elif [ -d "nltk_data_download/taggers/$tagger" ]; then
            move_nltk_data "nltk_data_download/taggers/$tagger" "$NLTK_DATA_DIR/taggers/$tagger"
        fi
    done
    
    # Clean up
    rm -rf nltk_data_download
    
    # Set up NLTK data path
    echo "Setting up NLTK data path..."
    mkdir -p "$VENV_DIR/bin/activate.d"
    echo "export NLTK_DATA=$NLTK_DATA_DIR" > "$VENV_DIR/bin/activate.d/nltk.sh"
    chmod +x "$VENV_DIR/bin/activate.d/nltk.sh"
    
    # Set for current session
    export NLTK_DATA="$NLTK_DATA_DIR"
    
    # Create sitecustomize.py
    SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
    echo "Creating sitecustomize.py..."
    cat > "$SITE_PACKAGES/sitecustomize.py" << EOF
import nltk
nltk.data.path.insert(0, "$NLTK_DATA_DIR")
EOF
else
    echo "Error: nltk_data.tar.gz not found. Please run download_nltk_local.sh on  local machine first."
    exit 1
fi

# List the contents of NLTK data directory to verify
echo "Contents of NLTK data directory:"
ls -R "$NLTK_DATA_DIR"

# Verify NLTK data installation
echo "Verifying NLTK data installation..."
python -c "
import nltk
import os
print('NLTK data path:', nltk.data.path)
print('NLTK_DATA environment variable:', os.getenv('NLTK_DATA'))
print('\nChecking for specific resources:')
resources = [
    'tokenizers/punkt',
    'corpora/wordnet',
    'corpora/omw-1.4',
    'taggers/averaged_perceptron_tagger',
    'taggers/averaged_perceptron_tagger_eng',
    'corpora/wordnet_ic',
    'corpora/brown',
    'corpora/stopwords'
]
for resource in resources:
    try:
        path = nltk.data.find(resource)
        print(f'Found {resource} at: {path}')
    except LookupError as e:
        print(f'Error: Could not find {resource}')
        exit(1)
print('\nAll NLTK data packages verified successfully')
"

# Install spaCy model
echo "Installing spaCy model..."
python -m spacy download en_core_web_sm

# Install the local sentence_concreteness package
if [ -d "sentence_concreteness" ]; then
    echo "Installing local package: sentence_concreteness"
    cd sentence_concreteness
    python -m pip install -e .
    cd ..
else
    echo "Warning: sentence_concreteness directory not found."
fi

# Extract and install local entity recognition package
if [ -f "local_entity_recognition.tar.gz" ]; then
    echo "Extracting local entity recognition package..."
    tar -xzf local_entity_recognition.tar.gz
    
    if [ -d "local_entity_recognition" ]; then
        echo "Setting up local entity recognition environment..."
        
        # Create local entity recognition data directory in the virtual environment
        LOCAL_ENTITY_DATA_DIR="$VENV_DIR/local_entity_data"
        mkdir -p "$LOCAL_ENTITY_DATA_DIR"
        
        # Remove existing local_entity_recognition directory if it exists
        if [ -d "$LOCAL_ENTITY_DATA_DIR/local_entity_recognition" ]; then
            echo "Removing existing local_entity_recognition directory..."
            rm -rf "$LOCAL_ENTITY_DATA_DIR/local_entity_recognition"
        fi
        
        # Move the package to the data directory
        echo "Moving local entity recognition to data directory..."
        mv local_entity_recognition "$LOCAL_ENTITY_DATA_DIR/"
        
        # Install the package from the data directory
        echo "Installing local package: local_entity_recognition"
        cd "$LOCAL_ENTITY_DATA_DIR/local_entity_recognition"
        python -m pip install -e .
        cd "$SCRATCH/complexity_project"
        
        # Set up local entity recognition data path
        echo "Setting up local entity recognition data path..."
        mkdir -p "$VENV_DIR/bin/activate.d"
        echo "export LOCAL_ENTITY_DATA=$LOCAL_ENTITY_DATA_DIR" >> "$VENV_DIR/bin/activate.d/nltk.sh"
        
        # Set for current session
        export LOCAL_ENTITY_DATA="$LOCAL_ENTITY_DATA_DIR"
        
        # Create cache directories for batch processing
        echo "Setting up cache directories for batch processing..."
        mkdir -p "$SCRATCH/dbpedia_cache"
        mkdir -p "$SCRATCH/complexity_cache"
        
        # Set cache environment variables
        export DBPEDIA_CACHE_DIR="$SCRATCH/dbpedia_cache"
        export COMPLEXITY_CACHE_DIR="$SCRATCH/complexity_cache"
        
        # Create sitecustomize.py for local entity recognition
        SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
        echo "Updating sitecustomize.py for local entity recognition..."
        cat >> "$SITE_PACKAGES/sitecustomize.py" << EOF

# Local entity recognition setup
import sys
import os
local_entity_path = os.getenv('LOCAL_ENTITY_DATA', '$LOCAL_ENTITY_DATA_DIR')
if local_entity_path and os.path.exists(local_entity_path):
    sys.path.insert(0, os.path.join(local_entity_path, 'local_entity_recognition'))
EOF
        
        echo "SUCCESS: Local entity recognition system installed"
        
        # List the contents of local entity recognition directory to verify
        echo "Contents of local entity recognition directory:"
        ls -R "$LOCAL_ENTITY_DATA_DIR"
        
        # Verify local entity recognition installation
        echo "Verifying local entity recognition installation..."
        python -c "
import os
import sys

print('Local entity data path:', os.getenv('LOCAL_ENTITY_DATA', '$LOCAL_ENTITY_DATA_DIR'))
print('Python path includes local entity recognition:', any('local_entity_recognition' in p for p in sys.path))

try:
    from local_entity_recognition import DBpediaRecognizer
    print('SUCCESS: DBpediaRecognizer imported successfully')
    
    recognizer = DBpediaRecognizer(
        cache_ttl_hours=24,
        max_retries=3,
        request_timeout=60,  # Increased timeout for batch processing
        confidence_threshold=0.5
    )
    print('SUCCESS: DBpediaRecognizer instantiated successfully')
    
    # Test basic entity extraction
    test_text = 'Barack Obama attended Harvard University.'
    entity_matches = recognizer.extract_entities(test_text)
    print(f'SUCCESS: Entity extraction working: {len(entity_matches)} entities found')
    
    # Test batch processing capabilities
    test_texts = ['Barack Obama attended Harvard University.', 'The Eiffel Tower is in Paris.']
    batch_results = recognizer.extract_entities_batch(test_texts, batch_size=2)
    print(f'SUCCESS: Batch processing working: {len(batch_results)} batches processed')
    
    # Test offline mode
    offline_results = recognizer.extract_entities_offline(test_text)
    print(f'SUCCESS: Offline mode working: {len(offline_results)} entities found')
    
    # Test connectivity (but don't fail if network is down)
    try:
        connectivity = recognizer.test_connectivity()
        print(f'SUCCESS: Connectivity test completed - SPARQL: {connectivity[\"sparql_accessible\"]}, Spotlight: {connectivity[\"spotlight_accessible\"]}')
    except Exception as e:
        print(f'WARNING: Connectivity test failed (this is OK for offline processing): {e}')
    
    print('SUCCESS: All DBpedia entity recognition components verified successfully')
except Exception as e:
    print(f'ERROR: Error verifying DBpedia entity recognition: {e}')
    exit(1)
"
    else
        echo "Warning: local_entity_recognition directory not found in package."
    fi
else
    echo "Warning: local_entity_recognition.tar.gz not found. DBpedia entity recognition not installed."
fi

echo "Setup completed." 