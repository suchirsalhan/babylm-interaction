#!/bin/bash

# Package local entity recognition for cluster transfer
echo "Packaging local entity recognition for cluster transfer..."

# First, create the package if it doesn't exist
if [ ! -d "local_entity_recognition" ]; then
    echo "Creating local entity recognition package..."
    chmod +x create_local_entity_recognition.sh
    ./create_local_entity_recognition.sh
fi

# Create tar.gz file
echo "Creating local_entity_recognition.tar.gz..."
tar -czf local_entity_recognition.tar.gz local_entity_recognition/

# Show the package details
echo "✅ Package created successfully!"
echo "File: local_entity_recognition.tar.gz"
echo "Size: $(du -h local_entity_recognition.tar.gz | cut -f1)"
echo ""
echo "📋 Transfer Instructions:"
echo "1. Upload local_entity_recognition.tar.gz to your cluster"
echo "2. Extract it on the cluster: tar -xzf local_entity_recognition.tar.gz"
echo "3. Run setup_environment.sh on the cluster"
echo ""
echo "📁 Package contents:"
ls -la local_entity_recognition/ 