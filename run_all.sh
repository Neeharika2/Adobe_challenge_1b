#!/bin/bash
set -e

if [ -z "$1" ]; then
    read -p "Enter the name of the collection folder (default: collections1): " user_input
    COLLECTION_FOLDER=${user_input:-collections1}
else
    COLLECTION_FOLDER=$1
fi

echo "Processing collection folder: $COLLECTION_FOLDER"
echo "============================================"

echo "Step 1: Running create_json_sections.py..."
python create_json_sections.py "$COLLECTION_FOLDER"

if [ $? -eq 0 ]; then
    echo "✅ Successfully created JSON sections"
    echo ""
    echo "Step 2: Running json_search_processor.py..."
    python json_search_processor.py "$COLLECTION_FOLDER"
    
    if [ $? -eq 0 ]; then
        echo "✅ Both scripts completed successfully!"
    else
        echo "❌ json_search_processor.py failed"
        exit 1
    fi
else
    echo "❌ create_json_sections.py failed. Exiting."
    exit 1
fi
