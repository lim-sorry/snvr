#!/bin/bash

# Create the destination directory if it doesn't exist
mkdir -p data/imagenet100/train

# Loop through each source directory
for i in {1..4}; do
    src_dir="data/imagenet100/train.X$i"
    
    # Check if the source directory exists
    if [ -d "$src_dir" ]; then
        # Copy the contents of the source directory to the destination directory
        cp -r "$src_dir"/* data/imagenet100/train/
    else
        echo "Directory $src_dir does not exist."
    fi
done

echo "Data combined into data/train."