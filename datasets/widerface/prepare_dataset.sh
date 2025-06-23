#!/usr/bin/env bash

echo "🚀 Running full WIDER FACE dataset preparation..."

# Step 1: Run setup
echo "🔧 Step 1: Downloading and organizing..."
bash setup_widerface.sh

# Step 2: Convert annotations
echo "🔄 Step 2: Converting annotations..."
python3 convert_labels.py

# Step 3: Cleanup
echo "🧹 Cleaning up annotation files and temp folders..."
rm -rf wider_face_split
rm -f WIDER_train.zip WIDER_val.zip wider_face_split.zip
rm -rf WIDER_train WIDER_val

echo "🎉 All done! Ready for YOLOv8 training."
